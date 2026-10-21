# -*- coding: utf-8 -*-
"""Statistics table, summary table and CSV writers.

Tables are written with ``arcpy.da.NumPyArrayToTable`` - a data-access call,
not a geoprocessing tool.  That distinction is the whole point: inside a large
ArcGIS Pro project every geoprocessing-tool call (``CreateTable``,
``AddFields``, ``CopyRows``, ``Delete``...) was measured at 20-35 seconds
regardless of what it did, which turned a 6 s run into 25 minutes.  The
``arcpy.da`` functions bypass the geoprocessing framework entirely.

``NumPyArrayToTable`` cannot write Null, so NaN is written and then replaced
with Null in one data-access pass; without that ArcGIS shows NaN as -1.#IND
and every downstream statistic breaks.
"""

from __future__ import annotations

import csv
import datetime
import os
import re

import numpy as np

__all__ = ["STATION_FIELDS", "SUMMARY_FIELDS", "safe_prefix", "write_table",
           "write_station_table", "write_summary_table", "write_csv"]


#: Long format: one row per (baseline, part, raster, station).  Wide format would
#: need a column block per raster, which breaks as soon as several rasters are
#: profiled and makes the table unusable for joins and charts.
STATION_FIELDS = [
    ("LINE_OID", "LONG", None),
    ("PART_ID", "LONG", None),
    ("LINE_NAME", "TEXT", 64),
    ("RASTER", "TEXT", 128),
    ("STATION", "LONG", None),
    ("DIST", "DOUBLE", None),
    ("X", "DOUBLE", None),
    ("Y", "DOUBLE", None),
    ("Z_MIN", "DOUBLE", None),
    ("Z_MAX", "DOUBLE", None),
    ("Z_MEAN", "DOUBLE", None),
    ("Z_MEDIAN", "DOUBLE", None),
    ("Z_Q1", "DOUBLE", None),
    ("Z_Q3", "DOUBLE", None),
    ("Z_PLO", "DOUBLE", None),
    ("Z_PHI", "DOUBLE", None),
    ("Z_CENTRE", "DOUBLE", None),
    ("Z_STD", "DOUBLE", None),
    ("RELIEF", "DOUBLE", None),
    ("HI", "DOUBLE", None),
    ("THI", "DOUBLE", None),
    ("THI_RAW", "DOUBLE", None),
    ("THI_STAR", "DOUBLE", None),
    ("N_VALID", "LONG", None),
    ("N_TOTAL", "LONG", None),
    ("KAPPA", "DOUBLE", None),
    ("FOLDED", "SHORT", None),
]

SUMMARY_FIELDS = [
    ("LINE_OID", "LONG", None),
    ("PART_ID", "LONG", None),
    ("LINE_NAME", "TEXT", 64),
    ("RASTER", "TEXT", 128),
    ("LENGTH", "DOUBLE", None),
    ("N_STATION", "LONG", None),
    ("N_LINES", "LONG", None),
    ("SWATH_W", "DOUBLE", None),
    ("STEP_REQ", "DOUBLE", None),
    ("STEP_ACT", "DOUBLE", None),
    ("PCT_LO", "DOUBLE", None),
    ("PCT_HI", "DOUBLE", None),
    ("MAX_RELIEF", "DOUBLE", None),
    ("MEAN_HI", "DOUBLE", None),
    ("MEAN_THI", "DOUBLE", None),
    ("MIN_RADIUS", "DOUBLE", None),
    ("PCT_NODATA", "DOUBLE", None),
    ("N_FOLDED", "LONG", None),
    # Percent of the swath band's area the sampling mask kept; Null without a
    # mask.  PCT_NODATA counts masked samples as missing, as it does folded ones.
    ("MASK_KEPT", "DOUBLE", None),
    ("METHOD", "TEXT", 32),
    ("CRS_NAME", "TEXT", 128),
    ("TOOL_VER", "TEXT", 32),
    ("RUN_UTC", "TEXT", 32),
    ("ELAPSED_S", "DOUBLE", None),
    ("PARAMS", "TEXT", 4000),
]


def safe_prefix(name: str, max_len: int = 20) -> str:
    """A raster name reduced to a field-name-safe prefix."""
    base = os.path.splitext(os.path.basename(str(name)))[0]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_") or "R"
    if safe[0].isdigit():
        safe = "R_" + safe
    return safe[:max_len].upper()


# ---------------------------------------------------------------------------
# numpy-backed table writing (no geoprocessing calls)
# ---------------------------------------------------------------------------

def _np_dtype(fields):
    out = []
    for name, ftype, length in fields:
        if ftype == "TEXT":
            out.append((name, f"<U{int(length or 255)}"))
        elif ftype == "LONG":
            out.append((name, "<i4"))
        elif ftype == "SHORT":
            out.append((name, "<i2"))
        else:
            out.append((name, "<f8"))
    return out


def _coerce(value, ftype):
    if ftype == "TEXT":
        return "" if value is None else str(value)
    if ftype in ("LONG", "SHORT"):
        return 0 if value is None else int(value)
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def rows_to_array(fields, rows):
    """Build the structured array ``NumPyArrayToTable`` wants."""
    arr = np.zeros(len(rows), dtype=_np_dtype(fields))
    types = [f[1] for f in fields]
    for i, row in enumerate(rows):
        arr[i] = tuple(_coerce(v, t) for v, t in zip(row, types))
    return arr


def nullify_nan(path, double_fields):
    """Replace NaN in *double_fields* with Null, in one data-access pass."""
    import arcpy

    if not double_fields:
        return 0
    n = 0
    with arcpy.da.UpdateCursor(path, list(double_fields)) as cur:
        for row in cur:
            if any(v is not None and v != v for v in row):      # NaN != NaN
                cur.updateRow([None if (v is not None and v != v) else v
                               for v in row])
                n += 1
    return n


def _is_lock_error(exc) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("000496", "being edited", "lock", "schema"))


def _write_via_memory(path, fields, rows):
    """Fallback for a geodatabase inside an edit session.

    Creating a dataset by copy is allowed in an edited workspace; modifying a
    schema in place is not.  Costs four geoprocessing calls, so it is only used
    when the direct write fails.
    """
    import uuid

    import arcpy

    tbl = arcpy.management.CreateTable("memory", f"swath_{uuid.uuid4().hex}")[0]
    arcpy.management.AddFields(
        tbl, [[n, t, n, (ln if ln else None)] for n, t, ln in fields])
    names = [f[0] for f in fields]
    types = [f[1] for f in fields]
    with arcpy.da.InsertCursor(tbl, names) as cur:
        for row in rows:
            cur.insertRow([None if (t == "DOUBLE" and (v is None or v != v))
                           else v for v, t in zip(row, types)])
    arcpy.management.CopyRows(tbl, path)
    try:
        arcpy.management.Delete(tbl)
    except Exception:
        pass
    return path


def write_table(path, fields, rows, messenger=None):
    """Write *rows* (tuples in *fields* order) to a geodatabase table."""
    import arcpy

    arr = rows_to_array(fields, rows)
    try:
        arcpy.da.NumPyArrayToTable(arr, path)
        nullify_nan(path, [f[0] for f in fields if f[1] == "DOUBLE"])
    except Exception as exc:
        if messenger:
            messenger.info(f"  (direct write failed - {type(exc).__name__}: "
                           f"{str(exc).splitlines()[0][:120]}; using the "
                           "in-memory route)")
        _write_via_memory(path, fields, rows)
    return path


def _null(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not np.isfinite(f) else f


def _opt(arr, j):
    return None if arr is None else _null(arr[j])


def write_station_table(path, result, messenger=None):
    """Write the per-station statistics table."""
    import time

    t0 = time.perf_counter()
    rows = []
    for oid, part_id in result.parts():
        g = result.geometry[(oid, part_id)]
        c, kappa = g["centreline"], g["kappa"]
        folded = g["folded"]
        fold_any = (folded.any(axis=0) if folded is not None
                    else np.zeros(len(c), dtype=bool))
        line_name = str(result.names.get((oid, part_id), ""))[:64]
        for raster_name, s in result.for_part(oid, part_id).items():
            for j in range(s.n_stations):
                rows.append((
                    oid, part_id, line_name, raster_name, j,
                    _null(s.distances[j]), _null(c[j, 0]), _null(c[j, 1]),
                    _null(s.z_min[j]), _null(s.z_max[j]), _null(s.z_mean[j]),
                    _null(s.z_median[j]), _null(s.z_q1[j]), _null(s.z_q3[j]),
                    _opt(s.z_plo, j), _opt(s.z_phi, j), _opt(s.z_centre, j),
                    _null(s.z_std[j]), _null(s.relief[j]), _null(s.hi[j]),
                    _null(s.thi[j]), _null(s.thi_raw[j]), _null(s.thi_star[j]),
                    int(s.n_valid[j]), int(s.n_total),
                    _null(kappa[j]), int(bool(fold_any[j])),
                ))
    write_table(path, STATION_FIELDS, rows, messenger)
    if messenger:
        messenger.info(f"  Statistics table ({len(rows):,} rows) -> {path} "
                       f"({time.perf_counter() - t0:.1f} s)")
    return path


def write_summary_table(path, result, messenger=None):
    """One row per baseline part per raster, including a provenance blob."""
    cfg = result.config
    run_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    crs_name = getattr(result.spatial_reference, "name", "") or ""
    params = cfg.to_json()[:4000]

    rows = []
    for oid, part_id in result.parts():
        g = result.geometry[(oid, part_id)]
        kmax = float(np.max(np.abs(g["kappa"]))) if len(g["kappa"]) else 0.0
        min_radius = (1.0 / kmax) if kmax > 0 else None
        folded = g["folded"]
        n_folded = int(np.count_nonzero(folded.any(axis=0))) if folded is not None else 0
        length = float(g["distances"][-1])
        line_name = str(result.names.get((oid, part_id), ""))[:64]
        for raster_name, s in result.for_part(oid, part_id).items():
            total = s.n_stations * max(s.n_total, 1)
            missing = total - int(np.sum(s.n_valid))
            rows.append((
                oid, part_id, line_name, raster_name, length, s.n_stations,
                int(g.get("n_lines", cfg.n_lines if cfg.is_offset else 0)),
                float(g.get("width", cfg.swath_width)), cfg.step, g["step_actual"],
                cfg.pct_low, cfg.pct_high,
                _null(s.max_relief),
                _null(np.nanmean(s.hi) if np.isfinite(s.hi).any() else np.nan),
                _null(np.nanmean(s.thi) if np.isfinite(s.thi).any() else np.nan),
                _null(min_radius),
                _null(100.0 * missing / total if total else np.nan),
                n_folded, _null(g.get("mask_kept")), cfg.method, crs_name,
                cfg.tool_version,
                run_utc, _null(result.elapsed), params,
            ))
    write_table(path, SUMMARY_FIELDS, rows, messenger)
    if messenger:
        messenger.info(f"  Summary table -> {path}")
    return path


def write_csv(folder, result, messenger=None):
    """One CSV per baseline part, wide across rasters where possible.

    NaN is written as an empty field, never the token ``nan``: Excel imports the
    latter as text, which silently turns the whole column into strings.
    """
    os.makedirs(folder, exist_ok=True)
    written = []

    for oid, part_id in result.parts():
        g = result.geometry[(oid, part_id)]
        by_raster = result.for_part(oid, part_id)
        c = g["centreline"]
        line_name = str(result.names.get((oid, part_id), ""))
        stem = safe_prefix(line_name, 32).lower() if line_name else f"L{oid}"
        path = os.path.join(folder, f"swath_{stem}_p{part_id}.csv")

        header = ["STATION", "DIST", "X", "Y", "KAPPA"]
        cols = [None] * 5
        for rname, s in by_raster.items():
            safe = safe_prefix(rname)
            for label, arr in (("MIN", s.z_min), ("MAX", s.z_max), ("MEAN", s.z_mean),
                               ("MEDIAN", s.z_median), ("Q1", s.z_q1), ("Q3", s.z_q3),
                               ("PLO", s.z_plo), ("PHI", s.z_phi),
                               ("CENTRE", s.z_centre),
                               ("STD", s.z_std), ("RELIEF", s.relief), ("HI", s.hi),
                               ("THI", s.thi), ("THI_STAR", s.thi_star),
                               ("N_VALID", s.n_valid)):
                if arr is None:
                    continue
                header.append(f"{safe}_{label}")
                cols.append(arr)

        any_stats = next(iter(by_raster.values()))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for j in range(any_stats.n_stations):
                row = [j,
                       f"{any_stats.distances[j]:.4f}",
                       f"{c[j, 0]:.6f}", f"{c[j, 1]:.6f}",
                       f"{g['kappa'][j]:.10f}"]
                for arr in cols[5:]:
                    v = arr[j]
                    if isinstance(v, (np.integer, int)):
                        row.append(int(v))
                    else:
                        row.append("" if not np.isfinite(v) else f"{float(v):.4f}")
                w.writerow(row)
        written.append(path)

    if messenger:
        messenger.info(f"  {len(written)} CSV file(s) -> {folder}")
    return written


# The native Pro chart lives on the station-points layer and is written as CIM
# JSON into its layer file (see ``cimchart`` and ``lyrx``).  ``arcpy.charts`` is
# deliberately not used anywhere in this package: it raised an access violation
# in testing, which inside Pro would end the user's session.
