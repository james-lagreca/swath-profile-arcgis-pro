# -*- coding: utf-8 -*-
"""Serialise a run to a ``.swath.npz`` bundle for the interactive viewer.

The viewer runs in a separate process (see :mod:`viewer.swath_viewer`), so the
results have to cross a process boundary.  A single npz of raw arrays plus a JSON
metadata blob keeps that cheap and keeps the viewer free of any arcpy import.

The *raw* sample arrays are stored, not just the reduced statistics, so the
viewer can re-apply smoothing and show cross-sections without recomputing.

Format version 2 adds the configurable percentiles, per-part names and the
run's elapsed time.  Version-1 files still load; missing arrays come back as
None.
"""

from __future__ import annotations

import json

import numpy as np

from .stats import SwathStats

__all__ = ["save_bundle", "load_bundle", "BUNDLE_SUFFIX", "BUNDLE_VERSION"]

BUNDLE_SUFFIX = ".swath.npz"
BUNDLE_VERSION = 2

_REQUIRED = ("distances", "z_min", "z_max", "z_mean", "z_median", "z_q1",
             "z_q3", "z_std", "relief", "hi", "thi", "thi_raw", "thi_star",
             "n_valid")
_OPTIONAL = ("z_plo", "z_phi", "z_centre")


def _finite(value):
    """JSON cannot hold NaN or numpy scalars; a missing bearing becomes null."""
    if value is None:
        return None
    v = float(value)
    return v if v == v and abs(v) != float("inf") else None


def save_bundle(path, result, title="", elapsed=None):
    """Write *result* to *path*."""
    arrays, entries = {}, []

    for i, (oid, part_id) in enumerate(result.parts()):
        g = result.geometry[(oid, part_id)]
        arrays[f"g{i}_centreline"] = g["centreline"]
        arrays[f"g{i}_kappa"] = g["kappa"]
        arrays[f"g{i}_offsets"] = np.asarray(g["offsets"], dtype=np.float64)

        rasters = []
        for j, (rname, s) in enumerate(result.for_part(oid, part_id).items()):
            for key in _REQUIRED:
                arrays[f"s{i}_{j}_{key}"] = np.asarray(getattr(s, key))
            for key in _OPTIONAL:
                arr = getattr(s, key, None)
                if arr is not None:
                    arrays[f"s{i}_{j}_{key}"] = np.asarray(arr)
            if s.values is not None:
                arrays[f"s{i}_{j}_values"] = s.values.astype(np.float32)
            rasters.append({"name": rname, "index": j,
                            "n_total": int(s.n_total),
                            "max_relief": float(s.max_relief),
                            "has_values": s.values is not None})
        entries.append({"index": i, "oid": int(oid), "part_id": int(part_id),
                        "name": str(result.names.get((oid, part_id), "")),
                        "width": float(g.get("width", 0.0) or 0.0),
                        "n_lines": int(g.get("n_lines", 0) or 0),
                        "method": str(g.get("method", "")),
                        # Which compass direction the +offset side faces, and
                        # how straight the baseline is.  Per part, not per
                        # raster: it is a property of the line, not the DEM.
                        "normal_bearing": _finite(g.get("normal_bearing")),
                        "normal_resultant": _finite(g.get("normal_resultant")),
                        "rasters": rasters})

    cfg = json.loads(result.config.to_json()) if result.config else {}
    meta = {
        "version": BUNDLE_VERSION,
        "title": title,
        "elapsed": None if elapsed is None else float(elapsed),
        "thi_raster": result.thi_raster,
        "raster_names": list(result.raster_names),
        "crs": getattr(result.spatial_reference, "name", "") or "",
        "config": cfg,
        "pct": [cfg.get("pct_low", 5.0), cfg.get("pct_high", 95.0)],
        "parts": entries,
    }
    arrays["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    return path


class LoadedBundle:
    """A bundle read back from disk, in the shape the viewer wants."""

    def __init__(self, meta, parts):
        self.meta = meta
        self.parts = parts                      # list of dicts

    @property
    def title(self):
        return self.meta.get("title", "")

    @property
    def thi_raster(self):
        return self.meta.get("thi_raster", "")

    @property
    def pct(self):
        return tuple(self.meta.get("pct", (5.0, 95.0)))

    def labels(self):
        return [p["name"] or f"OID {p['oid']} part {p['part_id']}"
                for p in self.parts]


def load_bundle(path):
    """Read a bundle written by :func:`save_bundle` (any version)."""
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        meta = json.loads(bytes(z["__meta__"]).decode("utf-8"))
        parts = []
        for entry in meta["parts"]:
            i = entry["index"]
            by_raster = {}
            for r in entry["rasters"]:
                j = r["index"]
                kw = {k: z[f"s{i}_{j}_{k}"] for k in _REQUIRED}
                for k in _OPTIONAL:
                    key = f"s{i}_{j}_{k}"
                    kw[k] = z[key] if key in files else None
                values = z[f"s{i}_{j}_values"].astype(np.float64) \
                    if r.get("has_values") else None
                by_raster[r["name"]] = SwathStats(
                    n_total=r["n_total"], max_relief=r["max_relief"],
                    values=values, offsets=z[f"g{i}_offsets"],
                    name=r["name"],
                    normal_bearing=entry.get("normal_bearing"),
                    normal_resultant=entry.get("normal_resultant"), **kw)
            parts.append({
                "oid": entry["oid"], "part_id": entry["part_id"],
                "name": entry.get("name", ""),
                "width": entry.get("width") or meta.get("config", {}).get("swath_width"),
                "n_lines": entry.get("n_lines") or meta.get("config", {}).get("n_lines"),
                "method": entry.get("method") or meta.get("config", {}).get("method"),
                "centreline": z[f"g{i}_centreline"], "kappa": z[f"g{i}_kappa"],
                "by_raster": by_raster,
            })
    return LoadedBundle(meta, parts)
