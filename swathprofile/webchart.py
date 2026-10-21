# -*- coding: utf-8 -*-
"""The browser chart: one self-contained HTML page per run.  No arcpy.

The chart window (``viewer/swath_viewer.py``) is a separate Python program,
and on a computer where security software lets only approved programs run it
can be blocked.  This page needs no Python at all: it opens in any modern
browser, works offline from a file on disk, and can be sent to someone without
ArcGIS.

Everything lives in one file.  ``templates/webchart.html`` carries the CSS and
the JavaScript (no libraries, no network: nothing to download or approve), and
:func:`render` injects the run's data as one JSON block.  Arrays travel as
base64 of little-endian floats, which keeps NaN - NoData - intact and is about
a third of the size of decimal text: float64 for distances and offsets, which
are coordinates, float32 for elevations and statistics.  Side colours, side
labels and the footer are computed here by the same functions the saved charts
use, so the page cannot disagree with them.

Size.  The per-line samples (they draw the individual profile lines and the
cross-sections) are capped at :data:`MAX_SAMPLES`: if every raster's fit they
are all kept, otherwise only the THi raster's are, profile by profile in
order until the budget is used.  Statistics are always kept.  A page over
:data:`PAGE_LIMIT` is not written at all, and pages over :data:`EMAIL_HINT`
are reported as too big for most email.
"""

from __future__ import annotations

import base64
import datetime
import html
import json
import os
import re

import numpy as np

from . import __version__

__all__ = ["MAX_SAMPLES", "EMAIL_HINT", "PAGE_LIMIT", "SCHEMA", "TEMPLATE",
           "build_payload", "render", "write_web_chart", "extract_payload", "decode"]

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates",
                        "webchart.html")
_DATA_SLOT = "/*__SWATH_DATA__*/"
_TITLE_SLOT = "__SWATH_TITLE__"
SCHEMA = 2

#: Per-line samples kept in one page: 16 MB of float32, about 21 MB of text.
MAX_SAMPLES = 4_000_000
#: Above this many bytes a page is too big for most email.
EMAIL_HINT = 20_000_000
#: A page larger than this is not written: browsers open it slowly or not at all.
PAGE_LIMIT = 150_000_000

#: Per-station series sent for every raster: what the page draws or reads out.
#: Relief is max - min, recomputed in the page.  n_valid is an integer count,
#: exact in float32 below 2**24.
_SERIES = ("z_min", "z_max", "z_mean", "z_median", "z_q1", "z_q3", "z_plo", "z_phi",
           "z_centre", "thi", "n_valid")


def _b64(arr, dtype="<f4"):
    """Base64 of an array as little-endian floats, or None."""
    if arr is None:
        return None
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    return base64.b64encode(a.tobytes()).decode("ascii")


def decode(b64, dtype="<f4"):
    """The inverse of the encoding used in the page (for tests and tools)."""
    return None if b64 is None else np.frombuffer(base64.b64decode(b64), dtype=dtype)


def _finite(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _hexes(colours):
    from matplotlib.colors import to_hex

    return [to_hex(c) for c in colours]


def _lines_kept(result, budget):
    """Which (part, raster) per-line samples go in the page, and the rule used."""
    sizes = {}
    for oid, part_id in result.parts():
        for rname, s in result.for_part(oid, part_id).items():
            if s.values is not None:
                sizes[(oid, part_id, rname)] = int(s.values.size)
    total = sum(sizes.values())
    if total <= budget:
        return set(sizes), "all", total, total
    keep, used = set(), 0
    for oid, part_id in result.parts():
        names = list(result.for_part(oid, part_id))
        rname = result.thi_raster if result.thi_raster in names else names[0]
        n = sizes.get((oid, part_id, rname))
        if n is not None and used + n <= budget:
            keep.add((oid, part_id, rname))
            used += n
    return keep, "thi", total, used


def build_payload(result, title: str = "", max_samples: int = MAX_SAMPLES,
                  created=None) -> dict:
    """Everything the page draws, as a JSON-ready dict (see the module notes)."""
    from .geometry import side_of
    from .plotting import PALETTE, chart_footer, profile_colours, side_labels

    cfg = result.config
    cfg_dict = json.loads(cfg.to_json()) if cfg is not None else {}
    keep, rule, total, kept = _lines_kept(result, int(max_samples))
    left_out, parts = [], []
    for oid, part_id in result.parts():
        g = result.geometry.get((oid, part_id), {})
        by_raster = result.for_part(oid, part_id)
        first = next(iter(by_raster.values()))
        name = result.names.get((oid, part_id)) or f"OID {oid} part {part_id}"
        offsets = first.offsets if first.values is not None else None
        had_lines = offsets is not None
        kept_here = had_lines and any((oid, part_id, r) in keep for r in by_raster)
        if had_lines and not kept_here:
            left_out.append(name)
        part = {
            "key": f"{oid}:{part_id}", "name": str(name), "oid": int(oid),
            "part_id": int(part_id), "n_stations": int(first.n_stations),
            "width": _finite(g.get("width")), "n_lines": int(g.get("n_lines") or 0),
            "method": str(g.get("method") or ""),
            "step": _finite(cfg.step) if cfg is not None else None,
            "spacing": _finite(g.get("step_actual")),
            "mask_kept": _finite(g.get("mask_kept")),
            "has_data": any(int(np.sum(s.n_valid)) > 0 for s in by_raster.values()),
            "footer": chart_footer(cfg, g) if cfg is not None else "",
            "side_labels": list(side_labels(first)),
            "distances": _b64(first.distances, "<f8"),
            # "included", "omitted" (size limit) or "none" (All cells mode).
            "lines": "included" if kept_here else ("omitted" if had_lines else "none"),
            "n_line_rows": int(len(offsets)) if kept_here else 0,
            "offsets": _b64(offsets, "<f8") if kept_here else None,
            "sides": [int(v) for v in side_of(offsets)] if kept_here else None,
            "colours": ({"side": _hexes(profile_colours(offsets, "side")),
                         "ramp": _hexes(profile_colours(offsets, "ramp"))}
                        if kept_here else None),
            "rasters": [],
        }
        for rname, s in by_raster.items():
            r = {"name": str(rname), "n_total": int(s.n_total),
                 "max_relief": _finite(s.max_relief)}
            for key in _SERIES:
                r[key] = _b64(getattr(s, key, None))
            r["values"] = (_b64(s.values) if (oid, part_id, rname) in keep else None)
            part["rasters"].append(r)
        parts.append(part)

    notes = []
    if rule == "thi" and len(result.raster_names) > 1:
        notes.append(f"To keep this page a manageable size, profile lines and "
                     f"cross-sections are shown for {result.thi_raster} only.")
    if left_out:
        shown = ", ".join(left_out[:8]) + (", ..." if len(left_out) > 8 else "")
        notes.append(f"Profile lines and cross-sections are left out for "
                     f"{len(left_out)} profile(s) to keep this page a manageable size: "
                     f"{shown}. Their envelopes and statistics are all here.")
    pct = (cfg.pct_low, cfg.pct_high) if cfg is not None else (5.0, 95.0)
    return {
        "schema": SCHEMA,
        "generator": f"Swath Profile for ArcGIS Pro {__version__}",
        "title": str(title),
        "created": created or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tool_version": (getattr(cfg, "tool_version", "") or __version__),
        "crs": str(getattr(result.spatial_reference, "name", "") or ""),
        "pct": [float(pct[0]), float(pct[1])],
        "thi_raster": str(result.thi_raster or ""),
        "raster_names": [str(n) for n in result.raster_names],
        "elapsed": _finite(result.elapsed),
        "config": cfg_dict,
        "palette": dict(PALETTE),
        "samples": {"budget": int(max_samples), "total": total, "kept": kept,
                    "rule": rule},
        "lines_note": " ".join(notes),
        "parts": parts,
    }


def render(payload: dict) -> str:
    """The page for *payload*: the template with the data and title filled in.

    Both slots are filled in one pass, so nothing in the title can be mistaken
    for the data slot.  Every ``<`` in the JSON is written as ``\\u003c``, so no
    profile name can close the data block or open markup, whatever it holds.
    """
    with open(TEMPLATE, encoding="utf-8") as fh:
        template = fh.read()
    for slot in (_DATA_SLOT, _TITLE_SLOT):
        if template.count(slot) != 1:
            raise ValueError(f"The browser chart template must hold {slot!r} once.")
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).replace("<", "\\u003c")
    title = html.escape(f"Swath profile - {payload.get('title') or 'result'}")
    fill = {_DATA_SLOT: data, _TITLE_SLOT: title}
    pattern = re.compile("|".join(re.escape(s) for s in fill))
    return pattern.sub(lambda m: fill[m.group(0)], template)


def extract_payload(page: str) -> dict:
    """The JSON data block of a rendered page (for tests and tools)."""
    m = re.search(r'<script type="application/json" id="swath-data">(.*?)</script>',
                  page, re.S)
    if m is None:
        raise ValueError("No browser chart data block in this page.")
    return json.loads(m.group(1))


def write_web_chart(path, result, title: str = "", max_samples: int = MAX_SAMPLES,
                    page_limit: int = PAGE_LIMIT, messenger=None) -> dict:
    """Write the browser chart for *result* to *path* (atomically).

    Returns ``{"path", "bytes", "written", "left_out", "note"}``.  A page over
    *page_limit* bytes is not written (``written`` False, ``path`` None).
    """
    payload = build_payload(result, title=title, max_samples=max_samples)
    page = render(payload).encode("utf-8")
    left_out = sum(1 for p in payload["parts"] if p["lines"] == "omitted")
    report = {"path": None, "bytes": len(page), "written": False,
              "left_out": left_out, "note": payload["lines_note"]}
    if len(page) > page_limit:
        return report
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(page)
    os.replace(tmp, path)
    report.update(path=path, written=True)
    if messenger is not None and payload["lines_note"]:
        messenger.info("  " + payload["lines_note"])
    return report
