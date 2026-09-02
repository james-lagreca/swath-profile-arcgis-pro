# -*- coding: utf-8 -*-
"""Display-time transforms of a swath profile.  Pure numpy - no arcpy, no Tk.

The viewer is a thin Tk shell; everything it does to a profile lives here so it
can be tested against closed forms without a display.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .stats import SwathStats

__all__ = ["Z_SERIES", "detrend_linear", "relative_to_min", "reorient",
           "cross_section", "measure", "vertical_exaggeration"]

#: Every elevation-valued series on a SwathStats.  Relief, HI and THi are
#: invariant under a vertical shift and are left untouched by the transforms.
Z_SERIES = ("z_min", "z_max", "z_mean", "z_median", "z_q1", "z_q3",
            "z_plo", "z_phi", "z_centre")


def _shift(stats: SwathStats, delta: np.ndarray) -> SwathStats:
    """Subtract the per-station vector *delta* from every elevation series."""
    changes = {}
    for key in Z_SERIES:
        arr = getattr(stats, key, None)
        if arr is not None:
            changes[key] = np.asarray(arr, dtype=np.float64) - delta
    if stats.values is not None:
        changes["values"] = stats.values - delta[None, :]
    return dataclasses.replace(stats, **changes)


def detrend_linear(stats: SwathStats, fit_on: str = "z_mean") -> SwathStats:
    """Remove the linear trend fitted to *fit_on* from every elevation series.

    The *same* trend is subtracted from min, max, mean, quartiles and the raw
    samples, so the envelope keeps its shape and only the regional slope goes.
    Returns a new SwathStats; the input is not modified.  A profile with fewer
    than two finite points is returned unchanged.
    """
    d = np.asarray(stats.distances, dtype=np.float64)
    y = np.asarray(getattr(stats, fit_on), dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(y)
    if ok.sum() < 2:
        return stats
    slope, intercept = np.polyfit(d[ok], y[ok], 1)
    trend = slope * d + intercept
    out = _shift(stats, trend)
    return dataclasses.replace(out, name=stats.name)


def relative_to_min(stats: SwathStats) -> SwathStats:
    """Express every series relative to the minimum profile (min becomes 0)."""
    base = np.asarray(stats.z_min, dtype=np.float64)
    base = np.where(np.isfinite(base), base, 0.0)
    return _shift(stats, base)


def reorient(distances, offsets=None, origin: float = 0.0, flip: bool = False):
    """Re-zero and optionally reverse the along-track axis.

    Returns ``(distances', offsets')`` with ``d' = +-(d - origin)``.  Flipping
    also negates the cross-track offsets so a cross-section mirrors to match
    the reversed viewing direction.  Arrays are not reordered - a monotonically
    decreasing x is fine for line plots and keeps station indices stable.
    """
    d = np.asarray(distances, dtype=np.float64) - float(origin)
    o = None if offsets is None else np.asarray(offsets, dtype=np.float64)
    if flip:
        d = -d
        if o is not None:
            o = -o
    return d, o


def cross_section(stats: SwathStats, j: int, flip: bool = False):
    """Elevation across the band at station *j*.

    Returns a dict with ``offsets`` (signed cross-track distance), ``z`` (the
    samples along that cross-section), and the station's ``z_min``, ``z_mean``,
    ``z_max``, or None when the profile carries no per-line samples (zonal
    mode).
    """
    if stats.values is None or stats.offsets is None:
        return None
    j = int(np.clip(j, 0, stats.n_stations - 1))
    offsets = np.asarray(stats.offsets, dtype=np.float64)
    z = np.asarray(stats.values[:, j], dtype=np.float64)
    if flip:
        offsets = -offsets
    return {
        "station": j,
        "distance": float(stats.distances[j]),
        "offsets": offsets,
        "z": z,
        "z_min": float(stats.z_min[j]),
        "z_mean": float(stats.z_mean[j]),
        "z_max": float(stats.z_max[j]),
    }


def measure(p1, p2):
    """Distance, height difference, gradient and angle between two chart points.

    *p1*, *p2* are ``(distance, elevation)`` pairs in the chart's units.
    """
    (d1, z1), (d2, z2) = p1, p2
    dd = float(d2 - d1)
    dz = float(z2 - z1)
    length = math.hypot(dd, dz)
    gradient = (dz / dd) if dd != 0 else float("inf")
    angle = math.degrees(math.atan2(dz, abs(dd))) if dd != 0 else 90.0
    return {
        "dd": dd, "dz": dz, "length": length,
        "gradient": gradient,
        "percent": 100.0 * gradient if math.isfinite(gradient) else float("inf"),
        "angle_deg": angle,
    }


def vertical_exaggeration(ax) -> float:
    """VE of a matplotlib axes: data units per pixel in x over the same in y."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent()
    if bbox.width <= 0 or bbox.height <= 0 or x1 == x0 or y1 == y0:
        return float("nan")
    x_per_px = abs(x1 - x0) / bbox.width
    y_per_px = abs(y1 - y0) / bbox.height
    return x_per_px / y_per_px
