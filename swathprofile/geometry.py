# -*- coding: utf-8 -*-
"""Centreline geometry: resampling, tangents, normals, curvature, offset grids.

Pure numpy - no arcpy.  A baseline arrives as an (V, 2) float64 vertex array and
leaves as an (N, M, 2) grid of sample coordinates plus the diagnostics needed to
tell the user when the swath has folded onto itself.

Conventions
-----------
* Stations are indexed by along-track distance from the start of the line.
* ``normal = (-t_y, t_x)`` is the LEFT normal, so a positive offset lies to the
  left of the digitising direction.
* All coordinates are float64.  This is not stylistic: NZTM2000 and southern
  hemisphere UTM northings reach 5-10 x 10**6 m, which float32 quantises to
  about 1 m.
"""

from __future__ import annotations

import numpy as np

from .errors import GeometryError

__all__ = [
    "clean_vertices",
    "polyline_length",
    "resample_polyline",
    "station_count",
    "tangents_normals",
    "curvature",
    "direction_curvature",
    "offset_grid",
    "fold_mask",
    "transects_along",
    "band_ring",
]


def _as_xy(xy) -> np.ndarray:
    a = np.asarray(xy, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise GeometryError(
            "Baseline vertices must be an (n, 2) array of coordinates; got shape "
            f"{a.shape}."
        )
    if not np.all(np.isfinite(a)):
        raise GeometryError("Baseline contains non-finite vertex coordinates.")
    return a


def clean_vertices(xy, tol: float = 0.0) -> np.ndarray:
    """Drop consecutive duplicate vertices.

    A zero-length segment produces a zero tangent, which poisons the normal and
    therefore every offset point at that station.  ``tol`` is an absolute
    distance; callers normally pass ``max(1e-9 * length, 1e-6 * cell_size)``.
    """
    a = _as_xy(xy)
    if len(a) < 2:
        return a
    step = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))
    keep = np.ones(len(a), dtype=bool)
    keep[1:] = step > tol
    out = a[keep]
    if len(out) < 2:
        raise GeometryError(
            "Baseline collapses to a single point after removing duplicate vertices."
        )
    return out


def polyline_length(xy) -> float:
    a = _as_xy(xy)
    if len(a) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))


def station_count(length: float, step: float, max_stations: int = 200_000) -> int:
    """Number of evenly spaced stations for a line of *length* at *step* spacing.

    Returns at least 2.  The realised spacing is ``length / (M - 1)``, which is
    not exactly *step*; callers report it as STEP_ACTUAL.
    """
    if step <= 0:
        raise GeometryError(f"Step size must be positive; got {step}.")
    if not np.isfinite(length) or length <= 0:
        raise GeometryError(f"Baseline length must be positive; got {length}.")
    m = int(round(length / step)) + 1
    return int(np.clip(m, 2, max_stations))


def resample_polyline(xy, n_stations: int):
    """Resample to *n_stations* points evenly spaced by arc length.

    Mirrors the reference ``_interpolate_polyline``: cumulative chord length plus
    linear interpolation, vectorised over both axes.

    Returns ``(C, s)`` where ``C`` is (M, 2) float64 and ``s`` is the (M,) array
    of along-track distances.
    """
    a = _as_xy(xy)
    if n_stations < 2:
        raise GeometryError(f"Need at least 2 stations; got {n_stations}.")
    seg = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        raise GeometryError("Baseline has zero length.")
    s = np.linspace(0.0, total, n_stations)
    c = np.column_stack([np.interp(s, cum, a[:, 0]), np.interp(s, cum, a[:, 1])])
    return np.ascontiguousarray(c, dtype=np.float64), s


def _moving_average_2d(v: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded moving average applied down axis 0 of an (M, 2) array."""
    if window < 3:
        return v
    if window % 2 == 0:
        window += 1
    if window >= len(v):
        window = len(v) - (1 - len(v) % 2)
        if window < 3:
            return v
    pad = window // 2
    box = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(v)
    for k in range(v.shape[1]):
        padded = np.pad(v[:, k], pad, mode="edge")
        out[:, k] = np.convolve(padded, box, mode="valid")
    return out


def tangents_normals(c: np.ndarray, smooth_window: int = 0):
    """Unit tangents and left normals at every station of an evenly spaced *c*.

    Central differences, one-sided at the ends.  Zero-length tangents (which
    survive only if the caller skipped :func:`clean_vertices`) are filled from
    the nearest valid neighbour rather than left as NaN.

    ``smooth_window`` smooths the *direction*, never the centreline.  Smoothing
    the centreline would change the reported along-track distances and displace
    the profile off the feature the user digitised; smoothing the tangent only
    stabilises the cross-section azimuth, which is the actual problem with
    hand-digitised baselines.
    """
    c = np.asarray(c, dtype=np.float64)
    m = len(c)
    if m < 2:
        raise GeometryError("Need at least 2 stations to compute tangents.")

    t = np.empty_like(c)
    t[1:-1] = c[2:] - c[:-2]
    t[0] = c[1] - c[0]
    t[-1] = c[-1] - c[-2]

    if smooth_window and smooth_window >= 3:
        t = _moving_average_2d(t, int(smooth_window))

    norm = np.hypot(t[:, 0], t[:, 1])
    bad = norm < 1e-12
    if bad.all():
        raise GeometryError("Baseline has zero length; every tangent is degenerate.")
    if bad.any():
        good_idx = np.flatnonzero(~bad)
        pos = np.searchsorted(good_idx, np.flatnonzero(bad)).clip(0, len(good_idx) - 1)
        t[bad] = t[good_idx[pos]]
        norm = np.hypot(t[:, 0], t[:, 1])

    t = t / norm[:, None]
    nrm = np.column_stack([-t[:, 1], t[:, 0]])   # left normal (+90 deg CCW)
    return np.ascontiguousarray(t), np.ascontiguousarray(nrm)


def curvature(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Signed curvature kappa at every station.

    kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

    Positive kappa means the line turns left.  Used for the fold test and
    written out as a per-station diagnostic.
    """
    c = np.asarray(c, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if len(c) < 3:
        return np.zeros(len(c), dtype=np.float64)
    # edge_order=2 matters: with the default first-order one-sided difference the
    # second derivative at the first and last station comes out at half its true
    # value, which would let a genuinely folded swath end slip through the fold
    # check.  Needs at least 3 points, which is guaranteed above.
    edge = 2 if len(c) >= 3 else 1
    dx = np.gradient(c[:, 0], s, edge_order=edge)
    dy = np.gradient(c[:, 1], s, edge_order=edge)
    ddx = np.gradient(dx, s, edge_order=edge)
    ddy = np.gradient(dy, s, edge_order=edge)
    denom = np.power(dx * dx + dy * dy, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (dx * ddy - dy * ddx) / denom
    return np.where(np.isfinite(k), k, 0.0)


def direction_curvature(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Curvature of the direction field actually used to build the band.

    ``d(theta)/ds`` where ``theta`` is the unwrapped heading of the unit
    tangents *t*.  For unsmoothed tangents this equals :func:`curvature` up to
    discretisation; for a smoothed direction field it is the curvature the
    offset construction really sees, which is what the fold test must use -
    testing against the raw centreline would mask samples that no longer fold.
    """
    t = np.asarray(t, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if len(t) < 3:
        return np.zeros(len(t), dtype=np.float64)
    theta = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    k = np.gradient(theta, s, edge_order=2)
    return np.where(np.isfinite(k), k, 0.0)


def offset_grid(c: np.ndarray, nrm: np.ndarray, width: float, n_lines: int):
    """Build the (N, M, 2) grid of sample coordinates.

    Returns ``(P, offsets)``.  ``offsets`` is the (N,) array of signed
    cross-track distances, negative to the right of the digitising direction.
    """
    if n_lines < 1:
        raise GeometryError(f"Number of profiles must be at least 1; got {n_lines}.")
    if width < 0:
        raise GeometryError(f"Swath width must not be negative; got {width}.")
    if n_lines == 1:
        offsets = np.zeros(1, dtype=np.float64)
    else:
        offsets = np.linspace(-width / 2.0, width / 2.0, n_lines)
    p = c[None, :, :] + offsets[:, None, None] * nrm[None, :, :]
    return np.ascontiguousarray(p, dtype=np.float64), offsets


def fold_mask(offsets: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Exact self-intersection test for a normal-offset band.

    For a parallel (normal-offset) curve the local area element scales by the
    Jacobian ``J = 1 - o * kappa``.  ``J <= 0`` means the offset point has
    crossed the centre of curvature, so the band has folded onto itself and that
    sample would double-count inside-of-bend terrain.

    Returns a boolean (N, M) array, True where folded.
    """
    j = 1.0 - np.asarray(offsets, dtype=np.float64)[:, None] * \
        np.asarray(kappa, dtype=np.float64)[None, :]
    return j <= 0.0


def transects_along(xy, spacing: float, length: float, side: str = "both",
                    start: float = 0.0, smooth_window: int = 0):
    """Perpendicular transects at regular spacing along a trace.

    Returns a dict with ``centres`` (K, 2), ``ends`` (K, 2, 2) - the two end
    points of each transect, ordered right-to-left of the digitising direction
    so ``ends[k, 0]`` is the right end - ``dist_along`` (K,), the position of
    each transect along the trace, and ``azimuth`` (K,), the transect's bearing
    in degrees clockwise from north.

    ``side`` chooses ``'both'`` (centred on the trace), ``'left'`` or
    ``'right'`` (one-sided, full *length* on that side).  Direction stabilising
    (*smooth_window*, in dense stations) is applied to the trace exactly as it
    is to a baseline, so a kinked fault trace gets smoothly rotating transects.
    """
    if spacing <= 0 or length <= 0:
        raise GeometryError("Transect spacing and length must be positive.")
    if side not in ("both", "left", "right"):
        raise GeometryError(f"Unknown transect side {side!r}.")

    a = clean_vertices(xy, 0.0)
    total = polyline_length(a)
    if total <= 0:
        raise GeometryError("Trace has zero length.")

    # Dense resample so the tangent at each transect is well defined.
    dense = max(int(round(total / max(spacing / 10.0, 1e-6))) + 1, 201)
    dense = min(dense, 200_000)
    c, s = resample_polyline(a, dense)
    t, nrm = tangents_normals(c, smooth_window)

    positions = np.arange(float(start), total + 1e-9, float(spacing))
    positions = positions[(positions >= 0.0) & (positions <= total)]
    if positions.size == 0:
        raise GeometryError(
            f"No transect fits: trace length {total:.1f}, start {start}, "
            f"spacing {spacing}.")
    idx = np.clip(np.rint(positions / total * (dense - 1)).astype(np.int64),
                  0, dense - 1)
    centres = c[idx]
    n_k = nrm[idx]
    t_k = t[idx]

    if side == "both":
        right = centres - 0.5 * length * n_k
        left = centres + 0.5 * length * n_k
    elif side == "left":
        right = centres
        left = centres + length * n_k
    else:
        right = centres - length * n_k
        left = centres
    ends = np.stack([right, left], axis=1)

    # Bearing of the transect (right -> left end), clockwise from north.
    dx = left[:, 0] - right[:, 0]
    dy = left[:, 1] - right[:, 1]
    azimuth = np.degrees(np.arctan2(dx, dy)) % 360.0

    return {"centres": centres, "ends": ends, "dist_along": positions,
            "azimuth": azimuth, "tangents": t_k, "normals": n_k}


def band_ring(c: np.ndarray, nrm: np.ndarray, width: float) -> np.ndarray:
    """Closed ring of the region actually sampled.

    Outermost +W/2 line forward, then the -W/2 line reversed, then closed.  This
    can self-intersect where the band folds, which is deliberate: the map should
    show what was measured, not a tidied-up buffer.
    """
    h = width / 2.0
    left = c + h * nrm
    right = c - h * nrm
    ring = np.vstack([left, right[::-1], left[:1]])
    return np.ascontiguousarray(ring, dtype=np.float64)
