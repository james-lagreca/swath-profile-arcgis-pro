# -*- coding: utf-8 -*-
"""Mode B: zonal statistics of every raster cell in the band, per station.

**Cross-section sweep.**  Every cell within the band is assigned to the station
whose cross-section line passes through it - the same construction the
offset-line mode uses, so the two modes measure the same thing and, in the
limit of many offset lines, give the same numbers on curves as well as straight
lines.  With direction stabilising on, a kink in a hand-drawn line is swept
round continuously; with it off, the outer wedge at a kink belongs to no
station and the inner overlap is removed by the fold test, which is the classic
rectangle behaviour.

Everything is expressed in the **raster's own coordinate system**: the station
centres, the unit along-track direction, the half-width vector across the band
and the local station spacing all arrive already transformed by the engine.
That is what lets a raster in a geographic CRS (a national DEM in lat/lon, say)
be profiled from a projected baseline without ever resampling the raster - and
without the per-station fallback that used to make that case take twenty
minutes.  In the raster's CRS the band is a slightly distorted ribbon; the
across-track position of a cell is measured as a *fraction* of the local
half-width vector, so "inside the band" means |fraction| <= 1 whatever the
units.

Assignment starts from a Euclidean distance transform on the block (O(cells),
a few nanoseconds per cell) and is refined by projecting each cell onto the
local direction - exact for a straight baseline, a short fixed-point iteration
on a curve.  All stations are then reduced at once with vectorised group
statistics.

Why not ``arcpy.sa.ZonalStatisticsAsTable``
-------------------------------------------
It cannot produce the Weibull-position quartiles the reference algorithm is
defined by (the QGIS plugin zero-fills Q1/Q3/THi in zonal mode - defect D17);
it rasterises zones and does not support overlapping polygons, which adjacent
slices are on the inside of any bend; and it needs a Spatial Analyst licence.
"""

from __future__ import annotations

import math

import numpy as np

from . import geometry as geom
from .raster_io import read_block, window_for_bbox
from .sampling import world_to_index

__all__ = ["group_stats", "zonal_profile", "empty_result", "STAT_KEYS"]

_KEYS = ("z_min", "z_max", "z_mean", "z_median", "z_q1", "z_q3", "z_std",
         "z_plo", "z_phi")
#: The per-station statistics a zonal result carries (besides n_valid).
STAT_KEYS = _KEYS

#: Upper bound on cells handled per block; the distance transform and its
#: index arrays cost ~25 bytes per cell.
_FAST_BLOCK_CAP = 12_000_000


def empty_result(m):
    """A zonal result for *m* stations with no data: what a wholly masked part gets."""
    out = {k: np.full(m, np.nan, dtype=np.float64) for k in _KEYS}
    out["n_valid"] = np.zeros(m, dtype=np.int64)
    out["n_folded"] = 0
    return out


# ---------------------------------------------------------------------------
# vectorised group statistics
# ---------------------------------------------------------------------------

def group_stats(grp, vals, n_groups, pct=(5.0, 95.0)):
    """Min/max/mean/std, Weibull quartiles and *pct* percentiles per group.

    ``grp`` are integer group ids in ``[0, n_groups)``, ``vals`` the values.
    Quantiles use the Weibull virtual index ``q * (n + 1)`` per group, exactly
    as :func:`numpy.quantile` with ``method='weibull'`` and as the reference.
    """
    grp = np.asarray(grp, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    out = empty_result(n_groups)
    if vals.size == 0:
        return out

    n = np.bincount(grp, minlength=n_groups)
    has = n > 0
    out["n_valid"] = n.astype(np.int64)

    order = np.lexsort((vals, grp))
    v = vals[order]
    g = grp[order]
    starts = np.searchsorted(g, np.arange(n_groups), side="left")
    ends = starts + n - 1

    out["z_min"][has] = v[starts[has]]
    out["z_max"][has] = v[ends[has]]

    s1 = np.bincount(grp, weights=vals, minlength=n_groups)
    s2 = np.bincount(grp, weights=vals * vals, minlength=n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / n
        var = s2 / n - mean * mean
    out["z_mean"][has] = mean[has]
    out["z_std"][has] = np.sqrt(np.maximum(var[has], 0.0))

    def quantile(p):
        h = np.clip(p * (n + 1.0), 1.0, np.maximum(n, 1).astype(np.float64))
        lo = np.floor(h).astype(np.int64)
        frac = h - lo
        i0 = starts + lo - 1
        i1 = np.minimum(i0 + 1, ends)
        res = np.full(n_groups, np.nan)
        res[has] = v[i0[has]] * (1.0 - frac[has]) + v[i1[has]] * frac[has]
        return res

    out["z_q1"] = quantile(0.25)
    out["z_median"] = quantile(0.5)
    out["z_q3"] = quantile(0.75)
    out["z_plo"] = quantile(float(pct[0]) / 100.0)
    out["z_phi"] = quantile(float(pct[1]) / 100.0)
    return out


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def zonal_profile(raster, info, centres, tangents, across, step_local, *,
                  kappa=None, half_width_work=None, mask_folds=True,
                  max_block_cells: int = 40_000_000, messenger=None,
                  pct=(5.0, 95.0), progress_cb=None, cell_mask=None):
    """Per-station statistics from every cell in the band.

    All geometry is in the **raster's** CRS:

    * ``centres`` (M, 2) - station positions;
    * ``tangents`` (M, 2) - unit along-track direction at each station (after
      any direction stabilising);
    * ``across`` (M, 2) - the half-width vector at each station, pointing to
      the *left* edge of the band; its length is the local half-width in
      raster units;
    * ``step_local`` (M,) - local station spacing.

    ``kappa`` (per station, in *working* units) and ``half_width_work`` (the
    scalar half-width in working units) drive the same fold test the offset
    mode uses; pass ``mask_folds=False`` to keep folded cells.

    ``cell_mask`` is an optional :class:`~swathprofile.mask.SampleMask` **in the
    raster's CRS**.  A cell whose centre it drops never becomes a candidate, so
    it counts in neither ``n_valid`` nor the fold count; the assignment of the
    cells that remain is unchanged, because it depends only on the station
    positions.  In keep mode a block the mask cannot reach is not even read.
    """
    from scipy.ndimage import distance_transform_edt

    centres = np.ascontiguousarray(centres, dtype=np.float64)
    tang = np.ascontiguousarray(tangents, dtype=np.float64)
    across = np.ascontiguousarray(across, dtype=np.float64)
    step_local = np.asarray(step_local, dtype=np.float64)
    m = len(centres)
    out = empty_result(m)
    if m < 2:
        return out

    hw_r = np.linalg.norm(across, axis=1)                 # per-station, raster units
    across2 = np.where(hw_r > 0, hw_r * hw_r, np.inf)
    finite = np.isfinite(hw_r) & np.isfinite(step_local)
    if not finite.any():
        return out
    hw_max = float(np.nanmax(hw_r[finite]))
    step_med = max(float(np.nanmedian(step_local[finite])), 1e-12)
    step_safe = np.where(np.isfinite(step_local) & (step_local > 0),
                         step_local, step_med)

    if kappa is None:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(centres, axis=0).T))])
        kappa = geom.direction_curvature(tang, s)
        half_width_work = hw_max
    kappa = np.asarray(kappa, dtype=np.float64)
    hw_work = float(half_width_work if half_width_work is not None else hw_max)

    cell_diag = math.hypot(info.cell_w, info.cell_h)
    # Station positions are quantised to cell centres for the distance
    # transform, and far from the line that is enough to pick the wrong
    # station: at distance u the true distances to stations j and j+k differ
    # by only (k*step)^2 / (2u), which half a cell diagonal of quantisation
    # noise exceeds for k up to about sqrt(2 * u * diag/2) / step.  Cells whose
    # coarse pick lies within that many stations of the chunk are candidates;
    # the projection refinement below then finds the true station.
    k_coarse = int(math.ceil(math.sqrt(2.0 * hw_max * 0.5 * cell_diag) / step_med)) + 3
    cap = min(int(max_block_cells), _FAST_BLOCK_CAP)
    n_folded = 0
    # Ties are real: a whole lattice row of cells can sit exactly on the
    # boundary between two stations.  A deterministic half-open interval with a
    # bias larger than the float drift of `along` across a row sends the row
    # wholly to one station; rounding would split it by noise and bias both
    # means.
    tie_bias = 1e-6

    def process(j0, j1):
        nonlocal n_folded
        cs = centres[j0:j1]
        ok = np.isfinite(cs).all(axis=1)
        if not ok.any():
            return
        cs = cs[ok]
        win = window_for_bbox(info,
                              float(cs[:, 0].min()) - hw_max,
                              float(cs[:, 1].min()) - hw_max,
                              float(cs[:, 0].max()) + hw_max,
                              float(cs[:, 1].max()) + hw_max,
                              margin_cells=2)
        if win is None:
            return
        if win[2] * win[3] > cap and (j1 - j0) > 1:
            mid = (j0 + j1) // 2
            process(j0, mid)
            process(mid, j1)
            return
        if cell_mask is not None:
            bx0 = info.x_min + win[0] * info.cell_w
            by1 = info.y_max - win[1] * info.cell_h
            if cell_mask.drops_all_in_box(bx0, by1 - win[3] * info.cell_h,
                                          bx0 + win[2] * info.cell_w, by1,
                                          pad=cell_diag):
                return                      # nothing here could be kept: skip the read

        block, grid = read_block(raster, info, win)
        n_rows, n_cols = block.shape
        dropped = cell_mask.drops_cells(grid) if cell_mask is not None else None

        fr, fc = world_to_index(grid, centres[:, 0], centres[:, 1])
        row = np.floor(fr + 0.5).astype(np.int64)
        col = np.floor(fc + 0.5).astype(np.int64)
        inside = ((row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols) &
                  np.isfinite(fr) & np.isfinite(fc))
        if not inside.any():
            return
        lab = np.full((n_rows, n_cols), -1, dtype=np.int32)
        lab[row[inside], col[inside]] = np.flatnonzero(inside).astype(np.int32)

        dist, (ri, ci) = distance_transform_edt(
            lab < 0, sampling=(info.cell_h, info.cell_w), return_indices=True)
        j_hat = lab[ri, ci]
        del ri, ci

        cand = ((dist <= hw_max + cell_diag) & np.isfinite(block) &
                (j_hat >= j0 - k_coarse) & (j_hat < j1 + k_coarse))
        del dist
        if dropped is not None:
            cand &= ~dropped
            del dropped
        rows, cols = np.nonzero(cand)
        del cand
        if rows.size == 0:
            return

        # Anchored on the raster origin, so identical in every tiling.
        xs, ys = grid.cell_centres()
        px, py = xs[cols], ys[rows]
        jh = j_hat[rows, cols].astype(np.int64)
        del j_hat

        best_j = np.empty(rows.size, dtype=np.int64)
        q_f = np.empty(rows.size, dtype=np.float64)
        u_f = np.empty(rows.size, dtype=np.float64)
        slab = 2_000_000
        for a in range(0, rows.size, slab):
            b = min(a + slab, rows.size)
            j_est = jh[a:b].copy()
            for _ in range(8):
                dx = px[a:b] - centres[j_est, 0]
                dy = py[a:b] - centres[j_est, 1]
                along = dx * tang[j_est, 0] + dy * tang[j_est, 1]
                shift = np.floor(along / step_safe[j_est] + 0.5 + tie_bias
                                 ).astype(np.int64)
                if not shift.any():
                    break
                j_est = np.clip(j_est + shift, 0, m - 1)
            dx = px[a:b] - centres[j_est, 0]
            dy = py[a:b] - centres[j_est, 1]
            q_f[a:b] = (dx * tang[j_est, 0] + dy * tang[j_est, 1]) / step_safe[j_est]
            u_f[a:b] = (dx * across[j_est, 0] + dy * across[j_est, 1]) / across2[j_est]
            best_j[a:b] = j_est

        keep = ((q_f >= -0.5 - tie_bias) & (q_f < 0.5 - tie_bias) &
                (np.abs(u_f) <= 1.0) & (best_j >= j0) & (best_j < j1))
        if mask_folds:
            # Same test as offset mode, in working units: beyond the centre of
            # curvature the cross-section lines cross.
            folded = keep & ((1.0 - (u_f * hw_work) * kappa[best_j]) <= 0.0)
            n_folded += int(folded.sum())
            keep &= ~folded
        if not keep.any():
            return

        vals = block[rows[keep], cols[keep]]
        grp = best_j[keep] - j0
        part = group_stats(grp, vals, j1 - j0, pct=pct)
        for key in _KEYS:
            out[key][j0:j1] = part[key]
        out["n_valid"][j0:j1] = part["n_valid"]

    # Chunk length about one swath width minimises bounding-box waste on
    # oblique baselines; bisection handles the memory cap from there.
    chunk = int(np.clip(round(2.0 * hw_max / step_med), 16, m))
    starts = list(range(0, m, chunk))
    for k, start in enumerate(starts):
        process(start, min(start + chunk, m))
        if progress_cb is not None:
            progress_cb(k + 1, len(starts))

    out["n_folded"] = n_folded
    if messenger is not None:
        empty = int(np.count_nonzero(out["n_valid"] == 0))
        if empty:
            why = (" (outside the sampling mask, outside the raster, or NoData)"
                   if cell_mask is not None else "")
            messenger.info(
                f"  {info.name}: {empty} of {m} stations received no valid cells{why}.")
        if n_folded:
            messenger.info(
                f"  {info.name}: {n_folded:,} cells excluded where the band folds "
                "onto itself on the inside of bends.")
    return out
