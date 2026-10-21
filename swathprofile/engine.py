# -*- coding: utf-8 -*-
"""Orchestration: baselines x rasters -> statistics.

Everything is computed in a projected *working* CRS.  The point grid is built
once per baseline part in that CRS and reused for every raster; a raster in a
different CRS gets the grid transformed into its own system for sampling, so
lines and rasters may each be in any coordinate system.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import geometry as geom
from . import raster_io as rio
from .config import SwathConfig
from .errors import ConfigError, CrsError, GeometryError
from .features import same_crs, transform_points
from .messaging import Messenger
from .stats import (SwathStats, compute_stats, hypsometric_integral,
                    terrain_hypsometric_index)
from .zonal import STAT_KEYS, empty_result, zonal_profile

__all__ = ["SwathResult", "ProfileKey", "run_swath", "stats_from_zonal"]


@dataclass(frozen=True)
class ProfileKey:
    oid: int
    part_id: int
    raster: str

    def __str__(self):
        return f"OID {self.oid}.{self.part_id} / {self.raster}"


@dataclass
class SwathResult:
    """Everything one run produced, ready for the writers and the viewer."""

    stats: dict = field(default_factory=dict)         # ProfileKey -> SwathStats
    geometry: dict = field(default_factory=dict)      # (oid, part) -> dict
    names: dict = field(default_factory=dict)         # (oid, part) -> label
    raster_names: list = field(default_factory=list)
    config: SwathConfig = None
    spatial_reference: object = None
    thi_raster: str = ""
    elapsed: float = 0.0

    def parts(self):
        """Ordered unique (oid, part_id) pairs."""
        seen, out = set(), []
        for k in self.stats:
            if (k.oid, k.part_id) not in seen:
                seen.add((k.oid, k.part_id))
                out.append((k.oid, k.part_id))
        return out

    def for_part(self, oid, part_id):
        return {k.raster: v for k, v in self.stats.items()
                if k.oid == oid and k.part_id == part_id}

    def merge(self, other):
        """Fold another run's parts into this result (different widths allowed)."""
        self.stats.update(other.stats)
        self.geometry.update(other.geometry)
        self.names.update(other.names)
        for n in other.raster_names:
            if n not in self.raster_names:
                self.raster_names.append(n)
        self.elapsed += other.elapsed
        return self


def stats_from_zonal(z: dict, distances: np.ndarray, cfg: SwathConfig,
                     name: str = "") -> SwathStats:
    """Wrap zonal reductions in a :class:`SwathStats`.

    Unlike the reference, Q1/Q3 are genuine here (computed from the full cell
    population), so HI and THi are meaningful in zonal mode too.

    *Minimum Valid Samples per Station* applies here exactly as in
    :func:`compute_stats`: a station with fewer valid cells has every statistic
    nulled before relief, the maximum relief and HI are derived from them.  (It
    used to be ignored in this mode, which mattered most at the edge of a
    sampling mask, where a station may keep only a sliver of cells.)
    """
    if cfg.min_valid > 1:
        poor = np.asarray(z["n_valid"]) < int(cfg.min_valid)
        if poor.any():
            z = dict(z)
            for k in STAT_KEYS:
                arr = np.array(z[k], dtype=np.float64, copy=True)
                arr[poor] = np.nan
                z[k] = arr
    relief = z["z_max"] - z["z_min"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        max_relief = float(np.nanmax(relief)) if relief.size else 0.0
    if not np.isfinite(max_relief):
        max_relief = 0.0

    hi = hypsometric_integral(z["z_min"], z["z_max"], z["z_mean"], max_relief,
                              flat="reference" if cfg.legacy_stats else cfg.flat_hi)
    clip = False if cfg.legacy_stats else cfg.thi_clip_factor
    return SwathStats(
        distances=distances,
        z_min=z["z_min"], z_max=z["z_max"], z_mean=z["z_mean"],
        z_median=z["z_median"], z_q1=z["z_q1"], z_q3=z["z_q3"], z_std=z["z_std"],
        relief=relief, hi=hi,
        thi=terrain_hypsometric_index(hi, relief, max_relief, False, clip,
                                      cfg.thi_clip_range),
        thi_raw=terrain_hypsometric_index(hi, relief, max_relief, False, False, False),
        thi_star=terrain_hypsometric_index(hi, relief, max_relief, True, clip,
                                           cfg.thi_clip_range),
        n_valid=z["n_valid"],
        n_total=int(np.max(z["n_valid"])) if z["n_valid"].size else 0,
        max_relief=max_relief, values=None, offsets=None, name=name,
        z_plo=z.get("z_plo"), z_phi=z.get("z_phi"),
    )


def _warn_sampling_density(cfg: SwathConfig, cell: float, msg: Messenger):
    """D9/D10: tell the user when the offset lines under- or over-sample the band."""
    if not cfg.is_offset or cfg.n_lines < 2:
        return
    spacing = cfg.cross_track_spacing
    if spacing > cell * 1.001:
        suggested = int(np.ceil(cfg.swath_width / cell)) + 1
        if suggested > 1001:
            hint = ("For a band this wide on a raster this fine, use the zonal "
                    "sampling method, which uses every cell - matching it with "
                    f"offset lines would need about {suggested:,} profiles.")
        else:
            hint = (f"Consider about {suggested} profiles, or the zonal sampling "
                    "method, which uses every cell.")
        msg.warn(
            f"Cross-track spacing is {spacing:.1f} map units but the cell size is "
            f"{cell:.1f}, so roughly {100.0 * (1.0 - cell / spacing):.0f}% of the "
            f"cells in the band never contribute. {hint}"
        )
    elif spacing < cell * 0.999:
        msg.info(
            f"  Cross-track spacing ({spacing:.2f}) is finer than the cell size "
            f"({cell:.2f}); adjacent profile lines resample the same cells.")
    if cfg.n_lines % 2 == 0:
        msg.info("  An even number of profiles means no line falls exactly on "
                 "the baseline.")


def _to_raster_crs(xy, working_sr, info, cfg):
    """Transform coordinates from the working CRS into a raster's own CRS."""
    if info.spatial_reference is None or same_crs(working_sr, info.spatial_reference):
        return xy
    return transform_points(xy, working_sr, info.spatial_reference,
                            cfg.geo_transform or None)


def _same_as_working(info, working_sr) -> bool:
    return info.spatial_reference is None or same_crs(working_sr, info.spatial_reference)


def _station_runs(keep):
    """(start, stop) pairs of consecutive True entries in a 1-D boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(np.int8), [0]))))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _sample_kept_runs(raster, info, p, masked, working_sr, cfg, msg, progress_cb):
    """Sample only the stations that keep at least one sample.

    A stretch of the profile the mask removes completely is never read.  The
    values that are read are bitwise those of a single full read, by the
    tiling guarantee of :func:`raster_io.sample_points`: a sample depends only
    on the cells around it, never on which block they came from.  One NoData
    message is reported, counted over the samples the mask keeps.
    """
    n_lines, m = masked.shape
    values = np.full((n_lines, m), np.nan, dtype=np.float64)
    runs = _station_runs(~masked.all(axis=0))
    for k, (j0, j1) in enumerate(runs):
        p_r = _to_raster_crs(p[:, j0:j1], working_sr, info, cfg)
        values[:, j0:j1] = rio.sample_points(
            raster, info, p_r, method=cfg.interpolation,
            max_block_cells=cfg.max_block_cells, edge_nearest=cfg.edge_nearest)
        if progress_cb is not None:
            progress_cb(k + 1, len(runs))
    kept = ~masked
    n_kept = int(np.count_nonzero(kept))
    n_bad = int(np.count_nonzero(np.isnan(values[kept])))
    if n_bad and n_kept:
        msg.info(f"  {info.name}: {n_bad:,} of the {n_kept:,} samples inside the "
                 f"sampling mask ({100.0 * n_bad / n_kept:.1f}%) are NoData or "
                 "outside the raster.")
    return values


def run_swath(baselines, rasters, cfg: SwathConfig, working_sr=None,
              messenger: Messenger = None, mask=None) -> SwathResult:
    """Extract swath statistics for every baseline part against every raster.

    ``rasters`` is a sequence of ``(arcpy_raster, RasterInfo)`` pairs.  Baselines
    must already be in *working_sr*; rasters may be in any CRS.

    ``mask`` is an optional :class:`~swathprofile.mask.SampleMask` in
    *working_sr*.  In offset-lines mode it is decided once per baseline on the
    sample positions, so every raster sees exactly the same samples and the
    masked samples are NaN before smoothing or any statistic sees them.  In
    all-cells mode each raster gets the mask in its own CRS and cells whose
    centre it drops never enter the sweep.  Parts, stretches and blocks the
    mask removes completely are not read at all.
    """
    msg = messenger or Messenger()
    if mask is not None and cfg.mask_mode and mask.mode != cfg.mask_mode:
        raise ConfigError(
            f"The sampling mask is in {mask.mode!r} mode but the configuration "
            f"records {cfg.mask_mode!r}.")
    result = SwathResult(config=cfg, spatial_reference=working_sr,
                         raster_names=[info.name for _, info in rasters],
                         thi_raster=cfg.thi_raster or rasters[0][1].name)

    # Cell size in working units, for validation messages.  A geographic raster
    # reports degrees; approximate with ~111 km per degree so the advice is sane.
    primary_info = rasters[0][1]
    primary_cell = primary_info.cell_size
    if primary_info.spatial_reference is not None and \
            primary_info.spatial_reference.type != "Projected":
        primary_cell *= 111_000.0
    _warn_sampling_density(cfg, primary_cell, msg)

    # All-cells mode tests raster cells in each raster's own CRS, so the mask is
    # moved into each one, densified first so straight edges stay on their true
    # path.  Offset-lines mode tests the sample positions in the working CRS and
    # needs no per-raster copy.  Keyed by raster index: names need not be unique.
    cell_masks = {}
    for k_r, (raster, info) in enumerate(rasters):
        same = _same_as_working(info, working_sr)
        if not same:
            msg.info(f"  {info.name} is in {info.spatial_reference.name}; sample "
                     f"points will be transformed from {working_sr.name}.")
        if mask is None or cfg.is_offset or mask.is_empty:
            continue
        if same:
            cell_masks[k_r] = mask
            continue
        # Bound the densified vertex count; the step is the natural spacing.
        seg = max(cfg.step, mask.perimeter / 200_000.0)
        try:
            cell_masks[k_r] = mask.transformed(
                lambda xy, _i=info: _to_raster_crs(xy, working_sr, _i, cfg), max_seg=seg)
        except ValueError as exc:
            raise CrsError(
                f"The sampling mask could not be placed in the coordinate system of "
                f"{info.name} ({info.spatial_reference.name}): {exc}") from exc
        msg.info(f"  Sampling mask placed in {info.spatial_reference.name} for "
                 f"{info.name} ({cell_masks[k_r].n_vertices:,} vertices).")

    total = len(baselines)
    pct = (cfg.pct_low, cfg.pct_high)
    for n, bl in enumerate(baselines, 1):
        label = getattr(bl, "name", "") or f"OID {bl.oid}"
        msg.info(f"[{n}/{total}] Baseline {label} part {bl.part_id} "
                 f"({bl.length:,.1f} map units)")
        msg.progress((n - 1) * 100, total * 100, f"Baseline {n} of {total}: {label}")
        result.names[(bl.oid, bl.part_id)] = getattr(bl, "name", "") or ""

        def chunk_progress(k, n_chunks, _n=n):
            msg.progress(int((_n - 1) * 100 + 100.0 * k / max(n_chunks, 1)),
                         total * 100)

        if bl.length < cfg.step:
            raise GeometryError(
                f"Baseline OID {bl.oid} part {bl.part_id} is {bl.length:.1f} map "
                f"units long, shorter than the step size ({cfg.step:.1f}). Reduce "
                "the step size or remove this feature."
            )

        tol = max(1e-9 * bl.length, 1e-6 * primary_cell)
        verts = geom.clean_vertices(bl.vertices, tol)
        m = geom.station_count(bl.length, cfg.step, cfg.max_stations)
        c, s = geom.resample_polyline(verts, m)
        t, nrm = geom.tangents_normals(c, cfg.tangent_smooth_window)
        # Which way "left of the line" actually points, so a chart legend can
        # name the side by compass instead of by digitising direction.  The
        # resultant says whether one bearing describes the whole baseline.
        normal_bearing, normal_resultant = geom.mean_normal_bearing(nrm)
        # Curvature of the direction field that builds the band - after any
        # stabilising - since that is what decides whether the band folds.
        kappa = geom.direction_curvature(t, s)
        step_actual = float(s[1] - s[0]) if m > 1 else cfg.step

        if cfg.is_offset:
            p, offsets = geom.offset_grid(c, nrm, cfg.swath_width, cfg.n_lines)
            n_samples = cfg.n_lines * m
            if n_samples > 20_000_000:
                msg.warn(
                    f"  {n_samples / 1e6:.0f} million samples for this baseline "
                    f"(roughly {n_samples * 32 / 1e9:.1f} GB of memory). If that is "
                    "too much, use a larger step, fewer profiles, or the zonal "
                    "method, which uses every cell without a sample grid.")

            folded = geom.fold_mask(offsets, kappa)
            if folded.any():
                n_st = int(np.count_nonzero(folded.any(axis=0)))
                # Its own name: `pct` is the percentile pair used further down.
                fold_pct = 100.0 * float(folded.mean())
                kmax = float(np.max(np.abs(kappa)))
                radius = (1.0 / kmax) if kmax > 0 else float("inf")
                if cfg.fold_policy == "mask":
                    effect = ("Those samples are excluded, so N_VALID is lower on the "
                              "inside of those bends")
                else:
                    effect = ("They are still included and double-count ground on "
                              "the inside of those bends")
                if cfg.tangent_smooth_window:
                    advice = ("Direction stabilising is already on. A band cannot "
                              "follow a bend tighter than its own half-width without "
                              "overlapping itself; if these stations matter, use a "
                              "straighter baseline or a narrower swath.")
                else:
                    advice = ("Turn on 'Stabilise Cross-Section Direction' (rounds "
                              "corners without moving the line), straighten the "
                              "baseline, or reduce the swath width.")
                msg.warn(
                    f"  Baseline OID {bl.oid} part {bl.part_id}: the swath folds onto "
                    f"itself at {n_st} of {m} stations ({fold_pct:.1f}% of samples), where "
                    f"the half-width ({cfg.half_width:,.0f}) exceeds the local radius "
                    f"of curvature (minimum {radius:,.0f}). {effect}. {advice}"
                )
        else:
            p, offsets, folded = None, np.zeros(1), None

        ring = geom.band_ring(c, nrm, cfg.swath_width)

        # The sampling mask.  Offset lines: decided once on the sample positions
        # in the working CRS, so every raster sees the same samples.  `masked`
        # stays None when nothing is dropped, which leaves that path exactly as
        # an unmasked run.  All cells: only whether the whole band is out of
        # reach; the cells themselves are tested inside the sweep.
        masked, mask_empty = None, False
        if mask is not None:
            if cfg.is_offset:
                drop = mask.drops(p[..., 0], p[..., 1])
                mask_empty = bool(drop.all())
                masked = drop if drop.any() else None
            else:
                with np.errstate(invalid="ignore"):
                    rx0, ry0 = np.nanmin(ring, axis=0)
                    rx1, ry1 = np.nanmax(ring, axis=0)
                mask_empty = mask.drops_all_in_box(rx0, ry0, rx1, ry1, pad=cfg.step)

        result.geometry[(bl.oid, bl.part_id)] = {
            "oid": bl.oid, "part_id": bl.part_id,
            "centreline": c, "distances": s, "tangents": t, "normals": nrm,
            "kappa": kappa, "offsets": offsets,
            "normal_bearing": normal_bearing, "normal_resultant": normal_resultant,
            "folded": folded, "step_actual": step_actual,
            "ring": ring,
            # (N, M) True where the sampling mask removed a sample, or None.
            "masked": masked,
            # True when the mask leaves this part nothing at all (it was not read).
            "mask_empty": mask_empty,
            "points": p,
            # Per-part settings, so results run at different widths (e.g.
            # transects vs. drawn baselines) can be merged into one result.
            "width": cfg.swath_width,
            "n_lines": cfg.n_lines if cfg.is_offset else 0,
            "method": cfg.method,
        }

        for k_r, (raster, info) in enumerate(rasters):
            key = ProfileKey(bl.oid, bl.part_id, info.name)
            with rio.env_guard():
                if cfg.is_offset:
                    if mask_empty:
                        values = np.full((len(offsets), m), np.nan, dtype=np.float64)
                    elif masked is not None:
                        values = _sample_kept_runs(raster, info, p, masked, working_sr,
                                                   cfg, msg, chunk_progress)
                    else:
                        p_r = _to_raster_crs(p, working_sr, info, cfg)
                        values = rio.sample_points(
                            raster, info, p_r, method=cfg.interpolation,
                            max_block_cells=cfg.max_block_cells,
                            edge_nearest=cfg.edge_nearest, messenger=msg,
                            progress_cb=chunk_progress)
                    if cfg.z_factor != 1.0:
                        values = values * cfg.z_factor
                    # Masked and folded samples become NaN before compute_stats,
                    # so neither smoothing nor any statistic ever sees them.
                    drop = masked
                    if folded is not None and cfg.fold_policy == "mask":
                        drop = folded if drop is None else (drop | folded)
                    if drop is not None:
                        values = np.where(drop, np.nan, values)
                    stats = compute_stats(
                        values, s, offsets=offsets, name=info.name,
                        smooth_window=cfg.smooth_window,
                        smooth_dilate=cfg.smooth_dilate,
                        quantile_method=cfg.quantile_method,
                        flat_hi=cfg.flat_hi,
                        thi_clip_factor=cfg.thi_clip_factor,
                        thi_clip_range=cfg.thi_clip_range,
                        min_valid=cfg.min_valid, legacy=cfg.legacy_stats,
                        pct_low=cfg.pct_low, pct_high=cfg.pct_high)
                else:
                    # The sweep runs in the raster's own CRS.  Transform the
                    # station centres, the (stabilised) along-track direction
                    # and the half-width vector; in a geographic raster these
                    # become a slightly distorted ribbon, which the sweep
                    # handles by measuring across-track position as a fraction
                    # of the local half-width vector.
                    same = _same_as_working(info, working_sr)
                    hw = cfg.half_width
                    if mask_empty:
                        pass                        # nothing to read; see below
                    elif same:
                        c_r, t_r = c, t
                        across_r = hw * nrm
                        step_r = np.full(m, step_actual)
                    else:
                        c_r = _to_raster_crs(c, working_sr, info, cfg)
                        fwd = _to_raster_crs(c + 0.5 * step_actual * t, working_sr,
                                             info, cfg)
                        back = _to_raster_crs(c - 0.5 * step_actual * t, working_sr,
                                              info, cfg)
                        chord = fwd - back
                        step_r = np.hypot(chord[:, 0], chord[:, 1])
                        with np.errstate(invalid="ignore", divide="ignore"):
                            t_r = chord / step_r[:, None]
                        left = _to_raster_crs(c + hw * nrm, working_sr, info, cfg)
                        right = _to_raster_crs(c - hw * nrm, working_sr, info, cfg)
                        across_r = 0.5 * (left - right)
                    if mask_empty:
                        z = empty_result(m)
                    else:
                        z = zonal_profile(raster, info, c_r, t_r, across_r, step_r,
                                          kappa=kappa, half_width_work=hw,
                                          mask_folds=(cfg.fold_policy == "mask"),
                                          max_block_cells=cfg.max_block_cells,
                                          messenger=msg, pct=pct,
                                          progress_cb=chunk_progress,
                                          cell_mask=cell_masks.get(k_r))
                    if cfg.z_factor != 1.0:
                        for k in ("z_min", "z_max", "z_mean", "z_median",
                                  "z_q1", "z_q3", "z_std", "z_plo", "z_phi"):
                            z[k] = z[k] * cfg.z_factor
                    stats = stats_from_zonal(z, s, cfg, name=info.name)

            # The profile along the baseline itself - the classic single-line
            # profile - always available whatever the sampling mode.  With an
            # odd number of offset lines it is the centre line as sampled (so
            # smoothing and masking apply consistently); otherwise it is
            # sampled once, one line of M points, and the sampling mask is
            # applied to it here, before smoothing, exactly as to the lines.
            stats.z_centre = None
            if (cfg.is_offset and stats.values is not None
                    and offsets is not None and np.any(np.isclose(offsets, 0.0))):
                stats.z_centre = np.array(
                    stats.values[int(np.argmin(np.abs(offsets)))], dtype=np.float64)
            else:
                c_drop = mask.drops(c[:, 0], c[:, 1]) if mask is not None else None
                if c_drop is not None and c_drop.all():
                    zc = np.full(m, np.nan, dtype=np.float64)     # nothing to read
                else:
                    with rio.env_guard():
                        c_line = _to_raster_crs(c[None, :, :], working_sr, info, cfg)
                        zc = rio.sample_points(raster, info, c_line,
                                               method=cfg.interpolation,
                                               max_block_cells=cfg.max_block_cells,
                                               edge_nearest=cfg.edge_nearest)[0]
                    if cfg.z_factor != 1.0:
                        zc = zc * cfg.z_factor
                    if c_drop is not None and c_drop.any():
                        zc = np.where(c_drop, np.nan, zc)
                    if cfg.smooth_window:
                        from .smoothing import savgol_nan
                        zc = savgol_nan(zc, cfg.smooth_window, dilate=cfg.smooth_dilate)
                stats.z_centre = np.asarray(zc, dtype=np.float64)

            stats.normal_bearing = normal_bearing
            stats.normal_resultant = normal_resultant

            n_empty = stats.n_empty_stations
            if n_empty and mask is None:
                msg.warn(
                    f"  {info.name}: {n_empty} of {m} stations have no valid data "
                    "(outside the raster or entirely NoData); their statistics are "
                    "Null.")
            elif n_empty and cfg.is_offset and not mask_empty:
                by_mask = (int(np.count_nonzero(masked.all(axis=0)))
                           if masked is not None else 0)
                other = n_empty - by_mask
                if other <= 0:
                    # Expected, not a problem: say so quietly.
                    msg.info(f"  {info.name}: {n_empty} of {m} stations lie wholly "
                             "outside the sampling mask; their statistics are Null.")
                else:
                    msg.warn(
                        f"  {info.name}: {n_empty} of {m} stations have no valid data "
                        f"- {by_mask} lie wholly outside the sampling mask and {other} "
                        "are outside the raster or entirely NoData. Their statistics "
                        "are Null.")
            # All cells with a mask: the sweep has already reported empty
            # stations, with the mask named as one of the reasons.
            result.stats[key] = stats

    if mask is not None:
        gone = [result.names.get(key) or f"OID {key[0]} part {key[1]}"
                for key, g in result.geometry.items() if g.get("mask_empty")]
        if gone:
            shown = ", ".join(gone[:10]) + (", ..." if len(gone) > 10 else "")
            msg.info(f"  The sampling mask removes the whole swath of {len(gone)} "
                     f"part(s), which were not read: {shown}.")

    msg.progress(total * 100, total * 100, "Done")
    return result
