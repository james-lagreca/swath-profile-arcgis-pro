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
from .errors import GeometryError
from .features import same_crs, transform_points
from .messaging import Messenger
from .stats import (SwathStats, compute_stats, hypsometric_integral,
                    terrain_hypsometric_index)
from .zonal import zonal_profile

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
    """
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


def run_swath(baselines, rasters, cfg: SwathConfig, working_sr=None,
              messenger: Messenger = None) -> SwathResult:
    """Extract swath statistics for every baseline part against every raster.

    ``rasters`` is a sequence of ``(arcpy_raster, RasterInfo)`` pairs.  Baselines
    must already be in *working_sr*; rasters may be in any CRS.
    """
    msg = messenger or Messenger()
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

    for raster, info in rasters:
        if info.spatial_reference is not None and \
                not same_crs(working_sr, info.spatial_reference):
            msg.info(f"  {info.name} is in {info.spatial_reference.name}; sample "
                     f"points will be transformed from {working_sr.name}.")

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
                pct = 100.0 * float(folded.mean())
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
                    f"itself at {n_st} of {m} stations ({pct:.1f}% of samples), where "
                    f"the half-width ({cfg.half_width:,.0f}) exceeds the local radius "
                    f"of curvature (minimum {radius:,.0f}). {effect}. {advice}"
                )
        else:
            p, offsets, folded = None, np.zeros(1), None

        result.geometry[(bl.oid, bl.part_id)] = {
            "oid": bl.oid, "part_id": bl.part_id,
            "centreline": c, "distances": s, "tangents": t, "normals": nrm,
            "kappa": kappa, "offsets": offsets,
            "normal_bearing": normal_bearing, "normal_resultant": normal_resultant,
            "folded": folded, "step_actual": step_actual,
            "ring": geom.band_ring(c, nrm, cfg.swath_width),
            "points": p,
            # Per-part settings, so results run at different widths (e.g.
            # transects vs. drawn baselines) can be merged into one result.
            "width": cfg.swath_width,
            "n_lines": cfg.n_lines if cfg.is_offset else 0,
            "method": cfg.method,
        }

        for raster, info in rasters:
            key = ProfileKey(bl.oid, bl.part_id, info.name)
            with rio.env_guard():
                if cfg.is_offset:
                    p_r = _to_raster_crs(p, working_sr, info, cfg)
                    values = rio.sample_points(
                        raster, info, p_r, method=cfg.interpolation,
                        max_block_cells=cfg.max_block_cells,
                        edge_nearest=cfg.edge_nearest, messenger=msg,
                        progress_cb=chunk_progress)
                    if cfg.z_factor != 1.0:
                        values = values * cfg.z_factor
                    if folded is not None and cfg.fold_policy == "mask":
                        values = np.where(folded, np.nan, values)
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
                    same = (info.spatial_reference is None or
                            same_crs(working_sr, info.spatial_reference))
                    hw = cfg.half_width
                    if same:
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
                    z = zonal_profile(raster, info, c_r, t_r, across_r, step_r,
                                      kappa=kappa, half_width_work=hw,
                                      mask_folds=(cfg.fold_policy == "mask"),
                                      max_block_cells=cfg.max_block_cells,
                                      messenger=msg, pct=pct,
                                      progress_cb=chunk_progress)
                    if cfg.z_factor != 1.0:
                        for k in ("z_min", "z_max", "z_mean", "z_median",
                                  "z_q1", "z_q3", "z_std", "z_plo", "z_phi"):
                            z[k] = z[k] * cfg.z_factor
                    stats = stats_from_zonal(z, s, cfg, name=info.name)

            # The profile along the baseline itself - the classic single-line
            # profile - always available whatever the sampling mode.  With an
            # odd number of offset lines it is the centre line as sampled (so
            # smoothing and fold masking apply consistently); otherwise it is
            # sampled once, one line of M points.
            stats.z_centre = None
            if (cfg.is_offset and stats.values is not None
                    and offsets is not None and np.any(np.isclose(offsets, 0.0))):
                stats.z_centre = np.array(
                    stats.values[int(np.argmin(np.abs(offsets)))], dtype=np.float64)
            else:
                with rio.env_guard():
                    c_line = _to_raster_crs(c[None, :, :], working_sr, info, cfg)
                    zc = rio.sample_points(raster, info, c_line,
                                           method=cfg.interpolation,
                                           max_block_cells=cfg.max_block_cells,
                                           edge_nearest=cfg.edge_nearest)[0]
                if cfg.z_factor != 1.0:
                    zc = zc * cfg.z_factor
                if cfg.smooth_window:
                    from .smoothing import savgol_nan
                    zc = savgol_nan(zc, cfg.smooth_window, dilate=cfg.smooth_dilate)
                stats.z_centre = np.asarray(zc, dtype=np.float64)

            stats.normal_bearing = normal_bearing
            stats.normal_resultant = normal_resultant

            if stats.n_empty_stations:
                msg.warn(
                    f"  {info.name}: {stats.n_empty_stations} of {m} stations have no "
                    "valid data (outside the raster or entirely NoData); their "
                    "statistics are Null.")
            result.stats[key] = stats

    msg.progress(total * 100, total * 100, "Done")
    return result
