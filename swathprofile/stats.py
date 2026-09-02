# -*- coding: utf-8 -*-
"""Swath statistics: min, max, mean, quartiles, relief, HI, THi.

Pure numpy - no arcpy.  This is the port of ``SwathProfile`` from the QGIS
plugin's ``swath_core.py``, which itself re-implements Perez-Pena et al. (2017),
*SwathProfiler and NProfiler*, Computers & Geosciences 104, 135-150.

Every deliberate divergence from the reference is tagged with its defect number
(D1..D17) and documented in ``docs/ALGORITHM.md``.  Setting ``legacy=True``
restores the reference behaviour for reproducing previously published numbers.

Definitions
-----------
::

    relief = max - min
    HI     = (mean - min) / relief
    factor = 0.2 * ln(relief / max_relief) + 1
    THi    = ((HI - 0.5) * factor + 0.5) * 100
    THi*   = as THi, with HI first rescaled to clip((HI - 0.2) / 0.6, 0, 1)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .smoothing import smooth_rows

__all__ = [
    "position_quantile",
    "quantiles",
    "hypsometric_integral",
    "terrain_hypsometric_index",
    "SwathStats",
    "compute_stats",
]

#: Below this fraction of max relief the THi relief factor turns negative and
#: inverts the index.  ln(ratio) * 0.2 + 1 == 0  =>  ratio == exp(-5).
THI_FACTOR_ZERO_RATIO = float(np.exp(-5.0))


def _nanreduce(fn, a, axis=0):
    """Run a nan-aware reduction without leaking All-NaN RuntimeWarnings (D2).

    A station where every sample is NoData legitimately has no minimum.  numpy
    warns and returns NaN; we want the NaN without the warning noise, and we
    report the affected station count separately via ``n_valid``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def position_quantile(sorted_matrix: np.ndarray, q: float) -> np.ndarray:
    """The reference's position-based quantile, kept verbatim for testing.

    The QGIS/ArcGIS implementation interpolates at the fractional index
    ``q * (n + 1) - 1`` (0-based) of a column-sorted matrix, clamping at both
    ends.  This function reproduces that exactly, including its NaN bug, so the
    test suite can prove that :func:`quantiles` agrees with it on clean data and
    diverges from it (correctly) on data containing NoData.

    Do not use this in production code - it is the buggy reference.
    """
    n = sorted_matrix.shape[0]
    idx = q * (n + 1) - 1.0
    if idx < 0:
        return sorted_matrix[0, :].copy()
    if idx >= n - 1:
        return sorted_matrix[-1, :].copy()
    lo = int(np.floor(idx))
    frac = idx - lo
    if frac == 0:
        return sorted_matrix[lo, :].copy()
    return sorted_matrix[lo, :] * (1.0 - frac) + sorted_matrix[lo + 1, :] * frac


def quantiles(values: np.ndarray, probs=(0.25, 0.5, 0.75), method: str = "weibull"):
    """Column-wise quantiles that respect NaN (D1).

    The reference sorts with ``np.sort``, which is *not* NaN-aware - NaN sorts to
    the end - and then indexes using the full profile count ``n``.  With any
    NoData present, both quartiles are read at the wrong rank of the valid
    subset, and Q3 can land inside the NaN region and return NaN outright.

    ``np.nanquantile(..., method='weibull')`` uses virtual index ``q * (n + 1)``
    (1-based), algebraically identical to the reference's ``q * (n + 1) - 1``
    0-based index with the same end clamping, *and* reduces ``n`` to each
    column's valid count.  Measured agreement with :func:`position_quantile` on
    NaN-free data: 2.8e-14 over n = 2..60.
    """
    v = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(v, list(probs), axis=0, method=method)


def hypsometric_integral(zmin, zmax, zmean, max_relief, flat: str = "nan",
                         eps_rel: float = 1e-9):
    """Hypsometric integral, with a scale-relative flat-station guard (D3).

    The reference substitutes ``1e-5`` for a zero denominator, which returns
    ``HI = 0`` for a genuinely flat cross-section.  That is wrong in two ways:
    a flat cross-section has *undefined*, not zero, hypsometry; and ``1e-5``
    means something entirely different for a DEM in metres, centimetres or feet.
    Returning 0 drags THi down by 50 points at every lake and valley floor.

    Here the threshold is relative to the profile's own relief scale, and flat
    stations are NaN by default.

    ``flat`` selects the behaviour: ``'nan'`` (default), ``'half'`` (0.5, the
    neutral value), or ``'reference'`` (0.0, bug-compatible).
    """
    zmin = np.asarray(zmin, dtype=np.float64)
    zmax = np.asarray(zmax, dtype=np.float64)
    zmean = np.asarray(zmean, dtype=np.float64)

    relief = zmax - zmin
    if flat == "reference":
        denom = np.where(relief == 0, 1e-5, relief)
        return (zmean - zmin) / denom

    scale = max_relief if np.isfinite(max_relief) and max_relief > 0 else 0.0
    eps = max(eps_rel, 1e-6 * scale)
    good = relief > eps
    safe = np.where(good, relief, 1.0)
    hi = np.where(good, (zmean - zmin) / safe, np.nan)
    if flat == "half":
        hi = np.where(good, hi, 0.5)
    return hi


def terrain_hypsometric_index(hi, relief, max_relief, enhanced: bool = False,
                              clip_factor: bool = True, clip_range: bool = False):
    """THi (or THi* when *enhanced*), with the relief factor guarded (D4, D5, D6).

    ``factor = 0.2 * ln(relief / max_relief) + 1`` crosses zero at
    ``ratio = exp(-5) = 0.0067``.  Below that the factor is **negative**, which
    inverts the sign of the HI deviation: a low-relief station with HI > 0.5 is
    reported as THi < 50, the opposite of its meaning.  As ratio -> 0 the factor
    -> -inf.  The reference's ``1e-10`` floor does not save it, because that
    floor is only applied in the ``relief <= 0`` branch - a genuinely small
    *positive* relief passes straight through - and even the floored value gives
    ``factor = -3.6``, putting THi in [-130, +230].

    Clamping the factor at zero sends negligible-relief stations to the neutral
    ``THi = 50`` ("no information"), which is the defensible limit.
    """
    hi = np.asarray(hi, dtype=np.float64)
    relief = np.asarray(relief, dtype=np.float64)

    if not np.isfinite(max_relief) or max_relief <= 0:   # D6
        return np.full(hi.shape, np.nan, dtype=np.float64)

    h = np.clip((hi - 0.2) / 0.6, 0.0, 1.0) if enhanced else hi

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(relief > 0, relief / max_relief, np.nan)   # D5
        factor = 0.2 * np.log(ratio) + 1.0
    if clip_factor:
        factor = np.maximum(factor, 0.0)

    thi = ((h - 0.5) * factor + 0.5) * 100.0
    if clip_range:
        thi = np.clip(thi, 0.0, 100.0)
    return thi


@dataclass
class SwathStats:
    """Per-station statistics for one baseline / raster combination."""

    distances: np.ndarray            # (M,)
    z_min: np.ndarray
    z_max: np.ndarray
    z_mean: np.ndarray
    z_median: np.ndarray
    z_q1: np.ndarray
    z_q3: np.ndarray
    z_std: np.ndarray
    relief: np.ndarray
    hi: np.ndarray
    thi: np.ndarray                  # relief factor clamped at 0 (recommended)
    thi_raw: np.ndarray              # reference formula, unclamped
    thi_star: np.ndarray             # enhanced, HI rescaled from [0.2, 0.8]
    n_valid: np.ndarray              # (M,) int
    n_total: int
    max_relief: float
    values: np.ndarray | None = field(default=None, repr=False)
    offsets: np.ndarray | None = field(default=None, repr=False)
    name: str = ""
    z_plo: np.ndarray | None = None  # configurable low percentile (default 5)
    z_phi: np.ndarray | None = None  # configurable high percentile (default 95)
    z_centre: np.ndarray | None = None  # the profile along the baseline itself

    @property
    def n_stations(self) -> int:
        return int(self.distances.size)

    @property
    def n_empty_stations(self) -> int:
        """Stations where every sample was NoData."""
        return int(np.count_nonzero(self.n_valid == 0))


def compute_stats(values, distances, offsets=None, *, name: str = "",
                  smooth_window: int = 0, smooth_dilate: bool = False,
                  quantile_method: str = "weibull", flat_hi: str = "nan",
                  thi_clip_factor: bool = True, thi_clip_range: bool = False,
                  min_valid: int = 1, legacy: bool = False,
                  keep_values: bool = True, pct_low: float = 5.0,
                  pct_high: float = 95.0) -> SwathStats:
    """Reduce an (N, M) sample array to per-station statistics.

    Parameters
    ----------
    values : (N, M) array
        One row per offset line (mode A) or a ragged-padded population
        (mode B builds its own stats and does not use this path).  NaN is NoData.
    distances : (M,) array
        Along-track distance of each station.
    smooth_window : int
        Savitzky-Golay window applied to the *input lines* before reducing,
        matching the reference.  Note this is not the same as smoothing the
        output curves: ``min(smooth(y)) >= smooth(min(y))``, so smoothing the
        lines systematically raises the minimum envelope and narrows the band
        (D13).
    legacy : bool
        Restore the reference's HI and THi behaviour (D3, D4) for reproducing
        previously published results.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[None, :]
    d = np.asarray(distances, dtype=np.float64)
    if v.shape[1] != d.size:
        raise ValueError(
            f"values has {v.shape[1]} stations but distances has {d.size}."
        )

    if smooth_window:
        v = smooth_rows(v, smooth_window, dilate=smooth_dilate)

    n_total = int(v.shape[0])
    n_valid = np.count_nonzero(~np.isnan(v), axis=0).astype(np.int64)

    z_min = _nanreduce(np.nanmin, v)
    z_max = _nanreduce(np.nanmax, v)
    z_mean = _nanreduce(np.nanmean, v)
    z_std = _nanreduce(np.nanstd, v)

    p_lo, p_hi = float(pct_low) / 100.0, float(pct_high) / 100.0
    if legacy:
        # Reference path, including its NaN bug, for bit-comparable output.
        srt = np.sort(v, axis=0)
        z_q1 = position_quantile(srt, 0.25)
        z_median = position_quantile(srt, 0.50)
        z_q3 = position_quantile(srt, 0.75)
        z_plo = position_quantile(srt, p_lo)
        z_phi = position_quantile(srt, p_hi)
    else:
        z_q1, z_median, z_q3, z_plo, z_phi = quantiles(
            v, (0.25, 0.5, 0.75, p_lo, p_hi), method=quantile_method)

    # Stations with too few valid samples are not reported at all.
    if min_valid > 1:
        poor = n_valid < int(min_valid)
        for arr in (z_min, z_max, z_mean, z_std, z_q1, z_median, z_q3,
                    z_plo, z_phi):
            arr[poor] = np.nan

    relief = z_max - z_min
    max_relief = float(_nanreduce(np.nanmax, relief[None, :])[0]) if relief.size else 0.0
    if not np.isfinite(max_relief):
        max_relief = 0.0

    hi = hypsometric_integral(z_min, z_max, z_mean, max_relief,
                              flat="reference" if legacy else flat_hi)

    clip_factor = False if legacy else thi_clip_factor
    thi = terrain_hypsometric_index(hi, relief, max_relief, enhanced=False,
                                    clip_factor=clip_factor,
                                    clip_range=thi_clip_range)
    thi_raw = terrain_hypsometric_index(hi, relief, max_relief, enhanced=False,
                                        clip_factor=False, clip_range=False)
    thi_star = terrain_hypsometric_index(hi, relief, max_relief, enhanced=True,
                                         clip_factor=clip_factor,
                                         clip_range=thi_clip_range)

    return SwathStats(
        distances=d,
        z_min=z_min, z_max=z_max, z_mean=z_mean, z_median=z_median,
        z_q1=z_q1, z_q3=z_q3, z_std=z_std,
        relief=relief, hi=hi, thi=thi, thi_raw=thi_raw, thi_star=thi_star,
        n_valid=n_valid, n_total=n_total, max_relief=max_relief,
        values=v if keep_values else None,
        offsets=None if offsets is None else np.asarray(offsets, dtype=np.float64),
        name=name, z_plo=z_plo, z_phi=z_phi,
    )
