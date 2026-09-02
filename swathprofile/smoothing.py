# -*- coding: utf-8 -*-
"""Savitzky-Golay smoothing that survives NoData.

scipy is a hard dependency here.  The QGIS reference silently falls back to a box
moving average when scipy is missing, which changes the algorithm and produces
different numbers under identical parameters with no indication to the user
(defect D12).  scipy 1.15.3 ships with ArcGIS Pro, so failing loudly costs
nothing and removes a silent-wrong-answer path.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError as exc:  # pragma: no cover - scipy ships with Pro
    raise ImportError(
        "scipy is required for swath profile smoothing but could not be imported. "
        "It is part of the standard arcgispro-py3 environment; if it is missing, "
        "the environment has been modified. Run the tool with smoothing disabled, "
        "or repair the Python environment."
    ) from exc

__all__ = ["SMOOTH_PRESETS", "normalise_window", "savgol_nan", "smooth_rows"]


#: Named smoothing strengths exposed in the toolbox and the viewer.
SMOOTH_PRESETS = {
    "None": 0,
    "Low": 5,
    "Medium": 11,
    "High": 21,
    "Ultra": 41,
}


def normalise_window(window: int) -> int:
    """Coerce *window* to 0 (off) or an odd integer >= 5."""
    w = int(window)
    if w <= 0:
        return 0
    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1
    return w


def savgol_nan(y, window: int, polyorder: int | None = None, dilate: bool = False):
    """Savitzky-Golay filter a 1-D series containing NaN.

    NaNs are linearly interpolated before filtering and re-injected afterwards,
    matching the reference behaviour.  Two caveats the reference does not
    document, both handled here:

    * ``np.interp`` extrapolates *constant* across leading and trailing NaN runs,
      which biases the filter at the profile ends.
    * Filtered values adjacent to a hole were computed partly from interpolated
      (fabricated) data.  ``dilate=True`` widens the NaN mask by ``window // 2``
      so those values are not presented as real measurements.

    Guards ``window > len(y)`` and the all-NaN case, both of which crash or
    misbehave in the reference (defect D11).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    w = normalise_window(window)
    if w == 0 or n == 0:
        return y.copy()
    if w > n:
        return y.copy()

    nan = np.isnan(y)
    if nan.all():
        return y.copy()

    poly = min(3, w - 2) if polyorder is None else int(polyorder)
    poly = max(1, min(poly, w - 1))

    filled = y.copy()
    if nan.any():
        idx = np.arange(n)
        filled[nan] = np.interp(idx[nan], idx[~nan], y[~nan])

    out = savgol_filter(filled, w, poly, mode="interp")

    if dilate and nan.any():
        half = w // 2
        widened = np.convolve(nan.astype(np.int8), np.ones(2 * half + 1, np.int8),
                              mode="same") > 0
        out[widened] = np.nan
    else:
        out[nan] = np.nan
    return out


def smooth_rows(values, window: int, polyorder: int | None = None,
                dilate: bool = False):
    """Apply :func:`savgol_nan` to every row of an (N, M) array.

    Rows with no NaN are filtered in a single vectorised ``savgol_filter`` call
    along ``axis=-1``; only rows that actually contain NaN take the per-row path.
    The reference loops over every row in Python regardless.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        return savgol_nan(v, window, polyorder, dilate)
    w = normalise_window(window)
    if w == 0 or v.shape[1] < w:
        return v.copy()

    out = np.empty_like(v)
    has_nan = np.isnan(v).any(axis=1)

    if (~has_nan).any():
        poly = min(3, w - 2) if polyorder is None else int(polyorder)
        poly = max(1, min(poly, w - 1))
        out[~has_nan] = savgol_filter(v[~has_nan], w, poly, mode="interp", axis=-1)

    for i in np.flatnonzero(has_nan):
        out[i] = savgol_nan(v[i], w, polyorder, dilate)
    return out
