# -*- coding: utf-8 -*-
"""arcpy raster access: window maths, NoData handling and tiled sampling.

This is the highest-risk module in the package - everything else assumes its
index conventions are right.  They were pinned down empirically against
ArcGIS Pro 3.4 rather than taken from the documentation:

* ``RasterToNumPyArray`` returns row 0 as the **topmost** row (confirmed: a
  round-trip against an analytic ramp differs by exactly 0, and by 1680 against
  the vertically flipped ramp).
* ``lower_left_corner`` is snapped by **rounding to the nearest cell**, not
  flooring.  At a 0.4-cell offset the read matches the floored block; at 0.6 it
  matches the block one cell over.  We therefore always pass an exactly
  cell-aligned corner and assert the returned shape.
* A window extending past the raster edge does **not** raise - it returns the
  requested shape padded with the NoData sentinel.  So the pad must be masked,
  and we clamp anyway.
* ``nodata_to_value=np.nan`` raises ``ValueError`` on integer rasters
  ("Integer types do not support NaN or infinite values"), so integer rasters
  take a separate read-raw-then-mask path.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import numpy as np

from .errors import MemoryBudgetError, RasterError
from .sampling import GridSpec, sample_block

__all__ = ["RasterInfo", "env_guard", "to_float_with_nan", "window_for_bbox",
           "read_block", "sample_points"]

#: Env settings that silently change what a Raster resolves to.
_GUARDED_ENV = ("mask", "extent", "cellSize", "snapRaster",
                "outputCoordinateSystem", "geographicTransformations")

_INT_KINDS = "iub"


@contextlib.contextmanager
def env_guard(**overrides):
    """Neutralise geoprocessing environment settings around raster access.

    ``arcpy.env.mask``, ``extent``, ``cellSize``, ``snapRaster`` and
    ``outputCoordinateSystem`` all silently alter what a ``Raster`` object
    resolves to.  A user with a leftover mask from a previous tool would get
    quietly wrong profiles with no error and no way to reproduce the report.
    """
    import arcpy

    saved = {}
    for key in _GUARDED_ENV:
        try:
            saved[key] = getattr(arcpy.env, key)
        except Exception:      # pragma: no cover - defensive
            pass
    try:
        for key in saved:
            setattr(arcpy.env, key, None)
        for key, value in overrides.items():
            setattr(arcpy.env, key, value)
        yield
    finally:
        for key, value in saved.items():
            try:
                setattr(arcpy.env, key, value)
            except Exception:  # pragma: no cover - defensive
                pass


@dataclass(frozen=True)
class RasterInfo:
    """Immutable georeferencing summary of a source raster."""

    path: str
    name: str
    x_min: float          # left edge
    y_max: float          # top edge
    cell_w: float
    cell_h: float
    n_cols: int
    n_rows: int
    nodata: object
    pixel_type: str
    spatial_reference: object = None

    @property
    def x_max(self) -> float:
        return self.x_min + self.n_cols * self.cell_w

    @property
    def y_min(self) -> float:
        return self.y_max - self.n_rows * self.cell_h

    @property
    def cell_size(self) -> float:
        return max(self.cell_w, self.cell_h)

    @property
    def is_integer(self) -> bool:
        pt = (self.pixel_type or "").upper()
        return pt.startswith(("S", "U"))     # S8/S16/S32, U1/U2/U4/U8/U16/U32

    def describe_extent(self) -> str:
        return (f"{self.x_min:.3f}, {self.y_min:.3f} to "
                f"{self.x_max:.3f}, {self.y_max:.3f}")


def raster_info(raster) -> RasterInfo:
    """Build a :class:`RasterInfo` from a path, layer or arcpy Raster."""
    import arcpy

    r = raster if isinstance(raster, arcpy.Raster) else arcpy.Raster(raster)
    if r.bandCount != 1:
        raise RasterError(
            f"'{r.name}' has {r.bandCount} bands. Swath Profile needs a single-band "
            "raster; select one band explicitly, e.g. "
            f"{r.name}\\Band_1."
        )
    ext = r.extent
    return RasterInfo(
        path=str(getattr(r, "catalogPath", "") or r.name),
        name=str(r.name),
        x_min=float(ext.XMin), y_max=float(ext.YMax),
        cell_w=float(r.meanCellWidth), cell_h=float(r.meanCellHeight),
        n_cols=int(r.width), n_rows=int(r.height),
        nodata=r.noDataValue, pixel_type=str(r.pixelType),
        spatial_reference=ext.spatialReference,
    )


def to_float_with_nan(a: np.ndarray, nodata) -> np.ndarray:
    """Cast a raw block to float and turn NoData into NaN.

    The sentinel is round-tripped into the *source* dtype and compared there,
    which makes the float32 ``-3.4028235e+38`` case an exact ``==`` rather than a
    tolerance hack, and handles integer sentinels such as -9999 exactly.

    Replaces the reference's ``abs(v) < 1e10`` filter (defect D8), which happens
    to catch float sentinels but misses -9999 and -32768 entirely, and silently
    discards legitimate data in unusual units.
    """
    a = np.asarray(a)
    mask = None
    if nodata is not None:
        nd = nodata[0] if isinstance(nodata, (tuple, list)) else nodata
        try:
            nd_native = np.array([nd], dtype=a.dtype)[0]
            mask = (a == nd_native)
        except (OverflowError, ValueError, TypeError):
            mask = None            # sentinel not representable in this dtype

    out = a.astype(np.float64 if a.dtype == np.float64 else np.float32, copy=True)
    if mask is not None and mask.any():
        out[mask] = np.nan
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.nan          # sweeps +-inf and pre-existing NaN
    return out


def window_for_bbox(info: RasterInfo, x0, y0, x1, y1, margin_cells: int = 2):
    """Cell-aligned read window covering a world bounding box.

    Returns ``(col0, row0, n_cols, n_rows)`` or None when the box misses the
    raster entirely.  ``margin_cells`` provides the bilinear halo (1 cell) plus
    snap slack (1 cell).
    """
    if not all(np.isfinite([x0, y0, x1, y1])):
        return None

    col0 = int(math.floor((x0 - info.x_min) / info.cell_w)) - margin_cells
    col1 = int(math.ceil((x1 - info.x_min) / info.cell_w)) + margin_cells
    row0 = int(math.floor((info.y_max - y1) / info.cell_h)) - margin_cells
    row1 = int(math.ceil((info.y_max - y0) / info.cell_h)) + margin_cells

    col0 = max(col0, 0)
    row0 = max(row0, 0)
    col1 = min(col1, info.n_cols)
    row1 = min(row1, info.n_rows)

    if col1 <= col0 or row1 <= row0:
        return None
    return col0, row0, col1 - col0, row1 - row0


#: Sentinel substituted for NoData when reading an integer raster, keyed by the
#: arcpy pixelType.  Each is that type's minimum value, which cannot plausibly
#: occur as real elevation data.
_INT_SENTINELS = {
    "U1": 1, "U2": 3, "U4": 15, "U8": 255, "U16": 65535, "U32": 4294967295,
    "S8": -128, "S16": -32768, "S32": -2147483648,
}


def _int_sentinel(pixel_type: str) -> int:
    return _INT_SENTINELS.get((pixel_type or "").upper(), -2147483648)


def read_block(raster, info: RasterInfo, window):
    """Read one cell-aligned window as ``(block, GridSpec)`` with NaN NoData.

    ``nodata_to_value`` is always passed explicitly rather than relying on the
    raster's own ``noDataValue``.  That property can be ``None`` even when the
    raster genuinely contains NoData cells - confirmed here for both an S32 and
    a Float32 raster written from numpy - in which case sentinel comparison
    alone silently lets NoData through as real data.
    """
    import arcpy

    col0, row0, n_cols, n_rows = window
    x_min_edge = info.x_min + col0 * info.cell_w
    y_max_edge = info.y_max - row0 * info.cell_h
    y_min_edge = y_max_edge - n_rows * info.cell_h

    corner = arcpy.Point(x_min_edge, y_min_edge)
    if info.is_integer:
        # nodata_to_value=np.nan raises ValueError on integer rasters
        # ("Integer types do not support NaN or infinite values"), so substitute
        # the pixel type's minimum and mask it after the float cast.
        sentinel = _int_sentinel(info.pixel_type)
        raw = arcpy.RasterToNumPyArray(raster, corner, n_cols, n_rows,
                                       nodata_to_value=sentinel)
        block = to_float_with_nan(raw, sentinel)
        if info.nodata is not None:
            block = to_float_with_nan(block, info.nodata)
    else:
        raw = arcpy.RasterToNumPyArray(raster, corner, n_cols, n_rows,
                                       nodata_to_value=np.nan)
        block = to_float_with_nan(raw, info.nodata)

    if block.shape != (n_rows, n_cols):
        # arcpy rounds lower_left_corner to the nearest cell; we always pass an
        # aligned corner, so a shape mismatch means an assumption broke.
        raise RasterError(
            f"Requested a {n_rows}x{n_cols} window from '{info.name}' but received "
            f"{block.shape[0]}x{block.shape[1]}. The raster grid may not be "
            "axis-aligned as assumed."
        )

    grid = GridSpec(x_min_edge=x_min_edge, y_max_edge=y_max_edge,
                    cell_w=info.cell_w, cell_h=info.cell_h,
                    n_rows=n_rows, n_cols=n_cols,
                    ref_x_min_edge=info.x_min, ref_y_max_edge=info.y_max,
                    col_offset=col0, row_offset=row0)
    return block, grid


def sample_points(raster, info: RasterInfo, p: np.ndarray, *,
                  method: str = "bilinear", max_block_cells: int = 40_000_000,
                  edge_nearest: bool = True, messenger=None,
                  progress_cb=None) -> np.ndarray:
    """Sample an (N, M, 2) coordinate grid, tiling along the station axis.

    Tiling is recursive bisection on the station axis.  Every chunk re-reads its
    own window with its own >= 2-cell margin, so bilinear interpolation at chunk
    seams is exactly as correct as in a single-block read - the halo is intrinsic
    to the scheme rather than bolted on.  ``test_tiling_equivalence`` asserts
    single-block and heavily-tiled runs are bitwise identical.

    The recursion is seeded with an along-track chunk length rather than starting
    from the whole array.  An axis-aligned envelope around a 45-degree baseline
    covers about ``L**2 / 2`` cells against the band's true ``L * W``; splitting
    into chunks of along-track length near the swath width drops that waste to
    O(1) at any azimuth.  There is no rotated-window read in arcpy, so this is
    the mitigation.

    Returns an (N, M) float64 array with NaN wherever the sample is unavailable.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 2:
        raise ValueError(f"Expected an (N, M, 2) point grid; got {p.shape}.")
    n_lines, n_stations = p.shape[0], p.shape[1]
    out = np.full((n_lines, n_stations), np.nan, dtype=np.float64)

    def process(j0: int, j1: int) -> None:
        sub = p[:, j0:j1, :]
        finite = np.isfinite(sub[..., 0]) & np.isfinite(sub[..., 1])
        if not finite.any():
            return
        xs, ys = sub[..., 0][finite], sub[..., 1][finite]
        win = window_for_bbox(info, float(xs.min()), float(ys.min()),
                              float(xs.max()), float(ys.max()))
        if win is None:
            return                                   # wholly outside; leave NaN

        cells = win[2] * win[3]
        if cells > max_block_cells:
            if (j1 - j0) > 1:
                mid = (j0 + j1) // 2
                process(j0, mid)
                process(mid, j1)
                return
            raise MemoryBudgetError(
                f"A single station of the swath needs a {win[3]}x{win[2]} window "
                f"({cells:,} cells) from '{info.name}', over the {max_block_cells:,} "
                "cell budget. Reduce the swath width, coarsen the raster, or raise "
                "the Max Raster Block setting."
            )

        block, grid = read_block(raster, info, win)
        # The edge rect is the SOURCE RASTER's extent, never the block's: if the
        # half-cell fallback ring were measured against the block, which points
        # received it would depend on how the read was tiled.
        out[:, j0:j1] = sample_block(
            block, grid, sub[..., 0], sub[..., 1], method=method,
            edge_nearest=edge_nearest,
            edge_rect=(info.x_min, info.y_min, info.x_max, info.y_max))

    # Seed with chunks whose along-track length is about the swath width.
    span = np.nanmax(np.hypot(np.diff(p[0, :, 0]), np.diff(p[0, :, 1]))) \
        if n_stations > 1 else 0.0
    step = float(span) if np.isfinite(span) and span > 0 else info.cell_size
    width = float(np.hypot(p[-1, 0, 0] - p[0, 0, 0], p[-1, 0, 1] - p[0, 0, 1]))
    target_len = max(width, 64.0 * info.cell_size)
    chunk = int(np.clip(round(target_len / step), 16, max(16, n_stations)))

    starts = list(range(0, n_stations, chunk))
    for k, start in enumerate(starts):
        process(start, min(start + chunk, n_stations))
        if progress_cb is not None:
            progress_cb(k + 1, len(starts))

    if messenger is not None:
        n_bad = int(np.isnan(out).sum())
        if n_bad:
            messenger.info(
                f"  {info.name}: {n_bad:,} of {out.size:,} samples "
                f"({100.0 * n_bad / out.size:.1f}%) are NoData or outside the raster."
            )
    return out
