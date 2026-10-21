# -*- coding: utf-8 -*-
"""Vectorised raster sampling on an in-memory block.

Pure numpy - no arcpy.  :mod:`raster_io` is responsible for turning an arcpy
Raster into a ``(GridSpec, ndarray)`` pair; everything here works on that pair
and is therefore unit-testable without ArcGIS.

Grid conventions (verified empirically against ArcGIS Pro 3.4)
--------------------------------------------------------------
``arcpy.Raster.extent`` reports **cell-edge** coordinates, and
``RasterToNumPyArray`` returns a north-up array whose **row 0 is the topmost**
row - while its ``lower_left_corner`` argument is given in lower-left world
coordinates.  Two opposite conventions in one call, which is the classic source
of half-cell and vertical-flip bugs.  Everything here is therefore anchored on
the **top-left cell edge**::

    x_centre(col) = x_min_edge + (col + 0.5) * cell_w
    y_centre(row) = y_max_edge - (row + 0.5) * cell_h

and inverted into continuous *cell-centre index space*, where ``fc == 0.0`` is
the centre of column 0 and ``fc == -0.5`` is the raster's left edge::

    fc = (x - x_min_edge) / cell_w - 0.5
    fr = (y_max_edge - y) / cell_h - 0.5        # note the sign flip
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GridSpec", "world_to_index", "sample_nearest", "sample_bilinear",
           "sample_block"]


@dataclass(frozen=True)
class GridSpec:
    """Georeferencing of an in-memory raster block, anchored top-left.

    ``ref_x_min_edge`` / ``ref_y_max_edge`` and the integer ``col_offset`` /
    ``row_offset`` locate this block within its **source raster**.  Index
    arithmetic is deliberately done against the raster origin and then shifted by
    the integer block offset, rather than against the block's own origin.

    That is not pedantry.  ``(x - x_min_edge) / cell_w`` subtracts a large number
    (NZTM eastings are ~1.5e6, northings ~5e6) from another large number, so the
    fractional part - and therefore the interpolation weight - lands on a
    slightly different float64 depending on which origin was used.  Anchoring on
    the raster and subtracting an exactly-representable integer makes the weights
    independent of how the read was tiled, so the same swath returns the same
    numbers at any block size.  Without this, tiled and untiled reads differ in
    the last ULP.

    Defaults leave the block as its own reference, which is what standalone unit
    tests want.
    """

    x_min_edge: float     # world X of the LEFT edge of column 0
    y_max_edge: float     # world Y of the TOP edge of row 0
    cell_w: float
    cell_h: float
    n_rows: int
    n_cols: int
    ref_x_min_edge: float | None = None
    ref_y_max_edge: float | None = None
    col_offset: int = 0
    row_offset: int = 0

    @property
    def ref_x(self) -> float:
        return self.x_min_edge if self.ref_x_min_edge is None else self.ref_x_min_edge

    @property
    def ref_y(self) -> float:
        return self.y_max_edge if self.ref_y_max_edge is None else self.ref_y_max_edge

    @property
    def x_max_edge(self) -> float:
        return self.x_min_edge + self.n_cols * self.cell_w

    @property
    def y_min_edge(self) -> float:
        return self.y_max_edge - self.n_rows * self.cell_h

    @property
    def lower_left(self):
        """(x, y) of the lower-left corner - what arcpy.Point wants."""
        return self.x_min_edge, self.y_min_edge

    def cell_centres(self):
        """(xs, ys) 1-D arrays of cell-centre coordinates.

        Computed from the **source raster's** origin and the integer block
        offset, the inverse of :func:`world_to_index`, so a cell's centre is the
        same float64 whichever block it was read in.  ``(g + 0.5)`` is exact for
        any global index ``g``, leaving one rounding in the multiply and one in
        the add, both fixed by the raster alone.

        Building the centres from the block's own origin instead rounds twice
        through a large coordinate, and on a fractional grid (a 0.3 m LiDAR
        tile, say) about one centre in five then differs in the last bit
        between tilings - enough to make All-cells statistics, and the cells a
        sampling mask keeps, depend on the block size.  A block that is its own
        reference (the default) gets exactly the old values.
        """
        xs = self.ref_x + (np.arange(self.n_cols) + (self.col_offset + 0.5)) * self.cell_w
        ys = self.ref_y - (np.arange(self.n_rows) + (self.row_offset + 0.5)) * self.cell_h
        return xs, ys


def _check_coords(x, y):
    """Coordinates must be float64.

    Not stylistic: NZTM2000 and southern-hemisphere UTM northings reach
    5-10 x 10**6 m, which float32 quantises to roughly 1 m.  Silently sampling
    the wrong cell by a metre would be almost impossible to notice downstream,
    so this is asserted rather than left to convention.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.dtype != np.float64 or y.dtype != np.float64:
        raise TypeError(
            "Sample coordinates must be float64; got "
            f"x={x.dtype}, y={y.dtype}. float32 quantises large projected "
            "coordinates (e.g. NZTM northings) to about 1 m."
        )
    if x.shape != y.shape:
        raise ValueError(f"x and y shapes differ: {x.shape} vs {y.shape}.")
    return x, y


def world_to_index(grid: GridSpec, x, y):
    """World coordinates -> continuous (row, col) in cell-centre index space.

    Computed against the source raster origin, then shifted by the integer block
    offset.  Both steps are exact, so the result does not depend on how the read
    was tiled (see :class:`GridSpec`).
    """
    x, y = _check_coords(x, y)
    fc = (x - grid.ref_x) / grid.cell_w - 0.5 - grid.col_offset
    fr = (grid.ref_y - y) / grid.cell_h - 0.5 - grid.row_offset
    return fr, fc


def sample_nearest(block: np.ndarray, grid: GridSpec, x, y) -> np.ndarray:
    """Nearest-neighbour sample.  Points outside the block become NaN.

    Uses ``floor(f + 0.5)`` rather than ``np.rint``: rint is round-half-to-even
    (banker's rounding), which disagrees with GDAL/ArcGIS at exact half-cell
    boundaries and is non-monotonic under floating-point noise.
    """
    fr, fc = world_to_index(grid, x, y)
    row = np.floor(fr + 0.5).astype(np.int64)
    col = np.floor(fc + 0.5).astype(np.int64)

    valid = ((col >= 0) & (col < grid.n_cols) &
             (row >= 0) & (row < grid.n_rows))

    out = np.full(fr.shape, np.nan, dtype=np.float64)
    if valid.any():
        out[valid] = block[row[valid], col[valid]]
    return out


def sample_bilinear(block: np.ndarray, grid: GridSpec, x, y,
                    edge_nearest: bool = True, edge_rect=None) -> np.ndarray:
    """Bilinear sample.  Points outside the block become NaN, never clipped.

    NoData is already NaN in *block*, so NaN propagates through the weighted sum
    for free - that gives strict "any NoData neighbour implies NoData" semantics
    with no extra code.

    Points in the outer half-cell ring have no complete set of four neighbours.
    With ``edge_nearest=True`` those fall back to nearest-neighbour, so a swath
    that grazes the raster border does not lose a half-cell strip.

    ``edge_rect`` is ``(x_min, y_min, x_max, y_max)`` in world coordinates
    delimiting where that fallback is allowed - normally the **source raster's**
    extent, not the block's.  This matters: if the ring were measured against the
    block, which points received the fallback would depend on how the read was
    tiled, and the same swath would yield different numbers at different block
    sizes.  ``test_tiling_equivalence_is_bitwise`` exists to catch exactly that.
    When None, the block's own footprint is used (single-block callers only).
    """
    fr, fc = world_to_index(grid, x, y)

    r0 = np.floor(fr).astype(np.int64)
    c0 = np.floor(fc).astype(np.int64)
    ty = fr - r0
    tx = fc - c0
    r1 = r0 + 1
    c1 = c0 + 1

    valid = ((c0 >= 0) & (c1 < grid.n_cols) &
             (r0 >= 0) & (r1 < grid.n_rows))

    out = np.full(fr.shape, np.nan, dtype=np.float64)
    if valid.any():
        rr0, cc0 = r0[valid], c0[valid]
        rr1, cc1 = r1[valid], c1[valid]
        wx, wy = tx[valid], ty[valid]
        out[valid] = (block[rr0, cc0] * (1.0 - wx) * (1.0 - wy) +
                      block[rr0, cc1] * wx * (1.0 - wy) +
                      block[rr1, cc0] * (1.0 - wx) * wy +
                      block[rr1, cc1] * wx * wy)

    if edge_nearest:
        xa, ya = np.asarray(x), np.asarray(y)
        if edge_rect is None:
            inside = ((fc >= -0.5) & (fc <= grid.n_cols - 0.5) &
                      (fr >= -0.5) & (fr <= grid.n_rows - 0.5))
        else:
            ex0, ey0, ex1, ey1 = edge_rect
            inside = (xa >= ex0) & (xa <= ex1) & (ya >= ey0) & (ya <= ey1)
        ring = inside & ~valid
        if ring.any():
            out[ring] = sample_nearest(block, grid, xa[ring], ya[ring])
    return out


def sample_block(block: np.ndarray, grid: GridSpec, x, y,
                 method: str = "bilinear", edge_nearest: bool = True,
                 edge_rect=None):
    """Dispatch to the requested interpolation."""
    m = (method or "bilinear").strip().lower()
    if m.startswith("near"):
        return sample_nearest(block, grid, x, y)
    if m.startswith("bilin"):
        return sample_bilinear(block, grid, x, y, edge_nearest=edge_nearest,
                               edge_rect=edge_rect)
    raise ValueError(f"Unknown interpolation method {method!r}; "
                     "expected 'bilinear' or 'nearest'.")
