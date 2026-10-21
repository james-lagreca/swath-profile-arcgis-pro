# -*- coding: utf-8 -*-
"""A numpy stand-in for an arcpy raster, so :func:`run_swath` runs without ArcGIS.

The engine reaches raster data through exactly one function,
:func:`raster_io.read_block` (both the offset sampler and the zonal sweep call
it), and wraps every read in :func:`raster_io.env_guard`.  Patching those two is
enough to drive the whole engine on an in-memory array, which puts the sampling
mask, the zonal sweep and their block-size invariance under CI.

``read_block`` is mirrored exactly, including the :class:`GridSpec` anchored on
the raster origin, because that anchoring is what the tiling tests are about.
Every read is counted, so a test can prove a read was skipped.
"""

from __future__ import annotations

import contextlib
from unittest import mock

import numpy as np

from swathprofile import raster_io, zonal
from swathprofile.raster_io import RasterInfo
from swathprofile.sampling import GridSpec

__all__ = ["FakeRaster", "fake_read_block", "patched", "plane"]


class FakeRaster:
    """An in-memory raster: *array* row 0 is the top, as RasterToNumPyArray gives."""

    def __init__(self, array, x_min, y_max, cell_w, cell_h=None, name="fake"):
        self.array = np.asarray(array, dtype=np.float64)
        n_rows, n_cols = self.array.shape
        cell_h = float(cell_w if cell_h is None else cell_h)
        self.info = RasterInfo(path=name, name=name, x_min=float(x_min),
                               y_max=float(y_max), cell_w=float(cell_w), cell_h=cell_h,
                               n_cols=int(n_cols), n_rows=int(n_rows), nodata=None,
                               pixel_type="F64", spatial_reference=None)
        self.reads = 0
        self.cells_read = 0

    @property
    def pair(self):
        """The (raster, info) pair run_swath expects."""
        return self, self.info


def fake_read_block(raster, info, window):
    """What :func:`raster_io.read_block` returns, read from the array."""
    col0, row0, n_cols, n_rows = window
    raster.reads += 1
    raster.cells_read += int(n_cols) * int(n_rows)
    block = raster.array[row0:row0 + n_rows, col0:col0 + n_cols].copy()
    if block.shape != (n_rows, n_cols):
        raise AssertionError(f"window {window} is outside the fake raster")
    grid = GridSpec(info.x_min + col0 * info.cell_w, info.y_max - row0 * info.cell_h,
                    info.cell_w, info.cell_h, int(n_rows), int(n_cols),
                    ref_x_min_edge=info.x_min, ref_y_max_edge=info.y_max,
                    col_offset=int(col0), row_offset=int(row0))
    return block, grid


@contextlib.contextmanager
def patched():
    """Route every engine raster read to :func:`fake_read_block`."""
    with mock.patch.object(raster_io, "read_block", fake_read_block), \
            mock.patch.object(zonal, "read_block", fake_read_block), \
            mock.patch.object(raster_io, "env_guard",
                              lambda **_: contextlib.nullcontext()):
        yield


def plane(a, b, c, x_min, y_max, cell, n_cols, n_rows, name="plane"):
    """z = a*x + b*y + c sampled at cell centres, as a FakeRaster."""
    xs = x_min + (np.arange(n_cols) + 0.5) * cell
    ys = y_max - (np.arange(n_rows) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    return FakeRaster(a * X + b * Y + c, x_min, y_max, cell, name=name)
