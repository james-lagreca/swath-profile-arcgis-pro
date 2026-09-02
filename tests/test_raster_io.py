# -*- coding: utf-8 -*-
"""Ground-truth tests for the raster index conventions.

Everything else in the package assumes these hold.  Build this first and do not
trust the engine until it passes.
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arcpy  # noqa: E402

from swathprofile import geometry as geom  # noqa: E402
from swathprofile import raster_io as rio  # noqa: E402
from swathprofile.errors import RasterError  # noqa: E402
from swathprofile.sampling import GridSpec, sample_bilinear, sample_nearest  # noqa: E402

from . import synth_arcpy as synth  # noqa: E402

# Plane coefficients: z = A*x + B*y + C
A, B, C = 3.0, 7.0, 100.0
X0, Y0 = 1_500_000.0, 5_000_000.0        # NZTM-scale coordinates on purpose
CW = CH = 10.0
NC, NR = 300, 220


class RasterIoTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        arcpy.env.overwriteOutput = True
        cls.folder, cls.gdb = synth.make_scratch()
        cls.plane = synth.plane_array(A, B, C, X0, Y0, CW, CH, NC, NR)
        cls.path = synth.write_raster(cls.plane, cls.gdb, "plane", X0, Y0, CW, CH)
        cls.info = rio.raster_info(cls.path)
        cls.ras = arcpy.Raster(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.ras = None
        arcpy.management.ClearWorkspaceCache()
        shutil.rmtree(cls.folder, ignore_errors=True)

    # -- conventions ----------------------------------------------------------

    def test_raster_info_matches_source(self):
        i = self.info
        self.assertAlmostEqual(i.x_min, X0, places=6)
        self.assertAlmostEqual(i.y_max, Y0 + NR * CH, places=6)
        self.assertEqual((i.n_rows, i.n_cols), (NR, NC))
        self.assertAlmostEqual(i.cell_w, CW)
        self.assertFalse(i.is_integer)

    def test_window_origin_roundtrip(self):
        """A[0, 0] must equal the analytic value at the top-left cell CENTRE.

        A half-cell error shows up here as a constant bias of A*CW/2 = 15.0,
        and a vertical flip as a bias of order B*NR*CH.
        """
        win = rio.window_for_bbox(
            self.info,
            X0 + 40 * CW, Y0 + 30 * CH,
            X0 + 60 * CW, Y0 + 50 * CH, margin_cells=0)
        self.assertIsNotNone(win)
        block, grid = rio.read_block(self.ras, self.info, win)

        col0, row0, ncols, nrows = win
        self.assertEqual(block.shape, (nrows, ncols))

        expect = A * (grid.x_min_edge + 0.5 * CW) + \
            B * (grid.y_max_edge - 0.5 * CH) + C
        self.assertAlmostEqual(float(block[0, 0]), expect, places=6)

        # And the whole block, not just its corner.
        sub = self.plane[row0:row0 + nrows, col0:col0 + ncols]
        self.assertLess(float(np.max(np.abs(block - sub))), 1e-6)

    def test_grid_cell_centres_are_self_consistent(self):
        g = GridSpec(X0, Y0 + NR * CH, CW, CH, NR, NC)
        xs, ys = g.cell_centres()
        self.assertAlmostEqual(xs[0], X0 + 0.5 * CW)
        self.assertAlmostEqual(ys[0], Y0 + NR * CH - 0.5 * CH)
        self.assertAlmostEqual(g.y_min_edge, Y0)
        self.assertAlmostEqual(g.x_max_edge, X0 + NC * CW)

    # -- interpolation --------------------------------------------------------

    def test_bilinear_is_exact_on_a_plane(self):
        """Bilinear reproduces span{1, x, y, xy} exactly.

        This single assertion validates the entire index chain: the top-left row
        origin, the +-0.5 cell-centre convention, window snapping and tiling.
        """
        rng = np.random.default_rng(7)
        x = rng.uniform(X0 + 5 * CW, X0 + (NC - 5) * CW, 5000)
        y = rng.uniform(Y0 + 5 * CH, Y0 + (NR - 5) * CH, 5000)
        p = np.stack([x, y], axis=-1)[None, :, :]

        z = rio.sample_points(self.ras, self.info, p, method="bilinear")[0]
        expect = A * x + B * y + C
        self.assertFalse(np.isnan(z).any())
        self.assertLess(float(np.max(np.abs(z - expect))), 1e-4)

    def test_nearest_error_is_bounded_by_half_a_cell(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(X0 + 5 * CW, X0 + (NC - 5) * CW, 5000)
        y = rng.uniform(Y0 + 5 * CH, Y0 + (NR - 5) * CH, 5000)
        p = np.stack([x, y], axis=-1)[None, :, :]

        z = rio.sample_points(self.ras, self.info, p, method="nearest")[0]
        err = z - (A * x + B * y + C)
        bound = (abs(A) + abs(B)) * CW / 2.0
        self.assertLess(float(np.max(np.abs(err))), bound + 1e-9)
        self.assertLess(abs(float(np.mean(err))), bound * 0.1)   # unbiased

    def test_nearest_uses_floor_half_not_bankers_rounding(self):
        """floor(f + 0.5), not np.rint, at exact half-cell boundaries."""
        block = np.arange(12, dtype=np.float64).reshape(3, 4)
        g = GridSpec(0.0, 30.0, 10.0, 10.0, 3, 4)
        # x exactly on the boundary between col 0 and col 1 -> col 1
        z = sample_nearest(block, g, np.array([10.0]), np.array([25.0]))
        self.assertEqual(float(z[0]), 1.0)
        # and between col 1 and col 2 -> col 2 (rint would give 2 here too, but
        # would give 0 for the first case)
        z = sample_nearest(block, g, np.array([20.0]), np.array([25.0]))
        self.assertEqual(float(z[0]), 2.0)

    # -- edges and NoData -----------------------------------------------------

    def test_outside_raster_is_nan_not_clamped(self):
        """Off-raster samples must be NaN, never the edge cell's value.

        On a steep plane an index-clipping implementation returns the edge value,
        which is plausible enough to go unnoticed - hence testing for the
        specific wrong answer, not just for correctness.
        """
        y = np.full(200, Y0 + 100 * CH)
        x = np.linspace(X0 - 60 * CW, X0 - 2 * CW, 200)      # entirely west
        p = np.stack([x, y], axis=-1)[None, :, :]
        z = rio.sample_points(self.ras, self.info, p, method="bilinear")[0]

        self.assertTrue(np.isnan(z).all())
        edge_value = A * (X0 + 0.5 * CW) + B * y[0] + C
        self.assertFalse(np.any(np.isclose(np.nan_to_num(z, nan=-1e30), edge_value)))

    def test_partial_coverage_splits_cleanly(self):
        y = np.full(400, Y0 + 100 * CH)
        x = np.linspace(X0 - 20 * CW, X0 + 40 * CW, 400)     # straddles the edge
        p = np.stack([x, y], axis=-1)[None, :, :]
        z = rio.sample_points(self.ras, self.info, p, method="bilinear")[0]

        inside = x > X0 + CW
        self.assertTrue(np.isnan(z[x < X0 - CW]).all())
        self.assertFalse(np.isnan(z[inside]).any())
        self.assertLess(
            float(np.max(np.abs(z[inside] - (A * x[inside] + B * y[inside] + C)))),
            1e-4)

    def test_float_nodata_becomes_nan(self):
        arr = self.plane.astype(np.float32).copy()
        arr[40:60, 40:60] = -9999.0
        path = synth.write_raster(arr, self.gdb, "plane_nd", X0, Y0, CW, CH,
                                  nodata=-9999.0)
        info = rio.raster_info(path)
        block, _ = rio.read_block(arcpy.Raster(path), info,
                                  rio.window_for_bbox(info, X0, Y0,
                                                      info.x_max, info.y_max,
                                                      margin_cells=0))
        self.assertEqual(int(np.isnan(block).sum()), 400)

    def test_integer_nodata_sentinel_is_masked(self):
        """-9999 in an integer raster - the case abs(v) < 1e10 misses entirely."""
        arr = np.tile(np.arange(NC, dtype=np.int32), (NR, 1))
        arr[10:20, 10:20] = -9999
        path = synth.write_raster(arr, self.gdb, "int_nd", X0, Y0, CW, CH,
                                  nodata=-9999)
        info = rio.raster_info(path)
        self.assertTrue(info.is_integer)
        block, _ = rio.read_block(arcpy.Raster(path), info,
                                  rio.window_for_bbox(info, X0, Y0,
                                                      info.x_max, info.y_max,
                                                      margin_cells=0))
        self.assertEqual(block.dtype.kind, "f")
        self.assertEqual(int(np.isnan(block).sum()), 100)
        self.assertFalse(np.any(np.nan_to_num(block, nan=0.0) == -9999))

    def test_bilinear_propagates_nodata_strictly(self):
        block = np.array([[1.0, 2.0], [3.0, np.nan]])
        g = GridSpec(0.0, 20.0, 10.0, 10.0, 2, 2)
        z = sample_bilinear(block, g, np.array([10.0]), np.array([10.0]),
                            edge_nearest=False)
        self.assertTrue(np.isnan(z[0]))

    # -- tiling ---------------------------------------------------------------

    def test_tiling_equivalence_is_bitwise(self):
        """Single-block and heavily-tiled reads must agree bit for bit.

        Halo and off-by-one errors in the tiling code produce small differences
        on a handful of columns, which is nearly invisible in any statistical
        check.  Bitwise equality catches them immediately.
        """
        verts = np.array([[X0 + 20 * CW, Y0 + 20 * CH],
                          [X0 + 280 * CW, Y0 + 200 * CH]])   # diagonal
        c, s = geom.resample_polyline(verts, 900)
        _, nrm = geom.tangents_normals(c)
        p, _ = geom.offset_grid(c, nrm, 400.0, 25)

        for method in ("bilinear", "nearest"):
            big = rio.sample_points(self.ras, self.info, p, method=method,
                                    max_block_cells=10 ** 9)
            small = rio.sample_points(self.ras, self.info, p, method=method,
                                      max_block_cells=4000)
            self.assertTrue(np.array_equal(big, small, equal_nan=True),
                            f"{method}: tiled result differs from single-block")

    def test_memory_budget_raises_actionably(self):
        verts = np.array([[X0 + 50 * CW, Y0 + 50 * CH],
                          [X0 + 250 * CW, Y0 + 150 * CH]])
        c, _ = geom.resample_polyline(verts, 50)
        _, nrm = geom.tangents_normals(c)
        p, _ = geom.offset_grid(c, nrm, 2000.0, 20)
        with self.assertRaises(Exception) as ctx:
            rio.sample_points(self.ras, self.info, p, max_block_cells=4)
        self.assertIn("swath width", str(ctx.exception).lower())

    # -- environment hygiene --------------------------------------------------

    def test_env_guard_neutralises_a_hostile_mask(self):
        """A leftover env.mask must not silently alter the samples."""
        rng = np.random.default_rng(3)
        x = rng.uniform(X0 + 5 * CW, X0 + (NC - 5) * CW, 500)
        y = rng.uniform(Y0 + 5 * CH, Y0 + (NR - 5) * CH, 500)
        p = np.stack([x, y], axis=-1)[None, :, :]
        clean = rio.sample_points(self.ras, self.info, p)

        small = arcpy.Extent(X0, Y0, X0 + 10 * CW, Y0 + 10 * CH)
        try:
            arcpy.env.extent = small
            with rio.env_guard():
                guarded = rio.sample_points(self.ras, self.info, p)
        finally:
            arcpy.env.extent = None

        self.assertTrue(np.array_equal(clean, guarded, equal_nan=True))
        self.assertIsNone(arcpy.env.extent)

    def test_multiband_raster_is_rejected_with_guidance(self):
        arr = np.stack([self.plane, self.plane + 1.0], axis=0)
        path = os.path.join(self.gdb, "multi")
        arcpy.NumPyArrayToRaster(arr, arcpy.Point(X0, Y0), CW, CH).save(path)
        with self.assertRaises(RasterError) as ctx:
            rio.raster_info(path)
        self.assertIn("Band_1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
