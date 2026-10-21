# -*- coding: utf-8 -*-
"""The sampling mask, from the crossing rule up to the engine.  No ArcGIS needed.

The geometry tests pin the one crossing rule both the point test and the cell
test use.  The engine tests drive :func:`run_swath` on an in-memory plane
(``tests/fake_raster.py``), where every masked statistic has a closed form, and
prove reads that cannot contribute are skipped.  Zonal block-size invariance is
here too: it depends on the cell-centre fix that the zonal mask builds on.
"""

from __future__ import annotations

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swathprofile.config import SwathConfig          # noqa: E402
from swathprofile.engine import run_swath            # noqa: E402
from swathprofile.errors import ConfigError          # noqa: E402
from swathprofile.features import Baseline           # noqa: E402
from swathprofile.mask import SampleMask, densify    # noqa: E402
from swathprofile.messaging import CollectingMessenger  # noqa: E402
from swathprofile.sampling import GridSpec           # noqa: E402

from tests.fake_raster import FakeRaster, patched, plane  # noqa: E402

SQUARE = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
HOLE = [[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]]

# A fractional grid like a real LiDAR tile, where the last bits of a cell
# centre depend on how it is computed.
FX0, FY0, FW = 1567890.123456789, 5123456.987654321, 0.3


def _star(rng, n, cx, cy, r0, r1):
    ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    rad = rng.uniform(r0, r1, n)
    return np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])


def _barycentric_inside(p, a, b, c):
    v0, v1, v2 = c - a, b - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    return (u > 0) & (v > 0) & (u + v < 1)


def _edge_distance(points, ring):
    """Distance from each point to the nearest edge of a closed ring."""
    a = np.asarray(ring, dtype=np.float64)
    b = np.roll(a, -1, axis=0)
    best = np.full(len(points), np.inf)
    for (ax, ay), (bx, by) in zip(a, b):
        dx, dy = bx - ax, by - ay
        t = np.clip(((points[:, 0] - ax) * dx + (points[:, 1] - ay) * dy) /
                    (dx * dx + dy * dy), 0.0, 1.0)
        best = np.minimum(best, np.hypot(points[:, 0] - (ax + t * dx),
                                         points[:, 1] - (ay + t * dy)))
    return best


class CrossingRuleTest(unittest.TestCase):

    def test_square_interior_exterior_and_boundary_rule(self):
        m = SampleMask([SQUARE])
        pts = np.array([[5, 5], [12, 5], [-1, 5], [0, 5], [10, 5], [5, 0], [5, 10],
                        [0, 0], [10, 10]], float)
        # A boundary point counts as if nudged east, then north: west and south
        # edges are inside, east and north edges outside.
        self.assertEqual(m.inside(pts[:, 0], pts[:, 1]).tolist(),
                         [True, False, False, True, False, True, False, True, False])
        # Reversing the ring's direction changes nothing.
        r = SampleMask([SQUARE[::-1]])
        self.assertEqual(r.inside(pts[:, 0], pts[:, 1]).tolist(),
                         m.inside(pts[:, 0], pts[:, 1]).tolist())

    def test_triangle_matches_barycentric_oracle(self):
        rng = np.random.default_rng(1)
        tri = np.array([[0.3, 0.1], [9.7, 2.2], [4.1, 8.9]])
        p = rng.uniform(-1.0, 11.0, (100_000, 2))
        np.testing.assert_array_equal(SampleMask([tri]).inside(p[:, 0], p[:, 1]),
                                      _barycentric_inside(p, *tri))

    def test_holes_and_an_island_inside_a_hole(self):
        island = [[4.5, 4.5], [5.5, 4.5], [5.5, 5.5], [4.5, 5.5]]
        m = SampleMask([SQUARE, HOLE, island])
        x = np.array([1.0, 3.5, 5.0, 12.0])
        y = np.array([1.0, 3.5, 5.0, 5.0])
        self.assertEqual(m.inside(x, y).tolist(), [True, False, True, False])

    def test_multipart(self):
        far = [[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0]]
        m = SampleMask([SQUARE, far])
        x = np.array([5.0, 15.0, 25.0])
        self.assertEqual(m.inside(x, np.full(3, 5.0)).tolist(), [True, False, True])

    def test_random_stars_match_matplotlib_away_from_edges(self):
        from matplotlib.path import Path

        rng = np.random.default_rng(7)
        for _ in range(5):
            star = _star(rng, 40, 0.0, 0.0, 20.0, 60.0)
            p = rng.uniform(-70.0, 70.0, (20_000, 2))
            far = _edge_distance(p, star) > 1e-9
            ours = SampleMask([star]).inside(p[:, 0], p[:, 1])
            theirs = Path(star, closed=False).contains_points(p)
            np.testing.assert_array_equal(ours[far], theirs[far])

    def test_polygons_sharing_an_edge_never_both_claim_a_point(self):
        left = SampleMask([[[0, 0], [5, 0], [5, 10], [0, 10]]])
        right = SampleMask([[[5, 0], [10, 0], [10, 10], [5, 10]]])
        xs, ys = np.meshgrid(np.linspace(0.0, 10.0, 41), np.linspace(0.0, 10.0, 41))
        a = left.inside(xs, ys)
        b = right.inside(xs, ys)
        self.assertFalse((a & b).any())
        both = SampleMask([[[0, 0], [10, 0], [10, 10], [0, 10]]]).inside(xs, ys)
        np.testing.assert_array_equal(a | b, both)

    def test_non_finite_points_are_never_inside(self):
        m = SampleMask([SQUARE])
        self.assertEqual(m.inside([np.nan, 5.0, np.inf], [5.0, np.nan, 5.0]).tolist(),
                         [False, False, False])

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            SampleMask([SQUARE], mode="inside")
        with self.assertRaises(ValueError):
            SampleMask([[[0, 0], [1, np.nan], [1, 1]]])
        self.assertTrue(SampleMask([[[0, 0], [1, 1]]]).is_empty)   # too few vertices


class CellsTest(unittest.TestCase):

    def test_global_index_cell_centres_are_tile_invariant(self):
        full = GridSpec(FX0, FY0, FW, FW, 300, 2000, FX0, FY0, 0, 0)
        fx, fy = full.cell_centres()
        for c0, r0 in ((1, 1), (7, 3), (333, 17), (1234, 167), (1999, 299)):
            blk = GridSpec(FX0 + c0 * FW, FY0 - r0 * FW, FW, FW, 300 - r0, 2000 - c0,
                           FX0, FY0, c0, r0)
            bx, by = blk.cell_centres()
            np.testing.assert_array_equal(bx, fx[c0:])
            np.testing.assert_array_equal(by, fy[r0:])
        own = GridSpec(10.0, 20.0, 2.0, 2.0, 3, 4)            # its own reference
        np.testing.assert_array_equal(own.cell_centres()[0], [11.0, 13.0, 15.0, 17.0])
        np.testing.assert_array_equal(own.cell_centres()[1], [19.0, 17.0, 15.0])

    def _tied_star(self, grid):
        rng = np.random.default_rng(11)
        fx, fy = grid.cell_centres()
        star = _star(rng, 60, FX0 + 500 * FW, FY0 - 150 * FW, 20.0, 45.0)
        # Put vertices exactly on cell-centre coordinates to force ties.
        star[::7, 0] = fx[np.clip(((star[::7, 0] - FX0) / FW).astype(int), 0, grid.n_cols - 1)]
        star[::5, 1] = fy[np.clip(((FY0 - star[::5, 1]) / FW).astype(int), 0, grid.n_rows - 1)]
        return star

    def test_cells_match_points_bitwise_at_cell_centres(self):
        grid = GridSpec(FX0, FY0, FW, FW, 300, 1000, FX0, FY0, 0, 0)
        hole = _star(np.random.default_rng(2), 20, FX0 + 500 * FW, FY0 - 150 * FW, 3.0, 8.0)
        m = SampleMask([self._tied_star(grid), hole])
        cells = m.inside_cells(grid)
        gx, gy = np.meshgrid(*grid.cell_centres())
        np.testing.assert_array_equal(cells, m.inside(gx, gy))
        self.assertGreater(int(cells.sum()), 1000)

    def test_cells_are_identical_whatever_the_tiling(self):
        whole = GridSpec(FX0, FY0, FW, FW, 300, 1000, FX0, FY0, 0, 0)
        m = SampleMask([self._tied_star(whole)])
        ref = m.inside_cells(whole)
        tiled = np.zeros_like(ref)
        rows = [0, 97, 210, 300]
        cols = [0, 333, 640, 1000]
        for r0, r1 in zip(rows, rows[1:]):
            for c0, c1 in zip(cols, cols[1:]):
                g = GridSpec(FX0 + c0 * FW, FY0 - r0 * FW, FW, FW, r1 - r0, c1 - c0,
                             FX0, FY0, c0, r0)
                tiled[r0:r1, c0:c1] = m.inside_cells(g)
        np.testing.assert_array_equal(tiled, ref)

    def test_keep_and_exclude_are_complementary(self):
        grid = GridSpec(-1.0, 12.0, 0.5, 0.5, 26, 26)
        keep, excl = SampleMask([SQUARE, HOLE]), SampleMask([SQUARE, HOLE], "exclude")
        np.testing.assert_array_equal(keep.drops_cells(grid), ~excl.drops_cells(grid))
        x = np.array([5.0, 1.0, 12.0, np.nan])
        y = np.array([5.0, 5.0, 5.0, 1.0])
        self.assertEqual(keep.drops(x, y).tolist(), [True, False, True, False])
        self.assertEqual(excl.drops(x, y).tolist(), [False, True, False, False])

    def test_empty_mask(self):
        grid = GridSpec(0.0, 10.0, 1.0, 1.0, 10, 10)
        keep, excl = SampleMask([]), SampleMask([], "exclude")
        self.assertTrue(keep.drops_cells(grid).all())
        self.assertFalse(excl.drops_cells(grid).any())
        self.assertTrue(keep.drops_all_in_box(0, 0, 1, 1))
        self.assertFalse(excl.drops_all_in_box(0, 0, 1, 1))


class MaskHelpersTest(unittest.TestCase):

    def test_densify_keeps_vertices_and_bounds_edges(self):
        d = densify([SQUARE + [SQUARE[0]]], 3.0)[0]          # closed input is fine
        for v in np.array(SQUARE, float):
            self.assertTrue((d == v).all(axis=1).any())
        edges = np.hypot(*(np.roll(d, -1, axis=0) - d).T)
        self.assertLessEqual(float(edges.max()), 3.0 + 1e-12)
        self.assertEqual(len(d), 16)
        with self.assertRaises(ValueError):
            densify([SQUARE], 0.0)

    def test_transformed_with_an_affine_map(self):
        m = SampleMask([SQUARE, HOLE], "exclude")
        moved = m.transformed(lambda xy: xy * 2.0 + np.array([100.0, -50.0]), max_seg=1.0)
        self.assertEqual(moved.mode, "exclude")
        x, y = np.array([5.0, 1.0]), np.array([5.0, 1.0])
        np.testing.assert_array_equal(moved.inside(x * 2 + 100, y * 2 - 50), m.inside(x, y))
        with self.assertRaises(ValueError):
            m.transformed(lambda xy: np.full_like(xy, np.nan))

    def test_drops_all_in_box(self):
        m = SampleMask([SQUARE])
        self.assertTrue(m.drops_all_in_box(20, 20, 30, 30))
        self.assertFalse(m.drops_all_in_box(20, 20, 30, 30, pad=15.0))
        self.assertFalse(m.drops_all_in_box(5, 5, 30, 30))
        self.assertFalse(SampleMask([SQUARE], "exclude").drops_all_in_box(20, 20, 30, 30))

    def test_speed(self):
        rng = np.random.default_rng(5)
        m = SampleMask([_star(rng, 2000, 0.0, 0.0, 50.0, 100.0)])
        q = rng.uniform(-110.0, 110.0, (1_000_000, 2))
        t0 = time.perf_counter()
        m.inside(q[:, 0], q[:, 1])
        self.assertLess(time.perf_counter() - t0, 5.0)

    def test_config_mask_mode(self):
        with self.assertRaises(ConfigError):
            SwathConfig(swath_width=10.0, step=1.0, mask_mode="inside")
        cfg = SwathConfig(swath_width=10.0, step=1.0, mask_mode="keep",
                          mask_note="x" * 500)
        self.assertEqual(len(cfg.mask_note), 200)


# ---------------------------------------------------------------------------
# Engine on an in-memory plane z = 3x + 7y + 100
# ---------------------------------------------------------------------------

A, B, C = 3.0, 7.0, 100.0
X0, YTOP, CW = 1_500_000.0, 5_003_000.0, 10.0
LX = X0 + 2000.0                       # a cell edge, so no cell centre sits on it


def _plane():
    return plane(A, B, C, X0, YTOP, CW, 400, 300)


def _north_line(oid=1, x=LX, y0=YTOP - 2500.0, y1=YTOP - 300.0, name="north"):
    """Drawn northwards: its left (positive offsets) is west."""
    return Baseline(oid, 0, np.array([[x, y0], [x, y1]]), abs(y1 - y0), name)


def _rect(x0, y0, x1, y1, mode="keep"):
    return SampleMask([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], mode)


EAST = (LX + 3.0, YTOP - 3000.0, LX + 400.0, YTOP)     # east of the line


def _run(baselines, raster, cfg, mask=None):
    msg = CollectingMessenger()
    with patched():
        res = run_swath(baselines, [raster.pair], cfg, None, msg, mask=mask)
    return res, msg


def _only(res):
    return next(iter(res.stats.values()))


class EngineOffsetMaskTest(unittest.TestCase):

    def setUp(self):
        self.cfg = SwathConfig(swath_width=300.0, step=10.0, n_lines=51)

    def test_keeping_the_east_half_gives_the_analytic_statistics(self):
        res, _ = _run([_north_line()], _plane(), self.cfg, _rect(*EAST))
        s = _only(res)
        # Lines 6 m apart; east of LX + 3 are the 25 at offsets -6 ... -150.
        self.assertTrue((s.n_valid == 25).all())
        np.testing.assert_allclose(s.relief, A * 144.0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(s.hi, 0.5, rtol=0, atol=1e-9)
        y = res.geometry[(1, 0)]["centreline"][:, 1]
        np.testing.assert_allclose(s.z_mean, A * (LX + 78.0) + B * y + C, rtol=0, atol=1e-6)
        masked = res.geometry[(1, 0)]["masked"]
        self.assertEqual(masked.shape, (51, len(y)))
        self.assertEqual(int((~masked).sum(axis=0)[0]), 25)

    def test_keep_and_exclude_partition_the_samples(self):
        r_all, _ = _run([_north_line()], _plane(), self.cfg)
        r_keep, _ = _run([_north_line()], _plane(), self.cfg, _rect(*EAST))
        r_excl, _ = _run([_north_line()], _plane(), self.cfg, _rect(*EAST, mode="exclude"))
        np.testing.assert_array_equal(_only(r_keep).n_valid + _only(r_excl).n_valid,
                                      _only(r_all).n_valid)

    def test_a_mask_that_drops_nothing_is_bitwise_no_mask(self):
        r_all, _ = _run([_north_line()], _plane(), self.cfg)
        everything = _rect(X0 - 10, YTOP - 4000, X0 + 5000, YTOP + 10)
        elsewhere = _rect(X0 + 3500, YTOP - 200, X0 + 3600, YTOP - 100, "exclude")
        for mask in (everything, elsewhere):
            s = _only(_run([_north_line()], _plane(), self.cfg, mask)[0])
            for key in ("z_min", "z_max", "z_mean", "z_q1", "z_q3", "hi", "thi",
                        "z_centre"):
                np.testing.assert_array_equal(getattr(s, key), getattr(_only(r_all), key),
                                              err_msg=key)

    def test_masked_values_never_reach_smoothing(self):
        cfg = SwathConfig(swath_width=300.0, step=10.0, n_lines=51, smooth_window=11)
        # Keep the east half, plus the west half's northern stretch only, so
        # the western lines are partly masked: their southern samples drop.
        mask = SampleMask([[[LX + 3, YTOP - 3000], [LX + 400, YTOP - 3000],
                            [LX + 400, YTOP], [LX - 400, YTOP],
                            [LX - 400, YTOP - 1200], [LX + 3, YTOP - 1200]]])
        clean = _plane()
        spiked = _plane()
        # A huge spike in the dropped south-west, well clear of the boundary.
        spiked.array[195:245, 180:192] += 1.0e6
        s0 = _only(_run([_north_line()], clean, cfg, mask)[0])
        s1 = _only(_run([_north_line()], spiked, cfg, mask)[0])
        for key in ("z_min", "z_max", "z_mean", "z_median", "z_q1", "z_q3", "relief"):
            np.testing.assert_array_equal(getattr(s1, key), getattr(s0, key), err_msg=key)
        # The western lines really were part-masked: some stations keep them.
        self.assertGreater(int(s0.n_valid.max()), 25)

    def test_baseline_profile_is_masked_with_odd_and_even_line_counts(self):
        for n in (51, 50):
            cfg = SwathConfig(swath_width=300.0, step=10.0, n_lines=n)
            s = _only(_run([_north_line()], _plane(), cfg, _rect(*EAST))[0])
            self.assertTrue(np.isnan(s.z_centre).all(), n)   # the line is west of LX+3
            wide = _rect(LX - 50, YTOP - 3000, LX + 400, YTOP)
            s2 = _only(_run([_north_line()], _plane(), cfg, wide)[0])
            s3 = _only(_run([_north_line()], _plane(), cfg)[0])
            np.testing.assert_array_equal(s2.z_centre, s3.z_centre)

    def test_a_part_the_mask_removes_is_never_read(self):
        near, far = _north_line(1), _north_line(2, x=LX - 1500.0, name="far")
        ras = _plane()
        _run([near], ras, self.cfg, _rect(*EAST))
        reads_one = ras.reads
        ras2 = _plane()
        res, msg = _run([near, far], ras2, self.cfg, _rect(*EAST))
        self.assertEqual(ras2.reads, reads_one)
        self.assertTrue(res.geometry[(2, 0)]["mask_empty"])
        self.assertTrue(np.isnan(res.stats[next(k for k in res.stats if k.oid == 2)].z_mean).all())
        self.assertTrue(any("removes the whole swath" in m for m in msg.infos))

    def test_a_masked_stretch_is_not_read_and_kept_values_are_bitwise(self):
        north_half = _rect(LX - 400, YTOP - 1400, LX + 400, YTOP)
        ras_all, ras_m = _plane(), _plane()
        r_all, _ = _run([_north_line()], ras_all, self.cfg)
        r_m, _ = _run([_north_line()], ras_m, self.cfg, north_half)
        self.assertLess(ras_m.cells_read, ras_all.cells_read)
        kept = ~r_m.geometry[(1, 0)]["masked"]
        np.testing.assert_array_equal(_only(r_m).values[kept], _only(r_all).values[kept])

    def test_wholly_masked_stations_are_reported_quietly(self):
        north_half = _rect(LX - 400, YTOP - 1400, LX + 400, YTOP)
        _, msg = _run([_north_line()], _plane(), self.cfg, north_half)
        self.assertTrue(any("wholly outside the sampling mask" in m for m in msg.infos))
        self.assertFalse(any("stations have no valid data" in w for w in msg.warnings))

    def test_mode_mismatch_raises(self):
        cfg = SwathConfig(swath_width=300.0, step=10.0, mask_mode="keep")
        with self.assertRaises(ConfigError):
            _run([_north_line()], _plane(), cfg, _rect(*EAST, mode="exclude"))

    def test_the_bundle_carries_the_mask_to_the_chart_window(self):
        import tempfile

        from swathprofile.bundle import load_bundle, save_bundle

        cfg = SwathConfig(swath_width=300.0, step=10.0, n_lines=51, mask_mode="keep",
                          mask_note="keep only inside 1 polygon(s) (1 drawn)",
                          min_valid=5)
        res, _ = _run([_north_line()], _plane(), cfg, _rect(*EAST))
        res.geometry[(1, 0)]["mask_kept"] = 49.0
        path = os.path.join(tempfile.mkdtemp(), "m.swath.npz")
        save_bundle(path, res, title="masked")
        back = load_bundle(path)
        self.assertEqual(back.parts[0]["mask_kept"], 49.0)
        self.assertEqual(back.meta["config"]["mask_mode"], "keep")
        self.assertEqual(back.meta["config"]["min_valid"], 5)
        stats = next(iter(back.parts[0]["by_raster"].values()))
        np.testing.assert_array_equal(stats.n_valid, _only(res).n_valid)


class EngineZonalMaskTest(unittest.TestCase):

    def setUp(self):
        self.cfg = SwathConfig(swath_width=300.0, step=10.0, method="zonal")

    def test_keeping_the_east_half(self):
        s_all = _only(_run([_north_line()], _plane(), self.cfg)[0])
        s = _only(_run([_north_line()], _plane(), self.cfg, _rect(*EAST))[0])
        inner = slice(5, -5)                        # stay clear of the line ends
        np.testing.assert_array_equal(s.n_valid[inner] * 2, s_all.n_valid[inner])
        np.testing.assert_allclose(s.z_max[inner], s_all.z_max[inner], rtol=0, atol=1e-9)
        # Westmost kept cell centre LX + 5 against LX - 145 unmasked: 150 m east.
        np.testing.assert_allclose(s.z_min[inner] - s_all.z_min[inner], A * 150.0,
                                   rtol=0, atol=1e-6)

    def test_keep_and_exclude_partition_the_cells(self):
        s_all = _only(_run([_north_line()], _plane(), self.cfg)[0])
        diag = SampleMask([[[LX - 300, YTOP - 2600], [LX + 300, YTOP - 900],
                            [LX - 300, YTOP - 200]]])
        excl = SampleMask(diag.rings, "exclude")
        s_k = _only(_run([_north_line()], _plane(), self.cfg, diag)[0])
        s_e = _only(_run([_north_line()], _plane(), self.cfg, excl)[0])
        np.testing.assert_array_equal(s_k.n_valid + s_e.n_valid, s_all.n_valid)

    def test_minimum_valid_cells_is_honoured(self):
        # 15 cells per station survive the mask.
        for min_valid, expect_null in ((15, False), (16, True)):
            cfg = SwathConfig(swath_width=300.0, step=10.0, method="zonal",
                              min_valid=min_valid)
            s = _only(_run([_north_line()], _plane(), cfg, _rect(*EAST))[0])
            inner = slice(5, -5)
            self.assertTrue((s.n_valid[inner] == 15).all())
            self.assertEqual(bool(np.isnan(s.z_mean[inner]).all()), expect_null, min_valid)
            self.assertEqual(bool(np.isnan(s.hi[inner]).all()), expect_null, min_valid)

    def test_block_size_does_not_change_a_single_bit(self):
        """Fractional grid, ties forced on |u| == 1; with and without a mask."""
        n_cols, n_rows = 400, 300
        xs = FX0 + (np.arange(n_cols) + 0.5) * FW
        ys = FY0 - (np.arange(n_rows) + 0.5) * FW
        X, Y = np.meshgrid(xs, ys)
        arr = 0.37 * (X - FX0) + 0.91 * (FY0 - Y) + np.sin((X - FX0) * 0.7) * 3.0
        x_line = xs[200]                            # through a column of centres
        line = Baseline(1, 0, np.array([[x_line, ys[260]], [x_line, ys[40]]]),
                        float(ys[40] - ys[260]), "tie")
        mask = SampleMask([[[x_line - 9.0, ys[290]], [x_line + 5.0, ys[250]],
                            [x_line + 2.0, ys[10]], [x_line - 4.0, ys[30]]]])
        for m in (None, mask):
            out = []
            for budget in (10 ** 9, 2000):
                cfg = SwathConfig(swath_width=40 * FW, step=FW, method="zonal",
                                  max_block_cells=budget)
                ras = FakeRaster(arr, FX0, FY0, FW)
                out.append(_only(_run([line], ras, cfg, m)[0]))
            for key in ("z_min", "z_max", "z_mean", "z_median", "z_q1", "z_q3",
                        "z_plo", "z_phi", "n_valid"):
                np.testing.assert_array_equal(getattr(out[0], key), getattr(out[1], key),
                                              err_msg=f"{key} (mask={m is not None})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
