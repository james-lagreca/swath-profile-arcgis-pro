# -*- coding: utf-8 -*-
"""Analytic tests for the arcpy-free core: geometry, statistics, smoothing.

These run under any Python 3.11 with numpy + scipy; no ArcGIS needed.
Where a closed form exists it is asserted exactly, so a failure localises the
bug rather than merely signalling one.
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swathprofile import geometry as geom          # noqa: E402
from swathprofile import stats as st               # noqa: E402
from swathprofile.errors import GeometryError      # noqa: E402
from swathprofile.smoothing import (               # noqa: E402
    normalise_window, savgol_nan, smooth_rows,
)


class GeometryTest(unittest.TestCase):

    def test_resample_is_evenly_spaced_and_hits_the_ends(self):
        xy = np.array([[0.0, 0.0], [30.0, 40.0], [30.0, 140.0]])   # 50 + 100
        c, s = geom.resample_polyline(xy, 151)
        self.assertAlmostEqual(float(s[-1]), 150.0)
        step = np.diff(s)
        self.assertLess(float(np.ptp(step)), 1e-9)
        np.testing.assert_allclose(c[0], xy[0])
        np.testing.assert_allclose(c[-1], xy[-1])
        # midpoint of the first leg
        np.testing.assert_allclose(c[25], [15.0, 20.0], atol=1e-9)

    def test_left_normal_convention(self):
        c, _ = geom.resample_polyline(np.array([[0.0, 0.0], [100.0, 0.0]]), 11)
        t, n = geom.tangents_normals(c)
        np.testing.assert_allclose(t[0], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(n[0], [0.0, 1.0], atol=1e-12)
        p, offs = geom.offset_grid(c, n, 10.0, 3)
        np.testing.assert_allclose(offs, [-5.0, 0.0, 5.0])
        np.testing.assert_allclose(p[0, 0], [0.0, -5.0], atol=1e-12)   # right
        np.testing.assert_allclose(p[-1, 0], [0.0, 5.0], atol=1e-12)   # left

    def test_curvature_of_a_circle_is_one_over_radius(self):
        th = np.linspace(0.0, 2.0 * np.pi, 4001)
        circle = np.column_stack([100.0 * np.cos(th), 100.0 * np.sin(th)])
        c, s = geom.resample_polyline(circle, 4001)
        k = geom.curvature(c, s)
        self.assertAlmostEqual(float(np.median(k)), 0.01, places=6)

    def test_fold_mask_matches_the_jacobian_condition(self):
        """A band folds exactly where |offset| exceeds the radius of curvature."""
        th = np.linspace(0.0, 2.0 * np.pi, 2001)
        circle = np.column_stack([100.0 * np.cos(th), 100.0 * np.sin(th)])
        c, s = geom.resample_polyline(circle, 2001)
        k = geom.curvature(c, s)
        _, n = geom.tangents_normals(c)
        _, offs = geom.offset_grid(c, n, 300.0, 5)      # -150..150, R = 100
        folded = geom.fold_mask(offs, k)
        # Only the +150 line has 1 - o*kappa = 1 - 1.5 < 0.
        self.assertAlmostEqual(float(folded.mean()), 0.2, places=2)
        self.assertTrue(folded[-1].all())
        self.assertFalse(folded[:-1].any())

    def test_direction_curvature_matches_geometric_curvature(self):
        th = np.linspace(0.0, 2.0 * np.pi, 4001)
        circle = np.column_stack([100.0 * np.cos(th), 100.0 * np.sin(th)])
        c, s = geom.resample_polyline(circle, 4001)
        t, _ = geom.tangents_normals(c)
        kd = geom.direction_curvature(t, s)
        kg = geom.curvature(c, s)
        self.assertLess(float(np.max(np.abs(kd[5:-5] - kg[5:-5]))), 1e-6)

    def test_stabilising_removes_folds_at_a_kink(self):
        """A hand-drawn line has a corner; a wide band folds there unless the
        direction field is smoothed - and the fold test must then see the
        smoothed field, not the raw centreline."""
        verts = np.array([[0.0, 0.0], [3000.0, 0.0], [6500.0, 1600.0]])
        m = geom.station_count(geom.polyline_length(verts), 0.75)
        c, s = geom.resample_polyline(verts, m)
        _, offs = geom.offset_grid(c, np.zeros_like(c), 500.0, 51)

        t_raw, _ = geom.tangents_normals(c, 0)
        raw = geom.fold_mask(offs, geom.direction_curvature(t_raw, s))
        self.assertTrue(raw.any())

        window = int(np.clip(round(0.5 * 500.0 / 0.75), 5, 4001)) | 1
        t_sm, _ = geom.tangents_normals(c, window)
        smooth = geom.fold_mask(offs, geom.direction_curvature(t_sm, s))
        self.assertFalse(smooth.any())

    def test_straight_line_never_folds(self):
        c, s = geom.resample_polyline(np.array([[0.0, 0.0], [1000.0, 0.0]]), 201)
        k = geom.curvature(c, s)
        _, n = geom.tangents_normals(c)
        _, offs = geom.offset_grid(c, n, 5000.0, 21)
        self.assertFalse(geom.fold_mask(offs, k).any())

    def test_duplicate_vertices_are_removed(self):
        xy = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0],
                       [10.0, 0.0], [20.0, 0.0]])
        clean = geom.clean_vertices(xy)
        self.assertEqual(len(clean), 3)
        self.assertAlmostEqual(geom.polyline_length(clean), 20.0)

    def test_zero_length_line_is_rejected(self):
        with self.assertRaises(GeometryError):
            geom.clean_vertices(np.array([[5.0, 5.0], [5.0, 5.0]]))

    def test_station_count_and_actual_step(self):
        self.assertEqual(geom.station_count(1000.0, 10.0), 101)
        self.assertEqual(geom.station_count(5.0, 10.0), 2)      # shorter than step
        self.assertEqual(geom.station_count(1000.0, 3.0), 334)

    def test_band_ring_is_closed_and_correctly_sided(self):
        c, _ = geom.resample_polyline(np.array([[0.0, 0.0], [100.0, 0.0]]), 11)
        _, n = geom.tangents_normals(c)
        ring = geom.band_ring(c, n, 20.0)
        np.testing.assert_allclose(ring[0], ring[-1])
        self.assertAlmostEqual(float(ring[:, 1].max()), 10.0)
        self.assertAlmostEqual(float(ring[:, 1].min()), -10.0)

    def test_tangent_smoothing_stabilises_a_jittery_baseline(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 1000.0, 201)
        y = rng.normal(scale=2.0, size=x.size)          # noisy about y = 0
        c, _ = geom.resample_polyline(np.column_stack([x, y]), 201)
        _, n_raw = geom.tangents_normals(c, smooth_window=0)
        _, n_smooth = geom.tangents_normals(c, smooth_window=21)
        # The true normal is (0, 1); smoothing must reduce the scatter.
        self.assertLess(float(np.std(n_smooth[:, 0])),
                        float(np.std(n_raw[:, 0])) * 0.5)


class QuantileTest(unittest.TestCase):

    def test_weibull_matches_the_reference_position_formula(self):
        """The substitution that fixes D1 must be provably equivalent on clean data.

        This is what makes replacing the reference formula defensible rather
        than a leap of faith.
        """
        rng = np.random.default_rng(42)
        worst = 0.0
        for n in range(2, 61):
            v = rng.normal(size=(n, 37)) * 100.0
            srt = np.sort(v, axis=0)
            for q in (0.25, 0.5, 0.75):
                ref = st.position_quantile(srt, q)
                new = np.nanquantile(v, q, axis=0, method="weibull")
                worst = max(worst, float(np.max(np.abs(ref - new))))
        self.assertLess(worst, 1e-12)

    def test_nodata_breaks_the_reference_but_not_ours(self):
        """D1 regression: NaN sorts last, so the reference reads the wrong rank."""
        v = np.arange(1.0, 51.0).reshape(50, 1).repeat(3, axis=1)
        v[40:, :] = np.nan                       # valid population is 1..40
        expect_q1 = float(np.nanquantile(np.arange(1.0, 41.0), 0.25,
                                        method="weibull"))
        expect_q3 = float(np.nanquantile(np.arange(1.0, 41.0), 0.75,
                                        method="weibull"))

        ours = st.compute_stats(v, np.arange(3.0))
        self.assertAlmostEqual(float(ours.z_q1[0]), expect_q1, places=9)
        self.assertAlmostEqual(float(ours.z_q3[0]), expect_q3, places=9)

        legacy = st.compute_stats(v, np.arange(3.0), legacy=True)
        self.assertGreater(abs(float(legacy.z_q1[0]) - expect_q1), 1.0)
        self.assertGreater(abs(float(legacy.z_q3[0]) - expect_q3), 1.0)


class StatsTest(unittest.TestCase):

    def _plane_profile(self, n_lines=50, m=400, grad=3.0, width=2000.0):
        offs = np.linspace(-width / 2.0, width / 2.0, n_lines)
        along = np.linspace(0.0, 700.0, m)
        values = (grad * offs)[:, None] + along[None, :] + 100.0
        return values, np.linspace(0.0, 5000.0, m), along, offs, grad * width

    def test_plane_cross_section_gives_hi_half_and_thi_fifty(self):
        """The closed-form anchor.

        A linear cross-section is symmetric about its mean, so HI is exactly 0.5
        and, with relief constant along the profile, the relief factor is exactly
        1 and THi is exactly 50.
        """
        v, d, along, _, relief = self._plane_profile()
        s = st.compute_stats(v, d)
        self.assertLess(float(np.nanmax(np.abs(s.hi - 0.5))), 1e-12)
        self.assertLess(float(np.nanmax(np.abs(s.thi - 50.0))), 1e-9)
        self.assertLess(float(np.ptp(s.relief)), 1e-9)
        self.assertAlmostEqual(float(s.relief[0]), relief, places=6)
        self.assertLess(float(np.nanmax(np.abs(s.z_mean - (along + 100.0)))), 1e-9)

    def test_quartiles_of_a_uniform_cross_section(self):
        v, d, along, offs, _ = self._plane_profile(n_lines=50)
        s = st.compute_stats(v, d)
        # Weibull virtual index for n = 50: 0.25*51 = 12.75 (1-based) -> 11.75 0-based
        idx = 0.25 * 51.0 - 1.0
        lo = int(np.floor(idx))
        frac = idx - lo
        expect = 3.0 * (offs[lo] + frac * (offs[lo + 1] - offs[lo])) + along + 100.0
        self.assertLess(float(np.nanmax(np.abs(s.z_q1 - expect))), 1e-9)

    def test_flat_cross_section_is_undefined_not_zero(self):
        """D3: the reference's 1e-5 guard reports HI = 0 for flat ground."""
        v = np.full((20, 5), 500.0)
        v[:, 0] = np.linspace(0.0, 100.0, 20)     # one station with real relief
        s = st.compute_stats(v, np.arange(5.0))
        self.assertTrue(np.isnan(s.hi[1:]).all())
        self.assertTrue(np.isnan(s.thi[1:]).all())
        legacy = st.compute_stats(v, np.arange(5.0), legacy=True)
        np.testing.assert_allclose(legacy.hi[1:], 0.0)

    def test_thi_relief_factor_does_not_invert(self):
        """D4: below ratio exp(-5) the unclamped factor flips the index."""
        self.assertAlmostEqual(st.THI_FACTOR_ZERO_RATIO, float(np.exp(-5.0)))
        hi = np.array([0.9, 0.9])
        relief = np.array([1000.0, 1.0])          # ratio 1.0 and 0.001
        raw = st.terrain_hypsometric_index(hi, relief, 1000.0, clip_factor=False)
        fixed = st.terrain_hypsometric_index(hi, relief, 1000.0, clip_factor=True)
        self.assertGreater(raw[0], 50.0)
        self.assertLess(raw[1], 50.0)             # inverted: high HI reads low
        self.assertAlmostEqual(float(fixed[1]), 50.0)

    def test_thi_star_rescales_hi(self):
        hi = np.array([0.2, 0.5, 0.8])
        relief = np.full(3, 100.0)
        star = st.terrain_hypsometric_index(hi, relief, 100.0, enhanced=True)
        np.testing.assert_allclose(star, [0.0, 50.0, 100.0], atol=1e-9)

    def test_all_nodata_station_is_silent(self):
        """D2: an all-NaN column must not leak a RuntimeWarning."""
        v = np.full((10, 5), 1.0)
        v[:, 2] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = st.compute_stats(v, np.arange(5.0))
        self.assertEqual(int(s.n_valid[2]), 0)
        self.assertTrue(np.isnan(s.z_min[2]))
        self.assertEqual(s.n_empty_stations, 1)

    def test_single_profile_line_yields_null_hi(self):
        """D7: with one line relief is identically zero, so HI/THi are undefined."""
        v = np.linspace(0.0, 100.0, 20)[None, :]
        s = st.compute_stats(v, np.arange(20.0))
        self.assertTrue(np.isnan(s.hi).all())
        self.assertTrue(np.isnan(s.thi).all())
        self.assertEqual(s.n_total, 1)

    def test_min_valid_nulls_undersampled_stations(self):
        v = np.full((10, 4), 5.0)
        v[2:, 1] = np.nan                     # only 2 valid samples here
        s = st.compute_stats(v, np.arange(4.0), min_valid=5)
        self.assertTrue(np.isnan(s.z_mean[1]))
        self.assertFalse(np.isnan(s.z_mean[0]))


class GroupStatsTest(unittest.TestCase):
    """The vectorised per-station reduction must match numpy group by group."""

    def test_matches_numpy_per_group(self):
        from swathprofile.zonal import group_stats

        rng = np.random.default_rng(12)
        n_groups = 40
        grp = rng.integers(0, n_groups, size=5000)
        grp[grp == 7] = 8                       # leave group 7 empty on purpose
        vals = rng.normal(size=grp.size) * 50.0 + 300.0
        out = group_stats(grp, vals, n_groups)

        for gi in range(n_groups):
            v = vals[grp == gi]
            if v.size == 0:
                self.assertEqual(out["n_valid"][gi], 0)
                self.assertTrue(np.isnan(out["z_mean"][gi]))
                continue
            self.assertEqual(out["n_valid"][gi], v.size)
            self.assertAlmostEqual(out["z_min"][gi], v.min(), places=9)
            self.assertAlmostEqual(out["z_max"][gi], v.max(), places=9)
            self.assertAlmostEqual(out["z_mean"][gi], v.mean(), places=9)
            self.assertAlmostEqual(out["z_std"][gi], v.std(), places=7)
            q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75], method="weibull")
            self.assertAlmostEqual(out["z_q1"][gi], q1, places=9)
            self.assertAlmostEqual(out["z_median"][gi], med, places=9)
            self.assertAlmostEqual(out["z_q3"][gi], q3, places=9)

    def test_single_member_groups(self):
        from swathprofile.zonal import group_stats

        out = group_stats(np.array([0, 2]), np.array([5.0, 9.0]), 3)
        self.assertEqual(out["z_q1"][0], 5.0)
        self.assertEqual(out["z_q3"][2], 9.0)
        self.assertEqual(out["z_std"][0], 0.0)
        self.assertTrue(np.isnan(out["z_mean"][1]))


class SmoothingTest(unittest.TestCase):

    def test_savgol_reproduces_low_order_polynomials_exactly(self):
        x = np.arange(200.0)
        y = 3.0 + 0.5 * x - 0.001 * x ** 2      # degree 2 <= polyorder 3
        out = savgol_nan(y, 11)
        self.assertLess(float(np.max(np.abs(out - y))), 1e-8)

    def test_nan_positions_are_preserved(self):
        y = np.sin(np.linspace(0.0, 10.0, 200))
        y[50:60] = np.nan
        out = savgol_nan(y, 11)
        np.testing.assert_array_equal(np.isnan(out), np.isnan(y))

    def test_dilate_widens_the_hole_by_the_half_window(self):
        y = np.sin(np.linspace(0.0, 10.0, 200))
        y[100] = np.nan
        out = savgol_nan(y, 11, dilate=True)
        self.assertEqual(int(np.isnan(out).sum()), 11)

    def test_window_longer_than_series_is_a_no_op(self):
        y = np.arange(5.0)
        np.testing.assert_array_equal(savgol_nan(y, 11), y)

    def test_all_nan_series_survives(self):
        y = np.full(50, np.nan)
        self.assertTrue(np.isnan(savgol_nan(y, 11)).all())

    def test_normalise_window_forces_odd_and_minimum(self):
        self.assertEqual(normalise_window(0), 0)
        self.assertEqual(normalise_window(3), 5)
        self.assertEqual(normalise_window(10), 11)
        self.assertEqual(normalise_window(21), 21)

    def test_smooth_rows_matches_per_row_smoothing(self):
        rng = np.random.default_rng(5)
        v = rng.normal(size=(8, 300))
        v[3, 100:110] = np.nan
        bulk = smooth_rows(v, 11)
        for i in range(v.shape[0]):
            np.testing.assert_allclose(bulk[i], savgol_nan(v[i], 11), atol=1e-12)

    def test_smoothing_narrows_the_band(self):
        """D13: smoothing lines then reducing is not the same as smoothing curves."""
        rng = np.random.default_rng(9)
        v = rng.normal(size=(40, 500)) * 10.0 + 100.0
        raw = st.compute_stats(v, np.arange(500.0))
        sm = st.compute_stats(v, np.arange(500.0), smooth_window=21)
        self.assertGreater(float(np.nanmean(sm.z_min)), float(np.nanmean(raw.z_min)))
        self.assertLess(float(np.nanmean(sm.relief)), float(np.nanmean(raw.relief)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
