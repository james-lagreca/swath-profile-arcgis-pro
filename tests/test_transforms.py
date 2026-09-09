# -*- coding: utf-8 -*-
"""Analytic tests for the display transforms, transect geometry, percentiles
and the run estimate.  No ArcGIS needed."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swathprofile import geometry as geom          # noqa: E402
from swathprofile import plotting                  # noqa: E402
from swathprofile import stats as st               # noqa: E402
from swathprofile import transforms as tf          # noqa: E402
from swathprofile.config import SwathConfig, estimate_run   # noqa: E402
from swathprofile.errors import ConfigError, GeometryError  # noqa: E402
from swathprofile.zonal import group_stats                  # noqa: E402


def _plane_stats(n_lines=21, m=200, grad=2.0, width=100.0, along=0.5,
                 base=100.0):
    offs = np.linspace(-width / 2.0, width / 2.0, n_lines)
    d = np.linspace(0.0, 2000.0, m)
    values = (grad * offs)[:, None] + (along * d)[None, :] + base
    return st.compute_stats(values, d, offsets=offs, pct_low=5, pct_high=95), d


class TransformsTest(unittest.TestCase):

    def test_detrend_of_a_plane_is_flat_and_keeps_the_envelope(self):
        s, d = _plane_stats()
        out = tf.detrend_linear(s)
        self.assertLess(float(np.nanmax(np.abs(out.z_mean))), 1e-6)
        # Same trend removed from every series: envelope width unchanged.
        np.testing.assert_allclose(out.z_max - out.z_min, s.z_max - s.z_min,
                                   atol=1e-9)
        np.testing.assert_allclose(out.relief, s.relief)      # untouched
        np.testing.assert_allclose(out.values, s.values - (0.5 * d + 100.0)[None, :],
                                   atol=1e-9)
        self.assertIsNot(out, s)                              # input unchanged
        self.assertGreater(float(np.nanmax(s.z_mean)), 100.0)

    def test_relative_to_min_zeroes_the_minimum(self):
        s, _ = _plane_stats()
        out = tf.relative_to_min(s)
        np.testing.assert_allclose(out.z_min, 0.0, atol=1e-9)
        np.testing.assert_allclose(out.z_mean, s.z_mean - s.z_min, atol=1e-9)
        np.testing.assert_allclose(out.z_plo, s.z_plo - s.z_min, atol=1e-9)

    def test_centreline_shifts_with_the_other_series(self):
        s, d = _plane_stats(n_lines=21)
        s.z_centre = s.values[10].copy()
        out = tf.detrend_linear(s)
        self.assertLess(float(np.nanmax(np.abs(out.z_centre))), 1e-6)
        rel = tf.relative_to_min(s)
        np.testing.assert_allclose(rel.z_centre, s.z_centre - s.z_min, atol=1e-9)
        # A stats object without a centreline still transforms.
        s.z_centre = None
        self.assertIsNone(tf.detrend_linear(s).z_centre)

    def test_reorient_round_trips_and_flips_offsets(self):
        d = np.linspace(0.0, 100.0, 11)
        o = np.linspace(-5.0, 5.0, 3)
        d2, o2 = tf.reorient(d, o, origin=40.0, flip=True)
        np.testing.assert_allclose(d2, -(d - 40.0))
        np.testing.assert_allclose(o2, -o)
        d3, o3 = tf.reorient(-d2, -o2, origin=-40.0, flip=False)
        np.testing.assert_allclose(d3, d)
        np.testing.assert_allclose(o3, o)

    def test_cross_section_of_a_plane_is_linear_in_offset(self):
        s, _ = _plane_stats(grad=2.0)
        xs = tf.cross_section(s, 50)
        self.assertEqual(xs["station"], 50)
        slope = np.polyfit(xs["offsets"], xs["z"], 1)[0]
        self.assertAlmostEqual(slope, 2.0, places=9)
        self.assertAlmostEqual(xs["z_min"], float(s.z_min[50]))
        flipped = tf.cross_section(s, 50, flip=True)
        np.testing.assert_allclose(flipped["offsets"], -xs["offsets"])

    def test_cross_section_is_none_without_samples(self):
        s, _ = _plane_stats()
        s2 = st.compute_stats(s.values, s.distances, keep_values=False)
        self.assertIsNone(tf.cross_section(s2, 3))

    def test_measure(self):
        m = tf.measure((0.0, 10.0), (100.0, 20.0))
        self.assertAlmostEqual(m["dd"], 100.0)
        self.assertAlmostEqual(m["dz"], 10.0)
        self.assertAlmostEqual(m["gradient"], 0.1)
        self.assertAlmostEqual(m["percent"], 10.0)
        self.assertAlmostEqual(m["angle_deg"], np.degrees(np.arctan(0.1)))
        self.assertAlmostEqual(m["length"], np.hypot(100.0, 10.0))
        v = tf.measure((5.0, 0.0), (5.0, 3.0))
        self.assertEqual(v["angle_deg"], 90.0)


class SideColouringTest(unittest.TestCase):
    """Colouring the profile lines by which side of the baseline they came from."""

    @staticmethod
    def _north_line(m=40):
        """A baseline drawn due north, so its left side faces west."""
        return np.column_stack([np.zeros(m), np.linspace(0.0, 400.0, m)])

    def test_side_of_splits_evenly_and_puts_the_centre_line_on_the_baseline(self):
        odd = geom.side_of(np.linspace(-150.0, 150.0, 21))
        self.assertEqual(int((odd == geom.LEFT).sum()), 10)
        self.assertEqual(int((odd == geom.RIGHT).sum()), 10)
        self.assertEqual(int((odd == geom.CENTRE).sum()), 1)
        # The centre class must be the same line the engine calls z_centre.
        offs = np.linspace(-1000.0, 1000.0, 101)
        self.assertEqual(int(np.flatnonzero(geom.side_of(offs) == geom.CENTRE)[0]),
                         int(np.argmin(np.abs(offs))))
        # An even count places no line on the baseline.
        self.assertEqual(int((geom.side_of(np.linspace(-8.0, 8.0, 4))
                              == geom.CENTRE).sum()), 0)

    def test_positive_offsets_are_left_of_the_digitising_direction(self):
        # Walking north, your left hand points west: a positive offset must
        # land at negative x.
        c = self._north_line()
        _, nrm = geom.tangents_normals(c)
        p, offs = geom.offset_grid(c, nrm, 100.0, 3)
        self.assertGreater(offs[-1], 0.0)
        self.assertLess(p[-1, 0, 0], 0.0)                      # +offset -> west
        bearing, resultant = geom.mean_normal_bearing(nrm)
        self.assertAlmostEqual(bearing, 270.0, places=6)
        self.assertAlmostEqual(resultant, 1.0, places=9)
        self.assertEqual(geom.compass_point(bearing), "W")

    def test_compass_points_and_wrapping(self):
        for bearing, expected in ((0.0, "N"), (90.0, "E"), (180.0, "S"),
                                  (270.0, "W"), (359.9, "N"), (22.5, "NNE"),
                                  (360.0 + 45.0, "NE")):
            self.assertEqual(geom.compass_point(bearing), expected, bearing)
        self.assertEqual(geom.compass_point(float("nan")), "")

    def test_a_curved_baseline_gets_no_compass_label(self):
        th = np.linspace(0.0, np.pi / 2, 60)
        c = np.column_stack([np.cos(th), np.sin(th)]) * 100.0
        _, nrm = geom.tangents_normals(c)
        bearing, resultant = geom.mean_normal_bearing(nrm)
        self.assertLess(resultant, 0.97)
        s, _ = _plane_stats()
        s.normal_bearing, s.normal_resultant = bearing, resultant
        # No single bearing is honest here, so the words stand alone.
        for label in plotting.side_labels(s):
            self.assertNotIn("(", label)

    def test_labels_name_the_side_the_ground_is_actually_on(self):
        c = self._north_line()
        _, nrm = geom.tangents_normals(c)
        p, offs = geom.offset_grid(c, nrm, 200.0, 11)
        z = 100.0 + 0.05 * p[..., 0]              # ground rises to the east
        s = st.compute_stats(z, np.linspace(0.0, 400.0, 40), offsets=offs)
        s.normal_bearing, s.normal_resultant = geom.mean_normal_bearing(nrm)

        for flip in (False, True):
            colours = plotting.profile_colours(offs, "side", flip=flip)
            left_rows = [i for i, col in enumerate(colours)
                         if col == plotting.PALETTE["side_left"]]
            right_rows = [i for i, col in enumerate(colours)
                          if col == plotting.PALETTE["side_right"]]
            higher = ("left" if s.values[left_rows].mean()
                      > s.values[right_rows].mean() else "right")
            label = plotting.side_labels(s, flip)[0 if higher == "left" else 1]
            # Whichever way round the display is, the higher ground is the east.
            self.assertIn("(E)", label, f"flip={flip}")

    def test_flipping_swaps_which_colour_each_side_gets(self):
        offs = np.linspace(-50.0, 50.0, 5)
        plain = plotting.profile_colours(offs, "side")
        flipped = plotting.profile_colours(offs, "side", flip=True)
        self.assertEqual(plain[0], flipped[-1])
        self.assertEqual(plain[-1], flipped[0])
        self.assertEqual(plain[2], plotting.PALETTE["centre"])
        self.assertEqual(flipped[2], plotting.PALETTE["centre"])

    def test_ramp_keeps_the_side_hues_and_fades_towards_the_baseline(self):
        from matplotlib.colors import to_rgb

        offs = np.linspace(-100.0, 100.0, 11)
        ramp = plotting.profile_colours(offs, "ramp")
        self.assertEqual(len(set(map(str, ramp))), len(offs))
        outer, inner = to_rgb(ramp[-1]), to_rgb(ramp[-2])
        far = to_rgb(plotting.PALETTE["side_left"])
        # The outermost line is the closest to the pure side colour.
        self.assertLess(sum((a - b) ** 2 for a, b in zip(outer, far)),
                        sum((a - b) ** 2 for a, b in zip(inner, far)))

    def test_figure_legend_names_both_sides_and_default_is_unchanged(self):
        c = self._north_line()
        _, nrm = geom.tangents_normals(c)
        p, offs = geom.offset_grid(c, nrm, 100.0, 9)
        z = 50.0 + 0.01 * p[..., 1]
        s = st.compute_stats(z, np.linspace(0.0, 400.0, 40), offsets=offs)
        s.normal_bearing, s.normal_resultant = geom.mean_normal_bearing(nrm)

        fig, axes = plotting.build_figure({"dem": s}, profile_colour="side")
        labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        self.assertIn("Left (W)", labels)
        self.assertIn("Right (E)", labels)

        # Default: no side entries, and the lines keep the historic grey.
        fig2, axes2 = plotting.build_figure({"dem": s})
        labels2 = [t.get_text() for t in axes2[0].get_legend().get_texts()]
        self.assertNotIn("Left (W)", labels2)
        greys = [ln for ln in axes2[0].get_lines()
                 if ln.get_color() == plotting.PALETTE["profiles"]]
        self.assertEqual(len(greys), len(offs))

    def test_all_cells_mode_has_nothing_to_colour_and_does_not_raise(self):
        s, d = _plane_stats()
        s.values, s.offsets = None, None          # what zonal sampling produces
        fig, axes = plotting.build_figure({"dem": s}, profile_colour="side")
        labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        self.assertNotIn("Left of line", labels)
        for text in labels:
            self.assertFalse(text.startswith("Left ("))


class TransectsTest(unittest.TestCase):

    def test_spacing_length_and_perpendicularity_on_a_straight_trace(self):
        trace = np.array([[0.0, 0.0], [1000.0, 0.0]])
        tr = geom.transects_along(trace, spacing=100.0, length=300.0)
        self.assertEqual(len(tr["centres"]), 11)
        np.testing.assert_allclose(tr["dist_along"], np.arange(0, 1001, 100))
        np.testing.assert_allclose(tr["centres"][:, 1], 0.0, atol=1e-9)
        lengths = np.linalg.norm(tr["ends"][:, 1] - tr["ends"][:, 0], axis=1)
        np.testing.assert_allclose(lengths, 300.0)
        # Perpendicular to an E-W trace: transects run N-S, right end south.
        np.testing.assert_allclose(tr["ends"][:, 0, 1], -150.0, atol=1e-9)
        np.testing.assert_allclose(tr["ends"][:, 1, 1], 150.0, atol=1e-9)
        np.testing.assert_allclose(tr["azimuth"], 0.0, atol=1e-6)   # towards north

    def test_one_sided_transects(self):
        trace = np.array([[0.0, 0.0], [1000.0, 0.0]])
        left = geom.transects_along(trace, 250.0, 200.0, side="left")
        np.testing.assert_allclose(left["ends"][:, 0], left["centres"])
        np.testing.assert_allclose(left["ends"][:, 1, 1], 200.0)
        right = geom.transects_along(trace, 250.0, 200.0, side="right")
        np.testing.assert_allclose(right["ends"][:, 1], right["centres"])
        np.testing.assert_allclose(right["ends"][:, 0, 1], -200.0)

    def test_transects_follow_a_curved_trace(self):
        th = np.linspace(0.0, np.pi / 2, 200)
        trace = np.column_stack([500.0 * np.cos(th), 500.0 * np.sin(th)])
        tr = geom.transects_along(trace, spacing=50.0, length=100.0)
        # On a circle the transect direction is radial: ends align with centre.
        for c, ends in zip(tr["centres"], tr["ends"]):
            radial = c / np.linalg.norm(c)
            direction = (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0])
            self.assertGreater(abs(float(np.dot(radial, direction))), 0.999)

    def test_start_offset_and_validation(self):
        trace = np.array([[0.0, 0.0], [1000.0, 0.0]])
        tr = geom.transects_along(trace, 300.0, 50.0, start=120.0)
        np.testing.assert_allclose(tr["dist_along"], [120.0, 420.0, 720.0])
        with self.assertRaises(GeometryError):
            geom.transects_along(trace, 0.0, 50.0)
        with self.assertRaises(GeometryError):
            geom.transects_along(trace, 100.0, 50.0, start=5000.0)


class PercentilesTest(unittest.TestCase):

    def test_compute_stats_percentiles_match_numpy(self):
        rng = np.random.default_rng(3)
        v = rng.normal(size=(60, 20)) * 10.0
        s = st.compute_stats(v, np.arange(20.0), pct_low=10, pct_high=90)
        np.testing.assert_allclose(
            s.z_plo, np.nanquantile(v, 0.10, axis=0, method="weibull"), atol=1e-12)
        np.testing.assert_allclose(
            s.z_phi, np.nanquantile(v, 0.90, axis=0, method="weibull"), atol=1e-12)
        self.assertTrue(np.all(s.z_plo <= s.z_q1 + 1e-12))
        self.assertTrue(np.all(s.z_q3 <= s.z_phi + 1e-12))

    def test_group_stats_percentiles_match_numpy(self):
        rng = np.random.default_rng(4)
        grp = rng.integers(0, 5, size=800)
        vals = rng.normal(size=800)
        out = group_stats(grp, vals, 5, pct=(2.5, 97.5))
        for g in range(5):
            v = vals[grp == g]
            lo, hi = np.quantile(v, [0.025, 0.975], method="weibull")
            self.assertAlmostEqual(out["z_plo"][g], lo, places=12)
            self.assertAlmostEqual(out["z_phi"][g], hi, places=12)

    def test_config_rejects_bad_percentiles(self):
        with self.assertRaises(ConfigError):
            SwathConfig(swath_width=10.0, step=1.0, pct_low=95, pct_high=5)
        with self.assertRaises(ConfigError):
            SwathConfig(swath_width=10.0, step=1.0, pct_low=0, pct_high=50)


class RidgelineTest(unittest.TestCase):

    def _entries(self, n=4):
        entries = []
        for k in range(n):
            s, _ = _plane_stats(n_lines=11, m=120, grad=1.0, width=50.0,
                                along=0.2 * (k + 1), base=100.0 * k)
            s.z_centre = s.values[5].copy()
            entries.append((f"T{k + 1:03d}", {"dem": s}, None))
        return entries

    def test_ridgeline_orders_labels_and_offsets(self):
        from swathprofile import plotting

        plotting.use_agg()
        fig, axes = plotting.build_ridgeline_figure(self._entries(), title="t")
        self.assertEqual(len(axes), 1)
        ax = axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(labels, ["T001", "T002", "T003", "T004"])
        ticks = list(ax.get_yticks())
        # First profile on top: offsets decrease down the list.
        self.assertTrue(all(a > b for a, b in zip(ticks, ticks[1:])))
        spacing = ticks[0] - ticks[1]
        self.assertGreater(spacing, 0.0)
        self.assertAlmostEqual(ticks[-1], 0.0)

    def test_ridgeline_options_and_save(self):
        import tempfile

        from swathprofile import plotting

        plotting.use_agg()
        for series, align, colour in (("z_centre", "mean", "thi"),
                                      ("z_max", "none", "elevation"),
                                      ("z_mean", "min", "none")):
            fig, axes = plotting.build_ridgeline_figure(
                self._entries(3), series=series, align=align, colour_by=colour,
                envelope=(align == "min"), spacing=25.0, first_on_top=False)
            ticks = list(axes[0].get_yticks())
            np.testing.assert_allclose(ticks, [0.0, 25.0, 50.0])
        path = os.path.join(tempfile.gettempdir(), "ridge_test.png")
        plotting.save_figure(fig, path, dpi=72)
        self.assertGreater(os.path.getsize(path), 5000)
        os.remove(path)


class CimChartTest(unittest.TestCase):

    def test_line_chart_structure(self):
        from swathprofile import cimchart

        ch = cimchart.line_chart("DIST", [("A_MIN", "Min", (1, 2, 3)),
                                          ("A_MEAN", "Mean", (4, 5, 6))],
                                 title="t", x_title="x", y_title="y")
        self.assertEqual(ch["type"], "CIMChart")
        self.assertEqual([s["fields"] for s in ch["series"]],
                         [["DIST", "A_MIN"], ["DIST", "A_MEAN"]])
        self.assertEqual(ch["mapSelectionHandling"], "Highlight")
        self.assertEqual(len(ch["axes"]), 2)
        self.assertEqual(ch["axes"][0]["title"], "x")
        self.assertFalse(ch["axes"][0]["useAutomaticTitle"])
        self.assertEqual(ch["generalProperties"]["title"], "t")

    def test_inject_chart_round_trip(self):
        import json
        import tempfile

        from swathprofile import cimchart

        doc = {"type": "CIMLayerDocument", "layerDefinitions": [{"type": "CIMFeatureLayer"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".lyrx", delete=False) as fh:
            json.dump(doc, fh)
            path = fh.name
        cimchart.inject_chart(path, cimchart.line_chart("X", [("Y", "y", (0, 0, 0))], "t"))
        with open(path, encoding="utf-8") as fh:
            back = json.load(fh)
        self.assertEqual(len(back["layerDefinitions"][0]["charts"]), 1)
        os.remove(path)


class EstimateTest(unittest.TestCase):

    def test_estimate_is_monotonic_and_labelled(self):
        base = estimate_run(3000.0, 0.75, 51, 500.0, 0.5, "offset")
        longer = estimate_run(6000.0, 0.75, 51, 500.0, 0.5, "offset")
        wider = estimate_run(3000.0, 0.75, 51, 1000.0, 0.5, "zonal")
        self.assertGreater(longer["work"], base["work"])
        self.assertGreater(longer["seconds"], base["seconds"])
        self.assertIn("samples", base["text"])
        self.assertIn("cells", wider["text"])
        self.assertEqual(base["stations"], int(3000.0 / 0.75) + 1)
        more_lines = estimate_run(3000.0, 0.75, 401, 500.0, 0.5, "offset")
        self.assertGreater(more_lines["work"], base["work"])
        self.assertIn("min", estimate_run(50_000.0, 0.5, 1001, 2000.0, 0.5)["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
