# -*- coding: utf-8 -*-
"""Drive the geoprocessing tool through its real arcpy Parameter objects.

Everything else in the suite tests the package; this tests the wiring in
``tools/extract_swath_profile.py`` - parameter reading, unit conversion,
transect generation, every writer, layer files and charts - the way ArcGIS Pro
would call it, minus the map and the viewer window.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import arcpy  # noqa: E402

from swathprofile import chartwindow, plotting         # noqa: E402
from swathprofile.errors import SwathError              # noqa: E402
from swathprofile.messaging import CollectingMessenger  # noqa: E402

from . import synth_arcpy as synth  # noqa: E402

A, B, C = 3.0, 7.0, 100.0
X0, Y0 = 1_500_000.0, 5_000_000.0
CW = CH = 10.0


class ToolEndToEndTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        arcpy.env.overwriteOutput = True
        cls.folder, cls.gdb = synth.make_scratch()
        cls.raster = synth.write_raster(
            synth.plane_array(A, B, C, X0, Y0, CW, CH, 400, 300),
            cls.gdb, "plane", X0, Y0, CW, CH)
        cls.trace = synth.write_line(
            [(X0 + 600.0, Y0 + 1500.0), (X0 + 3400.0, Y0 + 1500.0)],
            cls.gdb, "fault_trace")
        cls.out_gdb = arcpy.management.CreateFileGDB(cls.folder, "results.gdb")[0]
        module = SourceFileLoader("swpyt_e2e", os.path.join(ROOT, "SwathProfile.pyt")
                                  ).load_module()
        cls.tool_cls = module.SwathProfileTool

    @classmethod
    def tearDownClass(cls):
        arcpy.management.ClearWorkspaceCache()
        shutil.rmtree(cls.folder, ignore_errors=True)

    def _params(self):
        tool = self.tool_cls()
        params = tool.getParameterInfo()
        pr = {p.name: p for p in params}
        pr["in_rasters"].values = [self.raster]
        pr["swath_width"].value = "300 Meters"
        pr["step"].value = "15 Meters"
        pr["open_viewer"].value = "Off"
        pr["bundle_path"].value = os.path.join(self.folder, "live.swath.npz")
        return tool, params, pr

    def test_transects_end_to_end(self):
        tool, params, pr = self._params()
        pr["transect_source"].value = self.trace
        pr["transect_spacing"].value = "700 Meters"
        pr["transect_length"].value = "1200 Meters"
        for name in ("out_table", "out_band", "out_stations", "out_transects",
                     "out_offset_lines", "out_profiles_3d", "out_summary"):
            pr[name].value = os.path.join(self.out_gdb, name)
        pr["out_csv_folder"].value = os.path.join(self.folder, "csv")
        pr["chart_formats"].values = ["PNG"]
        pr["out_chart_folder"].value = os.path.join(self.folder, "charts")

        tool.updateParameters(params)
        tool.updateMessages(params)
        self.assertFalse(any(p.hasError() for p in params),
                         [(p.name, p.message) for p in params if p.hasError()])

        msg = CollectingMessenger()
        tool._run(params, msg)

        names = [r[0] for r in arcpy.da.SearchCursor(
            os.path.join(self.out_gdb, "out_transects"), ["LINE_NAME"])]
        self.assertEqual(names, ["T001", "T002", "T003", "T004", "T005"])
        table_names = {r[0] for r in arcpy.da.SearchCursor(
            os.path.join(self.out_gdb, "out_table"), ["LINE_NAME"])}
        self.assertEqual(table_names, set(names))
        self.assertEqual(int(arcpy.management.GetCount(
            os.path.join(self.out_gdb, "out_band"))[0]), 5)
        self.assertEqual(int(arcpy.management.GetCount(
            os.path.join(self.out_gdb, "out_stations"))[0]), 405)

        # Every styled output got a layer file; the stations one carries the chart.
        for name in ("out_band", "out_stations", "out_transects", "out_offset_lines"):
            self.assertTrue(pr[name].symbology and os.path.exists(pr[name].symbology),
                            name)

        # The offset lines say which side of the baseline each came from, and
        # the layer file symbolises on that field in the chart's colours.
        offsets_fc = os.path.join(self.out_gdb, "out_offset_lines")
        rows = [(r[0], r[1]) for r in
                arcpy.da.SearchCursor(offsets_fc, ["OFFSET", "SIDE"])]
        self.assertTrue(rows)
        for offset, side in rows:
            expected = ("Centre" if abs(offset) < 1e-9
                        else ("Left" if offset > 0 else "Right"))
            self.assertEqual(side, expected, offset)
        self.assertEqual({s for _, s in rows}, {"Left", "Centre", "Right"})
        with open(pr["out_offset_lines"].symbology, encoding="utf-8") as fh:
            renderer = json.load(fh)["layerDefinitions"][0]["renderer"]
        self.assertEqual(renderer["fields"], ["SIDE"])
        seen = {}
        for cls in renderer["groups"][0]["classes"]:
            rgb = cls["symbol"]["symbol"]["symbolLayers"][0]["color"]["values"][:3]
            seen[cls["values"][0]["fieldValues"][0]] = "#%02X%02X%02X" % tuple(rgb)
        self.assertEqual(seen["Left"], plotting.PALETTE["side_left"].upper())
        self.assertEqual(seen["Right"], plotting.PALETTE["side_right"].upper())
        self.assertEqual(seen["Centre"], plotting.PALETTE["centre"].upper())
        with open(pr["out_stations"].symbology, encoding="utf-8") as fh:
            self.assertIn('"charts"', fh.read())

        charts = sorted(os.listdir(os.path.join(self.folder, "charts")))
        self.assertEqual(charts, ["swath_stacked.png"] +
                         [f"swath_t00{i}_p0.png" for i in range(1, 6)])
        self.assertTrue(os.path.exists(pr["bundle_path"].valueAsText))
        self.assertTrue(any("5 transect(s) generated" in m for m in msg.infos))
        # The browser chart is written beside the permanent result copy on every
        # run, whatever Show Interactive Chart says, and holds all five transects.
        page = os.path.join(self.folder, "swath_results", "out_table.html")
        self.assertTrue(os.path.exists(page))
        with open(page, encoding="utf-8") as fh:
            text = fh.read()
        for name in ("T001", "T005"):
            self.assertIn(f'"name":"{name}"', text)
        self.assertTrue(any("Browser chart ->" in m for m in msg.infos))

        # On a plane every transect gives HI = 0.5 (centre column of the table).
        hi = [r[0] for r in arcpy.da.SearchCursor(
            os.path.join(self.out_gdb, "out_table"), ["HI"]) if r[0] is not None]
        self.assertLess(float(np.max(np.abs(np.array(hi) - 0.5))), 1e-6)

    def test_transects_can_use_their_own_swath_width(self):
        """A drawn/layer baseline at 300 m and transects at 120 m in one run."""
        tool, params, pr = self._params()
        baseline = synth.write_line(
            [(X0 + 2000.0, Y0 + 400.0), (X0 + 2000.0, Y0 + 2600.0)],
            self.gdb, "long_baseline")
        pr["in_lines"].value = baseline
        pr["transect_source"].value = self.trace
        pr["transect_spacing"].value = "1400 Meters"
        pr["transect_length"].value = "800 Meters"
        pr["transect_width"].value = "120 Meters"
        pr["out_table"].value = os.path.join(self.out_gdb, "mixed_table")
        pr["out_band"].value = os.path.join(self.out_gdb, "mixed_band")
        pr["out_stations"].value = os.path.join(self.out_gdb, "mixed_stations")
        pr["out_summary"].value = os.path.join(self.out_gdb, "mixed_summary")
        pr["out_transects"].value = os.path.join(self.out_gdb, "mixed_transects")
        pr["chart_formats"].values = ["PNG"]
        pr["out_chart_folder"].value = os.path.join(self.folder, "charts_mixed")

        tool.updateParameters(params)
        tool.updateMessages(params)
        msg = CollectingMessenger()
        tool._run(params, msg)

        widths = {r[0]: r[1] for r in arcpy.da.SearchCursor(
            os.path.join(self.out_gdb, "mixed_band"), ["LINE_NAME", "SWATH_W"])}
        self.assertEqual(widths[""], 300.0)                       # layer baseline
        self.assertEqual({v for k, v in widths.items() if k.startswith("T")}, {120.0})
        self.assertEqual(len(widths), 4)                          # 1 + 3 transects
        summary = {r[0]: r[1] for r in arcpy.da.SearchCursor(
            os.path.join(self.out_gdb, "mixed_summary"), ["LINE_NAME", "SWATH_W"])}
        self.assertEqual(summary["T001"], 120.0)
        self.assertTrue(any("Transects use a swath width of 120" in m for m in msg.infos))
        # Relief on the plane scales with the band width.  Both the baseline
        # and the transects (perpendicular to an E-W trace) run N-S, so the
        # band spans x and relief is A (3 m/m) times the width.
        relief = {r[0]: r[1] for r in arcpy.da.SearchCursor(
            os.path.join(self.out_gdb, "mixed_summary"), ["LINE_NAME", "MAX_RELIEF"])}
        self.assertAlmostEqual(relief[""], 3.0 * 300.0, delta=1.0)
        self.assertAlmostEqual(relief["T001"], 3.0 * 120.0, delta=1.0)
        # Several profiles: the stacked figure is written alongside the singles.
        self.assertIn("swath_stacked.png", os.listdir(os.path.join(self.folder,
                                                                   "charts_mixed")))

    def test_layer_baseline_with_label_field(self):
        tool, params, pr = self._params()
        arcpy.management.AddField(self.trace, "NAME", "TEXT", field_length=32)
        with arcpy.da.UpdateCursor(self.trace, ["NAME"]) as cur:
            for row in cur:
                cur.updateRow(["Alpine Fault"])
        pr["in_lines"].value = self.trace
        pr["label_field"].value = "NAME"
        pr["out_table"].value = os.path.join(self.out_gdb, "named_table")
        pr["out_band"].value = os.path.join(self.out_gdb, "named_band")
        pr["out_stations"].value = os.path.join(self.out_gdb, "named_stations")

        tool.updateParameters(params)
        tool.updateMessages(params)
        msg = CollectingMessenger()
        tool._run(params, msg)

        names = {r[0] for r in arcpy.da.SearchCursor(
            os.path.join(self.out_gdb, "named_table"), ["LINE_NAME"])}
        self.assertEqual(names, {"Alpine Fault"})
        centre = [r[0] for r in arcpy.da.SearchCursor(
            os.path.join(self.out_gdb, "named_table"), ["Z_CENTRE"])]
        self.assertTrue(all(v is not None for v in centre))

    def test_add_to_map_off_sets_the_environment(self):
        tool, params, pr = self._params()
        pr["in_lines"].value = self.trace
        pr["out_table"].value = os.path.join(self.out_gdb, "quick_table")
        pr["out_band"].value = ""
        pr["out_stations"].value = ""
        pr["add_to_map"].value = False
        seen = {}

        original = tool._run

        def spy(params_, msg_):
            seen["add"] = arcpy.env.addOutputsToMap
            return original(params_, msg_)

        tool._run = spy
        before = arcpy.env.addOutputsToMap
        tool.execute(params, None)
        self.assertFalse(seen["add"])
        self.assertEqual(arcpy.env.addOutputsToMap, before)     # restored

    # -- sampling mask --------------------------------------------------------

    LX = X0 + 2000.0                     # a cell edge: no cell centre on the line

    def _ns_line(self, name):
        """A baseline drawn northwards; its left (positive offsets) is west."""
        return synth.write_line([(self.LX, Y0 + 400.0), (self.LX, Y0 + 2600.0)],
                                self.gdb, name)

    def _east(self, name, y0=Y0 - 100.0, y1=Y0 + 3100.0):
        return synth.write_polygons([synth.rect_wkt(self.LX + 3.0, y0,
                                                    self.LX + 400.0, y1)],
                                    self.gdb, name)

    def _outputs(self, pr, stem):
        for name in ("out_table", "out_band", "out_stations", "out_offset_lines",
                     "out_summary"):
            pr[name].value = os.path.join(self.out_gdb, f"{stem}_{name}")
        return lambda name: os.path.join(self.out_gdb, f"{stem}_{name}")

    def _ready(self, tool, params):
        tool.updateParameters(params)
        tool.updateMessages(params)
        self.assertFalse(any(p.hasError() for p in params),
                         [(p.name, p.message) for p in params if p.hasError()])

    def test_keep_mask_from_a_layer(self):
        tool, params, pr = self._params()
        pr["in_lines"].value = self._ns_line("mk_line")
        pr["mask_layer"].value = self._east("mk_poly")
        out = self._outputs(pr, "mk")
        self._ready(tool, params)
        self.assertTrue(pr["mask_mode"].enabled)
        msg = CollectingMessenger()
        tool._run(params, msg)

        # 51 lines 6 m apart; east of LX + 3 are the 25 at offsets -6 ... -150.
        n_valid = [r[0] for r in arcpy.da.SearchCursor(out("out_table"), ["N_VALID"])]
        self.assertTrue(n_valid and all(v == 25 for v in n_valid))
        kept, params_json = next(arcpy.da.SearchCursor(out("out_summary"),
                                                       ["MASK_KEPT", "PARAMS"]))
        self.assertAlmostEqual(kept, 100.0 * 147.0 / 300.0, places=3)
        self.assertIn('"mask_mode": "keep"', params_json)
        area = sum(r[0] for r in arcpy.da.SearchCursor(out("out_band"), ["SHAPE@AREA"]))
        self.assertAlmostEqual(area, 147.0 * 2200.0, delta=1.0)
        offsets = [r[0] for r in arcpy.da.SearchCursor(out("out_offset_lines"),
                                                       ["OFFSET"])]
        self.assertEqual(len(offsets), 25)
        self.assertTrue(all(o < 0 for o in offsets))
        self.assertTrue(any("Sampling mask keeps 49%" in m for m in msg.infos))
        self.assertTrue(any("Minimum Valid Samples" in m for m in msg.infos))

    def test_exclude_mask_cuts_lines_part_way(self):
        tool, params, pr = self._params()
        pr["in_lines"].value = self._ns_line("mx_line")
        pr["mask_layer"].value = self._east("mx_poly", y0=Y0 + 100.0, y1=Y0 + 1500.0)
        pr["mask_mode"].value = "Exclude inside"
        out = self._outputs(pr, "mx")
        self._ready(tool, params)
        tool._run(params, CollectingMessenger())

        n_stations = sum(1 for _ in arcpy.da.SearchCursor(out("out_table"), ["OID@"]))
        rows = list(arcpy.da.SearchCursor(out("out_offset_lines"),
                                          ["OFFSET", "N_VALID", "SHAPE@"]))
        self.assertEqual(len(rows), 51)            # every line keeps something
        for offset, n_valid, shape in rows:
            if offset < 0:                         # east: cut where the mask ends
                self.assertAlmostEqual(shape.extent.YMin, Y0 + 1500.0, places=3)
                self.assertLess(n_valid, n_stations)
            else:                                  # west: untouched
                self.assertAlmostEqual(shape.extent.YMin, Y0 + 400.0, places=3)
                self.assertEqual(n_valid, n_stations)

    def test_an_unfinished_mask_sketch_is_ignored_with_a_warning(self):
        """Outside Pro a sketch parameter keeps only the schema of what it is
        given - exactly the state of a sketch the user has started but not
        finished.  Reading a sketch that holds polygons is tested directly in
        test_end_to_end.test_read_mask_from_a_dialog_sketch."""
        tool, params, pr = self._params()
        pr["in_lines"].value = self._ns_line("me_line")
        fs = arcpy.FeatureSet()
        fs.load(self._east("me_poly"))
        pr["mask_draw"].value = fs
        out = self._outputs(pr, "me")
        tool.updateParameters(params)
        tool.updateMessages(params)
        self.assertTrue(pr["mask_draw"].hasWarning())
        self.assertIn("double-click", pr["mask_draw"].message)
        msg = CollectingMessenger()
        tool._run(params, msg)
        self.assertTrue(any("not masked" in w for w in msg.warnings))
        n_valid = [r[0] for r in arcpy.da.SearchCursor(out("out_table"), ["N_VALID"])]
        self.assertTrue(all(v == 51 for v in n_valid))
        kept = next(arcpy.da.SearchCursor(out("out_summary"), ["MASK_KEPT"]))[0]
        self.assertIsNone(kept)

    def test_a_keep_mask_that_misses_every_swath_stops_the_run(self):
        tool, params, pr = self._params()
        pr["in_lines"].value = self._ns_line("mf_line")
        pr["mask_layer"].value = synth.write_polygons(
            [synth.rect_wkt(X0 + 10, Y0 + 10, X0 + 100, Y0 + 100)], self.gdb, "mf_poly")
        out = self._outputs(pr, "mf")
        self._ready(tool, params)
        with self.assertRaises(SwathError) as ctx:
            tool._run(params, CollectingMessenger())
        self.assertIn("does not overlap any swath", str(ctx.exception))
        self.assertFalse(arcpy.Exists(out("out_table")))

    def test_zonal_keep_mask_halves_the_cells(self):
        def run(stem, mask):
            tool, params, pr = self._params()
            pr["in_lines"].value = self._ns_line(f"{stem}_line")
            pr["method"].value = "All cells (exact envelope)"
            if mask:
                pr["mask_layer"].value = self._east(f"{stem}_poly")
            out = self._outputs(pr, stem)
            self._ready(tool, params)
            tool._run(params, CollectingMessenger())
            return out, np.array([r[0] for r in arcpy.da.SearchCursor(
                out("out_table"), ["N_VALID"])])

        _, full = run("mz0", False)
        out, half = run("mz1", True)
        inner = slice(3, -3)
        np.testing.assert_array_equal(half[inner] * 2, full[inner])
        kept = next(arcpy.da.SearchCursor(out("out_summary"), ["MASK_KEPT"]))[0]
        self.assertAlmostEqual(kept, 49.0, places=3)

    def test_transects_with_their_own_width_carry_the_mask(self):
        tool, params, pr = self._params()
        pr["transect_source"].value = self.trace
        pr["transect_spacing"].value = "700 Meters"
        pr["transect_length"].value = "1200 Meters"
        pr["transect_width"].value = "120 Meters"
        # Keep only the northern half of the transects (the trace runs E-W).
        pr["mask_layer"].value = synth.write_polygons(
            [synth.rect_wkt(X0, Y0 + 1500.0, X0 + 4000.0, Y0 + 3000.0)], self.gdb,
            "mt_poly")
        out = self._outputs(pr, "mt")
        self._ready(tool, params)
        tool._run(params, CollectingMessenger())
        rows = list(arcpy.da.SearchCursor(out("out_summary"),
                                          ["SWATH_W", "MASK_KEPT", "PARAMS"]))
        self.assertEqual(len(rows), 5)
        for width, kept, params_json in rows:
            self.assertAlmostEqual(width, 120.0)
            self.assertAlmostEqual(kept, 50.0, places=3)
            self.assertIn('"mask_mode": "keep"', params_json)


class ChartDisplayWiringTest(unittest.TestCase):
    """The tool's side of Show Interactive Chart.  The decisions themselves are
    in swathprofile.chartwindow and tested in CI (tests/test_chartwindow.py)."""

    @classmethod
    def setUpClass(cls):
        module = SourceFileLoader("swpyt_display", os.path.join(ROOT, "SwathProfile.pyt")
                                  ).load_module()
        cls.tool_cls = module.SwathProfileTool
        cls.mod = sys.modules[module.SwathProfileTool.__module__]

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="swathwire_")
        self.env = mock.patch.dict(os.environ, {chartwindow.STATE_DIR_ENV: self.tmp})
        self.env.start()
        chartwindow.SESSION.clear()

    def tearDown(self):
        chartwindow.SESSION.clear()
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _params(self):
        tool = self.tool_cls()
        params = tool.getParameterInfo()
        return tool, params, {p.name: p for p in params}

    def test_the_default_is_automatic(self):
        _, _, pr = self._params()
        self.assertEqual(pr["open_viewer"].value, "Automatic")
        self.assertEqual(list(pr["open_viewer"].filter.list),
                         ["Automatic", "Chart window", "Browser page", "Off"])

    def test_old_true_and_false_become_choices_before_validation(self):
        for given, want in ((False, "Off"), ("false", "Off"), (True, "Automatic"),
                            ("TRUE", "Automatic"), ("browser page", "Browser page")):
            tool, params, pr = self._params()
            pr["open_viewer"].value = given
            tool.updateParameters(params)
            self.assertEqual(pr["open_viewer"].valueAsText, want, given)

    def test_choosing_the_window_warns_where_it_is_known_not_to_start(self):
        chartwindow.remember("failed", "denied", persist=True)
        chartwindow.SESSION.clear()
        tool, params, pr = self._params()
        pr["open_viewer"].value = "Chart window"
        tool.updateParameters(params)
        tool.updateMessages(params)
        self.assertTrue(pr["open_viewer"].hasWarning())
        self.assertIn("could not start on this computer", pr["open_viewer"].message)
        tool, params, pr = self._params()
        tool.updateParameters(params)
        tool.updateMessages(params)
        self.assertFalse(pr["open_viewer"].hasWarning())      # Automatic: no warning

    def test_the_tool_hands_its_viewer_and_python_to_chartwindow(self):
        with mock.patch.object(chartwindow, "show_chart", return_value="page") as show:
            self.tool_cls._show_chart("Browser page", "b.swath.npz", "p.html",
                                      CollectingMessenger())
        kwargs = show.call_args.kwargs
        self.assertEqual(show.call_args.args, ("Browser page",))
        self.assertEqual(kwargs["viewer"], self.mod.VIEWER_SCRIPT)
        self.assertEqual(kwargs["root"], self.mod.ROOT)
        self.assertIs(kwargs["find_python"], self.mod._python_for_viewer)
        self.assertEqual((kwargs["bundle"], kwargs["page"]), ("b.swath.npz", "p.html"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
