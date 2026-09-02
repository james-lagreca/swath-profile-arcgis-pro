# -*- coding: utf-8 -*-
"""Drive the geoprocessing tool through its real arcpy Parameter objects.

Everything else in the suite tests the package; this tests the wiring in
``tools/extract_swath_profile.py`` - parameter reading, unit conversion,
transect generation, every writer, layer files and charts - the way ArcGIS Pro
would call it, minus the map and the viewer window.
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from importlib.machinery import SourceFileLoader

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import arcpy  # noqa: E402

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
        pr["open_viewer"].value = False
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
        for name in ("out_band", "out_stations", "out_transects"):
            self.assertTrue(pr[name].symbology and os.path.exists(pr[name].symbology),
                            name)
        with open(pr["out_stations"].symbology, encoding="utf-8") as fh:
            self.assertIn('"charts"', fh.read())

        charts = sorted(os.listdir(os.path.join(self.folder, "charts")))
        self.assertEqual(charts, ["swath_stacked.png"] +
                         [f"swath_t00{i}_p0.png" for i in range(1, 6)])
        self.assertTrue(os.path.exists(pr["bundle_path"].valueAsText))
        self.assertTrue(any("5 transect(s) generated" in m for m in msg.infos))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
