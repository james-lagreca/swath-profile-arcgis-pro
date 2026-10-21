# -*- coding: utf-8 -*-
"""arcpy-dependent tests for the new outputs: station points, transects, bundle
v2, the help XML and the tool's parameter set."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
import xml.etree.ElementTree as ET
from importlib.machinery import SourceFileLoader

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import arcpy  # noqa: E402

from swathprofile import features, outputs, raster_io as rio  # noqa: E402
from swathprofile.bundle import BUNDLE_VERSION, load_bundle, save_bundle  # noqa: E402
from swathprofile.config import SwathConfig                    # noqa: E402
from swathprofile.engine import run_swath                      # noqa: E402
from swathprofile.messaging import CollectingMessenger         # noqa: E402

from . import synth_arcpy as synth                             # noqa: E402

A, B, C = 3.0, 7.0, 100.0
X0, Y0 = 1_500_000.0, 5_000_000.0
CW = CH = 10.0
NC, NR = 400, 300
LX = X0 + 200 * CW


class OutputsV2Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        arcpy.env.overwriteOutput = True
        cls.folder, cls.gdb = synth.make_scratch()
        cls.plane = synth.plane_array(A, B, C, X0, Y0, CW, CH, NC, NR)
        cls.plane_path = synth.write_raster(cls.plane, cls.gdb, "plane", X0, Y0, CW, CH)
        cls.sr = arcpy.Describe(cls.plane_path).spatialReference
        # A labelled baseline layer.
        cls.line_fc = synth.write_line([(LX, Y0 + 50 * CH), (LX, Y0 + 250 * CH)],
                                       cls.gdb, "base")
        arcpy.management.AddField(cls.line_fc, "NAME", "TEXT", field_length=32)
        with arcpy.da.UpdateCursor(cls.line_fc, ["NAME"]) as cur:
            for row in cur:
                cur.updateRow(["Fault A"])

    @classmethod
    def tearDownClass(cls):
        arcpy.management.ClearWorkspaceCache()
        shutil.rmtree(cls.folder, ignore_errors=True)

    def _run(self, baselines=None, **kw):
        cfg = SwathConfig(**dict(dict(swath_width=1000.0, step=15.0, n_lines=51,
                                      tool_version="test"), **kw))
        msg = CollectingMessenger()
        pairs = [(arcpy.Raster(self.plane_path), rio.raster_info(self.plane_path))]
        if baselines is None:
            baselines = features.read_baselines(self.line_fc, self.sr, cfg.multipart,
                                                msg, label_field="NAME")
        return run_swath(baselines, pairs, cfg, self.sr, msg), msg

    def test_label_field_flows_into_names_and_outputs(self):
        result, _ = self._run()
        self.assertEqual(result.names[result.parts()[0]], "Fault A")
        path = os.path.join(self.gdb, "named_stats")
        outputs.write_station_table(path, result, None)
        names = {r[0] for r in arcpy.da.SearchCursor(path, ["LINE_NAME"])}
        self.assertEqual(names, {"Fault A"})

    def test_station_points_one_per_station_with_wide_stats(self):
        result, _ = self._run()
        path = os.path.join(self.gdb, "stations")
        features.write_station_points(path, result, self.sr, None)
        s = next(iter(result.stats.values()))
        self.assertEqual(int(arcpy.management.GetCount(path)[0]), s.n_stations)
        fields = {f.name for f in arcpy.ListFields(path)}
        pfx = outputs.safe_prefix("plane")
        for suffix in ("MIN", "MEAN", "MAX", "PLO", "PHI", "CENTRE", "THI"):
            self.assertIn(f"{pfx}_{suffix}", fields)
        self.assertEqual(arcpy.Describe(path).shapeType, "Point")
        means = [r[0] for r in arcpy.da.SearchCursor(path, [f"{pfx}_MEAN"])]
        np.testing.assert_allclose(means, s.z_mean, atol=1e-6)

    def test_transect_baselines_and_writer(self):
        trace = np.array([[LX - 800.0, Y0 + 100 * CH], [LX + 800.0, Y0 + 100 * CH]])
        bls, recs = features.transect_baselines(trace, 400.0, 600.0)
        self.assertEqual(len(bls), 5)
        self.assertEqual([b.name for b in bls], ["T001", "T002", "T003", "T004", "T005"])
        self.assertTrue(all(b.oid >= 2_000_000 for b in bls))
        path = os.path.join(self.gdb, "transects")
        features.write_transects(path, recs, self.sr, None)
        self.assertEqual(int(arcpy.management.GetCount(path)[0]), 5)
        rows = list(arcpy.da.SearchCursor(path, ["TRANSECT_ID", "LINE_NAME",
                                                 "DIST_ALONG", "LENGTH"]))
        self.assertEqual(rows[0][1], "T001")
        self.assertAlmostEqual(rows[-1][2], 1600.0)
        self.assertAlmostEqual(rows[0][3], 600.0, places=6)

        # And they profile like any other baseline, keeping their names.
        result, _ = self._run(baselines=bls, swath_width=200.0)
        self.assertEqual(len(result.parts()), 5)
        self.assertEqual(result.names[result.parts()[0]], "T001")
        for s in result.stats.values():
            self.assertLess(float(np.nanmax(np.abs(s.hi - 0.5))), 1e-6)

    def test_bundle_v2_round_trip_and_v1_tolerance(self):
        result, _ = self._run(pct_low=10, pct_high=90)
        path = os.path.join(self.folder, "v2.swath.npz")
        save_bundle(path, result, title="v2", elapsed=1.5)
        b = load_bundle(path)
        self.assertEqual(b.meta["version"], BUNDLE_VERSION)
        self.assertEqual(b.pct, (10.0, 90.0))
        self.assertEqual(b.parts[0]["name"], "Fault A")
        self.assertEqual(b.labels()[0], "Fault A")
        back = next(iter(b.parts[0]["by_raster"].values()))
        orig = next(iter(result.stats.values()))
        np.testing.assert_allclose(back.z_plo, orig.z_plo, atol=1e-9)
        np.testing.assert_allclose(back.z_centre, orig.z_centre, atol=1e-9)
        self.assertAlmostEqual(b.meta["elapsed"], 1.5)

        # Simulate a version-1 file: drop the optional arrays.
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files if not k.endswith(("z_plo", "z_phi"))}
        v1 = os.path.join(self.folder, "v1.swath.npz")
        np.savez_compressed(v1, **arrays)
        old = load_bundle(v1)
        self.assertIsNone(next(iter(old.parts[0]["by_raster"].values())).z_plo)

    def test_progress_is_reported(self):
        _, msg = self._run()
        self.assertTrue(msg.progress_calls)
        done, total, _ = msg.progress_calls[-1]
        self.assertEqual(done, total)

    def test_help_xml_covers_every_parameter(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import build_help  # noqa: E402

        module = SourceFileLoader("swpyt_test", os.path.join(ROOT, "SwathProfile.pyt")
                                  ).load_module()
        tool = module.SwathProfileTool()
        root, missing = build_help.build_tool_xml(tool)
        self.assertEqual(missing, [])
        params = root.findall("./tool/parameters/param")
        self.assertEqual(len(params), len(tool.getParameterInfo()))
        for p in params:
            ref = p.find("dialogReference")
            self.assertIsNotNone(ref, p.get("name"))
            self.assertTrue((ref.findtext("para") or "").strip(), p.get("name"))
        # The committed file matches the current parameter set.
        committed = ET.parse(os.path.join(ROOT, "SwathProfile.SwathProfileTool.pyt.xml"))
        self.assertEqual(len(committed.getroot().findall("./tool/parameters/param")),
                         len(params))

    def test_tool_parameter_set(self):
        module = SourceFileLoader("swpyt_test2", os.path.join(ROOT, "SwathProfile.pyt")
                                  ).load_module()
        tool = module.SwathProfileTool()
        params = tool.getParameterInfo()
        names = [p.name for p in params]
        self.assertEqual(len(names), len(set(names)))
        for expected in ("draw_lines", "label_field", "transect_source",
                         "transect_spacing", "pct_low", "pct_high", "out_stations",
                         "out_transects", "chart_preset", "add_to_map",
                         "transect_width", "mask_draw", "mask_layer", "mask_mode"):
            self.assertIn(expected, names)
        # The mask parameters sit together, so they form one dialog category.
        cats = [p.category for p in params if p.name.startswith("mask_")]
        self.assertEqual(cats, ["Sampling mask"] * 3)
        mode = next(p for p in params if p.name == "mask_mode")
        self.assertEqual(mode.value, "Keep only inside")
        method = next(p for p in params if p.name == "method")
        self.assertEqual(method.parameterType, "Required")
        tool.updateParameters(params)
        tool.updateMessages(params)

    def test_sketch_parameter_is_identical_on_every_dialog_open(self):
        """The draw-a-baseline sketch must not depend on hidden process state.

        Regression test.  The sketch parameter used to be given a POLYLINE
        template created in the "memory" workspace.  That workspace lives as
        long as the ArcGIS Pro process, so the first dialog open created the
        template and bound the parameter to it, and every later open raised
        ERROR 000258 (already exists), swallowed it, and left the parameter
        unbound - so drawing worked only every other time the tool was opened.
        Opening the dialog must now produce the same parameter every time,
        whatever ``overwriteOutput`` happens to be, and it must still restrict
        the sketch to lines.
        """
        module = SourceFileLoader("swpyt_test3", os.path.join(ROOT, "SwathProfile.pyt")
                                  ).load_module()
        seen = []
        for overwrite in (False, True, False):
            arcpy.env.overwriteOutput = overwrite
            draw = next(p for p in module.SwathProfileTool().getParameterInfo()
                        if p.name == "draw_lines")
            seen.append((draw.value, [str(g).lower() for g in draw.filter.list]))
        arcpy.env.overwriteOutput = True

        self.assertEqual(len(set(map(str, seen))), 1, f"sketch differs by open: {seen}")
        for value, geometries in seen:
            self.assertIsNone(value, "a bound template disables the sketch tool")
            self.assertEqual(geometries, ["polyline"])
        self.assertFalse(arcpy.Exists("memory/swath_baseline_template"),
                         "the dialog must not leave a template in the memory workspace")

    def test_mask_sketch_is_identical_on_every_dialog_open(self):
        """The mask sketch follows the draw_lines rules: no value, polygon filter."""
        module = SourceFileLoader("swpyt_test4", os.path.join(ROOT, "SwathProfile.pyt")
                                  ).load_module()
        seen = []
        for overwrite in (False, True, False):
            arcpy.env.overwriteOutput = overwrite
            draw = next(p for p in module.SwathProfileTool().getParameterInfo()
                        if p.name == "mask_draw")
            seen.append((draw.value, [str(g).lower() for g in draw.filter.list]))
        arcpy.env.overwriteOutput = True
        self.assertEqual(len(set(map(str, seen))), 1, f"sketch differs by open: {seen}")
        for value, geometries in seen:
            self.assertIsNone(value)
            self.assertEqual(geometries, ["polygon"])

    def test_a_line_drawn_as_the_mask_is_rejected(self):
        module = SourceFileLoader("swpyt_test5", os.path.join(ROOT, "SwathProfile.pyt")
                                  ).load_module()
        tool = module.SwathProfileTool()
        params = tool.getParameterInfo()
        pr = {p.name: p for p in params}
        fs = arcpy.FeatureSet()
        fs.load(self.line_fc)                    # a polyline where a polygon belongs
        pr["mask_draw"].value = fs
        tool.updateParameters(params)
        tool.updateMessages(params)
        self.assertTrue(pr["mask_draw"].hasError())
        self.assertIn("not a polygon", pr["mask_draw"].message)
        self.assertTrue(pr["mask_mode"].enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
