# -*- coding: utf-8 -*-
"""End-to-end tests: engine, both sampling modes, outputs, bundle, CRS."""

from __future__ import annotations

import os
import shutil
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arcpy  # noqa: E402

from swathprofile import features, outputs, plotting, raster_io as rio  # noqa: E402
from swathprofile.bundle import load_bundle, save_bundle               # noqa: E402
from swathprofile.config import SwathConfig                            # noqa: E402
from swathprofile.engine import run_swath                              # noqa: E402
from swathprofile.errors import CrsError, GeometryError                # noqa: E402
from swathprofile.messaging import CollectingMessenger                 # noqa: E402

from . import synth_arcpy as synth                                     # noqa: E402

A, B, C = 3.0, 7.0, 100.0
X0, Y0 = 1_500_000.0, 5_000_000.0
CW = CH = 10.0
NC, NR = 400, 300
LX = X0 + 200 * CW                       # N-S baseline easting


class EndToEndTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        arcpy.env.overwriteOutput = True
        cls.folder, cls.gdb = synth.make_scratch()
        cls.plane = synth.plane_array(A, B, C, X0, Y0, CW, CH, NC, NR)
        cls.plane_path = synth.write_raster(cls.plane, cls.gdb, "plane",
                                            X0, Y0, CW, CH)
        cls.line_fc = synth.write_line(
            [(LX, Y0 + 50 * CH), (LX, Y0 + 250 * CH)], cls.gdb, "base")
        cls.sr = arcpy.Describe(cls.plane_path).spatialReference

    @classmethod
    def tearDownClass(cls):
        arcpy.management.ClearWorkspaceCache()
        shutil.rmtree(cls.folder, ignore_errors=True)

    # -- helpers --------------------------------------------------------------

    def _run(self, line_fc=None, rasters=None, mask=None, **kw):
        cfg = SwathConfig(**dict(
            dict(swath_width=1000.0, step=15.0, n_lines=51,
                 tool_version="test"), **kw))
        msg = CollectingMessenger()
        ras_list = rasters or [self.plane_path]
        pairs = [(arcpy.Raster(r), rio.raster_info(r)) for r in ras_list]
        bl = features.read_baselines(line_fc or self.line_fc, self.sr,
                                     cfg.multipart, msg)
        return run_swath(bl, pairs, cfg, self.sr, msg, mask=mask), msg

    # -- the analytic anchor --------------------------------------------------

    def test_plane_gives_hi_half_and_thi_fifty(self):
        result, _ = self._run()
        s = next(iter(result.stats.values()))
        self.assertLess(float(np.nanmax(np.abs(s.hi - 0.5))), 1e-7)
        self.assertLess(float(np.nanmax(np.abs(s.thi - 50.0))), 1e-5)
        self.assertLess(float(np.ptp(s.relief)), 1e-4)
        self.assertAlmostEqual(float(s.relief[0]), A * 1000.0, places=3)

    def test_mean_matches_the_analytic_plane(self):
        result, _ = self._run()
        key = next(iter(result.stats))
        s = result.stats[key]
        y = result.geometry[(key.oid, key.part_id)]["centreline"][:, 1]
        self.assertLess(float(np.nanmax(np.abs(s.z_mean - (A * LX + B * y + C)))),
                        1e-3)

    def test_odd_profile_count_samples_the_centreline(self):
        result, _ = self._run(n_lines=51)
        g = result.geometry[next(iter(result.stats)).oid, 0]
        self.assertTrue(np.any(np.isclose(g["offsets"], 0.0)))

    def test_centreline_profile_in_every_mode(self):
        """The baseline's own profile is always present and on the plane."""
        truth = lambda res: (A * res.geometry[res.parts()[0]]["centreline"][:, 0] +
                             B * res.geometry[res.parts()[0]]["centreline"][:, 1] + C)
        odd, _ = self._run(n_lines=51)
        s = next(iter(odd.stats.values()))
        self.assertIsNotNone(s.z_centre)
        np.testing.assert_array_equal(s.z_centre, s.values[25])      # exact row
        self.assertLess(float(np.nanmax(np.abs(s.z_centre - truth(odd)))), 1e-3)

        even, _ = self._run(n_lines=50)
        s = next(iter(even.stats.values()))
        self.assertIsNotNone(s.z_centre)
        self.assertLess(float(np.nanmax(np.abs(s.z_centre - truth(even)))), 1e-3)

        zon, _ = self._run(method="zonal")
        s = next(iter(zon.stats.values()))
        self.assertIsNotNone(s.z_centre)
        self.assertLess(float(np.nanmax(np.abs(s.z_centre - truth(zon)))), 1e-3)

    # -- sampling modes -------------------------------------------------------

    def test_offset_and_zonal_agree_on_a_plane(self):
        """Both modes must measure the same physical quantity."""
        off, _ = self._run(method="offset", n_lines=101)
        zon, _ = self._run(method="zonal")
        so = next(iter(off.stats.values()))
        sz = next(iter(zon.stats.values()))
        n = min(so.n_stations, sz.n_stations)
        inner = slice(2, n - 2)
        self.assertLess(
            float(np.nanmax(np.abs(so.z_mean[inner] - sz.z_mean[inner]))),
            0.02 * float(np.nanmean(so.relief)))

    def test_offset_and_zonal_agree_on_a_curved_baseline(self):
        """The two modes must use the same cross-section construction on curves.

        With enough offset lines, mode A converges on mode B.  A kinked line
        with direction stabilising is the case that showed up as spikes in
        zonal output when it used a nearest-station partition instead.
        """
        verts = [(LX - 600.0, Y0 + 60 * CH), (LX, Y0 + 150 * CH),
                 (LX + 700.0, Y0 + 230 * CH)]
        fc = synth.write_lines([verts], self.gdb, "kinked")
        # Step at 1.5 x cell, as the tool defaults: a slice thinner than a cell
        # catches an irregular handful of cells and the comparison is unfair.
        kw = dict(line_fc=fc, swath_width=300.0, step=15.0,
                  tangent_smooth_window=21)
        off, _ = self._run(method="offset", n_lines=301, **kw)
        zon, _ = self._run(method="zonal", **kw)
        so = next(iter(off.stats.values()))
        sz = next(iter(zon.stats.values()))
        n = min(so.n_stations, sz.n_stations)
        inner = slice(3, n - 3)
        # Zonal reads whole cells, so its across-track positions are quantised
        # to the lattice.  The mean averages that out to within one cell of
        # across-track error times the cross-slope (relief / width * cell).
        # An extreme differs for two reasons: the outermost cell centre inside
        # the band can sit up to a cell diagonal short of the edge, and a zonal
        # slice has a finite along-track length (step) where an offset line has
        # none, so on a surface that also slopes along the baseline the slice's
        # max exceeds the line's by up to along-slope * step / 2.  Both are
        # properties of the definitions, not errors.
        cross_slope = float(np.nanmean(so.relief)) / 300.0
        along_half_step = 0.5 * float(np.nanmedian(np.abs(np.diff(so.z_mean))))
        tol = cross_slope * CW
        tol_extreme = cross_slope * CW * np.sqrt(2.0) + along_half_step
        self.assertLess(float(np.nanmax(np.abs(so.z_mean[inner] - sz.z_mean[inner]))), tol)
        self.assertLess(float(np.nanmax(np.abs(so.z_max[inner] - sz.z_max[inner]))),
                        tol_extreme)
        self.assertLess(float(np.nanmax(np.abs(so.z_min[inner] - sz.z_min[inner]))),
                        tol_extreme)
        # No single-station spikes in zonal relative to offset mode.  The means
        # themselves change by ~100 m per station here (the plane's along-track
        # slope), so the spike test is on the *difference* between the modes.
        jumps = np.abs(np.diff(sz.z_mean[inner] - so.z_mean[inner]))
        self.assertLess(float(np.nanmax(jumps)), tol)

    def test_zonal_produces_real_quartiles(self):
        """D17: the reference zero-fills Q1/Q3/THi in zonal mode."""
        result, _ = self._run(method="zonal")
        s = next(iter(result.stats.values()))
        self.assertTrue(np.isfinite(s.z_q1).any())
        self.assertTrue(np.isfinite(s.z_q3).any())
        self.assertTrue(np.all(s.z_q1[np.isfinite(s.z_q1)] <=
                               s.z_q3[np.isfinite(s.z_q3)] + 1e-9))
        self.assertLess(float(np.nanmax(np.abs(s.hi - 0.5))), 0.02)

    # -- multi-raster ---------------------------------------------------------

    def test_multiple_rasters_share_one_point_grid(self):
        second = synth.write_raster(self.plane * 2.0, self.gdb, "plane2",
                                    X0, Y0, CW, CH)
        result, _ = self._run(rasters=[self.plane_path, second])
        names = {k.raster for k in result.stats}
        self.assertEqual(len(names), 2)
        self.assertEqual(len(result.geometry), 1)          # one grid, two rasters
        a, b = (result.stats[k] for k in sorted(result.stats, key=lambda k: k.raster))
        self.assertEqual(a.n_stations, b.n_stations)
        self.assertLess(float(np.nanmax(np.abs(b.z_mean - 2.0 * a.z_mean))), 1e-2)

    def test_raster_at_a_different_cell_size(self):
        coarse = synth.plane_array(A, B, C, X0, Y0, 40.0, 40.0, NC // 4, NR // 4)
        coarse_path = synth.write_raster(coarse, self.gdb, "coarse", X0, Y0,
                                         40.0, 40.0)
        result, _ = self._run(rasters=[self.plane_path, coarse_path])
        self.assertEqual(len({k.raster for k in result.stats}), 2)
        for s in result.stats.values():
            self.assertLess(float(np.nanmax(np.abs(s.hi - 0.5))), 1e-3)

    def test_raster_not_covering_the_band_warns_but_does_not_fail(self):
        small = synth.plane_array(A, B, C, X0, Y0, CW, CH, 60, 60)
        small_path = synth.write_raster(small, self.gdb, "small", X0, Y0, CW, CH)
        result, msg = self._run(rasters=[self.plane_path, small_path])
        self.assertEqual(len({k.raster for k in result.stats}), 2)
        s = [v for k, v in result.stats.items() if "small" in k.raster][0]
        self.assertGreater(s.n_empty_stations, 0)
        self.assertTrue(msg.has_warning("no valid data"))

    # -- geometry behaviour ---------------------------------------------------

    def test_curved_baseline_folds_are_flagged_and_masked(self):
        th = np.linspace(-np.pi / 2, np.pi / 2, 60)
        r = 300.0
        cx, cy = X0 + 200 * CW, Y0 + 150 * CH
        verts = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])
        fc = synth.write_lines([verts], self.gdb, "arc")
        result, msg = self._run(line_fc=fc, swath_width=1200.0, step=15.0)
        g = result.geometry[next(iter(result.stats)).oid, 0]
        self.assertIsNotNone(g["folded"])
        self.assertTrue(g["folded"].any())
        self.assertTrue(msg.has_warning("folds onto itself"))

    def test_baseline_shorter_than_step_is_rejected(self):
        fc = synth.write_lines([[(LX, Y0 + 100 * CH), (LX, Y0 + 100 * CH + 5.0)]],
                               self.gdb, "tiny")
        with self.assertRaises(GeometryError) as ctx:
            self._run(line_fc=fc, step=50.0)
        self.assertIn("shorter than the step size", str(ctx.exception))

    def test_multipart_lines_get_separate_part_ids(self):
        sr = arcpy.SpatialReference(synth.NZTM)
        fc = arcpy.management.CreateFeatureclass(
            self.gdb, "multi_line", "POLYLINE", spatial_reference=sr)[0]
        parts = arcpy.Array([
            arcpy.Array([arcpy.Point(LX, Y0 + 60 * CH),
                         arcpy.Point(LX, Y0 + 120 * CH)]),
            arcpy.Array([arcpy.Point(LX + 500, Y0 + 60 * CH),
                         arcpy.Point(LX + 500, Y0 + 120 * CH)]),
        ])
        with arcpy.da.InsertCursor(fc, ["SHAPE@"]) as cur:
            cur.insertRow([arcpy.Polyline(parts, sr)])

        result, msg = self._run(line_fc=fc)
        self.assertEqual(len(result.parts()), 2)
        self.assertEqual({p for _, p in result.parts()}, {0, 1})
        self.assertTrue(msg.has_warning("multipart"))

    def test_selection_is_honoured(self):
        fc = synth.write_lines(
            [[(LX, Y0 + 60 * CH), (LX, Y0 + 200 * CH)],
             [(LX + 400, Y0 + 60 * CH), (LX + 400, Y0 + 200 * CH)],
             [(LX + 800, Y0 + 60 * CH), (LX + 800, Y0 + 200 * CH)]],
            self.gdb, "three")
        layer = arcpy.management.MakeFeatureLayer(fc, "three_lyr")[0]
        try:
            arcpy.management.SelectLayerByAttribute(
                layer, "NEW_SELECTION", "OBJECTID = 2")
            note = features.selection_note(layer)
            self.assertIn("selection active", note)
            result, _ = self._run(line_fc=layer)
            self.assertEqual(len(result.parts()), 1)
        finally:
            arcpy.management.Delete(layer)

    # -- CRS ------------------------------------------------------------------

    def test_undefined_crs_is_rejected_with_guidance(self):
        path = os.path.join(self.gdb, "nosr")
        arcpy.NumPyArrayToRaster(self.plane[:20, :20], arcpy.Point(X0, Y0),
                                 CW, CH).save(path)
        with self.assertRaises(CrsError) as ctx:
            features.require_spatial_reference(arcpy.Describe(path), "nosr")
        self.assertIn("Define Projection", str(ctx.exception))

    def test_transform_points_is_one_to_one_and_survives_duplicates(self):
        """arcpy multipoints merge coincident points; the mapping must not."""
        nztm = arcpy.SpatialReference(synth.NZTM)
        wgs = arcpy.SpatialReference(synth.WGS84)
        xy = np.array([[LX, Y0 + 100 * CH]] * 5 + [[LX + 10.0, Y0 + 100 * CH]],
                      dtype=np.float64)
        out = features.transform_points(xy, nztm, wgs)
        self.assertEqual(out.shape, xy.shape)
        # Degrees are large numbers with tiny meaningful differences: 10 m is
        # about 1e-4 degrees, so compare with an absolute, not relative, tolerance.
        self.assertTrue(np.allclose(out[0], out[4], rtol=0, atol=1e-12))
        self.assertGreater(float(np.max(np.abs(out[5] - out[0]))), 5e-5)
        back = features.transform_points(out, wgs, nztm)
        self.assertLess(float(np.max(np.abs(back - xy))), 1e-3)
        # Shape is preserved for an (N, M, 2) grid, NaNs pass through.
        grid = np.full((3, 4, 2), np.nan)
        grid[1, 2] = [LX, Y0 + 100 * CH]
        g2 = features.transform_points(grid, nztm, wgs)
        self.assertEqual(g2.shape, grid.shape)
        self.assertTrue(np.isnan(g2[0, 0]).all())
        self.assertTrue(np.isfinite(g2[1, 2]).all())

    def test_geographic_raster_is_sampled_via_the_working_crs(self):
        """A WGS84 DEM with a NZTM baseline: the plane identities must still hold.

        The raster's values are a plane in NZTM coordinates, stored on a lon/lat
        grid.  Sampling in the NZTM working CRS, with points transformed into the
        raster's CRS, must recover HI = 0.5 and THi = 50 (to bilinear error on a
        surface that is only approximately bilinear in degrees).
        """
        nztm = arcpy.SpatialReference(synth.NZTM)
        wgs = arcpy.SpatialReference(synth.WGS84)
        lon0, lat0, dd, n = 170.0, -44.05, 0.0001, 300
        lons = lon0 + (np.arange(n) + 0.5) * dd
        lats = (lat0 + n * dd) - (np.arange(n) + 0.5) * dd
        ll = np.stack(np.meshgrid(lons, lats), axis=-1)          # (n, n, 2)
        xy = features.transform_points(ll, wgs, nztm)
        z = A * xy[..., 0] + B * xy[..., 1] + C
        path = synth.write_raster(z, self.gdb, "geo_plane", lon0, lat0, dd, dd,
                                  epsg=synth.WGS84)

        centre = xy[n // 2, n // 2]
        fc = synth.write_lines([[(centre[0], centre[1] - 600.0),
                                 (centre[0], centre[1] + 600.0)]],
                               self.gdb, "geo_base")
        result, msg = self._run(line_fc=fc, rasters=[path], swath_width=600.0,
                                step=12.0, n_lines=41)
        s = next(iter(result.stats.values()))
        self.assertGreater(int(np.sum(s.n_valid == s.n_total)), s.n_stations * 0.9)
        good = s.n_valid == s.n_total
        self.assertLess(float(np.nanmax(np.abs(s.hi[good] - 0.5))), 0.01)
        self.assertLess(float(np.nanmax(np.abs(s.thi[good] - 50.0))), 1.0)
        self.assertTrue(any("transformed" in m for m in msg.infos))

    def test_geographic_raster_all_cells_mode_is_fast_and_right(self):
        """Zonal on a lat/lon raster must take the sweep, not a per-station path.

        The plane identities must hold to lattice tolerance, and the run must
        finish in seconds: this is the case that used to take twenty minutes.
        """
        import time

        nztm = arcpy.SpatialReference(synth.NZTM)
        wgs = arcpy.SpatialReference(synth.WGS84)
        lon0, lat0, dd, n = 170.0, -44.05, 0.0001, 300
        lons = lon0 + (np.arange(n) + 0.5) * dd
        lats = (lat0 + n * dd) - (np.arange(n) + 0.5) * dd
        ll = np.stack(np.meshgrid(lons, lats), axis=-1)
        xy = features.transform_points(ll, wgs, nztm)
        z = A * xy[..., 0] + B * xy[..., 1] + C
        path = synth.write_raster(z, self.gdb, "geo_plane_z", lon0, lat0, dd, dd,
                                  epsg=synth.WGS84)
        centre = xy[n // 2, n // 2]
        fc = synth.write_lines([[(centre[0] - 400.0, centre[1] - 500.0),
                                 (centre[0] + 400.0, centre[1] + 500.0)]],
                               self.gdb, "geo_base_z")
        t0 = time.perf_counter()
        result, msg = self._run(line_fc=fc, rasters=[path], swath_width=600.0,
                                step=12.0, method="zonal")
        elapsed = time.perf_counter() - t0
        s = next(iter(result.stats.values()))
        good = s.n_valid > 0
        self.assertGreater(int(good.sum()), s.n_stations * 0.9)
        # Cells are ~8-11 m; one cell of cross-slope is the honest tolerance.
        cross_slope = float(np.nanmean(s.relief[good])) / 600.0
        self.assertLess(float(np.nanmax(np.abs(s.hi[good] - 0.5))), 0.03)
        truth = A * result.geometry[result.parts()[0]]["centreline"][:, 0] + \
            B * result.geometry[result.parts()[0]]["centreline"][:, 1] + C
        self.assertLess(float(np.nanmax(np.abs(s.z_mean[good] - truth[good]))),
                        cross_slope * 12.0)
        self.assertLess(elapsed, 30.0)
        self.assertTrue(any("transformed" in m for m in msg.infos))

    def test_working_crs_prefers_a_projected_raster(self):
        nztm = arcpy.SpatialReference(synth.NZTM)
        wgs = arcpy.SpatialReference(synth.WGS84)
        self.assertTrue(features.same_crs(
            features.choose_working_crs([wgs, nztm], wgs), nztm))
        self.assertTrue(features.same_crs(
            features.choose_working_crs([wgs], nztm), nztm))
        ext = arcpy.Extent(169.9, -44.1, 170.1, -43.9)
        auto = features.choose_working_crs([wgs], wgs, ext)
        self.assertEqual(auto.type, "Projected")
        self.assertIn("59", auto.name)                 # UTM zone 59S for 170E

    def test_sketched_featureset_is_read_as_baselines(self):
        fs = arcpy.FeatureSet()
        fs.load(self.line_fc)
        bl = features.read_featureset(fs, self.sr, self.sr)
        self.assertEqual(len(bl), 1)
        self.assertGreaterEqual(bl[0].oid, 1_000_000)
        self.assertAlmostEqual(bl[0].length, 2000.0, places=3)
        self.assertEqual(features.read_featureset(None, self.sr), [])

    def test_baseline_is_reprojected_to_the_raster_crs(self):
        native, _ = self._run()
        wgs = arcpy.management.Project(
            self.line_fc, os.path.join(self.gdb, "base_wgs"),
            arcpy.SpatialReference(synth.WGS84))[0]
        reproj, _ = self._run(line_fc=wgs)
        a = next(iter(native.stats.values()))
        b = next(iter(reproj.stats.values()))
        n = min(a.n_stations, b.n_stations)
        # Tolerance from half a cell of positional error on this gradient.
        tol = 0.5 * CW * (abs(A) + abs(B))
        self.assertLess(float(np.nanmax(np.abs(a.z_mean[:n] - b.z_mean[:n]))), tol)

    # -- outputs --------------------------------------------------------------

    def test_station_table_row_count_and_nulls(self):
        result, _ = self._run()
        path = os.path.join(self.gdb, "stats_tbl")
        outputs.write_station_table(path, result, None)
        s = next(iter(result.stats.values()))
        self.assertEqual(int(arcpy.management.GetCount(path)[0]), s.n_stations)
        names = {f.name for f in arcpy.ListFields(path)}
        for expected in ("Z_MIN", "Z_Q1", "THI", "THI_RAW", "THI_STAR",
                         "N_VALID", "KAPPA", "FOLDED", "RASTER"):
            self.assertIn(expected, names)

    def test_outputs_can_be_written_into_an_edited_geodatabase(self):
        """Regression for ERROR 000496 'table is being edited'.

        Pro opens an edit session on the project geodatabase as soon as a layer
        from it is edited or a line is sketched, and AddField then refuses to
        modify any table there.  Outputs are therefore built in memory and
        copied in, which is permitted.
        """
        result, _ = self._run()
        editor = arcpy.da.Editor(self.gdb)
        editor.startEditing(False, True)
        editor.startOperation()
        try:
            with arcpy.da.UpdateCursor(self.line_fc, ["OID@"]):
                pass
            tbl = outputs.write_station_table(
                os.path.join(self.gdb, "edited_stats"), result, None)
            self.assertGreater(int(arcpy.management.GetCount(tbl)[0]), 0)
            g = result.geometry[result.parts()[0]]
            band = features.write_band_polygons(
                os.path.join(self.gdb, "edited_band"),
                [{"ring": g["ring"], "oid": 1, "part_id": 0, "width": 1000.0,
                  "step": g["step_actual"], "n_lines": 51,
                  "n_stations": len(g["distances"]), "method": "offset"}],
                self.sr, None)
            self.assertEqual(int(arcpy.management.GetCount(band)[0]), 1)
        finally:
            editor.stopOperation()
            editor.stopEditing(False)

    def test_nan_is_written_as_null_not_indeterminate(self):
        small = synth.plane_array(A, B, C, X0, Y0, CW, CH, 60, 60)
        small_path = synth.write_raster(small, self.gdb, "small_null", X0, Y0,
                                        CW, CH)
        result, _ = self._run(rasters=[small_path])
        path = os.path.join(self.gdb, "null_tbl")
        outputs.write_station_table(path, result, None)
        vals = [r[0] for r in arcpy.da.SearchCursor(path, ["Z_MEAN"])]
        self.assertIn(None, vals)
        self.assertFalse(any(v is not None and not np.isfinite(v) for v in vals))

    def test_geometry_outputs_are_written(self):
        result, _ = self._run()
        cfg = result.config
        rings = []
        for oid, part_id in result.parts():
            g = result.geometry[(oid, part_id)]
            rings.append({"ring": g["ring"], "oid": oid, "part_id": part_id,
                          "width": cfg.swath_width, "step": g["step_actual"],
                          "n_lines": cfg.n_lines,
                          "n_stations": len(g["distances"]),
                          "method": cfg.method})
        band = features.write_band_polygons(
            os.path.join(self.gdb, "band"), rings, self.sr, None)
        self.assertEqual(int(arcpy.management.GetCount(band)[0]), 1)

        s = next(iter(result.stats.values()))
        g = result.geometry[result.parts()[0]]
        profiles = [{"xy": g["centreline"], "z": s.z_mean,
                     "oid": 1, "part_id": 0, "raster": "plane",
                     "series": "MEAN", "offset": 0.0}]
        fc3d = features.write_profiles_3d(
            os.path.join(self.gdb, "prof3d"), profiles, self.sr, None)
        self.assertTrue(arcpy.Describe(fc3d).hasZ)

    def test_3d_lines_split_at_nodata_rather_than_faking_z(self):
        g_xy = np.column_stack([np.linspace(0.0, 100.0, 20),
                                np.zeros(20)]) + [X0, Y0]
        z = np.arange(20.0)
        z[8:12] = np.nan
        fc = features.write_profiles_3d(
            os.path.join(self.gdb, "split3d"),
            [{"xy": g_xy, "z": z, "oid": 1, "part_id": 0, "raster": "r",
              "series": "MEAN", "offset": 0.0}], self.sr, None)
        with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as cur:
            shape = next(iter(cur))[0]
        self.assertEqual(shape.partCount, 2)
        self.assertEqual(shape.pointCount, 16)          # 20 - 4 NaN

    def test_csv_writes_empty_string_for_nodata(self):
        small_path = synth.write_raster(
            synth.plane_array(A, B, C, X0, Y0, CW, CH, 60, 60),
            self.gdb, "small_csv", X0, Y0, CW, CH)
        result, _ = self._run(rasters=[small_path])
        folder = os.path.join(self.folder, "csv")
        paths = outputs.write_csv(folder, result, None)
        with open(paths[0], encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn(",,", text)
        self.assertNotIn("nan", text.lower())

    def test_chart_renders_to_png(self):
        plotting.use_agg()
        result, _ = self._run()
        fig, _ = plotting.build_figure(
            result.for_part(*result.parts()[0]), title="test",
            thi_raster=result.thi_raster)
        out = os.path.join(self.folder, "chart.png")
        plotting.save_figure(fig, out)
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 10_000)

    # -- bundle ---------------------------------------------------------------

    def test_bundle_round_trip_preserves_statistics(self):
        result, _ = self._run()
        path = os.path.join(self.folder, "r.swath.npz")
        save_bundle(path, result, title="round trip")
        loaded = load_bundle(path)

        self.assertEqual(len(loaded.parts), len(result.parts()))
        orig = next(iter(result.stats.values()))
        back = next(iter(loaded.parts[0]["by_raster"].values()))
        np.testing.assert_allclose(back.z_mean, orig.z_mean, rtol=0, atol=1e-9)
        np.testing.assert_allclose(back.thi, orig.thi, rtol=0, atol=1e-9,
                                   equal_nan=True)
        self.assertEqual(loaded.title, "round trip")
        # Raw samples must survive so the viewer can re-smooth without recomputing.
        self.assertIsNotNone(back.values)
        self.assertEqual(back.values.shape, orig.values.shape)

    def test_legacy_flag_changes_only_the_documented_statistics(self):
        flat = np.full((NR, NC), 500.0, dtype=np.float64)
        flat_path = synth.write_raster(flat, self.gdb, "flat", X0, Y0, CW, CH)
        modern, _ = self._run(rasters=[flat_path])
        legacy, _ = self._run(rasters=[flat_path], legacy_stats=True)
        m = next(iter(modern.stats.values()))
        l = next(iter(legacy.stats.values()))
        np.testing.assert_allclose(m.z_mean, l.z_mean)      # unchanged
        self.assertTrue(np.isnan(m.hi).all())               # undefined
        np.testing.assert_allclose(l.hi, 0.0)               # reference's 0

    # -- sampling mask --------------------------------------------------------

    EAST = (LX + 3.0, Y0 - 100.0, LX + 700.0, Y0 + 3100.0)   # east of the line

    def _mask_fc(self, name, wkts, epsg=synth.NZTM):
        return synth.write_polygons(wkts, self.gdb, name, epsg=epsg)

    def test_read_mask_honours_a_layer_selection(self):
        fc = self._mask_fc("mask_sel", [synth.rect_wkt(*self.EAST),
                                        synth.rect_wkt(X0, Y0, X0 + 100, Y0 + 100)])
        layer = arcpy.management.MakeFeatureLayer(fc, "mask_sel_lyr")[0]
        try:
            arcpy.management.SelectLayerByAttribute(layer, "NEW_SELECTION",
                                                    "OBJECTID = 1")
            msg = CollectingMessenger()
            mi = features.read_mask(layer, None, self.sr, "keep", messenger=msg)
            self.assertEqual(mi.n_layer, 1)
            self.assertTrue(any("selection active" in m for m in msg.infos))
            self.assertIn("keep only inside 1 polygon(s)", mi.description)
            self.assertAlmostEqual(mi.geometry.area, 697.0 * 3200.0, places=3)
        finally:
            arcpy.management.Delete(layer)

    def test_read_mask_from_a_dialog_sketch(self):
        fs = arcpy.FeatureSet()
        fs.load(self._mask_fc("mask_fs", [synth.rect_wkt(*self.EAST)]))
        mi = features.read_mask(None, fs, self.sr, "exclude")
        self.assertEqual((mi.n_layer, mi.n_sketch, mi.mode), (0, 1, "exclude"))
        self.assertEqual(mi.sample_mask.mode, "exclude")
        empty = arcpy.FeatureSet()
        empty.load(self._mask_fc("mask_fs_empty", []))
        msg = CollectingMessenger()
        self.assertIsNone(features.read_mask(None, empty, self.sr, messenger=msg))
        self.assertTrue(any("no finished polygon" in w for w in msg.warnings))

    def test_overlaps_are_unioned_and_holes_honoured(self):
        x, y = LX, Y0 + 1000.0
        fc = self._mask_fc("mask_ovl", [
            synth.rect_wkt(x, y, x + 200, y + 200),
            synth.rect_wkt(x + 100, y + 100, x + 300, y + 300),
            f"POLYGON(({x} {y + 400},{x + 400} {y + 400},{x + 400} {y + 700},"
            f"{x} {y + 700},{x} {y + 400}),({x + 100} {y + 500},{x + 100} {y + 600},"
            f"{x + 250} {y + 600},{x + 250} {y + 500},{x + 100} {y + 500}))"])
        mi = features.read_mask(fc, None, self.sr)
        sm = mi.sample_mask
        px = np.array([x + 150, x + 50, x + 250, x + 175, x + 50, x + 500])
        py = np.array([y + 150, y + 50, y + 250, y + 550, y + 450, y + 150])
        # The overlap of the two squares is inside, not cancelled; the hole is out.
        self.assertEqual(sm.inside(px, py).tolist(),
                         [True, True, True, False, True, False])

    def test_numpy_mask_agrees_with_arcpy_on_random_points(self):
        x, y = LX, Y0 + 1000.0
        multipart = (f"MULTIPOLYGON((({x} {y},{x + 300} {y},{x + 150} {y + 280},"
                     f"{x} {y})),(({x + 400} {y},{x + 700} {y + 50},{x + 600} {y + 400},"
                     f"{x + 400} {y})))")
        holed = (f"POLYGON(({x} {y + 500},{x + 600} {y + 500},{x + 600} {y + 900},"
                 f"{x} {y + 900},{x} {y + 500}),({x + 200} {y + 600},{x + 250} {y + 800},"
                 f"{x + 400} {y + 650},{x + 200} {y + 600}))")
        bowtie = (f"POLYGON(({x + 800} {y},{x + 1000} {y + 200},{x + 1000} {y},"
                  f"{x + 800} {y + 200},{x + 800} {y}))")
        mi = features.read_mask(self._mask_fc("mask_rand", [multipart, holed, bowtie]),
                                None, self.sr)
        rng = np.random.default_rng(4)
        px = rng.uniform(x - 50, x + 1050, 3000)
        py = rng.uniform(y - 50, y + 950, 3000)
        ours = mi.sample_mask.inside(px, py)
        theirs = np.array([mi.geometry.contains(arcpy.PointGeometry(arcpy.Point(a, b),
                                                                    self.sr))
                           for a, b in zip(px, py)])
        np.testing.assert_array_equal(ours, theirs)

    def test_a_geographic_mask_keeps_the_same_samples(self):
        nztm = arcpy.SpatialReference(synth.NZTM)
        wgs = arcpy.SpatialReference(synth.WGS84)
        x0, y0, x1, y1 = self.EAST
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        ll = features.transform_points(corners, nztm, wgs)
        wkt = "POLYGON((" + ",".join(f"{a:.10f} {b:.10f}" for a, b in
                                     list(ll) + [ll[0]]) + "))"
        geo = features.read_mask(self._mask_fc("mask_wgs", [wkt], epsg=synth.WGS84),
                                 None, self.sr)
        local = features.read_mask(self._mask_fc("mask_nztm", [synth.rect_wkt(*self.EAST)]),
                                   None, self.sr)
        r_geo, _ = self._run(mask=geo.sample_mask)
        r_loc, _ = self._run(mask=local.sample_mask)
        np.testing.assert_array_equal(next(iter(r_geo.stats.values())).n_valid,
                                      next(iter(r_loc.stats.values())).n_valid)
        self.assertEqual(int(next(iter(r_loc.stats.values())).n_valid[10]), 25)

    def test_a_mask_layer_with_nothing_usable_raises(self):
        with self.assertRaises(GeometryError):
            features.read_mask(self._mask_fc("mask_none", []), None, self.sr)

    def test_mask_coverage_of_a_band(self):
        ring = np.array([[LX - 500, Y0 + 500], [LX + 500, Y0 + 500],
                         [LX + 500, Y0 + 2500], [LX - 500, Y0 + 2500],
                         [LX - 500, Y0 + 500]])
        half = synth.rect_wkt(LX, Y0 - 100, LX + 700, Y0 + 3100)
        for mode, expect in (("keep", 50.0), ("exclude", 50.0)):
            mi = features.read_mask(self._mask_fc(f"cov_{mode}", [half]), None,
                                    self.sr, mode)
            pct, shape = features.mask_coverage(ring, self.sr, mi)
            self.assertAlmostEqual(pct, expect, places=6)
            self.assertAlmostEqual(shape.area, 1000.0 * 1000.0, places=3)
        far = features.read_mask(self._mask_fc("cov_far", [synth.rect_wkt(
            X0, Y0, X0 + 50, Y0 + 50)]), None, self.sr, "keep")
        self.assertEqual(features.mask_coverage(ring, self.sr, far), (0.0, None))
        whole = features.read_mask(self._mask_fc("cov_all", [synth.rect_wkt(
            X0, Y0, X0 + 4000, Y0 + 3000)]), None, self.sr, "keep")
        pct, shape = features.mask_coverage(ring, self.sr, whole)
        self.assertEqual(pct, 100.0)
        self.assertAlmostEqual(shape.area, 1000.0 * 2000.0, places=3)

    def test_zonal_mask_on_a_geographic_raster_partitions_the_cells(self):
        """The cell mask is densified and moved into the raster's lat/lon grid."""
        nztm = arcpy.SpatialReference(synth.NZTM)
        wgs = arcpy.SpatialReference(synth.WGS84)
        lon0, lat0, dd, n = 170.0, -44.05, 0.0001, 300
        lons = lon0 + (np.arange(n) + 0.5) * dd
        lats = (lat0 + n * dd) - (np.arange(n) + 0.5) * dd
        xy = features.transform_points(np.stack(np.meshgrid(lons, lats), axis=-1),
                                       wgs, nztm)
        z = A * xy[..., 0] + B * xy[..., 1] + C
        path = synth.write_raster(z, self.gdb, "geo_plane_mask", lon0, lat0, dd, dd,
                                  epsg=synth.WGS84)
        cx, cy = xy[n // 2, n // 2]
        fc = synth.write_lines([[(cx - 400.0, cy - 500.0), (cx + 400.0, cy + 500.0)]],
                               self.gdb, "geo_base_mask")
        mask_fc = self._mask_fc("mask_geo", [synth.rect_wkt(cx - 2000, cy - 2000,
                                                            cx + 60, cy + 2000)])
        kw = dict(line_fc=fc, rasters=[path], swath_width=600.0, step=12.0,
                  method="zonal")
        s_all = next(iter(self._run(**kw)[0].stats.values()))
        keep = features.read_mask(mask_fc, None, self.sr, "keep").sample_mask
        excl = features.read_mask(mask_fc, None, self.sr, "exclude").sample_mask
        r_k, msg = self._run(mask=keep, **kw)
        s_k = next(iter(r_k.stats.values()))
        s_e = next(iter(self._run(mask=excl, **kw)[0].stats.values()))
        np.testing.assert_array_equal(s_k.n_valid + s_e.n_valid, s_all.n_valid)
        self.assertGreater(int(s_k.n_valid.sum()), 0)
        self.assertGreater(int(s_e.n_valid.sum()), 0)
        self.assertTrue(any("Sampling mask placed in" in m for m in msg.infos))


if __name__ == "__main__":
    unittest.main(verbosity=2)
