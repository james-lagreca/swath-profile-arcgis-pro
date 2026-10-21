# -*- coding: utf-8 -*-
"""The browser chart writer.  No ArcGIS needed: runs in CI.

The page must carry exactly the run's numbers (checked by decoding its data
block and comparing with the statistics), use the same side colours, labels
and footer as the saved charts, stay a manageable size, work offline with no
way to reach the network, hold no file paths, and not let any profile name
break out of its data block.
"""

from __future__ import annotations

import html
import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swathprofile import plotting, webchart                   # noqa: E402
from swathprofile.config import SwathConfig                   # noqa: E402
from swathprofile.engine import run_swath                     # noqa: E402
from swathprofile.features import Baseline                    # noqa: E402
from swathprofile.mask import SampleMask                      # noqa: E402
from swathprofile.messaging import CollectingMessenger        # noqa: E402

from tests.fake_raster import patched, plane                  # noqa: E402

X0, YTOP, CW = 1_500_000.0, 5_003_000.0, 10.0
LX = X0 + 2000.0


def _line(oid=1, x=LX, name="north"):
    return Baseline(oid, 0, np.array([[x, YTOP - 2500.0], [x, YTOP - 300.0]]), 2200.0, name)


def _result(baselines=None, mask=None, rasters=1, **cfg):
    cfg = SwathConfig(**dict(dict(swath_width=300.0, step=10.0, n_lines=51,
                                  tool_version="test"), **cfg))
    pairs = [plane(3.0, 7.0, 100.0 + 10 * k, X0, YTOP, CW, 400, 300,
                   name=f"dem{k}" if rasters > 1 else "dem").pair for k in range(rasters)]
    with patched():
        return run_swath(baselines or [_line()], pairs, cfg, None, None, mask=mask)


class PayloadTest(unittest.TestCase):

    def test_arrays_round_trip_exactly(self):
        res = _result()
        payload = webchart.build_payload(res, title="t")
        part = payload["parts"][0]
        stats = next(iter(res.stats.values()))
        # Coordinates as float64, so exact; elevations and statistics float32.
        np.testing.assert_array_equal(webchart.decode(part["distances"], "<f8"),
                                      stats.distances)
        np.testing.assert_array_equal(webchart.decode(part["offsets"], "<f8"), stats.offsets)
        r = part["rasters"][0]
        for key in ("z_min", "z_max", "z_mean", "z_median", "z_q1", "z_q3", "z_plo",
                    "z_phi", "z_centre", "thi", "n_valid"):
            np.testing.assert_array_equal(webchart.decode(r[key]),
                                          np.asarray(getattr(stats, key), "<f4"),
                                          err_msg=key)
        # Not drawn by the page, so not sent: relief is max - min, recomputed.
        for key in ("hi", "thi_star", "relief"):
            self.assertNotIn(key, r)
        np.testing.assert_array_equal(webchart.decode(r["values"]).reshape(stats.values.shape),
                                      stats.values.astype("<f4"))
        self.assertEqual(part["lines"], "included")
        self.assertEqual(part["n_line_rows"], 51)
        self.assertTrue(part["has_data"])
        self.assertEqual(payload["schema"], webchart.SCHEMA)
        self.assertEqual(payload["samples"]["rule"], "all")
        self.assertEqual(payload["config"]["swath_width"], 300.0)

    def test_nodata_survives_as_nan(self):
        mask = SampleMask([[[LX + 3.0, YTOP - 3000.0], [LX + 400.0, YTOP - 3000.0],
                            [LX + 400.0, YTOP], [LX + 3.0, YTOP]]])
        r = webchart.build_payload(_result(mask=mask))["parts"][0]["rasters"][0]
        values = webchart.decode(r["values"]).reshape(51, -1)
        # Offsets run from -150 (right of a line drawn north: east) to +150, so
        # the east half the mask keeps is rows 0-24; the centre line at LX and
        # the west half are masked.
        self.assertTrue(np.isfinite(values[:25]).all())
        self.assertTrue(np.isnan(values[25:]).all())
        self.assertTrue(np.isnan(webchart.decode(r["z_centre"])).all())

    def test_side_colours_labels_and_footer_are_the_saved_charts(self):
        from matplotlib.colors import to_hex

        res = _result()
        res.geometry[(1, 0)]["mask_kept"] = 49.0
        part = webchart.build_payload(res)["parts"][0]
        stats = next(iter(res.stats.values()))
        for mode in ("side", "ramp"):
            self.assertEqual(part["colours"][mode],
                             [to_hex(c) for c in plotting.profile_colours(stats.offsets, mode)])
        self.assertEqual(part["side_labels"], list(plotting.side_labels(stats)))
        self.assertEqual(part["side_labels"], ["Left (W)", "Right (E)"])
        self.assertEqual([part["sides"][k] for k in (0, 25, 50)], [-1, 0, 1])
        self.assertEqual(part["footer"],
                         plotting.chart_footer(res.config, res.geometry[(1, 0)]))
        self.assertEqual(part["mask_kept"], 49.0)

    def test_transects_at_their_own_width_keep_it_in_the_footer(self):
        res = _result()
        narrow = _result([_line(2, x=LX - 800.0, name="T002")], swath_width=120.0)
        res.merge(narrow)
        footers = [p["footer"] for p in webchart.build_payload(res)["parts"]]
        self.assertIn("swath width 300", footers[0])
        self.assertIn("swath width 120", footers[1])

    def test_all_cells_profiles_carry_no_lines(self):
        part = webchart.build_payload(_result(method="zonal"))["parts"][0]
        self.assertEqual(part["lines"], "none")
        self.assertIsNone(part["offsets"])
        self.assertIsNone(part["sides"])
        self.assertIsNone(part["rasters"][0]["values"])
        self.assertIsNotNone(part["rasters"][0]["z_mean"])

    def test_over_budget_only_the_thi_rasters_lines_are_kept_in_order(self):
        res = _result([_line(1, name="first"), _line(2, x=LX - 800.0, name="second")],
                      rasters=2)
        one = res.stats[next(iter(res.stats))].values.size
        # Room for one raster of one profile: the THi raster of the first.
        payload = webchart.build_payload(res, max_samples=one + 1)
        self.assertEqual(payload["samples"]["rule"], "thi")
        first, second = payload["parts"]
        self.assertEqual((first["lines"], second["lines"]), ("included", "omitted"))
        kept = [r["name"] for r in first["rasters"] if r["values"] is not None]
        self.assertEqual(kept, [res.thi_raster])
        self.assertIsNone(second["rasters"][0]["values"])
        self.assertIsNotNone(second["rasters"][0]["z_max"])
        self.assertIn(f"shown for {res.thi_raster} only", payload["lines_note"])
        self.assertIn("1 profile(s)", payload["lines_note"])
        self.assertIn("second", payload["lines_note"])
        self.assertEqual(webchart.build_payload(res)["lines_note"], "")

        msg = CollectingMessenger()
        path = os.path.join(tempfile.mkdtemp(), "small.html")
        report = webchart.write_web_chart(path, res, max_samples=one + 1, messenger=msg)
        self.assertTrue(report["written"])
        self.assertEqual(report["left_out"], 1)
        self.assertTrue(any("manageable size" in m for m in msg.infos))

    def test_a_page_over_the_limit_is_not_written(self):
        path = os.path.join(tempfile.mkdtemp(), "huge.html")
        report = webchart.write_web_chart(path, _result(), page_limit=1000)
        self.assertFalse(report["written"])
        self.assertIsNone(report["path"])
        self.assertGreater(report["bytes"], 1000)
        self.assertFalse(os.path.exists(path))

    def test_parts_with_no_data_are_flagged(self):
        far = SampleMask([[[X0, YTOP - 10], [X0 + 5, YTOP - 10], [X0 + 5, YTOP - 5]]])
        res = _result([_line(1), _line(2, x=LX - 800.0, name="b")], mask=far)
        self.assertEqual([p["has_data"] for p in webchart.build_payload(res)["parts"]],
                         [False, False])


class PageTest(unittest.TestCase):

    def _page(self, res=None, title="run"):
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, "run.html")
        report = webchart.write_web_chart(path, res or _result(), title=title)
        with open(path, encoding="utf-8") as fh:
            return folder, path, report, fh.read()

    def test_written_atomically_self_contained_and_sealed_off_from_the_network(self):
        folder, path, report, page = self._page()
        self.assertFalse(os.path.exists(path + ".tmp"))
        self.assertEqual(report["bytes"], os.path.getsize(path))
        # Nothing may reach the network: no URLs, no external scripts or styles,
        # and a policy that forbids any request at run time.  The SVG namespace
        # is an identifier, not a download.
        scrubbed = page.replace("http://www.w3.org/2000/svg", "")
        for needle in ("http://", "https://", "<script src", "<link", "@import",
                       "fetch(", "XMLHttpRequest", "import("):
            self.assertNotIn(needle, scrubbed, needle)
        csp = re.search(r'http-equiv="Content-Security-Policy"\s+content="([^"]+)"', page)
        self.assertIsNotNone(csp)
        self.assertIn("default-src 'none'", csp.group(1))
        self.assertEqual(page.count("</script>"), 3)          # data, core, interface
        self.assertIn("<title>Swath profile - run</title>", page)
        self.assertEqual(webchart.extract_payload(page)["title"], "run")
        # The page is meant to be sent to people: it holds no file paths.
        self.assertNotIn(folder, page)
        self.assertNotIn(os.path.expanduser("~"), page)
        self.assertIsNone(re.search(r"[A-Za-z]:\\\\", page))

    def test_the_core_block_never_touches_the_page(self):
        _, _, _, page = self._page()
        core = page[page.index("/* SWATH-CORE-START"):page.index("/* SWATH-CORE-END */")]
        for needle in ("document", "window", "getElementById"):
            self.assertNotIn(needle, core, needle)

    def test_no_name_can_break_out_of_the_data_block(self):
        evil = '</script><img src=x onerror="alert(1)"><!-- /*__SWATH_DATA__*/'
        res = _result([_line(name=evil)])
        page = webchart.render(webchart.build_payload(res, title=evil))
        self.assertNotIn("</script><img", page)
        self.assertNotIn('<img src=x', page)
        self.assertEqual(page.count("</script>"), 3)
        data = webchart.extract_payload(page)
        self.assertEqual(data["parts"][0]["name"], evil)
        self.assertEqual(data["title"], evil)
        self.assertIn(html.escape(f"Swath profile - {evil}"), page)

    def test_the_template_must_have_each_slot_once(self):
        bad = os.path.join(tempfile.mkdtemp(), "bad.html")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("<html><title>__SWATH_TITLE__</title><body>no data slot</body></html>")
        original = webchart.TEMPLATE
        webchart.TEMPLATE = bad
        try:
            with self.assertRaises(ValueError):
                webchart.render(webchart.build_payload(_result()))
        finally:
            webchart.TEMPLATE = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
