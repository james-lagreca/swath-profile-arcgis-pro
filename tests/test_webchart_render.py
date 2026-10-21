# -*- coding: utf-8 -*-
"""The browser chart in a real browser engine.  No ArcGIS needed.

Each test builds a page from an in-memory run, adds a small script that drives
the page the way a person would - hover, click, drag, keys, the save buttons -
and records what happened in a ``data-test`` attribute, then loads it from disk
in headless Chrome or Edge and reads the resulting DOM.  The script is added
only to the test copy: the page users get has no test hooks.

Skipped when no Chromium-based browser is found.  GitHub's Linux runners ship
Google Chrome, so these run in CI; set ``SWATH_TEST_BROWSER`` to choose one.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swathprofile import webchart                             # noqa: E402
from swathprofile.config import SwathConfig                   # noqa: E402
from swathprofile.engine import run_swath                     # noqa: E402
from swathprofile.features import Baseline                    # noqa: E402

from tests.fake_raster import FakeRaster, patched             # noqa: E402

X0, YTOP, CW = 1_500_000.0, 5_003_000.0, 10.0


def _headless_args(exe, profile):
    args = [exe, "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", f"--user-data-dir={profile}"]
    if sys.platform.startswith("linux"):
        args.append("--no-sandbox")
    return args


def _answers(exe):
    """Whether *exe* really renders a page headlessly and prints its DOM.

    Not every installed browser does: a desktop Edge that is already running
    can take the request over and exit at once with no output.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        page = Path(tmp) / "probe.html"
        page.write_text("<html><body><script>document.body.setAttribute('data-ok', '1')"
                        "</script></body></html>", encoding="utf-8")
        try:
            proc = subprocess.run(_headless_args(exe, Path(tmp) / "profile") +
                                  ["--dump-dom", page.as_uri()],
                                  capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return False
    return 'data-ok="1"' in proc.stdout


def _browser():
    candidates = [os.environ.get("SWATH_TEST_BROWSER")]
    candidates += [shutil.which(n) for n in ("google-chrome", "google-chrome-stable",
                                             "chromium", "chromium-browser", "chrome",
                                             "msedge")]
    candidates += [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                   r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    seen = set()
    for exe in candidates:
        if not exe or exe in seen or not os.path.exists(exe):
            continue
        seen.add(exe)
        if _answers(exe):
            return exe
    return None


BROWSER = _browser()

# Drives the page and writes what it saw into <body data-test="...">.
HARNESS = r"""
<script>
(function () {
  var out = { downloads: [], csp: [] };
  document.addEventListener("securitypolicyviolation", function (e) {
    out.csp.push(e.violatedDirective + " " + e.blockedURI);
  });
  var captured = [];
  var origCreate = URL.createObjectURL.bind(URL);
  URL.createObjectURL = function (b) { captured.push(b); return origCreate(b); };
  HTMLAnchorElement.prototype.click = function () { out.downloads.push(this.getAttribute("download")); };
  window.alert = function (m) { out.alert = String(m); };
  function finish() { document.body.setAttribute("data-test", JSON.stringify(out)); }
  function chart() { return document.getElementById("chart"); }
  function overlay() { return chart().querySelector("g.ui rect[fill='transparent']"); }
  function fire(type, fx, target) {
    var r = overlay().getBoundingClientRect();
    (target || overlay()).dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true,
      button: 0, clientX: r.left + r.width * fx, clientY: r.top + 30 }));
  }
  function xsTitle() {
    var t = Array.prototype.filter.call(chart().querySelectorAll("text"),
      function (e) { return e.textContent.indexOf("Cross-section at") === 0; });
    return t.length ? t[0].textContent : "";
  }
  function meanLength() {
    var p = chart().querySelector("path[stroke='#01008a'][stroke-width='1.87']");
    return p ? p.getAttribute("d").length : 0;
  }
  function run() {
    try {
      out.ready = document.body.getAttribute("data-swath-ready");
      out.error = document.body.getAttribute("data-swath-error");
      out.parts = document.querySelectorAll("#parts button").length;
      out.partsHtml = document.getElementById("parts").innerHTML.indexOf("<img") >= 0;
      out.firstName = document.querySelector("#parts button span").textContent;
      out.linesDisabled = document.getElementById("show-profiles").disabled;
      out.notices = Array.prototype.map.call(chart().querySelectorAll("text"),
        function (e) { return e.textContent; }).filter(function (s) {
          return /No per-line|left out|No valid data/.test(s); });
      out.profilePaths = chart().querySelectorAll("path[stroke-opacity='0.6']").length;
      out.info = document.getElementById("info").textContent;
      out.note = document.getElementById("note").hidden ? "" : document.getElementById("note").textContent;
      out.chartText = Array.prototype.map.call(chart().querySelectorAll("text"),
        function (e) { return e.textContent; }).join(" | ");
      fire("mousemove", 0.5);
      var tip = document.getElementById("tip");
      out.tipVisible = !tip.hidden;
      out.tipText = tip.textContent;
      out.xsBefore = xsTitle();
      fire("mousedown", 0.3); fire("mouseup", 0.3);
      out.markers = chart().querySelectorAll("line[stroke='#d62728']").length;
      out.xsPinned = xsTitle();
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
      out.xsMoved = xsTitle();
      var before = meanLength();
      fire("mousedown", 0.2); fire("mousemove", 0.6); fire("mouseup", 0.6);
      var zoomed = meanLength();
      overlay().dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
      out.zoom = { before: before, zoomed: zoomed, reset: meanLength() };
      out.markersAfterZoom = chart().querySelectorAll("line[stroke='#d62728']").length;
      document.getElementById("save-svg").click();
      captured[0].text().then(function (t) {
        out.svg = { xml: t.indexOf("<?xml") === 0, svg: t.indexOf("<svg") >= 0,
                    ui: t.indexOf('class="ui"') >= 0, length: t.length,
                    footer: t.indexOf("SwathProfile test") >= 0 };
        document.getElementById("save-png").click();
        var waited = 0;
        (function poll() {
          if (captured.length > 1 || out.alert || waited > 3000) {
            if (captured.length > 1) out.png = { type: captured[1].type, size: captured[1].size };
            finish();
            return;
          }
          waited += 50;
          setTimeout(poll, 50);
        })();
      }, function (e) { out.svgError = String(e); finish(); });
    } catch (e) {
      out.harnessError = String(e && e.stack || e);
      finish();
    }
  }
  document.addEventListener("DOMContentLoaded", function () { setTimeout(run, 0); });
})();
</script>
"""


def _terrain():
    xs = X0 + (np.arange(400) + 0.5) * CW
    ys = YTOP - (np.arange(300) + 0.5) * CW
    X, Y = np.meshgrid(xs, ys)
    z = (200 + 0.02 * (YTOP - Y) + 6.0 / (1 + np.exp(-((YTOP - Y) - 1500) / 40))
         + 8 * np.sin((X - X0) / 230) * np.cos((YTOP - Y) / 310))
    return FakeRaster(z, X0, YTOP, CW, name="lidar_dem")


def _result(method="offset", name="Across the scarp", **cfg):
    line = Baseline(1, 0, np.array([[X0 + 2000, YTOP - 2700], [X0 + 2050, YTOP - 300]]),
                    float(np.hypot(50, 2400)), name)
    other = Baseline(2, 0, np.array([[X0 + 800, YTOP - 2600], [X0 + 900, YTOP - 400]]),
                     float(np.hypot(100, 2200)), "West profile")
    cfg = SwathConfig(**dict(dict(swath_width=300.0, step=12.0, n_lines=41, method=method,
                                  chart_profile_colour="side", tool_version="test"), **cfg))
    with patched():
        return run_swath([line, other], [_terrain().pair], cfg, None, None)


@unittest.skipIf(BROWSER is None, "no Chromium-based browser found")
class RenderTest(unittest.TestCase):

    def _drive(self, result, max_samples=None):
        page = webchart.render(webchart.build_payload(result, title="render",
                                                      max_samples=max_samples))
        page = page.replace("</body>", HARNESS + "</body>")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "page.html"
            path.write_text(page, encoding="utf-8")
            args = _headless_args(BROWSER, Path(tmp) / "profile") + [
                "--window-size=1400,900", "--virtual-time-budget=8000", "--dump-dom",
                path.as_uri()]
            proc = subprocess.run(args, capture_output=True, text=True, timeout=120,
                                  encoding="utf-8", errors="replace")
        m = re.search(r'data-test="([^"]*)"', proc.stdout)
        self.assertIsNotNone(m, f"the page never finished; stderr: {proc.stderr[-800:]}")
        out = json.loads(html.unescape(m.group(1)))
        self.assertNotIn("harnessError", out, out.get("harnessError"))
        return out

    def test_offset_profile_draws_and_responds(self):
        out = self._drive(_result())
        self.assertEqual(out["ready"], "1")
        self.assertIsNone(out["error"])
        self.assertEqual(out["parts"], 2)
        self.assertEqual(out["profilePaths"], 41)
        self.assertFalse(out["linesDisabled"])
        # Hover shows the readout.
        self.assertTrue(out["tipVisible"])
        for word in ("Distance", "Mean", "THi", "Valid samples"):
            self.assertIn(word, out["tipText"])
        # A click pins a station: a marker in the main and THi panels, and the
        # cross-section moves there; the arrow key steps it on.
        self.assertEqual(out["markers"], 2)
        self.assertNotEqual(out["xsPinned"], out["xsBefore"])
        self.assertNotEqual(out["xsMoved"], out["xsPinned"])
        # A drag zooms in (fewer stations drawn), a double-click zooms out.
        self.assertLess(out["zoom"]["zoomed"], out["zoom"]["before"])
        self.assertEqual(out["zoom"]["reset"], out["zoom"]["before"])
        self.assertEqual(out["markersAfterZoom"], 2)
        # Saving: an SVG without the mouse layer, and a real PNG.
        self.assertTrue(out["svg"]["xml"] and out["svg"]["svg"])
        self.assertFalse(out["svg"]["ui"])
        self.assertNotIn("alert", out)
        self.assertEqual(out["png"]["type"], "image/png")
        self.assertGreater(out["png"]["size"], 5000)
        self.assertEqual(out["downloads"], ["render_Across_the_scarp.svg",
                                            "render_Across_the_scarp.png"])
        # The title and the saved charts' footer are part of the chart itself,
        # so an exported image says what it shows.
        self.assertIn("Across the scarp", out["chartText"])
        self.assertIn("swath width 300 | step 12 | 41 profiles", out["chartText"])
        self.assertTrue(out["svg"]["footer"])
        # The page's security policy never had to block anything it does.
        self.assertEqual(out["csp"], [])

    def test_all_cells_profile_has_no_lines_or_cross_section(self):
        out = self._drive(_result(method="zonal"))
        self.assertEqual(out["ready"], "1")
        self.assertEqual(out["profilePaths"], 0)
        self.assertTrue(out["linesDisabled"])
        self.assertIn("No per-line samples (All-cells mode)", out["notices"])
        self.assertTrue(out["tipVisible"])

    def test_lines_left_out_by_the_size_limit_are_explained(self):
        out = self._drive(_result(), max_samples=10)
        self.assertEqual(out["ready"], "1")
        self.assertEqual(out["profilePaths"], 0)
        self.assertTrue(out["linesDisabled"])
        self.assertIn("Per-line samples left out of this page", out["notices"])
        self.assertGreater(out["zoom"]["before"], 0)          # the envelope is drawn

    def test_every_profile_line_is_drawn_even_in_a_very_wide_swath(self):
        out = self._drive(_result(n_lines=1001, swath_width=1000.0))
        self.assertEqual(out["ready"], "1")
        self.assertEqual(out["profilePaths"], 1001)
        self.assertNotIn("Showing", out["info"])
        self.assertEqual(out["csp"], [])

    def test_the_size_note_is_shown_above_the_chart(self):
        out = self._drive(_result(), max_samples=10)
        self.assertIn("too large for a browser to open", out["note"])

    def test_a_hostile_profile_name_stays_text(self):
        evil = '</script><img src=x onerror="alert(1)"><!--'
        out = self._drive(_result(name=evil))
        self.assertEqual(out["ready"], "1")
        self.assertEqual(out["firstName"], evil)
        self.assertFalse(out["partsHtml"])
        self.assertNotIn("alert", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
