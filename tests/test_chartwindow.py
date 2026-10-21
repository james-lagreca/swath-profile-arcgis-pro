# -*- coding: utf-8 -*-
"""Showing a run: the chart window, or the browser chart.  No ArcGIS needed.

Most tests replace the launch and os.startfile, so nothing starts.  One class
starts a real stand-in program - a few lines of Python playing the viewer -
to exercise the actual start, crash, block and silence paths.  Starting
programs on a computer whose security software allows only approved ones
raises notices there, so that class runs only in CI or when
``SWATH_TEST_SUBPROCESS`` is set.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swathprofile import chartwindow as cw                  # noqa: E402
from swathprofile.messaging import CollectingMessenger      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer", "swath_viewer.py")


class _StateMixin:

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="swathcw_")
        self.bundle = os.path.join(self.tmp, "live.swath.npz")
        self.page = os.path.join(self.tmp, "run.html")
        self.env = mock.patch.dict(os.environ, {cw.STATE_DIR_ENV: self.tmp})
        self.env.start()
        cw.SESSION.clear()

    def tearDown(self):
        cw.SESSION.clear()
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


class ModeTest(unittest.TestCase):

    def test_parse_mode(self):
        for value, want in ((True, cw.AUTO), (False, cw.OFF), ("false", cw.OFF),
                            ("TRUE", cw.AUTO), ("browser page", cw.PAGE),
                            ("Chart window", cw.WINDOW), (None, cw.AUTO), ("", cw.AUTO),
                            ("0", cw.OFF), ("off", cw.OFF), ("nonsense", cw.AUTO)):
            self.assertEqual(cw.parse_mode(value), want, value)

    def test_legacy_label_only_changes_what_needs_changing(self):
        self.assertEqual(cw.legacy_label("true"), cw.AUTO)
        self.assertEqual(cw.legacy_label("False"), cw.OFF)
        self.assertEqual(cw.legacy_label("automatic"), cw.AUTO)
        for keep in ("Automatic", "Chart window", "Browser page", "Off", "", None, "x"):
            self.assertIsNone(cw.legacy_label(keep), keep)


class StateTest(_StateMixin, unittest.TestCase):

    def test_round_trip_and_forget(self):
        self.assertIsNone(cw.remembered())
        cw.remember("failed", "PermissionError: no", persist=True)
        with open(cw.state_path(), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["chart_window"], "unavailable")
        cw.SESSION.clear()                                # a new Pro session
        self.assertEqual(cw.remembered()["kind"], "failed")
        self.assertIn("PermissionError", cw.remembered()["detail"])
        cw.forget()
        self.assertIsNone(cw.remembered())
        self.assertFalse(os.path.exists(cw.state_path()))

    def test_session_records_are_not_written_to_disk(self):
        cw.remember("crashed", "boom")
        self.assertEqual(cw.remembered()["kind"], "crashed")
        self.assertIsNone(cw.remembered(persistent_only=True))
        self.assertFalse(os.path.exists(cw.state_path()))

    def test_a_corrupt_state_file_counts_as_no_record(self):
        os.makedirs(os.path.dirname(cw.state_path()), exist_ok=True)
        with open(cw.state_path(), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertIsNone(cw.remembered())


class ShowChartTest(_StateMixin, unittest.TestCase):

    def _show(self, mode, outcome=("started", "", 1.0), find_python=lambda: "py.exe",
              page="default"):
        msg = CollectingMessenger()
        page = self.page if page == "default" else page
        with mock.patch.object(cw, "launch", return_value=outcome) as launch, \
                mock.patch.object(cw.os, "startfile", create=True) as start:
            status = cw.show_chart(mode, bundle=self.bundle, page=page, viewer=VIEWER,
                                   root=ROOT, find_python=find_python, msg=msg)
        return status, msg, launch, start

    def _said(self, msg):
        return " ".join(msg.infos + msg.warnings)

    def test_a_blocked_start_opens_the_page_and_is_remembered_on_disk(self):
        status, msg, launch, start = self._show(
            cw.AUTO, ("failed", "py.exe could not be started: PermissionError: denied", 0.0))
        self.assertEqual(status, "failed")
        start.assert_called_once_with(self.page)
        said = self._said(msg)
        self.assertIn("could not start on this computer", said)
        self.assertIn("ask IT to approve the viewer folder", said)
        self.assertIn("Browser chart opened", said)
        self.assertNotIn("SwathViewer.cmd", said)
        self.assertNotIn("opened in a separate window", said)
        # The warning states what happened, never the cause as fact.
        self.assertIn("looks the same as some other start-up faults", said)
        cw.SESSION.clear()                                # a new Pro session
        status, msg, launch, start = self._show(cw.AUTO)
        self.assertEqual(status, "skipped")
        launch.assert_not_called()
        start.assert_called_once_with(self.page)
        self.assertIn("could not start on this computer on", self._said(msg))
        self.assertFalse(msg.warnings)

    def test_chart_window_retries_and_a_start_clears_the_record(self):
        self._show(cw.AUTO, ("failed", "denied", 0.0))
        status, msg, launch, start = self._show(cw.WINDOW)
        self.assertEqual(status, "started")
        launch.assert_called_once()
        start.assert_not_called()
        self.assertIn("opened in a separate window", self._said(msg))
        self.assertIsNone(cw.remembered())
        self.assertFalse(os.path.exists(cw.state_path()))

    def test_a_crash_is_not_a_block_and_is_kept_for_the_session_only(self):
        status, msg, launch, start = self._show(
            cw.AUTO, ("crashed", "it closed again after 1.2 s with exit code 1", 1.2))
        self.assertEqual(status, "crashed")
        start.assert_called_once_with(self.page)
        said = self._said(msg)
        self.assertIn("rather than a block", said)
        self.assertNotIn("ask IT", said)
        self.assertIsNone(cw.remembered(persistent_only=True))
        status, *_ = self._show(cw.AUTO)
        self.assertEqual(status, "skipped")               # not retried this session
        cw.SESSION.clear()
        status, *_ = self._show(cw.AUTO)
        self.assertEqual(status, "started")               # a new session tries again

    def test_no_python_is_not_a_block(self):
        def missing():
            raise RuntimeError("Could not find a Python interpreter for the viewer")
        status, msg, launch, start = self._show(cw.AUTO, find_python=missing)
        self.assertEqual(status, "no_python")
        launch.assert_not_called()
        start.assert_called_once_with(self.page)
        self.assertNotIn("ask IT", self._said(msg))
        self.assertIsNone(cw.remembered(persistent_only=True))

    def test_unconfirmed_opens_the_page_and_a_late_start_is_noticed(self):
        status, msg, launch, start = self._show(cw.AUTO, ("unconfirmed", "", 4.0))
        self.assertEqual(status, "unconfirmed")
        start.assert_called_once_with(self.page)
        self.assertIn("may still appear", self._said(msg))
        self.assertIsNone(cw.remembered(persistent_only=True))
        status, *_ = self._show(cw.AUTO)
        self.assertEqual(status, "skipped")
        # The window did start in the end (its marker is there): try again.
        with open(self.bundle + ".started", "w") as fh:
            fh.write("1 0\n")
        status, *_ = self._show(cw.AUTO)
        self.assertEqual(status, "started")

    def test_an_open_window_is_refreshed_and_clears_any_record(self):
        cw.remember("failed", "denied", persist=True)
        with open(self.bundle + ".lock", "w") as fh:
            fh.write("1 0\n")
        status, msg, launch, start = self._show(cw.AUTO)
        self.assertEqual(status, "refreshed")
        launch.assert_not_called()
        start.assert_not_called()
        self.assertIsNone(cw.remembered())

    def test_browser_page_and_off(self):
        status, msg, launch, start = self._show(cw.PAGE)
        self.assertEqual(status, "page")
        launch.assert_not_called()
        start.assert_called_once_with(self.page)
        status, msg, launch, start = self._show(cw.OFF)
        self.assertEqual(status, "off")
        launch.assert_not_called()
        start.assert_not_called()

    def test_messages_follow_what_actually_happened(self):
        # No page written.
        status, msg, _, start = self._show(cw.PAGE, page=None)
        start.assert_not_called()
        self.assertIn("no browser chart for this run", self._said(msg))
        # The browser could not be started: say where the page is instead.
        msg = CollectingMessenger()
        with mock.patch.object(cw.os, "startfile", create=True,
                               side_effect=OSError("no browser")):
            cw.show_chart(cw.PAGE, bundle=self.bundle, page=self.page, viewer=VIEWER,
                          root=ROOT, find_python=lambda: "py", msg=msg)
        self.assertIn("double-clicking", self._said(msg))
        self.assertNotIn("Browser chart opened", self._said(msg))


_STANDIN = r'''
import os, sys, time
bundle = sys.argv[1]
mode = os.path.basename(bundle).split(".")[0]
marker = os.environ.get("SWATH_PROFILE_STARTED_FILE")
if mode == "block":
    sys.stderr.write("PermissionError: pretend policy stopped this\n")
    sys.exit(5)
if mode != "silent" and marker:
    open(marker, "w").write(str(os.getpid()))
if mode == "crash":
    time.sleep(0.2)
    sys.stderr.write("Traceback (most recent call last): pretend crash\n")
    sys.exit(3)
if mode == "heartbeat":
    open(bundle + ".lock", "w").write("1")
time.sleep(1.5)
'''


@unittest.skipUnless(os.environ.get("CI") or os.environ.get("SWATH_TEST_SUBPROCESS"),
                     "starts real programs; runs in CI or with SWATH_TEST_SUBPROCESS=1")
class RealLaunchTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="swathlaunch_")
        self.viewer = os.path.join(self.tmp, "standin.py")
        with open(self.viewer, "w", encoding="utf-8") as fh:
            fh.write(_STANDIN)

    def tearDown(self):
        time.sleep(1.8)                                   # let the stand-ins finish
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _launch(self, mode, wait=2.0, settle=0.4):
        bundle = os.path.join(self.tmp, f"{mode}.swath.npz")
        return cw.launch(sys.executable, self.viewer, bundle, self.tmp, wait=wait,
                         settle=settle)

    def test_a_normal_start_is_confirmed_by_its_marker(self):
        status, detail, seconds = self._launch("normal")
        self.assertEqual(status, "started")
        self.assertLess(seconds, 2.0)

    def test_a_heartbeat_confirms_at_once(self):
        status, _, seconds = self._launch("heartbeat", settle=5.0)
        self.assertEqual(status, "started")
        self.assertLess(seconds, 2.0)

    def test_stopped_before_running_is_failed_and_quotes_python(self):
        status, detail, _ = self._launch("block")
        self.assertEqual(status, "failed")
        self.assertIn("exit code 5", detail)
        self.assertIn("pretend policy", detail)

    def test_closing_after_starting_is_a_crash(self):
        status, detail, _ = self._launch("crash")
        self.assertEqual(status, "crashed")
        self.assertIn("exit code 3", detail)
        self.assertIn("pretend crash", detail)

    def test_silence_is_unconfirmed(self):
        status, _, seconds = self._launch("silent", wait=0.8)
        self.assertEqual(status, "unconfirmed")
        self.assertGreaterEqual(seconds, 0.8)

    def test_a_program_that_cannot_be_started_is_failed(self):
        bundle = os.path.join(self.tmp, "x.swath.npz")
        status, detail, _ = cw.launch(os.path.join(self.tmp, "no_such_python.exe"),
                                      self.viewer, bundle, self.tmp)
        self.assertEqual(status, "failed")
        self.assertIn("could not be started", detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
