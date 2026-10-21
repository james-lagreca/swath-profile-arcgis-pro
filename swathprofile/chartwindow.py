# -*- coding: utf-8 -*-
"""Showing a run: the chart window, or the browser chart when it cannot run.

No arcpy - standard library only - so all of it is tested in CI.

The chart window (``viewer/swath_viewer.py``) is a separate Python program.
On a computer where security software lets only approved programs run, that
program can be stopped before it starts, and every attempt raises a notice for
the user and is usually logged for IT.  So this module finds out whether the
window really started, says plainly when it did not, opens the browser chart
instead, and remembers - in memory for this ArcGIS Pro session, and for a
definite failure in a small file for this user and computer - so the window is
not tried again until the user asks for it.

Nothing here tries to get the window past such a block.  That is the
organisation's decision; the legitimate routes are IT approving the viewer, or
the browser chart, which needs no Python at all.

What "started" means
--------------------
The viewer writes a *start marker* (path given in :data:`MARKER_ENV`) before
it imports anything heavy - measured at a quarter of a second after launch -
and a heartbeat file once its window is up, about five seconds after.  From
the launching side the outcomes are:

``started``      the heartbeat appeared, or the marker did and the program
                 kept running for :data:`SETTLE` seconds;
``unconfirmed``  still running, but no marker within :data:`START_WAIT`;
``failed``       could not be started, or ended before writing the marker -
                 which is what a block looks like, and also some other faults;
``crashed``      ended after the marker: the script ran, so not a block.

A block and some other faults look identical from here; the messages say so
rather than claim a cause.  stderr goes to ``<bundle>.viewer.log`` and its last
line is quoted, which usually tells them apart for a human reader.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import time

__all__ = ["AUTO", "WINDOW", "PAGE", "OFF", "MODES", "STATE_DIR_ENV", "MARKER_ENV",
           "parse_mode", "legacy_label", "viewer_alive", "state_path", "remembered",
           "remember", "forget", "launch", "open_page", "show_chart"]

AUTO, WINDOW, PAGE, OFF = "Automatic", "Chart window", "Browser page", "Off"
MODES = (AUTO, WINDOW, PAGE, OFF)

#: Relocates the remembered state (tests).
STATE_DIR_ENV = "SWATH_PROFILE_STATE_DIR"
#: Tells the viewer where to write its start marker.  An environment variable,
#: not a path derived from the bundle, so a viewer opened by hand on a saved
#: result never leaves marker files beside it.
MARKER_ENV = "SWATH_PROFILE_STARTED_FILE"

#: Seconds since the viewer's last heartbeat before it is considered gone.
HEARTBEAT_STALE = 8.0
#: Seconds to wait for the start marker.
START_WAIT = 4.0
#: Seconds the program must keep running after the marker to count as started
#: (the imports that could fail take a second or two after it).
SETTLE = 2.5
POLL = 0.1

#: This process's record - one ArcGIS Pro session.
SESSION: dict = {}

_TRUE = ("true", "1", "yes", "on")
_FALSE = ("false", "0", "no", "off", "none")


def parse_mode(value) -> str:
    """Show Interactive Chart as one of :data:`MODES`.

    It used to be a check box, so a script may still pass True or False, or
    their text.  Blank means the default.
    """
    if isinstance(value, bool):
        return AUTO if value else OFF
    text = str(value if value is not None else "").strip().lower()
    for mode in MODES:
        if text == mode.lower():
            return mode
    if text in _FALSE and text:
        return OFF
    return AUTO


def legacy_label(text):
    """The dialog label for a legacy or differently-cased value, else None.

    Called from ``updateParameters``, which runs before the value list is
    checked, so ``open_viewer=False`` from an old script becomes "Off" instead
    of failing validation.  None means leave the value alone.
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw or raw in MODES:
        return None
    low = raw.lower()
    for mode in MODES:
        if low == mode.lower():
            return mode
    if low in _TRUE:
        return AUTO
    if low in _FALSE:
        return OFF
    return None


def viewer_alive(bundle) -> bool:
    """Whether a chart window is open on *bundle* (its heartbeat is fresh)."""
    try:
        return (time.time() - os.path.getmtime(bundle + ".lock")) < HEARTBEAT_STALE
    except OSError:
        return False


# -- remembering ---------------------------------------------------------------

def state_path() -> str:
    base = (os.environ.get(STATE_DIR_ENV) or os.environ.get("LOCALAPPDATA")
            or os.path.expanduser("~"))
    return os.path.join(base, "SwathProfile", "viewer_state.json")


def remembered(persistent_only: bool = False):
    """The last recorded failure, ``{"kind", "since", "detail"}``, or None.

    The session's record comes first; the file only ever holds a definite
    failure ("failed").  A corrupt or unreadable file counts as no record.
    """
    if SESSION and not persistent_only:
        return dict(SESSION)
    try:
        with open(state_path(), encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return None
    if isinstance(state, dict) and state.get("chart_window") == "unavailable":
        return {"kind": "failed", "since": str(state.get("since", "")),
                "detail": str(state.get("detail", ""))}
    return None


def remember(kind: str, detail: str = "", persist: bool = False) -> None:
    since = datetime.date.today().isoformat()
    SESSION.clear()
    SESSION.update(kind=kind, since=since, detail=detail)
    if not persist:
        return
    path = state_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"chart_window": "unavailable", "since": since,
                       "detail": detail}, fh, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass


def forget() -> None:
    """The window started: clear every record (this is how an IT approval lands)."""
    SESSION.clear()
    try:
        os.remove(state_path())
    except OSError:
        pass


# -- launching -------------------------------------------------------------------

def _log_tail(path, limit=200) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return ""
    return lines[-1][:limit] if lines else ""


def launch(python, viewer, bundle, root, wait=None, settle=None):
    """Start the chart window on *bundle* and find out whether it runs.

    Returns ``(status, detail, seconds)``; see the module docstring for the
    statuses.
    """
    wait = START_WAIT if wait is None else float(wait)
    settle = SETTLE if settle is None else float(settle)
    marker, lock, log = bundle + ".started", bundle + ".lock", bundle + ".viewer.log"
    for stale in (marker, lock):
        try:
            os.remove(stale)              # viewer_alive was False, so a lock is stale
        except OSError:
            pass
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env[MARKER_ENV] = marker
    flags = 0
    for name in ("CREATE_NO_WINDOW", "DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
        flags |= getattr(subprocess, name, 0)
    try:
        err = open(log, "w", encoding="utf-8")
    except OSError:
        err = subprocess.DEVNULL
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen([python, viewer, bundle], cwd=root, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=err, creationflags=flags, close_fds=True)
    except OSError as exc:
        return "failed", f"{python} could not be started: {type(exc).__name__}: {exc}", 0.0
    finally:
        if err is not subprocess.DEVNULL:
            err.close()
    marker_seen = None
    while True:
        now = time.monotonic()
        code = proc.poll()
        if code is not None:
            said = _log_tail(log)
            said = f"; Python said: {said}" if said else ""
            if marker_seen is None and not os.path.exists(marker):
                return ("failed", f"it ended after {now - t0:.1f} s with exit code "
                        f"{code}, before reporting that it was running{said}", now - t0)
            return ("crashed", f"it closed again after {now - t0:.1f} s with exit code "
                    f"{code}{said}", now - t0)
        if viewer_alive(bundle):
            return "started", "", now - t0
        if marker_seen is None and os.path.exists(marker):
            marker_seen = now
        if marker_seen is not None and now - marker_seen >= settle:
            return "started", "", now - t0
        if now - t0 >= wait:
            return ("started" if marker_seen is not None else "unconfirmed"), "", now - t0
        time.sleep(POLL)


def open_page(page):
    """Open the browser chart in the default browser.  None, or why it could not."""
    try:
        os.startfile(page)                      # Windows: the default browser
        return None
    except Exception as exc:                    # noqa: BLE001 - reported, never fatal
        return f"{type(exc).__name__}: {exc}"


def _show_page(page, msg, lead=""):
    """Open the page and report exactly what happened, after it happened."""
    if not page:
        msg.warn(f"  {lead}There is no browser chart for this run (see the messages "
                 "above); the results are in the tables.")
        return
    err = open_page(page)
    if err is None:
        msg.info(f"  {lead}Browser chart opened: {page}")
    else:
        msg.info(f"  {lead}Open the browser chart by double-clicking {page} (your "
                 f"browser could not be started from here: {err}).")


_SKIP_REASON = {
    "failed": "the chart window could not start on this computer on {since}",
    "crashed": "the chart window closed unexpectedly earlier in this session",
    "no_python": "no Python was found for the chart window earlier in this session",
    "unconfirmed": ("the chart window did not confirm that it started earlier in "
                    "this session"),
}


def show_chart(mode, *, bundle, page, viewer, root, find_python, msg) -> str:
    """Show a run per *mode*; returns what was done (for tests and messages).

    *find_python* returns the interpreter for the window or raises;
    *msg* is a Messenger.  Return values: ``off``, ``page``, ``refreshed``,
    ``skipped``, ``started``, ``unconfirmed``, ``failed``, ``crashed``,
    ``no_python``.
    """
    mode = parse_mode(mode)
    if mode == OFF:
        return "off"
    if mode == PAGE:
        _show_page(page, msg)
        return "page"
    if viewer_alive(bundle):
        forget()
        msg.info("  Interactive chart refreshed in the open viewer window.")
        return "refreshed"

    record = remembered()
    if record and record["kind"] == "unconfirmed" and os.path.exists(bundle + ".started"):
        record = None                           # it did start in the end
        forget()
    if record and mode == AUTO:
        reason = _SKIP_REASON.get(record["kind"], _SKIP_REASON["failed"])
        msg.info(f"  Showing the browser chart: {reason.format(since=record['since'])}. "
                 f"Choose '{WINDOW}' under Show Interactive Chart to try the window again.")
        _show_page(page, msg)
        return "skipped"

    try:
        python = find_python()
    except Exception as exc:                    # noqa: BLE001
        remember("no_python", str(exc))
        msg.warn(f"No Python was found to run the chart window ({exc}). Your results "
                 "are complete.")
        _show_page(page, msg)
        return "no_python"

    status, detail, _ = launch(python, viewer, bundle, root)
    if status == "started":
        forget()
        msg.info("  Interactive chart opened in a separate window.")
        return status
    if status == "unconfirmed":
        remember("unconfirmed", "it did not confirm that it started")
        msg.warn("The chart window has not confirmed that it started. It may still "
                 "appear; the browser chart is opening as well.")
        _show_page(page, msg)
        return status
    if status == "crashed":
        remember("crashed", detail)
        msg.warn(f"The chart window started but {detail}. Your results are complete.")
        _show_page(page, msg)
        msg.info("  That looks like a fault in the chart window rather than a block. It "
                 f"is tried again in the next ArcGIS Pro session, or now with '{WINDOW}'.")
        return status
    remember("failed", detail, persist=True)
    msg.warn(f"The chart window could not start on this computer ({detail}). Your "
             "results are complete.")
    _show_page(page, msg)
    msg.info("  The chart window is a separate Python program. Security software that "
             "lets only approved programs run can stop programs started this way, and "
             "from here that looks the same as some other start-up faults. To use the "
             "window, ask IT to approve the viewer folder "
             f"{os.path.dirname(os.path.abspath(viewer))}; if ArcGIS Pro uses a cloned "
             f"Python environment, its Python ({python}) may need approving too. Then "
             f"choose '{WINDOW}' under Show Interactive Chart once. Until then, "
             f"'{AUTO}' uses the browser chart on this computer and does not try the "
             "window.")
    return status
