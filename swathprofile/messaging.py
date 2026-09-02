# -*- coding: utf-8 -*-
"""Message and progress sinks.

The core modules must not import arcpy, but they still need to report warnings
and progress.  They take a ``Messenger`` instead.  ``ArcpyMessenger`` forwards
to the geoprocessing message pane and progressor; ``ConsoleMessenger`` prints;
``CollectingMessenger`` records, which is what the tests assert against.
"""

from __future__ import annotations


class Messenger:
    """Message sink protocol.  The default implementation discards everything."""

    def info(self, text: str) -> None:  # pragma: no cover - trivial
        pass

    def warn(self, text: str) -> None:  # pragma: no cover - trivial
        pass

    def error(self, text: str) -> None:  # pragma: no cover - trivial
        pass

    def progress(self, done: int, total: int, label: str = "") -> None:
        """Report progress; *done* of *total* units complete."""
        pass


class ConsoleMessenger(Messenger):
    def info(self, text):
        print(text)

    def warn(self, text):
        print("WARNING: " + text)

    def error(self, text):
        print("ERROR: " + text)

    def progress(self, done, total, label=""):
        if total and (done == total or done % max(total // 10, 1) == 0):
            print(f"  ... {done}/{total} {label}".rstrip())


class CollectingMessenger(Messenger):
    """Records messages so tests can assert on them."""

    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []
        self.progress_calls = []

    def info(self, text):
        self.infos.append(text)

    def warn(self, text):
        self.warnings.append(text)

    def error(self, text):
        self.errors.append(text)

    def progress(self, done, total, label=""):
        self.progress_calls.append((done, total, label))

    def has_warning(self, substring):
        return any(substring.lower() in w.lower() for w in self.warnings)


class ArcpyMessenger(Messenger):
    """Forwards to the geoprocessing message pane and progressor.

    Calling ``SetProgressorPosition`` is also what gives ArcGIS Pro a chance
    to honour Cancel inside a long loop.
    """

    def __init__(self):
        import arcpy  # noqa: F401  (deferred so the core stays arcpy-free)
        self._arcpy = arcpy
        self._total = None

    def info(self, text):
        self._arcpy.AddMessage(text)

    def warn(self, text):
        self._arcpy.AddWarning(text)

    def error(self, text):
        self._arcpy.AddError(text)

    def progress(self, done, total, label=""):
        try:
            if total != self._total:
                self._arcpy.SetProgressor("step", label or "Working...", 0,
                                          max(int(total), 1), 1)
                self._total = total
            elif label:
                self._arcpy.SetProgressorLabel(label)
            self._arcpy.SetProgressorPosition(int(done))
        except Exception:      # pragma: no cover - progressor is best-effort
            pass
