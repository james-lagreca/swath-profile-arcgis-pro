# -*- coding: utf-8 -*-
"""Swath Profile toolbox for ArcGIS Pro - a single tool.

A port of the Perez-Pena et al. (2017) SwathProfiler method
(Computers & Geosciences 104, 135-150) to arcpy/numpy.

The toolbox itself is deliberately thin: it only wires ArcGIS Pro to the
``swathprofile`` package, so the algorithm can be tested and scripted without
going through the geoprocessing framework.

Note: in the Catalog pane, "Remove" takes a toolbox out of the project;
"Delete" deletes this file from disk.
"""

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ArcGIS Pro caches imported packages across a toolbox refresh, so edits to the
# package appear not to take effect until the project is reopened.  Set
# SWATH_DEV_RELOAD=1 while developing.
if os.environ.get("SWATH_DEV_RELOAD"):
    for _name in [m for m in list(sys.modules)
                  if m.split(".")[0] in ("swathprofile", "tools")]:
        try:
            importlib.reload(sys.modules[_name])
        except Exception:
            sys.modules.pop(_name, None)

from tools.extract_swath_profile import SwathProfileTool  # noqa: E402


class Toolbox(object):
    def __init__(self):
        self.label = "Swath Profile"
        self.alias = "swathprofile"
        self.tools = [SwathProfileTool]


__all__ = ["Toolbox", "SwathProfileTool"]
