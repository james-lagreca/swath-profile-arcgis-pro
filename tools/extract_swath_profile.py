# -*- coding: utf-8 -*-
"""The Swath Profile geoprocessing tool - the only tool in the toolbox.

Minimal interaction: draw a line on the map (or pick a layer, or give a fault
trace and a spacing), choose a raster, Run.  Everything else has a sensible
default, outputs land in the project geodatabase, and the chart opens in the
viewer window - which stays open and refreshes on every subsequent run.
"""

from __future__ import annotations

import datetime
import os
import sys
import time

import numpy as np

import arcpy

from swathprofile import __version__
from swathprofile import chartwindow, features, outputs, plotting, raster_io
from swathprofile.bundle import BUNDLE_SUFFIX, save_bundle
from swathprofile.config import SwathConfig, estimate_run
from swathprofile.engine import run_swath
from swathprofile.errors import SwathError
from swathprofile.messaging import ArcpyMessenger
from swathprofile.smoothing import SMOOTH_PRESETS

METHOD_OFFSET = "Offset lines (classic, keeps profile curves)"
METHOD_ZONAL = "All cells (exact envelope)"
SIDES = {"Both sides": "both", "Left of trace": "left", "Right of trace": "right"}
#: Dialog wording -> plotting mode for the individual profile lines.  Offset
#: lines only: All-cells sampling keeps no per-line samples to colour.
PROFILE_COLOUR_CHOICES = {"Single colour": "single", "Side of line": "side",
                          "Side + distance": "ramp"}
#: Dialog wording -> sampling-mask mode.
MASK_MODES = {"Keep only inside": "keep", "Exclude inside": "exclude"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER_SCRIPT = os.path.join(ROOT, "viewer", "swath_viewer.py")

#: Warn before Run when the estimated time exceeds this.
ESTIMATE_WARN_SECONDS = 120.0

#: Choices for Show Interactive Chart (see swathprofile/chartwindow.py).
VIEWER_AUTO, VIEWER_WINDOW, VIEWER_PAGE, VIEWER_OFF = (
    chartwindow.AUTO, chartwindow.WINDOW, chartwindow.PAGE, chartwindow.OFF)
VIEWER_MODES = chartwindow.MODES

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _project():
    try:
        return arcpy.mp.ArcGISProject("CURRENT")
    except Exception:
        return None


_GDB_CACHE = {}


def _project_gdb():
    """The open project's default geodatabase, else the scratch geodatabase.

    Dialog-time only (never during execution), and cached: ``ArcGISProject
    ("CURRENT")`` can take a long time on a large project, and validation runs
    on every keystroke.
    """
    if "gdb" not in _GDB_CACHE:
        gdb = None
        aprx = _project()
        if aprx is not None:
            try:
                gdb = aprx.defaultGeodatabase
            except Exception:
                gdb = None
        _GDB_CACHE["gdb"] = gdb or arcpy.env.scratchGDB
    return _GDB_CACHE["gdb"]


def _drive_is_slow(path) -> bool:
    """True for network, removable or unknown drives (Windows only).

    File geodatabase writes are dozens of small file operations per table;
    on a network share or a cloud-synced/removable drive each one can take
    seconds, which turns a 20 s run into 25 minutes.
    """
    if not path:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            drive = os.path.splitdrive(os.path.abspath(path))[0]
            if not drive or path.startswith("\\\\"):
                return True                            # UNC share
            kind = ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")
            # 2 = removable, 3 = fixed, 4 = remote, 5 = CD-ROM, 6 = RAM disk
            if kind in (0, 1, 2, 4, 5):
                return True
        except Exception:
            pass
    # A drive can report "fixed" and still be slow (external disk, BitLocker,
    # antivirus scanning geodatabase files): time a small write and judge.
    return _write_latency(path) > 0.3


def _write_latency(path) -> float:
    """Seconds to create, flush and delete a small file in *path*'s folder."""
    import uuid

    folder = path
    while folder and not os.path.isdir(folder):
        parent = os.path.dirname(folder)
        if parent == folder:
            return 0.0
        folder = parent
    probe = os.path.join(folder, f".swath_probe_{uuid.uuid4().hex[:8]}")
    t0 = time.perf_counter()
    try:
        with open(probe, "wb") as fh:
            fh.write(b"0" * 1024)
            fh.flush()
            os.fsync(fh.fileno())
        os.remove(probe)
    except Exception:
        return 0.0
    return time.perf_counter() - t0


def _local_results_home():
    """A fast local folder for results when the project lives on a slow drive."""
    docs = os.path.join(os.path.expanduser("~"), "Documents", "ArcGIS")
    return docs if os.path.isdir(docs) else os.path.expanduser("~")


def _results_gdb():
    """The dedicated output geodatabase: <project folder>/swath_results/swath_results.gdb.

    Outputs deliberately do not default into the project's own geodatabase.
    That is the one the user is editing (sketches, fault traces), and creating
    a dataset in a geodatabase with an unsaved edit session blocks until the
    edits are saved.  A geodatabase nothing else edits has no such lock.

    When the project folder is on a network or removable drive the results
    geodatabase goes to a local folder instead (Documents\\ArcGIS\\swath_results),
    because file-geodatabase writes on such drives are painfully slow.
    Dialog-time only.
    """
    if "results_gdb" in _GDB_CACHE:
        return _GDB_CACHE["results_gdb"]
    project_gdb = _project_gdb()
    home = os.path.dirname(project_gdb) if project_gdb.lower().endswith(".gdb") \
        else arcpy.env.scratchFolder
    if _drive_is_slow(home):
        home = _local_results_home()
        _GDB_CACHE["relocated"] = True
    folder = os.path.join(home, "swath_results")
    gdb = os.path.join(folder, "swath_results.gdb")
    try:
        os.makedirs(folder, exist_ok=True)
        if not arcpy.Exists(gdb):
            arcpy.management.CreateFileGDB(folder, "swath_results.gdb")
    except Exception:
        gdb = arcpy.env.scratchGDB
    _GDB_CACHE["results_gdb"] = gdb
    return gdb


def _ensure_workspace(path):
    """Create the file geodatabase an output path points into, if missing."""
    if not path:
        return
    low = path.replace("/", os.sep).lower()
    idx = low.find(".gdb")
    if idx < 0:
        return
    gdb = path[:idx + 4]
    if not arcpy.Exists(gdb):
        folder, name = os.path.split(gdb)
        os.makedirs(folder, exist_ok=True)
        arcpy.management.CreateFileGDB(folder, name)


def _active_map():
    aprx = _project()
    if aprx is None:
        return None
    try:
        return aprx.activeMap
    except Exception:
        return None


def _active_map_sr():
    m = _active_map()
    try:
        return None if m is None else m.spatialReference
    except Exception:
        return None


def _stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def _by_name(params):
    return {p.name: p for p in params}


def _linear_units(param, target_sr, fallback=None):
    """Convert a GPLinearUnit parameter into the target CRS's linear units."""
    if param.value is None:
        return fallback
    parts = (param.valueAsText or "").strip().split()
    try:
        magnitude = float(parts[0])
    except (ValueError, IndexError):
        return fallback
    unit = parts[1] if len(parts) > 1 else "Unknown"
    target = getattr(target_sr, "linearUnitName", "Meter") or "Meter"
    if unit in ("Unknown", "") or target in ("Unknown", ""):
        return magnitude
    try:
        return magnitude * arcpy.LinearUnitConversionFactor(unit, target)
    except Exception:
        return magnitude


def _metres(param, fallback=None):
    """A GPLinearUnit parameter in metres (for estimates and validation)."""
    return _linear_units(param, arcpy.SpatialReference(3857), fallback)


def _cell_size_metres(raster_value):
    """Cell size in ground metres, whatever the raster's CRS."""
    d = arcpy.Describe(raster_value)
    cell = max(float(d.meanCellWidth), float(d.meanCellHeight))
    sr = getattr(d, "spatialReference", None)
    if sr is not None and sr.type == "Projected":
        try:
            return cell * arcpy.LinearUnitConversionFactor(sr.linearUnitName, "Meter")
        except Exception:
            return cell
    return cell * 111_000.0            # degrees -> rough metres


def _feature_count(feature_set, cap=1000):
    """Features in a dialog sketch, counted with a cursor (never GetCount).

    Only used on sketch record sets, which carry no selection a cursor could
    clear; counting stops at *cap*, because only zero versus some matters.
    """
    try:
        n = 0
        with arcpy.da.SearchCursor(feature_set, ["OID@"]) as cur:
            for _ in cur:
                n += 1
                if n >= cap:
                    break
        return n
    except Exception:
        return -1                           # unknown: say nothing


def _total_length_metres(source, cap=2000):
    """Sum of feature lengths (honours selection), in metres; None if unknown.

    Runs at validation time, so it must not disturb the user's map selection -
    and iterating a cursor over a layer object clears its selection.  Lengths
    are therefore read from the underlying feature class, restricted to the
    selected OIDs when there is a selection.
    """
    if source is None:
        return None
    try:
        d = arcpy.Describe(source)
        sr = d.spatialReference
        if sr is None or not sr.name or sr.name == "Unknown":
            return None
        target = getattr(d, "catalogPath", None) or source
        where = None
        fidset = getattr(d, "FIDSet", "") or ""
        if fidset:
            ids = [f.strip() for f in fidset.split(";") if f.strip()][:cap]
            oid_field = getattr(d, "OIDFieldName", "OBJECTID") or "OBJECTID"
            where = f"{oid_field} IN ({','.join(ids)})"
        total, n = 0.0, 0
        with arcpy.da.SearchCursor(target, ["SHAPE@LENGTH"], where) as cur:
            for (length,) in cur:
                total += float(length or 0.0)
                n += 1
                if n >= cap:
                    break
        if sr.type == "Projected":
            try:
                total *= arcpy.LinearUnitConversionFactor(sr.linearUnitName, "Meter")
            except Exception:
                pass
        else:
            total *= 111_000.0
        return total
    except Exception:
        return None


def _layer_stem(value):
    try:
        d = arcpy.Describe(value)
        name = getattr(d, "baseName", None) or getattr(d, "name", None) or str(value)
    except Exception:
        name = str(value)
    return outputs.safe_prefix(os.path.basename(name), 24).lower()


def _python_for_viewer():
    """Locate a real python interpreter for the detached viewer process.

    Inside ArcGIS Pro ``sys.executable`` is **ArcGISPro.exe**, not python - so
    "the interpreter next to sys.executable" launches a second copy of Pro.
    ``sys.prefix`` is the active conda environment and is reliable in both
    contexts.  ArcGISPro.exe is never returned, whatever the fallbacks find.
    """
    candidates = [os.path.join(sys.prefix, "pythonw.exe"),
                  os.path.join(sys.prefix, "python.exe")]
    try:
        install = arcpy.GetInstallInfo().get("InstallDir", "")
        env_dir = os.path.join(install, "bin", "Python", "envs", "arcgispro-py3")
        candidates += [os.path.join(env_dir, "pythonw.exe"),
                       os.path.join(env_dir, "python.exe")]
    except Exception:
        pass
    if "arcgispro.exe" not in os.path.basename(sys.executable).lower():
        candidates.append(sys.executable)
    for c in candidates:
        if c and os.path.exists(c) and "arcgispro.exe" not in c.lower():
            return c
    raise SwathError(
        "Could not find a Python interpreter for the viewer. Looked in "
        f"{sys.prefix}. The result bundle was still written.")


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------

class SwathProfileTool(object):

    def __init__(self):
        self.label = "Swath Profile"
        self.description = (
            "Draw or choose a baseline - or give a fault trace and a spacing - "
            "pick a DEM, run: per-station minimum, maximum, mean, quartiles, "
            "percentiles, relief, hypsometric integral and THi across a band of "
            "chosen width, with an interactive chart linked to the map. After "
            "Perez-Pena et al. (2017)."
        )
        self.canRunInBackground = False
        self._stamp = _stamp()

    def isLicensed(self):
        return True

    # -- parameters -----------------------------------------------------------

    def getParameterInfo(self):
        def p(name, label, dtype, ptype="Optional", direction="Input",
              multi=False, category=None):
            prm = arcpy.Parameter(displayName=label, name=name, datatype=dtype,
                                  parameterType=ptype, direction=direction,
                                  multiValue=multi)
            if category:
                prm.category = category
            return prm

        draw = p("draw_lines", "Draw Baseline on Map", "GPFeatureRecordSetLayer")
        # Set no value; constrain the sketch to lines with the parameter's own
        # filter.  Pro's interactive input button then builds the sketch layer
        # itself, in the active map's coordinate system, and the filter is what
        # narrows its feature-type menu to a line.  _run() still falls back to
        # asking the map if a sketch ever arrives without a coordinate system.
        #
        # Esri does document assigning a feature class or a .lyrx to .value as a
        # way to give a feature set its schema, but it is the wrong tool here:
        # arcpy materialises whatever path is assigned into an in-memory feature
        # set at assignment time and keeps no link to it, so the template's only
        # lasting effect was to make the dialog depend on process-lifetime
        # state.  The "memory" workspace lives as long as the ArcGIS Pro
        # process, so the first open created memory\swath_baseline_template and
        # gave the parameter a value, while every later open raised
        # "ERROR 000258: Output ... already exists", had it swallowed by the
        # except, and left the parameter unset - the state in which drawing
        # actually works.  Worse, whenever _active_map_sr() returned None the
        # template was created with no coordinate system at all, and a feature
        # set whose CRS is Unknown cannot be edited; that, plus execute()
        # leaking overwriteOutput = True for the rest of the session (restored
        # now), is the "sometimes" in the bug report.
        #
        # A .lyrx template would fix none of that and would bake one fixed CRS
        # into every user's sketch.  Dropping the template also takes a
        # geoprocessing call and an arcpy.mp map access out of dialog
        # construction, which matters where every framework call costs tens of
        # seconds.
        #
        # The filter is not documented to drive the sketch menu, so
        # updateMessages checks the shape type as well.  No try/except here:
        # the assignment cannot raise.  Measured on arcpy 3.4, an unsupported
        # keyword is silently coerced to an empty list instead ("Line" and
        # "Bogus" both give []), so "Polyline" is the spelling that survives and
        # a handler would only imply a failure mode that does not exist.
        draw.filter.list = ["Polyline"]

        lines = p("in_lines", "...or Baseline Layer", "GPFeatureLayer")
        lines.filter.list = ["Polyline"]

        label_field = p("label_field", "Baseline Label Field", "Field")
        label_field.parameterDependencies = ["in_lines"]
        label_field.filter.list = ["Text"]

        rasters = p("in_rasters", "Raster(s) - DEM first", "GPRasterLayer",
                    "Required", multi=True)

        width = p("swath_width", "Swath Width (total)", "GPLinearUnit", "Required")
        width.value = "2000 Meters"

        gdb = _results_gdb()
        table = p("out_table", "Output Statistics Table", "DETable",
                  direction="Output")
        table.value = os.path.join(gdb, f"swath_sketch_{self._stamp}")

        # -- Transects --------------------------------------------------------
        cat = "Transects along a trace"
        tsrc = p("transect_source", "Trace to Place Transects Along", "GPFeatureLayer",
                 category=cat)
        tsrc.filter.list = ["Polyline"]
        tspacing = p("transect_spacing", "Transect Spacing", "GPLinearUnit",
                     category=cat)
        tspacing.value = "500 Meters"
        tlength = p("transect_length", "Transect Length (total)", "GPLinearUnit",
                    category=cat)
        tlength.value = "3000 Meters"
        twidth = p("transect_width", "Transect Swath Width (blank = Swath Width)",
                   "GPLinearUnit", category=cat)
        tside = p("transect_side", "Transect Side", "GPString", category=cat)
        tside.filter.type = "ValueList"
        tside.filter.list = list(SIDES.keys())
        tside.value = "Both sides"
        tstart = p("transect_start", "Distance Along Trace to First Transect",
                   "GPLinearUnit", category=cat)
        tstart.value = "0 Meters"

        # -- Sampling mask ----------------------------------------------------
        cat = "Sampling mask"
        # A polygon sketch, set up exactly like draw_lines and for the same
        # reasons (see the comment there): no value, so no template is ever
        # bound, and the geometry type asked for by the parameter's own filter.
        mask_draw = p("mask_draw", "Draw Sampling Mask on Map",
                      "GPFeatureRecordSetLayer", category=cat)
        mask_draw.filter.list = ["Polygon"]
        mask_layer = p("mask_layer", "...or Sampling Mask Layer", "GPFeatureLayer",
                       category=cat)
        mask_layer.filter.list = ["Polygon"]
        mask_mode = p("mask_mode", "Mask Mode", "GPString", category=cat)
        mask_mode.filter.type = "ValueList"
        mask_mode.filter.list = list(MASK_MODES.keys())
        mask_mode.value = "Keep only inside"

        # -- Sampling ---------------------------------------------------------
        method = p("method", "Sampling Method", "GPString", "Required",
                   category="Sampling")
        method.filter.type = "ValueList"
        method.filter.list = [METHOD_OFFSET, METHOD_ZONAL]
        method.value = METHOD_OFFSET

        nlines = p("n_lines", "Number of Parallel Profiles", "GPLong",
                   category="Sampling")
        nlines.value = 51
        nlines.filter.type = "Range"
        nlines.filter.list = [1, 1001]

        step = p("step", "Step Along Profile (default 1.5 x cell)", "GPLinearUnit",
                 category="Sampling")

        interp = p("interpolation", "Interpolation", "GPString", category="Sampling")
        interp.filter.type = "ValueList"
        interp.filter.list = ["Bilinear", "Nearest"]
        interp.value = "Bilinear"

        zfac = p("z_factor", "Z Factor", "GPDouble", category="Sampling")
        zfac.value = 1.0

        multipart = p("multipart", "Multipart Handling", "GPString",
                      category="Sampling")
        multipart.filter.type = "ValueList"
        multipart.filter.list = ["Each part separately", "Largest part", "Error"]
        multipart.value = "Each part separately"

        tsmooth = p("tangent_smooth", "Stabilise Cross-Section Direction",
                    "GPBoolean", category="Sampling")
        tsmooth.value = True

        fold = p("fold_policy", "Tight-Bend Handling", "GPString",
                 category="Sampling")
        fold.filter.type = "ValueList"
        fold.filter.list = ["Mask folded samples", "Warn only", "Ignore"]
        fold.value = "Mask folded samples"

        # -- Statistics -------------------------------------------------------
        smooth = p("smoothing", "Curve Smoothing", "GPString", category="Statistics")
        smooth.filter.type = "ValueList"
        smooth.filter.list = list(SMOOTH_PRESETS.keys())
        smooth.value = "None"

        plo = p("pct_low", "Lower Percentile", "GPDouble", category="Statistics")
        plo.value = 5.0
        phi = p("pct_high", "Upper Percentile", "GPDouble", category="Statistics")
        phi.value = 95.0

        thi_src = p("thi_raster", "THi Source Raster (when several)", "GPString",
                    category="Statistics")

        legacy = p("legacy_stats", "Legacy (ArcGIS-compatible) Statistics",
                   "GPBoolean", category="Statistics")
        legacy.value = False

        flathi = p("flat_hi", "Flat Cross-Section HI", "GPString",
                   category="Statistics")
        flathi.filter.type = "ValueList"
        flathi.filter.list = ["Null", "0.5", "Reference (0.0)"]
        flathi.value = "Null"

        minvalid = p("min_valid", "Minimum Valid Samples per Station", "GPLong",
                     category="Statistics")
        minvalid.value = 1

        # -- Outputs ----------------------------------------------------------
        band = p("out_band", "Swath Band Polygons", "DEFeatureClass",
                 direction="Output", category="Outputs")
        band.value = os.path.join(gdb, f"swath_band_sketch_{self._stamp}")

        stations = p("out_stations", "Station Points (chart linked to map)",
                     "DEFeatureClass", direction="Output", category="Outputs")
        stations.value = os.path.join(gdb, f"swath_stations_sketch_{self._stamp}")

        transects = p("out_transects", "Transects", "DEFeatureClass",
                      direction="Output", category="Outputs")

        offs = p("out_offset_lines", "Offset Lines", "DEFeatureClass",
                 direction="Output", category="Outputs")
        prof3d = p("out_profiles_3d", "3D Profile Lines (Z-enabled)",
                   "DEFeatureClass", direction="Output", category="Outputs")
        summary = p("out_summary", "Summary Table", "DETable",
                    direction="Output", category="Outputs")
        csv_dir = p("out_csv_folder", "CSV Folder", "DEFolder",
                    direction="Output", category="Outputs")

        chart_fmt = p("chart_formats", "Also Save Chart As", "GPString",
                      multi=True, category="Outputs")
        chart_fmt.filter.type = "ValueList"
        chart_fmt.filter.list = ["PNG", "PDF", "SVG"]

        preset = p("chart_preset", "Chart Preset", "GPString", category="Outputs")
        preset.filter.type = "ValueList"
        preset.filter.list = list(plotting.PRESETS.keys())
        preset.value = "Screen"

        pcolour = p("chart_profile_colour", "Colour Profile Lines By", "GPString",
                    category="Outputs")
        pcolour.filter.type = "ValueList"
        pcolour.filter.list = list(PROFILE_COLOUR_CHOICES.keys())
        pcolour.value = "Single colour"

        chart_dir = p("out_chart_folder", "Chart Folder", "DEFolder",
                      direction="Output", category="Outputs")

        viewer = p("open_viewer", "Show Interactive Chart", "GPString",
                   category="Outputs")
        viewer.filter.type = "ValueList"
        viewer.filter.list = list(VIEWER_MODES)
        viewer.value = VIEWER_AUTO

        add_map = p("add_to_map", "Add Outputs to Map", "GPBoolean",
                    category="Outputs")
        add_map.value = True

        # -- Advanced ---------------------------------------------------------
        geotrans = p("geo_transform", "Datum Transformation", "GPString",
                     category="Advanced")
        geotrans.filter.type = "ValueList"
        geotrans.filter.list = []

        blockmb = p("max_block_mb", "Max Raster Block (MB)", "GPLong",
                    category="Advanced")
        blockmb.value = 512

        bundle = p("bundle_path", "Result Bundle (.swath.npz)", "DEFile",
                   direction="Output", category="Advanced")
        bundle.value = os.path.join(arcpy.env.scratchFolder,
                                    "swath_live" + BUNDLE_SUFFIX)

        return [draw, lines, label_field, rasters, width, table,
                tsrc, tspacing, tlength, twidth, tside, tstart,
                mask_draw, mask_layer, mask_mode,
                method, nlines, step, interp, zfac, multipart, tsmooth, fold,
                smooth, plo, phi, thi_src, legacy, flathi, minvalid,
                band, stations, transects, offs, prof3d, summary, csv_dir,
                chart_fmt, preset, pcolour, chart_dir, viewer, add_map,
                geotrans, blockmb, bundle]

    # -- validation -----------------------------------------------------------

    def _raster_values(self, pr):
        v = pr["in_rasters"].value
        if not v:
            return []
        try:
            return [v.getValue(i, 0) for i in range(v.rowCount)]
        except Exception:
            return [v]

    def _is_offset(self, pr):
        return (pr["method"].valueAsText or METHOD_OFFSET) == METHOD_OFFSET

    def updateParameters(self, params):
        pr = _by_name(params)
        rasters = self._raster_values(pr)
        names = [os.path.basename(str(r)) for r in rasters]

        thi = pr["thi_raster"]
        thi.filter.type = "ValueList"
        thi.filter.list = names
        thi.enabled = len(names) > 1
        if names and (not thi.value or thi.valueAsText not in names):
            thi.value = names[0]

        if pr["in_rasters"].altered and rasters and not pr["step"].altered:
            try:
                cell_m = _cell_size_metres(rasters[0])
                pr["step"].value = f"{1.5 * cell_m:g} Meters"
            except Exception:
                pass

        # Output names follow the baseline source, until the user edits them.
        src = pr["in_lines"].value or pr["transect_source"].value
        if (pr["in_lines"].altered or pr["transect_source"].altered):
            stem = _layer_stem(src) if src is not None else "sketch"
            gdb = _results_gdb()
            for name, prefix in (("out_table", "swath"), ("out_band", "swath_band"),
                                 ("out_stations", "swath_stations")):
                if not pr[name].altered:
                    pr[name].value = os.path.join(gdb, f"{prefix}_{stem}_{self._stamp}")
            if pr["transect_source"].value is not None and not pr["out_transects"].altered:
                pr["out_transects"].value = os.path.join(
                    gdb, f"swath_transects_{stem}_{self._stamp}")

        # Datum transformations between the baseline source and the primary raster.
        if src is not None and rasters and (pr["in_lines"].altered or
                                            pr["transect_source"].altered or
                                            pr["in_rasters"].altered):
            try:
                sr_l = arcpy.Describe(src).spatialReference
                sr_r = arcpy.Describe(rasters[0]).spatialReference
                pr["geo_transform"].filter.list = \
                    arcpy.ListTransformations(sr_l, sr_r) or []
            except Exception:
                pr["geo_transform"].filter.list = []

        is_offset = self._is_offset(pr)
        for name in ("n_lines", "interpolation", "fold_policy", "tangent_smooth",
                     "out_offset_lines", "out_profiles_3d"):
            pr[name].enabled = is_offset

        has_trace = pr["transect_source"].value is not None
        for name in ("transect_spacing", "transect_length", "transect_width",
                     "transect_side", "transect_start", "out_transects"):
            pr[name].enabled = has_trace

        pr["mask_mode"].enabled = (pr["mask_draw"].value is not None or
                                   pr["mask_layer"].value is not None)

        # Show Interactive Chart used to be a check box.  This runs before the
        # value list is checked, so a script still passing True or False gets
        # "Automatic" or "Off" instead of a validation error.
        label = chartwindow.legacy_label(pr["open_viewer"].valueAsText)
        if label:
            pr["open_viewer"].value = label

        fmts = set(pr["chart_formats"].values or [])
        pr["out_chart_folder"].enabled = bool(fmts & {"PNG", "PDF", "SVG"})
        pr["chart_preset"].enabled = bool(fmts & {"PNG", "PDF", "SVG"})
        # Colouring the profile lines needs per-line samples, which only offset
        # sampling produces.
        pr["chart_profile_colour"].enabled = (bool(fmts & {"PNG", "PDF", "SVG"})
                                              and is_offset)

    def updateMessages(self, params):
        pr = _by_name(params)
        rasters = self._raster_values(pr)

        # A swath needs a line to run along.  Pro's sketch menu is asked for a
        # line by the parameter's filter, but that is not documented to bind, so
        # the shape type is checked here too: a polygon would silently be read
        # as a baseline along its boundary, and a point fails part-way through
        # the run with a much less helpful message.  Describe on an in-memory
        # record set is not a geoprocessing-framework call, so this is free even
        # in a slow project, and it only runs once something has been drawn.
        sketch = pr["draw_lines"].value
        if sketch is not None:
            try:
                shape_type = getattr(arcpy.Describe(sketch), "shapeType", "")
            except Exception:
                shape_type = ""
            if shape_type and shape_type != "Polyline":
                pr["draw_lines"].setErrorMessage(
                    f"The sketch is {shape_type}, not a line. Click the sketch "
                    "button beside this parameter and choose the line tool, or "
                    "use the Baseline Layer parameter instead.")

        # The mask sketch gets the same check for polygons, and a nudge while it
        # is still empty: an unfinished sketch is ignored, so without the nudge
        # a run would quietly go ahead unmasked.
        mask_sketch = pr["mask_draw"].value
        if mask_sketch is not None:
            try:
                shape_type = getattr(arcpy.Describe(mask_sketch), "shapeType", "")
            except Exception:
                shape_type = ""
            if shape_type and shape_type != "Polygon":
                pr["mask_draw"].setErrorMessage(
                    f"The sketch is {shape_type}, not a polygon. Click the sketch "
                    "button beside this parameter and choose the polygon tool, or "
                    "use the Sampling Mask Layer parameter instead.")
            elif shape_type == "Polygon" and _feature_count(mask_sketch) == 0:
                pr["mask_draw"].setWarningMessage(
                    "No polygon has been finished yet: double-click to close it. "
                    "An empty sketch is ignored and the run is not masked.")

        if chartwindow.parse_mode(pr["open_viewer"].valueAsText) == VIEWER_WINDOW:
            known = chartwindow.remembered(persistent_only=True)
            if known:
                pr["open_viewer"].setWarningMessage(
                    f"The chart window could not start on this computer on "
                    f"{known['since']}. '{VIEWER_WINDOW}' tries it again, and security "
                    "software may show its notice again; the browser chart opens if "
                    "it still cannot start.")

        for value in rasters:
            try:
                d = arcpy.Describe(value)
            except Exception:
                continue
            label = os.path.basename(str(value))
            sr = getattr(d, "spatialReference", None)
            if sr is None or not sr.name or sr.name == "Unknown":
                pr["in_rasters"].setErrorMessage(
                    f"'{label}' has no coordinate system. Define one with "
                    "Define Projection first.")
                break
            if int(getattr(d, "bandCount", 1)) != 1:
                pr["in_rasters"].setErrorMessage(
                    f"'{label}' is multiband. Pick one band, e.g. {label}\\Band_1.")
                break

        if rasters and all(
                getattr(arcpy.Describe(r).spatialReference, "type", "") != "Projected"
                for r in rasters):
            pr["in_rasters"].setWarningMessage(
                "No raster is projected. The swath will be computed in a projected "
                "system chosen automatically (the baseline's, or a UTM zone) and "
                "samples transformed on the fly. Fine for a quick look; for "
                "published work, project the DEM explicitly.")

        cell_m = None
        if rasters:
            try:
                cell_m = _cell_size_metres(rasters[0])
            except Exception:
                cell_m = None
        step_m = _metres(pr["step"], None)
        width_m = _metres(pr["swath_width"], None)

        if cell_m and step_m and step_m < cell_m * 0.999:
            pr["step"].setWarningMessage(
                f"Step ({step_m:g} m) is finer than the cell size ({cell_m:g} m); "
                "adjacent stations resample the same cells without adding "
                "information.")

        n = pr["n_lines"].value
        if n is not None and self._is_offset(pr):
            n = int(n)
            if n == 1:
                pr["n_lines"].setWarningMessage(
                    "With one profile line the relief is identically zero, so HI "
                    "and THi are undefined and will be Null.")
            elif n % 2 == 0:
                pr["n_lines"].setWarningMessage(
                    "An even count places no line exactly on the baseline; use an "
                    "odd number to include the centreline.")

        if pr["min_valid"].value is not None and int(pr["min_valid"].value) < 1:
            pr["min_valid"].setErrorMessage("Must be at least 1.")
        if pr["z_factor"].value is not None and float(pr["z_factor"].value) == 0:
            pr["z_factor"].setErrorMessage("Z factor must not be zero.")

        lo, hi = pr["pct_low"].value, pr["pct_high"].value
        if lo is not None and hi is not None and not (0 < float(lo) < float(hi) < 100):
            pr["pct_high"].setErrorMessage(
                "Percentiles must satisfy 0 < lower < upper < 100.")

        # Transects.
        trace = pr["transect_source"].value
        if trace is not None:
            sp = _metres(pr["transect_spacing"], None)
            ln = _metres(pr["transect_length"], None)
            if sp is not None and sp <= 0:
                pr["transect_spacing"].setErrorMessage("Spacing must be positive.")
            if ln is not None and ln <= 0:
                pr["transect_length"].setErrorMessage("Length must be positive.")
            if sp and width_m and sp < width_m:
                pr["transect_spacing"].setWarningMessage(
                    f"Spacing ({sp:g} m) is less than the swath width ({width_m:g} m): "
                    "adjacent bands overlap. Allowed, but each patch of ground then "
                    "appears in more than one profile.")
            tw = _metres(pr["transect_width"], None) or width_m
            if tw is not None and tw <= 0:
                pr["transect_width"].setErrorMessage("Transect swath width must be positive.")
            if ln and tw and ln < tw:
                pr["transect_length"].setWarningMessage(
                    f"Each transect ({ln:g} m) is shorter than its band is wide "
                    f"({tw:g} m); the swath will be wider than it is long.")

        if pr["geo_transform"].filter.list and not pr["geo_transform"].value:
            pr["geo_transform"].setWarningMessage(
                "A datum transformation exists between the baseline and raster "
                "systems but none is selected; without one the baseline can be "
                "shifted by 100 m or more with no other symptom. Choose one under "
                "Advanced unless the datums are identical.")

        # Pre-run cost estimate, so a ten-minute run is never a surprise.
        if rasters and cell_m and step_m and width_m:
            method = "offset" if self._is_offset(pr) else "zonal"
            n_lines = int(pr["n_lines"].value or 51)
            seconds, work_text = 0.0, []
            total = 0.0
            for src in (pr["in_lines"].value, pr["draw_lines"].value):
                length = _total_length_metres(src)
                if length:
                    total += length
            if total > 0:
                est = estimate_run(total, step_m, n_lines, width_m, cell_m, method,
                                   len(rasters))
                seconds += est["seconds"]
                work_text.append(est["text"])
            if trace is not None:
                sp = _metres(pr["transect_spacing"], 0.0) or 0.0
                ln = _metres(pr["transect_length"], 0.0) or 0.0
                tw = _metres(pr["transect_width"], None) or width_m
                trace_len = _total_length_metres(trace) or 0.0
                if sp > 0 and ln > 0 and trace_len > 0:
                    est = estimate_run((int(trace_len / sp) + 1) * ln, step_m, n_lines,
                                       tw, cell_m, method, len(rasters))
                    seconds += est["seconds"]
                    work_text.append("transects: " + est["text"])
            if seconds > ESTIMATE_WARN_SECONDS:
                pr["swath_width"].setWarningMessage(
                    "Large run: " + " ".join(work_text) + " Consider a larger step, "
                    "a narrower band, fewer profiles, or the All cells method.")

    # -- execution ------------------------------------------------------------

    def execute(self, params, messages):
        msg = ArcpyMessenger()
        # Restored in the finally below.  Leaking it True for the rest of the
        # Pro session changed how the dialog behaved on later opens, which is
        # what made the sketch bug intermittent rather than reproducible.
        saved_overwrite = arcpy.env.overwriteOutput
        arcpy.env.overwriteOutput = True
        # Every geoprocessing call made inside the tool (CreateTable, AddFields,
        # CopyRows, MakeFeatureLayer, ...) would otherwise be appended to the
        # project's geoprocessing history and to dataset metadata.  On a project
        # with a long history that logging can cost seconds per call and turned
        # a 20 s run into 25 minutes.  The tool's own run is still recorded.
        saved = {}
        for getter, setter in (("GetLogHistory", "SetLogHistory"),
                               ("GetLogMetadata", "SetLogMetadata")):
            try:
                saved[setter] = getattr(arcpy, getter)()
                getattr(arcpy, setter)(False)
            except Exception:
                pass
        # "Add Outputs to Map" off: Pro's post-run add-and-redraw is skipped and
        # only the chart window appears.  Layer files are still written, so the
        # outputs can be added later with their styling.
        add_to_map = _by_name(params).get("add_to_map")
        saved_add = arcpy.env.addOutputsToMap
        if add_to_map is not None and add_to_map.value is False:
            arcpy.env.addOutputsToMap = False
            msg.info("Add Outputs to Map is off: outputs are written but not added "
                     "to the map.")
        try:
            self._run(params, msg)
        except SwathError as exc:
            arcpy.AddError(str(exc))
            raise arcpy.ExecuteError(str(exc))
        finally:
            try:
                arcpy.env.addOutputsToMap = saved_add
                arcpy.env.overwriteOutput = saved_overwrite
            except Exception:
                pass
            for setter, value in saved.items():
                try:
                    getattr(arcpy, setter)(value)
                except Exception:
                    pass

    def _run(self, params, msg):
        pr = _by_name(params)
        t_start = time.perf_counter()
        raster_vals = self._raster_values(pr)
        if not raster_vals:
            raise SwathError("Choose at least one raster.")

        rasters, raster_srs = [], []
        for value in raster_vals:
            ras = value if isinstance(value, arcpy.Raster) else arcpy.Raster(value)
            info = raster_io.raster_info(ras)
            features.require_spatial_reference(
                arcpy.Describe(value), os.path.basename(str(value)))
            rasters.append((ras, info))
            raster_srs.append(info.spatial_reference)

        lines_layer = pr["in_lines"].value
        sketch = pr["draw_lines"].value
        trace = pr["transect_source"].value

        # arcpy.mp.ArcGISProject("CURRENT") is never called during execution:
        # on a large project (many layers, network drive) it can block for
        # many minutes.  A line sketched on the map arrives in the map's own
        # coordinate system, so the map is only consulted in the rare case of a
        # feature set that reports no coordinate system at all.
        sketch_sr = None
        if sketch is not None:
            try:
                sr = getattr(arcpy.Describe(sketch), "spatialReference", None)
                if sr is not None and sr.name not in ("", "Unknown"):
                    sketch_sr = sr
            except Exception:
                sketch_sr = None
            if sketch_sr is None:
                msg.info("  Sketch has no coordinate system; asking the map.")
                sketch_sr = _active_map_sr()
        map_sr = sketch_sr

        # Baseline CRS and extent, for choosing the working CRS.
        base_sr, base_extent = None, None
        for src in (lines_layer, trace):
            if src is not None:
                d = arcpy.Describe(src)
                base_sr = features.require_spatial_reference(d, "Baseline layer")
                base_extent = getattr(d, "extent", None)
                break
        if base_sr is None and sketch is not None:
            base_sr = sketch_sr
            try:
                base_extent = getattr(arcpy.Describe(sketch), "extent", None)
            except Exception:
                base_extent = None

        working_sr = features.choose_working_crs(raster_srs, base_sr, base_extent, msg)
        msg.info(f"Working coordinate system: {working_sr.name}")

        width = _linear_units(pr["swath_width"], working_sr)
        step = _linear_units(pr["step"], working_sr,
                             fallback=1.5 * _cell_size_metres(raster_vals[0]))
        block_mb = int(pr["max_block_mb"].value or 512)
        has_trace = trace is not None

        cfg = SwathConfig(
            swath_width=float(width), step=float(step),
            n_lines=int(pr["n_lines"].value or 51),
            method="offset" if self._is_offset(pr) else "zonal",
            interpolation=(pr["interpolation"].valueAsText or "Bilinear").lower(),
            z_factor=float(pr["z_factor"].value or 1.0),
            multipart={"Each part separately": "each_part", "Largest part":
                       "largest_part", "Error": "error"}[pr["multipart"].valueAsText],
            tangent_smooth_window=(
                int(np.clip(round(0.5 * float(width) / float(step)), 5, 4001)) | 1
                if pr["tangent_smooth"].value else 0),
            fold_policy={"Mask folded samples": "mask", "Warn only": "warn",
                         "Ignore": "ignore"}[pr["fold_policy"].valueAsText],
            smooth_window=SMOOTH_PRESETS.get(pr["smoothing"].valueAsText, 0),
            pct_low=float(pr["pct_low"].value or 5.0),
            pct_high=float(pr["pct_high"].value or 95.0),
            chart_profile_colour=PROFILE_COLOUR_CHOICES.get(
                pr["chart_profile_colour"].valueAsText, "single"),
            flat_hi={"Null": "nan", "0.5": "half",
                     "Reference (0.0)": "reference"}[pr["flat_hi"].valueAsText],
            min_valid=int(pr["min_valid"].value or 1),
            legacy_stats=bool(pr["legacy_stats"].value),
            transect_spacing=float(_linear_units(pr["transect_spacing"], working_sr, 0.0)
                                   or 0.0) if has_trace else 0.0,
            transect_length=float(_linear_units(pr["transect_length"], working_sr, 0.0)
                                  or 0.0) if has_trace else 0.0,
            transect_side=SIDES.get(pr["transect_side"].valueAsText, "both"),
            transect_start=float(_linear_units(pr["transect_start"], working_sr, 0.0)
                                 or 0.0) if has_trace else 0.0,
            max_block_cells=max(1, block_mb * 1024 * 1024 // 4),
            geo_transform=pr["geo_transform"].valueAsText or "",
            label_field=pr["label_field"].valueAsText or "",
            thi_raster=pr["thi_raster"].valueAsText or "",
            tool_version=__version__,
        )
        if cfg.legacy_stats:
            msg.warn("Legacy statistics: reproduces published SwathProfiler numbers "
                     "but restores two known defects (docs/ALGORITHM.md, D3 and D4).")

        baselines, transect_records = [], []
        if lines_layer is not None:
            msg.info(features.selection_note(lines_layer))
            baselines += features.read_baselines(lines_layer, working_sr,
                                                 cfg.multipart, msg,
                                                 label_field=cfg.label_field)
        sketched = features.read_featureset(sketch, working_sr, map_sr,
                                            messenger=msg)
        if sketch is not None and not sketched:
            msg.info("  Sketch parameter contained no drawn features.")
        baselines += sketched

        if has_trace and cfg.uses_transects:
            msg.info(features.selection_note(trace).replace("baseline", "trace"))
            traces = features.read_baselines(trace, working_sr, "each_part", msg)
            for tr in traces:
                bl, recs = features.transect_baselines(
                    tr.vertices, cfg.transect_spacing, cfg.transect_length,
                    cfg.transect_side, cfg.transect_start,
                    smooth_window=cfg.tangent_smooth_window,
                    start_oid=2_000_000 + 10_000 * len(transect_records))
                # Continue numbering across multipart / multiple traces.
                offset = len(transect_records)
                for b, r in zip(bl, recs):
                    r["id"] += offset
                    r["name"] = f"T{r['id']:03d}"
                    b.name = r["name"]
                baselines += bl
                transect_records += recs
            msg.info(f"  {len(transect_records)} transect(s) generated along the trace.")

        if not baselines:
            raise SwathError("No baseline. Draw one on the map with the sketch tool "
                             "in 'Draw Baseline on Map', choose a line layer, or "
                             "give a trace to place transects along.")

        # Transects may use their own swath width: run them as a second batch
        # with a copy of the configuration and merge the results.
        transect_ids = {b.oid for b in baselines if b.oid >= 2_000_000}
        base_bl = [b for b in baselines if b.oid not in transect_ids]
        tr_bl = [b for b in baselines if b.oid in transect_ids]
        tw = _linear_units(pr["transect_width"], working_sr, None) if has_trace else None
        import dataclasses
        tr_cfg = cfg
        if tr_bl and tw and float(tw) > 0 and abs(float(tw) - cfg.swath_width) > 1e-9:
            tr_cfg = dataclasses.replace(cfg, swath_width=float(tw))
            msg.info(f"  Transects use a swath width of {float(tw):g} "
                     f"(baselines {cfg.swath_width:g}).")

        # Sampling mask: read once in the working CRS, like the baselines, and
        # clipped to the padded extent every swath can reach.
        mask_input = None
        mask_sketch, mask_layer = pr["mask_draw"].value, pr["mask_layer"].value
        if mask_sketch is not None or mask_layer is not None:
            mode = MASK_MODES.get(pr["mask_mode"].valueAsText, "keep")
            half = max(cfg.half_width, float(tr_cfg.half_width))
            pad = half + 5.0 * cfg.step
            allv = np.concatenate([b.vertices for b in baselines])
            box = (float(allv[:, 0].min()) - pad, float(allv[:, 1].min()) - pad,
                   float(allv[:, 0].max()) + pad, float(allv[:, 1].max()) + pad)
            mask_input = features.read_mask(
                mask_layer, mask_sketch, working_sr, mode,
                fallback_sr=map_sr if map_sr is not None else _active_map_sr,
                clip_box=box, densify_step=cfg.step, messenger=msg)
        if mask_input is not None and mask_input.is_empty:
            if mask_input.mode == "keep":
                raise SwathError(
                    "The sampling mask does not overlap any swath, so with 'Keep only "
                    "inside' nothing would be sampled. Move or redraw the mask, or "
                    "switch the Mask Mode to 'Exclude inside'.")
            msg.warn("The sampling mask does not overlap any swath, so nothing is "
                     "excluded.")
            mask_input = None
        if mask_input is not None:
            cfg = dataclasses.replace(cfg, mask_mode=mask_input.mode,
                                      mask_note=mask_input.description)
            tr_cfg = dataclasses.replace(tr_cfg, mask_mode=mask_input.mode,
                                         mask_note=mask_input.description)
            msg.info(f"Sampling mask: {mask_input.description}.")
            if cfg.min_valid <= 1:
                msg.info("  Stations at the edge of the mask may keep only a few "
                         "samples. Raise 'Minimum Valid Samples per Station' to leave "
                         "such slivers out of the statistics.")
        sample_mask = mask_input.sample_mask if mask_input is not None else None

        result = None
        if base_bl:
            result = run_swath(base_bl, rasters, cfg, working_sr, msg, mask=sample_mask)
        if tr_bl:
            tr_result = run_swath(tr_bl, rasters, tr_cfg, working_sr, msg,
                                  mask=sample_mask)
            result = tr_result if result is None else result.merge(tr_result)
        arcpy.ResetProgressor()
        result.elapsed = time.perf_counter() - t_start
        if mask_input is not None:
            self._mask_coverage(result, mask_input, working_sr, msg)

        # Point the geoprocessing environment at the (small) output geodatabase
        # while writing, not at the project's own: the few remaining
        # geoprocessing calls resolve names against env.workspace and a large
        # geodatabase makes that slow.
        saved_env = {k: getattr(arcpy.env, k) for k in ("workspace", "scratchWorkspace")}
        try:
            out_gdb = None
            low = (pr["out_table"].valueAsText or "").lower()
            idx = low.find(".gdb")
            if idx > 0:
                out_gdb = pr["out_table"].valueAsText[:idx + 4]
            if out_gdb and arcpy.Exists(out_gdb):
                arcpy.env.workspace = out_gdb
                arcpy.env.scratchWorkspace = out_gdb
            self._write_outputs(pr, result, working_sr, cfg, transect_records, msg,
                                t_start, mask_input=mask_input)
        finally:
            for k, v in saved_env.items():
                try:
                    setattr(arcpy.env, k, v)
                except Exception:
                    pass

    # -- outputs --------------------------------------------------------------

    @staticmethod
    def _unique(path):
        """Never overwrite a previous run's default-named output."""
        if not path or not arcpy.Exists(path):
            return path
        ws, name = os.path.split(path)
        return arcpy.CreateUniqueName(name, ws)

    @staticmethod
    def _has_data(result, oid, part_id) -> bool:
        """Whether any raster has a single valid sample for this part."""
        return any(int(np.sum(s.n_valid)) > 0
                   for s in result.for_part(oid, part_id).values())

    def _mask_coverage(self, result, mask_input, sr, msg):
        """Record how much of each band the mask keeps, and the clipped band.

        Sets ``mask_kept`` (percent of the band's area) and ``band_shape`` (the
        kept piece, or None) on every part's geometry record, reports the
        coverage - one line per profile, or a range when there are many - and
        stops the run if the mask has left nothing anywhere to report.
        """
        rows = []
        for oid, part_id in result.parts():
            g = result.geometry[(oid, part_id)]
            pct, shape = features.mask_coverage(g["ring"], sr, mask_input)
            g["mask_kept"], g["band_shape"] = pct, shape
            rows.append(((oid, part_id), pct))

        if not any(self._has_data(result, *key) for key, _ in rows):
            raise SwathError(
                "The sampling mask leaves no valid samples in any swath. With 'Keep "
                "only inside' the polygons must overlap the swath band; with "
                "'Exclude inside' they must not cover all of it.")

        def label(key):
            return result.names.get(key) or f"OID {key[0]} part {key[1]}"

        if len(rows) <= 5:
            for key, pct in rows:
                msg.info(f"  Sampling mask keeps {pct:.0f}% of the swath band of "
                         f"{label(key)}.")
        else:
            vals = np.array([pct for _, pct in rows])
            msg.info(f"  Sampling mask keeps {vals.min():.0f}-{vals.max():.0f}% of each "
                     f"swath band (median {np.median(vals):.0f}%) across {len(rows)} "
                     "profiles.")
        gone = [label(key) for key, _ in rows if not self._has_data(result, *key)]
        if gone:
            shown = ", ".join(gone[:10]) + (", ..." if len(gone) > 10 else "")
            msg.warn(f"  The sampling mask removes all of {len(gone)} profile(s): {shown}. "
                     "They stay in the tables with Null statistics but are left out "
                     "of the band, offset-line and 3D layers and the saved charts.")

    def _write_outputs(self, pr, result, sr, cfg, transect_records, msg, t_start,
                       mask_input=None):
        def tick(text):
            msg.info(f"{text}  [{time.perf_counter() - t_start:.1f} s]")

        tick("Writing outputs...")
        # Outputs are added to the map by Pro once, after the tool finishes.
        # Never add layers from inside execute: every add forces a map redraw,
        # and with a large LiDAR raster in the map (on a network drive, without
        # pyramids) each redraw can take minutes.  Styling, labels and the
        # native chart travel in layer files attached to the output parameters
        # instead, which Pro applies as it adds them.
        for name in ("out_table", "out_band", "out_stations", "out_transects",
                     "out_offset_lines", "out_profiles_3d", "out_summary"):
            try:
                _ensure_workspace(pr[name].valueAsText)
            except Exception as exc:
                msg.warn(f"  Could not create the workspace for {name}: {exc}")

        table_path = self._unique(pr["out_table"].valueAsText) or \
            os.path.join(arcpy.env.scratchGDB, f"swath_{_stamp()}")
        tick("  output workspace checked")
        results_dir = self._results_folder(table_path)
        lyrx_dir = os.path.join(results_dir, "layer_files")
        os.makedirs(lyrx_dir, exist_ok=True)
        tick("  results folder ready")

        project_gdb = _GDB_CACHE.get("gdb")
        if project_gdb and table_path.lower().startswith(project_gdb.lower()):
            msg.warn(
                "Writing into the project's default geodatabase. If it has unsaved "
                "edits, this will wait until you save them (Edit > Save). Leave the "
                "default output location - swath_results.gdb - to avoid that.")
        if _drive_is_slow(table_path):
            msg.warn(
                "Outputs are on a network or removable drive. File-geodatabase "
                "writes there can take minutes; the default location "
                f"({_local_results_home()}\\swath_results) is on the local disk.")
        elif _GDB_CACHE.get("relocated"):
            msg.info("  Outputs go to a local results geodatabase because the "
                     "project folder is on a network or removable drive.")

        outputs.write_station_table(table_path, result, msg)
        pr["out_table"].value = table_path
        tick("  statistics table written")

        if pr["out_summary"].valueAsText:
            path = self._unique(pr["out_summary"].valueAsText)
            outputs.write_summary_table(path, result, msg)
            pr["out_summary"].value = path

        band_layer = None
        if pr["out_band"].valueAsText:
            rings = []
            for oid, part_id in result.parts():
                g = result.geometry[(oid, part_id)]
                rec = {"ring": g["ring"], "oid": oid, "part_id": part_id,
                       "name": result.names.get((oid, part_id), ""),
                       "width": float(g.get("width", cfg.swath_width)),
                       "step": g["step_actual"],
                       "n_lines": int(g.get("n_lines",
                                            cfg.n_lines if cfg.is_offset else 0)),
                       "n_stations": len(g["distances"]),
                       "method": g.get("method", cfg.method)}
                if mask_input is not None:
                    # The band as sampled: clipped to the mask, or left out
                    # when the mask keeps none of it.
                    if g.get("band_shape") is None:
                        continue
                    rec["shape"] = g["band_shape"]
                rings.append(rec)
            if rings:
                path = self._unique(pr["out_band"].valueAsText)
                features.write_band_polygons(path, rings, sr, msg)
                pr["out_band"].value = path
                self._attach_layer_file(pr["out_band"], path, lyrx_dir, "band", msg)
                tick("  band polygons written")

        if transect_records and pr["out_transects"].valueAsText:
            path = self._unique(pr["out_transects"].valueAsText)
            features.write_transects(path, transect_records, sr, msg)
            pr["out_transects"].value = path
            self._attach_layer_file(pr["out_transects"], path, lyrx_dir, "transects",
                                    msg, label_field="LINE_NAME")
            tick("  transects written")

        if pr["out_stations"].valueAsText:
            path = self._unique(pr["out_stations"].valueAsText)
            features.write_station_points(path, result, sr, msg)
            pr["out_stations"].value = path
            self._attach_layer_file(pr["out_stations"], path, lyrx_dir, "stations",
                                    msg, chart=self._station_chart(result))
            tick("  station points written")

        if pr["out_offset_lines"].valueAsText and cfg.is_offset:
            lines = []
            for oid, part_id in result.parts():
                g = result.geometry[(oid, part_id)]
                any_stats = next(iter(result.for_part(oid, part_id).values()))
                masked = g.get("masked")
                for i, off in enumerate(g["offsets"]):
                    rec = {"xy": g["points"][i], "oid": oid, "part_id": part_id,
                           "name": result.names.get((oid, part_id), ""),
                           "index": i, "offset": float(off),
                           "n_valid": int(np.sum(np.isfinite(any_stats.values[i])))
                           if any_stats.values is not None else 0}
                    # With a mask, decide per line from the samples it kept:
                    # untouched lines are written as they are, lines the mask
                    # removed entirely are left out, and only lines it cut
                    # partway pay for a geometry clip.
                    if mask_input is not None and masked is not None:
                        row = masked[i]
                        if row.all():
                            continue
                        if row.any():
                            rec["shape"] = features.clip_line(g["points"][i], sr,
                                                              mask_input)
                            if rec["shape"] is None:
                                continue
                    lines.append(rec)
            if lines:
                path = self._unique(pr["out_offset_lines"].valueAsText)
                features.write_offset_lines(path, lines, sr, msg)
                pr["out_offset_lines"].value = path
                self._attach_layer_file(pr["out_offset_lines"], path, lyrx_dir,
                                        "offsets", msg)

        if pr["out_profiles_3d"].valueAsText and cfg.is_offset:
            profiles = []
            for oid, part_id in result.parts():
                if mask_input is not None and not self._has_data(result, oid, part_id):
                    continue
                g = result.geometry[(oid, part_id)]
                for rname, s in result.for_part(oid, part_id).items():
                    for series, arr in (("MIN", s.z_min), ("MEAN", s.z_mean),
                                        ("MAX", s.z_max)):
                        profiles.append({"xy": g["centreline"], "z": arr,
                                         "oid": oid, "part_id": part_id,
                                         "raster": rname, "series": series,
                                         "offset": 0.0})
            path = self._unique(pr["out_profiles_3d"].valueAsText)
            features.write_profiles_3d(path, profiles, sr, msg)
            pr["out_profiles_3d"].value = path

        if pr["out_csv_folder"].valueAsText:
            outputs.write_csv(pr["out_csv_folder"].valueAsText, result, msg)

        fmts = set(pr["chart_formats"].values or [])
        image_fmts = fmts & {"PNG", "PDF", "SVG"}
        if image_fmts:
            # Beside the saved results, not in Pro's scratch folder, which is
            # hidden away and where nobody would look for them.
            folder = pr["out_chart_folder"].valueAsText or results_dir
            self._write_charts(folder, result, cfg, image_fmts,
                               pr["chart_preset"].valueAsText or "Screen", msg,
                               cfg.chart_profile_colour)

        bundle = pr["bundle_path"].valueAsText or \
            os.path.join(arcpy.env.scratchFolder, "swath_live" + BUNDLE_SUFFIX)
        self._publish_bundle(bundle, result, os.path.basename(table_path),
                             results_dir, msg)
        page = self._write_page(result, os.path.basename(table_path), results_dir, msg)
        tick(f"Done: computation {result.elapsed:.1f} s, total")
        self._show_chart(pr["open_viewer"].valueAsText, bundle, page, msg)

    # -- layer files for the outputs ------------------------------------------

    @staticmethod
    def _attach_layer_file(param, fc_path, folder, kind, msg, chart=None,
                           label_field=None):
        """Write a .lyrx for an output from a JSON template and attach it to
        the parameter's symbology.

        Pure JSON: no ``MakeFeatureLayer``, no ``saveACopy``, no map access -
        nothing here is a geoprocessing call and nothing triggers a redraw.  Pro
        applies the layer file when it adds the output after the tool finishes.
        A native chart is written into the file as CIM JSON (``cimchart``);
        the ``arcpy.charts`` API is never used, because it can crash the
        process.  Best-effort: any failure leaves the output unstyled and says so.
        """
        from swathprofile import lyrx

        try:
            out = os.path.join(folder, f"{os.path.basename(fc_path)}_{kind}.lyrx")
            lyrx.build_layer_file(kind, fc_path, out, chart=chart,
                                  label_field=label_field)
            param.symbology = out
            if chart is not None:
                msg.info("  Native chart stored with the station points layer: in "
                         "the Contents pane switch to 'List By Charts' (or expand "
                         "the layer) and open 'Swath profile'. Selecting on the "
                         "chart highlights stations on the map, and vice versa.")
        except Exception as exc:
            msg.info(f"  ({kind}: layer file not created - "
                     f"{type(exc).__name__}: {exc})")

    @staticmethod
    def _station_chart(result):
        """CIM definition of the min / mean / max chart for the station points."""
        from swathprofile import cimchart

        primary = result.thi_raster or result.raster_names[0]
        pfx = outputs.safe_prefix(primary)
        return cimchart.line_chart(
            "DIST",
            [(f"{pfx}_MIN", "Min", (77, 212, 138)),
             (f"{pfx}_MEAN", "Mean", (1, 0, 138)),
             (f"{pfx}_MAX", "Max", (248, 186, 129))],
            title=f"Swath profile - {primary}",
            x_title="Distance along baseline", y_title=primary)

    # -- charts, bundle, viewer -----------------------------------------------

    @staticmethod
    def _chart_footer(cfg, g=None):
        """The saved charts' provenance line; shared with the browser chart."""
        return plotting.chart_footer(cfg, g)

    def _write_charts(self, folder, result, cfg, formats, preset, msg,
                      profile_colour="single"):
        plotting.use_agg()
        os.makedirs(folder, exist_ok=True)
        footer = self._chart_footer(cfg)
        # A part with no valid sample in any raster (the mask removed it, or it
        # lies off the rasters) would only give an empty chart.
        parts = [k for k in result.parts() if self._has_data(result, *k)]
        for oid, part_id in parts:
            name = result.names.get((oid, part_id), "") or f"OID {oid} part {part_id}"
            fig, _ = plotting.build_figure(
                result.for_part(oid, part_id), title=f"Swath profile - {name}",
                thi_raster=result.thi_raster,
                footer=self._chart_footer(cfg, result.geometry.get((oid, part_id))),
                preset=preset, pct=(cfg.pct_low, cfg.pct_high),
                profile_colour=profile_colour)
            stem = outputs.safe_prefix(name, 32).lower()
            for ext in sorted(formats):
                path = os.path.join(folder, f"swath_{stem}_p{part_id}.{ext.lower()}")
                plotting.save_figure(fig, path, dpi=300 if preset != "Screen" else 200)
                msg.info(f"  Chart -> {path}")

        # Several profiles (transects along a trace): also a stacked figure.
        if len(parts) > 1:
            entries = []
            for oid, part_id in parts:
                label = result.names.get((oid, part_id), "") or f"OID {oid}.{part_id}"
                entries.append((label, result.for_part(oid, part_id), None))
            fig, _ = plotting.build_ridgeline_figure(
                entries, raster=result.thi_raster, title="Stacked swath profiles",
                footer=footer, preset=preset)
            for ext in sorted(formats):
                path = os.path.join(folder, f"swath_stacked.{ext.lower()}")
                plotting.save_figure(fig, path, dpi=300 if preset != "Screen" else 200)
                msg.info(f"  Stacked chart -> {path}")

    @staticmethod
    def _results_folder(table_path):
        """Where permanent results go: <project home>/swath_results.

        The home folder is taken from the output table's geodatabase (the
        project geodatabase sits in the project folder), not from
        ``arcpy.mp.ArcGISProject("CURRENT")``, which can block for minutes on a
        large project.  Falls back to the scratch folder.
        """
        home = None
        p = (table_path or "").replace("/", os.sep)
        low = p.lower()
        idx = low.find(".gdb")
        if idx > 0:
            home = os.path.dirname(p[:idx + 4])
        elif p:
            home = os.path.dirname(p)
        if not home or not os.path.isdir(home):
            home = arcpy.env.scratchFolder
        # The default output geodatabase already lives in swath_results/.
        if os.path.basename(home).lower() == "swath_results":
            folder = home
        else:
            folder = os.path.join(home, "swath_results")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _publish_bundle(self, bundle, result, title, results_dir, msg):
        """Write the live bundle atomically, and keep a permanent copy per run."""
        import shutil

        os.makedirs(os.path.dirname(bundle), exist_ok=True)
        tmp = bundle + ".tmp.npz"
        save_bundle(tmp, result, title=title, elapsed=result.elapsed)
        os.replace(tmp, bundle)
        try:
            keep = os.path.join(results_dir, f"{title}{BUNDLE_SUFFIX}")
            shutil.copyfile(bundle, keep)
            msg.info(f"  Result saved for later -> {keep}")
        except Exception as exc:
            msg.warn(f"  Could not keep a permanent copy of the result "
                     f"({type(exc).__name__}: {exc}); the live bundle is {bundle}.")

    @staticmethod
    def _write_page(result, title, results_dir, msg):
        """Write this run's browser chart beside its permanent result copy.

        Written on every run, whatever Show Interactive Chart says: it needs no
        Python to open, so it is also the copy to send to someone without
        ArcGIS.  A failure here never fails the run.
        """
        from swathprofile import webchart

        path = os.path.join(results_dir, f"{title}.html")
        try:
            report = webchart.write_web_chart(path, result, title=title, messenger=msg)
        except Exception as exc:
            msg.warn(f"  Could not write the browser chart ({type(exc).__name__}: {exc}).")
            return None
        mb = report["bytes"] / 1e6
        if not report["written"]:
            msg.warn(f"  The browser chart would be {mb:.0f} MB, more than browsers open "
                     f"comfortably ({webchart.PAGE_LIMIT / 1e6:.0f} MB), so it was not "
                     "written. Run fewer profiles at a time, or use a larger step.")
            return None
        msg.info(f"  Browser chart -> {path} ({mb:.1f} MB)")
        if report["bytes"] > webchart.EMAIL_HINT:
            msg.info("  At that size the browser chart is too big for most email; zip "
                     "it, or share it through a folder.")
        return path

    @staticmethod
    def _show_chart(mode, bundle, page, msg):
        """Show the run in the chart window or the browser chart, per *mode*.

        All of the decision - including finding out whether the window really
        started, and remembering when it cannot on this computer - is in
        swathprofile.chartwindow, which needs no arcpy and is tested in CI.
        """
        return chartwindow.show_chart(mode, bundle=bundle, page=page,
                                      viewer=VIEWER_SCRIPT, root=ROOT,
                                      find_python=_python_for_viewer, msg=msg)
