# -*- coding: utf-8 -*-
"""Synthetic rasters with closed-form values, written to a scratch geodatabase.

Analytic surfaces are the whole point: for a plane, a cone and a sine ridge the
correct min / max / mean / HI / THi are known exactly, so a failing assertion
localises a bug instead of merely signalling one.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import arcpy

#: NZTM2000 - a southern-hemisphere projected CRS with ~5e6 m northings, which
#: is exactly the case that exposes float32 coordinate quantisation.
NZTM = 2193
WGS84 = 4326
UTM60S = 32760


def make_scratch(prefix: str = "swathtest_"):
    """Create a throwaway folder + file geodatabase.  Returns (folder, gdb)."""
    folder = tempfile.mkdtemp(prefix=prefix)
    arcpy.management.CreateFileGDB(folder, "test.gdb")
    return folder, os.path.join(folder, "test.gdb")


def plane_array(a, b, c, x0, y0, cw, ch, n_cols, n_rows):
    """z = a*x + b*y + c evaluated at cell centres, row 0 = TOP."""
    y_top = y0 + n_rows * ch
    xc = x0 + (np.arange(n_cols) + 0.5) * cw
    yc = y_top - (np.arange(n_rows) + 0.5) * ch
    return a * xc[None, :] + b * yc[:, None] + c


def cone_array(h, k, cx, cy, x0, y0, cw, ch, n_cols, n_rows):
    """z = h - k * sqrt((x-cx)^2 + (y-cy)^2)."""
    y_top = y0 + n_rows * ch
    xc = x0 + (np.arange(n_cols) + 0.5) * cw
    yc = y_top - (np.arange(n_rows) + 0.5) * ch
    dx = xc[None, :] - cx
    dy = yc[:, None] - cy
    return h - k * np.hypot(dx, dy)


def sine_ridge_array(amp, wavelength, x0, y0, cw, ch, n_cols, n_rows):
    """z = amp * sin(2*pi*y / wavelength) - constant in x."""
    y_top = y0 + n_rows * ch
    yc = y_top - (np.arange(n_rows) + 0.5) * ch
    col = amp * np.sin(2.0 * np.pi * yc / wavelength)
    return np.repeat(col[:, None], n_cols, axis=1)


def write_raster(arr, gdb, name, x0, y0, cw, ch, nodata=None, epsg=NZTM):
    """Save a numpy array as a raster with a defined projection."""
    path = os.path.join(gdb, name)
    if nodata is None:
        ras = arcpy.NumPyArrayToRaster(arr, arcpy.Point(x0, y0), cw, ch)
    else:
        ras = arcpy.NumPyArrayToRaster(arr, arcpy.Point(x0, y0), cw, ch, nodata)
    ras.save(path)
    arcpy.management.DefineProjection(path, arcpy.SpatialReference(epsg))
    return path


def write_line(vertices, gdb, name, epsg=NZTM):
    """Save an (n, 2) vertex array as a single-feature polyline feature class."""
    sr = arcpy.SpatialReference(epsg)
    fc = arcpy.management.CreateFeatureclass(
        gdb, name, "POLYLINE", spatial_reference=sr)[0]
    arr = arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in vertices])
    with arcpy.da.InsertCursor(fc, ["SHAPE@"]) as cur:
        cur.insertRow([arcpy.Polyline(arr, sr)])
    return fc


def write_lines(list_of_vertices, gdb, name, epsg=NZTM):
    """Save several polylines into one feature class."""
    sr = arcpy.SpatialReference(epsg)
    fc = arcpy.management.CreateFeatureclass(
        gdb, name, "POLYLINE", spatial_reference=sr)[0]
    with arcpy.da.InsertCursor(fc, ["SHAPE@"]) as cur:
        for verts in list_of_vertices:
            arr = arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in verts])
            cur.insertRow([arcpy.Polyline(arr, sr)])
    return fc


def rect_wkt(x0, y0, x1, y1):
    """WKT of an axis-aligned rectangle."""
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


def write_polygons(wkts, gdb, name, epsg=NZTM):
    """Save polygons given as WKT into one feature class.

    WKT is used because it states holes and multiple parts unambiguously:
    ``POLYGON((outer),(hole))`` and ``MULTIPOLYGON(((a)),((b)))``.
    """
    sr = arcpy.SpatialReference(epsg)
    fc = arcpy.management.CreateFeatureclass(
        gdb, name, "POLYGON", spatial_reference=sr)[0]
    with arcpy.da.InsertCursor(fc, ["SHAPE@"]) as cur:
        for wkt in wkts:
            cur.insertRow([arcpy.FromWKT(wkt, sr)])
    return fc
