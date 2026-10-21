# -*- coding: utf-8 -*-
"""Reading baselines and writing the geometry outputs.

All arcpy geometry work lives here.  Everything is computed in the *primary
raster's* coordinate system; baselines are reprojected on the fly by the cursor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .errors import CrsError, GeometryError

__all__ = ["Baseline", "require_spatial_reference", "same_crs", "choose_working_crs",
           "transform_points", "read_baselines", "read_featureset",
           "transect_baselines", "selection_note", "MaskInput", "read_mask",
           "mask_coverage", "clip_line", "write_band_polygons",
           "write_offset_lines", "write_profiles_3d", "write_transects",
           "write_station_points"]


def require_spatial_reference(describe_obj, label: str):
    """Return the spatial reference of *describe_obj*, or raise with guidance.

    Only an *undefined* CRS is fatal.  Geographic inputs are fine: everything
    is computed in a projected working CRS (see :func:`choose_working_crs`) and
    both baselines and rasters are transformed into it on the fly.
    """
    sr = getattr(describe_obj, "spatialReference", None)
    if sr is None or not getattr(sr, "name", "") or sr.name in ("Unknown", ""):
        raise CrsError(
            f"'{label}' has no spatial reference, so the tool cannot place it on "
            "the ground. Define one with the Define Projection tool and re-run."
        )
    return sr


def same_crs(a, b) -> bool:
    """True when two spatial references describe the same system."""
    if a is None or b is None:
        return False
    try:
        if a.factoryCode and b.factoryCode:
            return int(a.factoryCode) == int(b.factoryCode)
    except Exception:
        pass
    return a.name == b.name


def _utm_for(lon: float, lat: float):
    """UTM zone spatial reference containing a lon/lat."""
    import arcpy

    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    code = (32700 if lat < 0 else 32600) + zone
    return arcpy.SpatialReference(code)


def choose_working_crs(raster_srs, baseline_sr, baseline_extent=None,
                       messenger=None):
    """Pick the projected CRS everything is computed in.

    Preference order: the first *projected* raster (so the common case needs no
    reprojection at all); otherwise the baseline CRS if projected; otherwise a
    UTM zone chosen from the baseline's centre.  Swath width, step and distance
    are ground lengths, so a geographic working CRS is never acceptable.
    """
    import arcpy

    for sr in raster_srs:
        if sr is not None and sr.type == "Projected":
            return sr
    if baseline_sr is not None and baseline_sr.type == "Projected":
        if messenger:
            messenger.info("  No raster is projected; computing in the baseline's "
                           f"system ({baseline_sr.name}).")
        return baseline_sr

    # Everything is geographic: auto-select a UTM zone from the baseline centre.
    lon, lat = 0.0, 0.0
    if baseline_extent is not None and baseline_sr is not None:
        ext = baseline_extent
        centre = arcpy.PointGeometry(
            arcpy.Point((ext.XMin + ext.XMax) / 2.0, (ext.YMin + ext.YMax) / 2.0),
            baseline_sr).projectAs(arcpy.SpatialReference(4326)).firstPoint
        lon, lat = float(centre.X), float(centre.Y)
    sr = _utm_for(lon, lat)
    if messenger:
        messenger.warn(
            "All inputs are in geographic coordinate systems. Computing in "
            f"{sr.name} (chosen from the baseline centre at {lon:.2f}, {lat:.2f}); "
            "swath width and step are metres in that system. Outputs are written "
            "in it too. Supply a projected raster to control this explicitly."
        )
    return sr


def transform_points(xy, from_sr, to_sr, transformation=None):
    """Project an (..., 2) coordinate array between spatial references.

    Uses a single WKT MULTIPOINT round trip, which is ~20x faster than building
    Point objects.  arcpy multipoints silently merge coincident points, which
    would break the one-to-one mapping, so only the *unique* coordinates are
    sent and the result is scattered back; the count is asserted.
    """
    import arcpy

    a = np.asarray(xy, dtype=np.float64)
    shape = a.shape
    flat = a.reshape(-1, 2)
    finite = np.isfinite(flat).all(axis=1)
    out = np.full_like(flat, np.nan)
    if not finite.any():
        return out.reshape(shape)

    uniq, inverse = np.unique(flat[finite], axis=0, return_inverse=True)
    wkt = "MULTIPOINT (" + ",".join(f"{x:.10f} {y:.10f}" for x, y in uniq) + ")"
    geom = arcpy.FromWKT(wkt, from_sr)
    geom = geom.projectAs(to_sr, transformation) if transformation \
        else geom.projectAs(to_sr)
    part = geom.getPart()
    proj = np.array([[p.X, p.Y] for p in part], dtype=np.float64)
    if proj.shape[0] != uniq.shape[0]:
        raise CrsError(
            f"Coordinate transformation from '{from_sr.name}' to '{to_sr.name}' "
            f"returned {proj.shape[0]} points for {uniq.shape[0]} sent; points "
            "were merged or dropped. Check the datum transformation."
        )
    out[finite] = proj[inverse.reshape(-1)]
    return out.reshape(shape)


@dataclass
class Baseline:
    """One polyline part, already in the target CRS."""

    oid: int
    part_id: int
    vertices: np.ndarray          # (V, 2) float64
    length: float
    name: str = ""                # label carried into every output

    @property
    def key(self) -> str:
        return f"{self.oid}.{self.part_id}"


def read_featureset(feature_set, spatial_reference, fallback_sr=None,
                    start_oid: int = 1_000_000, messenger=None):
    """Read baselines sketched in the tool dialog (a GPFeatureRecordSetLayer).

    A sketch made on the map arrives in the map's coordinate system; if the
    feature set reports no spatial reference, *fallback_sr* (the active map's)
    is assumed.  Sketched lines get OIDs from *start_oid* upward so they never
    collide with features from an input layer used in the same run.
    """
    import arcpy

    if feature_set is None:
        return []
    try:
        with arcpy.da.SearchCursor(feature_set, ["OID@"]) as cur:
            n = sum(1 for _ in cur)
    except Exception:
        n = 0
    if n == 0:
        return []

    d = arcpy.Describe(feature_set)
    src_sr = getattr(d, "spatialReference", None)
    if src_sr is None or not src_sr.name or src_sr.name == "Unknown":
        src_sr = fallback_sr
        if src_sr is None:
            raise CrsError(
                "The sketched baseline has no coordinate system and the active "
                "map's could not be determined. Draw the line on a map with a "
                "defined coordinate system.")
        if messenger:
            messenger.info(f"  Sketched baseline assumed to be in {src_sr.name}.")

    out = []
    with arcpy.da.SearchCursor(feature_set, ["SHAPE@"]) as cur:
        for i, (shape,) in enumerate(cur):
            if shape is None:
                continue
            if not same_crs(src_sr, spatial_reference):
                shape = arcpy.Polyline(shape.getPart(), src_sr).projectAs(
                    spatial_reference) if shape.spatialReference is None or \
                    shape.spatialReference.name in ("", "Unknown") \
                    else shape.projectAs(spatial_reference)
            for part_id in range(shape.partCount):
                verts = _part_to_array(shape.getPart(part_id))
                if len(verts) < 2:
                    continue
                length = float(np.sum(np.hypot(np.diff(verts[:, 0]),
                                               np.diff(verts[:, 1]))))
                if length <= 0:
                    continue
                out.append(Baseline(oid=start_oid + i, part_id=part_id,
                                    vertices=verts, length=length,
                                    name=f"sketch {i + 1}"))
    if messenger and out:
        messenger.info(f"  {len(out)} baseline(s) taken from the map sketch.")
    return out


def transect_baselines(trace_vertices, spacing, length, side="both", start=0.0,
                       smooth_window=0, start_oid: int = 2_000_000):
    """Generate perpendicular transects along a trace as :class:`Baseline` objects.

    Also returns the transect records (id, name, ends, dist_along, azimuth)
    for :func:`write_transects`.  Transects are numbered from 1 and labelled
    ``T001, T002, ...``; their OIDs start at *start_oid* so they never collide
    with drawn or layer baselines in the same run.
    """
    from .geometry import transects_along

    tr = transects_along(trace_vertices, spacing, length, side, start,
                         smooth_window)
    baselines, records = [], []
    for k in range(len(tr["centres"])):
        ends = tr["ends"][k]
        verts = np.asarray(ends, dtype=np.float64)
        seg_len = float(np.hypot(*(verts[1] - verts[0])))
        if seg_len <= 0:
            continue
        name = f"T{k + 1:03d}"
        baselines.append(Baseline(oid=start_oid + k, part_id=0, vertices=verts,
                                  length=seg_len, name=name))
        records.append({"id": k + 1, "name": name, "ends": verts,
                        "dist_along": float(tr["dist_along"][k]),
                        "azimuth": float(tr["azimuth"][k]),
                        "length": seg_len})
    return baselines, records


def selection_note(layer, noun: str = "baseline feature(s)") -> str:
    """Describe how many features will be processed, and whether a selection is on.

    A ``GPFeatureLayer`` parameter carries the layer's selection set and
    definition query, and ``GetCount`` on the layer is selection-aware.  This is
    reported explicitly because the failure mode - silently processing all 40,000
    features when the user selected three - is invisible until the output is
    inspected.
    """
    import arcpy

    # Read the selection set BEFORE touching the layer with a cursor: iterating
    # a cursor over a layer object clears its selection (observed in arcpy
    # 3.4), so a count-then-read sequence would silently process everything.
    # GetCount is avoided too - it is a geoprocessing tool, and inside a large
    # Pro project every geoprocessing call costs seconds.
    try:
        fidset = getattr(arcpy.Describe(layer), "FIDSet", "") or ""
    except Exception:
        fidset = ""
    n_sel = len([f for f in fidset.split(";") if f.strip()]) if fidset else 0
    if n_sel:
        return f"Processing {n_sel} {noun} (selection active: {n_sel} selected)."
    with arcpy.da.SearchCursor(layer, ["OID@"]) as cur:
        total = sum(1 for _ in cur)
    return f"Processing {total} {noun} (no selection - the whole layer)."


def _part_to_array(part) -> np.ndarray:
    pts = [(p.X, p.Y) for p in part if p is not None]
    return np.asarray(pts, dtype=np.float64)


def read_baselines(layer, spatial_reference, multipart: str = "each_part",
                   messenger=None, label_field: str = ""):
    """Read polylines as :class:`Baseline` objects in *spatial_reference*.

    The layer object is passed straight to the cursor so its selection and
    definition query are honoured; resolving it to a catalog path first would
    silently discard both.

    Multipart features are kept as separate parts rather than concatenated -
    joining them would invent a connecting segment across the gap and report
    along-track distances that do not exist on the ground.  The reference
    implementation silently takes part 0 and discards the rest (defect D16).

    *label_field* names a text field whose value becomes the baseline's label
    in every output and in the viewer.
    """
    import arcpy

    out = []
    n_multipart = 0
    fields = ["OID@", "SHAPE@"] + ([label_field] if label_field else [])
    with arcpy.da.SearchCursor(layer, fields,
                               spatial_reference=spatial_reference) as cur:
        for row in cur:
            oid, shape = row[0], row[1]
            name = "" if len(row) < 3 or row[2] is None else str(row[2])
            if shape is None:
                if messenger:
                    messenger.warn(f"Baseline OID {oid} has null geometry; skipped.")
                continue

            parts = [_part_to_array(shape.getPart(i)) for i in range(shape.partCount)]
            parts = [p for p in parts if len(p) >= 2]
            if not parts:
                if messenger:
                    messenger.warn(f"Baseline OID {oid} has no usable vertices; skipped.")
                continue

            if len(parts) > 1:
                n_multipart += 1
                if multipart == "error":
                    raise GeometryError(
                        f"Baseline OID {oid} is multipart ({len(parts)} parts). Set "
                        "Multipart Handling to 'Each part separately' or "
                        "'Largest part', or repair the geometry."
                    )
                if multipart == "largest_part":
                    lengths = [float(np.sum(np.hypot(np.diff(p[:, 0]),
                                                     np.diff(p[:, 1])))) for p in parts]
                    parts = [parts[int(np.argmax(lengths))]]

            for part_id, verts in enumerate(parts):
                length = float(np.sum(np.hypot(np.diff(verts[:, 0]),
                                               np.diff(verts[:, 1]))))
                if length <= 0:
                    if messenger:
                        messenger.warn(
                            f"Baseline OID {oid} part {part_id} has zero length; skipped.")
                    continue
                out.append(Baseline(oid=int(oid), part_id=part_id,
                                    vertices=verts, length=length, name=name))

    if n_multipart and messenger:
        messenger.warn(
            f"{n_multipart} baseline feature(s) are multipart. Each part is profiled "
            "separately and identified by PART_ID; parts are never joined, because "
            "that would invent a connecting segment."
        )
    if not out:
        raise GeometryError("No usable baseline geometry was found.")
    return out


# ---------------------------------------------------------------------------
# Sampling mask
# ---------------------------------------------------------------------------

@dataclass
class MaskInput:
    """The user's sampling mask, read and ready for the engine and the writers.

    ``geometry`` is one arcpy Polygon in the working CRS: the union of every
    usable polygon, clipped to the extent the swaths can reach.  It is what the
    band and the offset lines are clipped with.  ``sample_mask`` holds the rings
    of that same polygon for the numpy engine, so the map and the statistics
    agree about what was kept.
    """

    geometry: object
    sample_mask: object                  # swathprofile.mask.SampleMask
    mode: str
    n_layer: int = 0
    n_sketch: int = 0
    description: str = ""

    @property
    def is_empty(self) -> bool:
        """Nothing of the mask lies where any swath can reach."""
        return self.sample_mask.is_empty


def _polygon_rings(poly):
    """Every ring of an arcpy Polygon as (K, 2) arrays, exteriors and holes alike.

    ``getPart(i)`` gives an exterior ring followed by its holes, each hole
    preceded by a ``None`` point.  :func:`_part_to_array` drops those separators,
    which would join every hole onto its exterior with a phantom edge; this
    splits there instead.  The even-odd rule in :mod:`.mask` then needs no
    telling which ring is a hole.
    """
    rings = []
    for i in range(poly.partCount):
        cur = []
        for pt in poly.getPart(i):
            if pt is None:
                if len(cur) >= 3:
                    rings.append(np.asarray(cur, dtype=np.float64))
                cur = []
            else:
                cur.append((pt.X, pt.Y))
        if len(cur) >= 3:
            rings.append(np.asarray(cur, dtype=np.float64))
    return rings


def _union_all(geoms):
    """Union by pairwise reduction: log-depth, rather than a growing running union."""
    geoms = list(geoms)
    while len(geoms) > 1:
        merged = [geoms[i].union(geoms[i + 1]) for i in range(0, len(geoms) - 1, 2)]
        if len(geoms) % 2:
            merged.append(geoms[-1])
        geoms = merged
    return geoms[0] if geoms else None


def _usable_polygon(shape, densify_step=None):
    """The polygon, ready to union, or None when it has nothing in it.

    True curves (arcs drawn with the sketch's curve tools) are densified first:
    ``getPart`` reports only an arc's end points, so the rings would cut the
    curve's corner.
    """
    if shape is None:
        return None
    try:
        if shape.partCount == 0:
            return None
    except Exception:
        return None
    if densify_step and getattr(shape, "hasCurves", False):
        step = float(densify_step)
        shape = shape.densify("DISTANCE", step, step / 20.0)
    return shape


def _read_mask_sketch(feature_set, spatial_reference, fallback_sr, densify_step,
                      messenger):
    """Polygons sketched in the dialog, in *spatial_reference* (mirrors read_featureset)."""
    import arcpy

    try:
        with arcpy.da.SearchCursor(feature_set, ["OID@"]) as cur:
            n = sum(1 for _ in cur)
    except Exception:
        n = 0
    if n == 0:
        return []

    d = arcpy.Describe(feature_set)
    src_sr = getattr(d, "spatialReference", None)
    if src_sr is None or not src_sr.name or src_sr.name == "Unknown":
        src_sr = fallback_sr() if callable(fallback_sr) else fallback_sr
        if src_sr is None:
            raise CrsError(
                "The sketched sampling mask has no coordinate system and the active "
                "map's could not be determined. Draw it on a map with a defined "
                "coordinate system.")
        if messenger:
            messenger.info(f"  Sketched mask assumed to be in {src_sr.name}.")

    out = []
    with arcpy.da.SearchCursor(feature_set, ["SHAPE@"]) as cur:
        for (shape,) in cur:
            if shape is None:
                continue
            if not same_crs(src_sr, spatial_reference):
                if shape.spatialReference is None or \
                        shape.spatialReference.name in ("", "Unknown"):
                    shape = arcpy.Polygon(shape.getPart(), src_sr)
                shape = shape.projectAs(spatial_reference)
            g = _usable_polygon(shape, densify_step)
            if g is not None:
                out.append(g)
    return out


def read_mask(layer, sketch, spatial_reference, mode: str = "keep", *,
              fallback_sr=None, clip_box=None, densify_step=None, messenger=None):
    """Read the sampling mask from a polygon layer and/or a dialog sketch.

    Returns a :class:`MaskInput` in *spatial_reference*, or None when only an
    empty sketch was given (the run then proceeds unmasked, with a warning).
    A layer that yields no polygon at all is an error: running unmasked when
    the user asked for a mask would be a wrong answer, not a degraded one.

    Polygons are read exactly the way baselines are - the layer's selection is
    noted before any cursor touches it, and the cursor projects on the fly - so
    mask and baselines are always mutually consistent.  Everything is then
    unioned (overlapping polygons must merge, not cancel under even-odd) and
    clipped to *clip_box* ``(x0, y0, x1, y1)``, the padded extent of every swath.
    That one topological operation also simplifies a self-intersecting sketch,
    so the rings the engine tests are those of the geometry the outputs are
    clipped with, and it bounds the work on a detailed mapped polygon.

    *fallback_sr* (an SR or a zero-argument callable) is used only when a
    sketch reports no coordinate system, so the map is consulted only then.
    """
    import arcpy

    from .mask import SampleMask

    geoms, notes = [], []
    n_layer = n_sketch = 0

    if layer is not None:
        d = arcpy.Describe(layer)
        require_spatial_reference(d, "Sampling mask layer")
        note = selection_note(layer, noun="mask polygon(s)")
        if messenger:
            messenger.info(f"  {note}")
        with arcpy.da.SearchCursor(layer, ["OID@", "SHAPE@"],
                                   spatial_reference=spatial_reference) as cur:
            for oid, shape in cur:
                g = _usable_polygon(shape, densify_step)
                if g is None:
                    if messenger:
                        messenger.warn(f"  Mask polygon OID {oid} is empty; skipped.")
                    continue
                geoms.append(g)
                n_layer += 1
        if n_layer == 0:
            raise GeometryError(
                "The sampling mask layer has no polygons to use. Check its selection "
                "and definition query, or clear the mask parameters.")
        notes.append(f"{n_layer} from {getattr(d, 'name', '') or 'the layer'}")

    if sketch is not None:
        drawn = _read_mask_sketch(sketch, spatial_reference, fallback_sr,
                                  densify_step, messenger)
        n_sketch = len(drawn)
        geoms.extend(drawn)
        if n_sketch:
            notes.append(f"{n_sketch} drawn")
        elif layer is None:
            if messenger:
                messenger.warn(
                    "The sampling mask sketch has no finished polygon (double-click to "
                    "finish drawing one), so the run is not masked.")
            return None

    if not geoms:
        return None

    union = _union_all(geoms)
    extent = arcpy.Extent(*clip_box) if clip_box is not None else union.extent
    clipped = union.clip(extent)
    rings = (_polygon_rings(clipped)
             if clipped is not None and clipped.partCount else [])
    verb = "keep only inside" if mode == "keep" else "exclude inside"
    description = f"{verb} {len(geoms)} polygon(s) ({', '.join(notes)})"
    return MaskInput(geometry=clipped, sample_mask=SampleMask(rings, mode), mode=mode,
                     n_layer=n_layer, n_sketch=n_sketch, description=description)


def _has_parts(geom) -> bool:
    """Whether an arcpy geometry has anything in it.

    Never test a geometry for truthiness: ``Geometry.__len__`` is its part
    count, so an empty clip result is falsy and so is ``None``, and ``or``
    between geometries silently picks the wrong one.
    """
    return geom is not None and geom.partCount > 0


def mask_coverage(ring, sr, mask_input):
    """``(percent kept, kept piece or None)`` for one swath band.

    Both halves come from the same two operators - ``intersect`` and
    ``difference`` - and the percentage is ``kept / (inside + outside)``.  The
    band's own area would be the wrong denominator where the band folds, because
    the ring is stored as drawn and self-intersects there.  A mask that misses
    or swallows the band is answered without either operation, and then the
    kept piece is the band exactly as drawn.
    """
    import arcpy

    band = arcpy.Polygon(arcpy.Array([arcpy.Point(float(x), float(y))
                                      for x, y in ring]), sr)
    keep = mask_input.mode == "keep"
    mg = mask_input.geometry
    if not _has_parts(mg) or mg.disjoint(band):
        return (0.0, None) if keep else (100.0, band)
    if mg.contains(band):
        return (100.0, band) if keep else (0.0, None)
    inside = band.intersect(mg, 4)
    outside = band.difference(mg)
    a_in = float(inside.area) if _has_parts(inside) else 0.0
    a_out = float(outside.area) if _has_parts(outside) else 0.0
    total = a_in + a_out
    if total <= 0.0:
        return 0.0, None
    kept, a_kept = (inside, a_in) if keep else (outside, a_out)
    return 100.0 * a_kept / total, (kept if a_kept > 0.0 else None)


def clip_line(xy, sr, mask_input):
    """The part of a polyline the mask keeps, or None if it keeps nothing."""
    import arcpy

    line = arcpy.Polyline(arcpy.Array([arcpy.Point(float(x), float(y))
                                       for x, y in xy]), sr)
    mg = mask_input.geometry
    if not _has_parts(mg):
        return None if mask_input.mode == "keep" else line
    piece = line.intersect(mg, 2) if mask_input.mode == "keep" else line.difference(mg)
    if not _has_parts(piece) or not piece.length > 0.0:
        return None
    return piece


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _field_spec(fields):
    return [[fname, ftype, fname, (rest[0] if rest else None)]
            for fname, ftype, *rest in fields]


def _create(out_path, geom_type, sr, fields, has_z=False, kind="fc"):
    """Create an empty feature class with *fields*, using as few geoprocessing
    calls as possible.

    Inside a large Pro project every geoprocessing-tool call was measured at
    20-35 seconds regardless of what it did, so the count matters more than
    anything else here.  A template feature class ``_swath_tmpl_<kind>`` is
    created once per geodatabase (two calls); every later output is one call,
    ``CreateFeatureclass`` with that template, which carries the fields.

    Returns ``(path, via_memory)``.  If the geodatabase is inside an edit
    session the direct creation fails (ERROR 000496); the feature class is then
    built in the ``memory`` workspace and copied in by :func:`_finalise_fc`.

    IMPORTANT: the template is keyed only by *kind*, and it is reused whenever
    it already exists.  A results geodatabase written by an earlier version
    therefore still holds that version's fields.  **When the field list for a
    kind changes, change its** ``kind`` **string too** (that is why the offset
    lines are ``offsets_side``), or every user with an existing results
    geodatabase gets the old schema and the insert cursor fails on the new
    field.  Hashing the fields into the name would do this automatically, but
    it would force a fresh template for every kind on the first run in every
    existing geodatabase, at 20-35 s per call in a slow project.
    """
    import uuid

    import arcpy

    ws, name = os.path.split(out_path)
    z = "ENABLED" if has_z else "DISABLED"
    try:
        template = os.path.join(ws, f"_swath_tmpl_{kind}") if ws.lower().endswith(".gdb") \
            else None
        if template and not arcpy.Exists(template):
            tmpl = arcpy.management.CreateFeatureclass(
                ws, os.path.basename(template), geom_type, spatial_reference=sr,
                has_z=z)[0]
            arcpy.management.AddFields(tmpl, _field_spec(fields))
        if template:
            fc = arcpy.management.CreateFeatureclass(
                ws, name, geom_type, template=template, spatial_reference=sr,
                has_z=z)[0]
        else:
            fc = arcpy.management.CreateFeatureclass(
                ws, name, geom_type, spatial_reference=sr, has_z=z)[0]
            arcpy.management.AddFields(fc, _field_spec(fields))
        return fc, False
    except Exception:
        fc = arcpy.management.CreateFeatureclass(
            "memory", f"swath_{uuid.uuid4().hex}", geom_type, spatial_reference=sr,
            has_z=z)[0]
        arcpy.management.AddFields(fc, _field_spec(fields))
        return fc, True


def _finalise_fc(fc, out_path, via_memory=True):
    """Copy a memory feature class to its destination (no-op if written direct)."""
    import arcpy

    if not via_memory:
        return out_path
    arcpy.management.CopyFeatures(fc, out_path)
    try:
        arcpy.management.Delete(fc)
    except Exception:
        pass
    return out_path


def write_band_polygons(out_path, rings, sr, messenger=None):
    """Write one polygon per baseline part: the region actually sampled.

    A record may carry a precomputed ``"shape"`` (the band already clipped to a
    sampling mask); otherwise the polygon is built from its ``"ring"``.
    """
    import arcpy

    fc, via_memory = _create(out_path, "POLYGON", sr, [
        ("LINE_OID", "LONG"), ("PART_ID", "LONG"), ("LINE_NAME", "TEXT", 64),
        ("SWATH_W", "DOUBLE"), ("STEP", "DOUBLE"),
        ("N_LINES", "LONG"), ("N_STATION", "LONG"),
        ("METHOD", "TEXT", 32),
    ], kind="band")
    with arcpy.da.InsertCursor(fc, ["SHAPE@", "LINE_OID", "PART_ID", "LINE_NAME",
                                    "SWATH_W", "STEP", "N_LINES", "N_STATION",
                                    "METHOD"]) as cur:
        for r in rings:
            shape = r.get("shape")
            if shape is None:
                arr = arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in r["ring"]])
                shape = arcpy.Polygon(arr, sr)
            cur.insertRow([shape, r["oid"], r["part_id"],
                           str(r.get("name", ""))[:64], r["width"], r["step"],
                           r["n_lines"], r["n_stations"], r["method"]])
    # The sampled ring self-intersects wherever the band folds; that is real and
    # deliberate, and the geodatabase accepts it as drawn.  (A band clipped to a
    # sampling mask has been through a topological operation and is simplified.)
    fc = _finalise_fc(fc, out_path, via_memory)
    if messenger:
        messenger.info(f"  Swath band polygons -> {out_path}")
    return fc


def write_offset_lines(out_path, lines, sr, messenger=None):
    """Write the N parallel profile lines per baseline part.

    ``SIDE`` records which side of the baseline each line came from, in the same
    words the chart legend uses, so a unique-value renderer on it reproduces the
    chart's colours on the map.  It is a text field rather than a reading of the
    signed ``OFFSET`` because Pro's unique-value renderer needs a categorical
    field, and because a definition query on ``SIDE = 'Left'`` cannot make the
    ``>`` / ``>=`` mistake at the centre line.

    A record may carry a precomputed ``"shape"`` (the line already clipped to a
    sampling mask); otherwise the polyline is built from its ``"xy"``.
    """
    import arcpy

    from .geometry import SIDE_NAMES, side_of

    fc, via_memory = _create(out_path, "POLYLINE", sr, [
        ("LINE_OID", "LONG"), ("PART_ID", "LONG"), ("LINE_NAME", "TEXT", 64),
        ("LINE_IDX", "LONG"), ("OFFSET", "DOUBLE"), ("N_VALID", "LONG"),
        ("SIDE", "TEXT", 8),
    ], kind="offsets_side")
    with arcpy.da.InsertCursor(fc, ["SHAPE@", "LINE_OID", "PART_ID", "LINE_NAME",
                                    "LINE_IDX", "OFFSET", "N_VALID",
                                    "SIDE"]) as cur:
        for ln in lines:
            shape = ln.get("shape")
            if shape is None:
                arr = arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in ln["xy"]])
                shape = arcpy.Polyline(arr, sr)
            side = SIDE_NAMES[int(side_of([ln["offset"]])[0])]
            cur.insertRow([shape, ln["oid"], ln["part_id"],
                           str(ln.get("name", ""))[:64], ln["index"], ln["offset"],
                           ln["n_valid"], side])
    fc = _finalise_fc(fc, out_path, via_memory)
    if messenger:
        messenger.info(f"  Offset lines -> {out_path}")
    return fc


def write_transects(out_path, records, sr, messenger=None):
    """Write generated transects with their id, label, position and bearing."""
    import arcpy

    fc, via_memory = _create(out_path, "POLYLINE", sr, [
        ("TRANSECT_ID", "LONG"), ("LINE_NAME", "TEXT", 64),
        ("DIST_ALONG", "DOUBLE"), ("AZIMUTH", "DOUBLE"), ("LENGTH", "DOUBLE"),
    ], kind="transects")
    with arcpy.da.InsertCursor(fc, ["SHAPE@", "TRANSECT_ID", "LINE_NAME",
                                    "DIST_ALONG", "AZIMUTH", "LENGTH"]) as cur:
        for r in records:
            arr = arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in r["ends"]])
            cur.insertRow([arcpy.Polyline(arr, sr), r["id"], r["name"],
                           r["dist_along"], r["azimuth"], r["length"]])
    fc = _finalise_fc(fc, out_path, via_memory)
    if messenger:
        messenger.info(f"  {len(records)} transect(s) -> {out_path}")
    return fc


def write_station_points(out_path, result, sr, messenger=None, max_rasters=4):
    """One point per station with the statistics as wide attributes.

    A *feature* layer is what makes Pro's native charts link to the map: brush
    the chart and the stations highlight, select on the map and the chart
    follows.  One point per station (not per raster) keeps that selection
    unambiguous; each raster's statistics become prefixed columns.
    """
    import arcpy

    from .outputs import nullify_nan, rows_to_array, safe_prefix

    names = list(result.raster_names)[:max_rasters]
    if len(result.raster_names) > max_rasters and messenger:
        messenger.warn(f"  Station points carry statistics for the first "
                       f"{max_rasters} rasters only; the table has all of them.")

    stat_keys = [("MIN", "z_min"), ("MAX", "z_max"), ("MEAN", "z_mean"),
                 ("MEDIAN", "z_median"), ("Q1", "z_q1"), ("Q3", "z_q3"),
                 ("PLO", "z_plo"), ("PHI", "z_phi"), ("CENTRE", "z_centre"),
                 ("RELIEF", "relief"), ("HI", "hi"), ("THI", "thi")]
    fields = [("LINE_OID", "LONG", None), ("PART_ID", "LONG", None),
              ("LINE_NAME", "TEXT", 64), ("STATION", "LONG", None),
              ("DIST", "DOUBLE", None), ("KAPPA", "DOUBLE", None),
              ("FOLDED", "SHORT", None), ("N_VALID", "LONG", None)]
    prefixes = {n: safe_prefix(n) for n in names}
    for n in names:
        for label, _ in stat_keys:
            fields.append((f"{prefixes[n]}_{label}", "DOUBLE", None))

    rows, xy = [], []
    for oid, part_id in result.parts():
        g = result.geometry[(oid, part_id)]
        c, kappa = g["centreline"], g["kappa"]
        folded = g["folded"]
        fold_any = (folded.any(axis=0) if folded is not None
                    else np.zeros(len(c), dtype=bool))
        by_raster = result.for_part(oid, part_id)
        first = next(iter(by_raster.values()))
        name = str(result.names.get((oid, part_id), ""))[:64]
        for j in range(first.n_stations):
            row = [oid, part_id, name, j, float(first.distances[j]),
                   float(kappa[j]), int(bool(fold_any[j])), int(first.n_valid[j])]
            for n in names:
                s = by_raster.get(n)
                for _, key in stat_keys:
                    arr = None if s is None else getattr(s, key, None)
                    row.append(np.nan if arr is None else float(arr[j]))
            rows.append(tuple(row))
            xy.append((float(c[j, 0]), float(c[j, 1])))

    # arcpy.da.NumPyArrayToFeatureClass is a data-access call, not a
    # geoprocessing tool: no framework overhead, no memory workspace, no copy.
    base = rows_to_array(fields, rows)
    dtype = [("SHAPE@XY", "<f8", 2)] + base.dtype.descr
    arr = np.zeros(len(rows), dtype=dtype)
    arr["SHAPE@XY"] = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    for f in base.dtype.names:
        arr[f] = base[f]
    try:
        arcpy.da.NumPyArrayToFeatureClass(arr, out_path, ["SHAPE@XY"], sr)
        nullify_nan(out_path, [f[0] for f in fields if f[1] == "DOUBLE"])
    except Exception as exc:
        if messenger:
            messenger.info(f"  (direct write failed - {type(exc).__name__}: "
                           f"{str(exc).splitlines()[0][:120]}; using the "
                           "in-memory route)")
        fc, via_memory = _create(out_path, "POINT", sr,
                                 [(f[0], f[1], f[2]) if f[2] else (f[0], f[1])
                                  for f in fields], kind="stations")
        cols = ["SHAPE@XY"] + [f[0] for f in fields]
        with arcpy.da.InsertCursor(fc, cols) as cur:
            for pt, row in zip(xy, rows):
                cur.insertRow([pt] + [None if isinstance(v, float) and not np.isfinite(v)
                                      else v for v in row])
        _finalise_fc(fc, out_path, via_memory)
    if messenger:
        messenger.info(f"  Station points ({len(rows):,}) -> {out_path}")
    return out_path


def _split_at_nan(xy, z):
    """Yield contiguous runs of valid samples as (xy, z) segments."""
    valid = np.isfinite(z)
    if not valid.any():
        return
    idx = np.flatnonzero(valid)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(idx) - 1]])
    for a, b in zip(starts, ends):
        run = idx[a:b + 1]
        if len(run) >= 2:
            yield xy[run], z[run]


def write_profiles_3d(out_path, profiles, sr, messenger=None):
    """Write Z-enabled profile lines.

    NaN cannot be stored as a Z value, so each line is split into multipart
    segments at NoData runs rather than interpolating across the hole or
    dropping to zero - either of which would draw terrain that does not exist.
    """
    import arcpy

    fc, via_memory = _create(out_path, "POLYLINE", sr, [
        ("LINE_OID", "LONG"), ("PART_ID", "LONG"), ("RASTER", "TEXT", 128),
        ("SERIES", "TEXT", 16), ("OFFSET", "DOUBLE"),
    ], has_z=True, kind="profiles3d")

    n_split = 0
    with arcpy.da.InsertCursor(fc, ["SHAPE@", "LINE_OID", "PART_ID", "RASTER",
                                    "SERIES", "OFFSET"]) as cur:
        for pr in profiles:
            segments = list(_split_at_nan(pr["xy"], pr["z"]))
            if not segments:
                continue
            if len(segments) > 1:
                n_split += 1
            outer = arcpy.Array()
            for seg_xy, seg_z in segments:
                outer.add(arcpy.Array([
                    arcpy.Point(float(x), float(y), float(z))
                    for (x, y), z in zip(seg_xy, seg_z)]))
            cur.insertRow([arcpy.Polyline(outer, sr, True), pr["oid"], pr["part_id"],
                           pr["raster"], pr["series"], pr.get("offset", 0.0)])

    fc = _finalise_fc(fc, out_path, via_memory)
    if messenger:
        messenger.info(f"  3D profile lines -> {out_path}")
        if n_split:
            messenger.info(
                f"  {n_split} 3D line(s) were split into multiple parts at NoData gaps.")
    return fc
