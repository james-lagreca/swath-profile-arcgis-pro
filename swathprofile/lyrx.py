# -*- coding: utf-8 -*-
"""Layer files (.lyrx) from JSON templates.  No arcpy, no geoprocessing calls.

A ``.lyrx`` is a CIM layer document in JSON.  The templates under
``templates/`` were saved once by ArcGIS Pro (via ``MakeFeatureLayer`` and
``saveACopy``) with the symbology and labelling we want; at run time only the
data connection, the name and optionally a chart are filled in.  Pro applies
the file when it adds the output to the map after the tool finishes.

This replaces ``MakeFeatureLayer`` + ``saveACopy`` + ``Delete`` at run time -
three geoprocessing calls per output - because inside a large Pro project each
geoprocessing call was measured at 20-35 seconds regardless of what it did.
"""

from __future__ import annotations

import json
import os
import uuid

__all__ = ["build_layer_file", "TEMPLATES"]

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = {
    "band": os.path.join(_HERE, "templates", "band.lyrx.json"),
    "transects": os.path.join(_HERE, "templates", "transects.lyrx.json"),
    "stations": os.path.join(_HERE, "templates", "stations.lyrx.json"),
    "offsets": os.path.join(_HERE, "templates", "offsets.lyrx.json"),
}

#: Unique-value class label -> the ``plotting.PALETTE`` key it takes its colour
#: from.  Patching at write time means the map symbology and the chart cannot
#: drift apart when a colour is changed in one place.
_CLASS_COLOURS = {
    "Left of line": "side_left",
    "Right of line": "side_right",
    "On the line": "centre",
}


def _rgb255(colour):
    """'#2C6FA8' -> [44, 111, 168, 100], the CIMRGBColor value list."""
    h = colour.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [100]


def _apply_palette(ld):
    """Colour a unique-value renderer's classes from the plotting palette."""
    from .plotting import PALETTE

    renderer = ld.get("renderer") or {}
    if renderer.get("type") != "CIMUniqueValueRenderer":
        return
    for group in renderer.get("groups") or []:
        for cls in group.get("classes") or []:
            key = _CLASS_COLOURS.get(cls.get("label"))
            if key is None:
                continue
            for layer in ((cls.get("symbol") or {}).get("symbol") or {}
                          ).get("symbolLayers") or []:
                if "color" in layer:
                    layer["color"]["values"] = _rgb255(PALETTE[key])


def _split_gdb_path(fc_path):
    """('<...>.gdb', 'dataset') from a file-geodatabase feature class path."""
    p = fc_path.replace("/", os.sep)
    low = p.lower()
    idx = low.find(".gdb")
    if idx < 0:
        raise ValueError(f"Not a file geodatabase path: {fc_path}")
    gdb = p[:idx + 4]
    rest = p[idx + 4:].strip(os.sep)
    dataset = rest.split(os.sep)[-1] if rest else ""
    return gdb, dataset


def build_layer_file(kind, fc_path, out_path, layer_name=None, chart=None,
                     label_field=None):
    """Write a .lyrx for *fc_path* from the *kind* template.

    ``chart`` is a CIM chart dict (see :mod:`cimchart`); ``label_field`` sets
    the Arcade label expression on templates that carry a label class.
    """
    with open(TEMPLATES[kind], encoding="utf-8") as fh:
        doc = json.load(fh)

    gdb, dataset = _split_gdb_path(fc_path)
    name = layer_name or dataset
    uri = f"CIMPATH=Internal_Map/{dataset}_{uuid.uuid4().hex[:8]}.json"

    ld = doc["layerDefinitions"][0]
    ld["name"] = name
    ld["uRI"] = uri
    doc["layers"] = [uri]

    table = ld.setdefault("featureTable", {"type": "CIMFeatureTable"})
    table["dataConnection"] = {
        "type": "CIMStandardDataConnection",
        "workspaceConnectionString": f"DATABASE={gdb}",
        "workspaceFactory": "FileGDB",
        "dataset": dataset,
        "datasetType": "esriDTFeatureClass",
    }
    # Field descriptions in the template refer to the template's own fields.
    table.pop("fieldDescriptions", None)
    table["displayField"] = "OBJECTID"

    if label_field and ld.get("labelClasses"):
        for lc in ld["labelClasses"]:
            lc["expression"] = f"$feature.{label_field}"
            lc["expressionEngine"] = "Arcade"
        ld["labelVisibility"] = True

    if chart is not None:
        ld["charts"] = [chart]
    else:
        ld.pop("charts", None)

    _apply_palette(ld)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return out_path
