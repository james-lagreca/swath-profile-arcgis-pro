# -*- coding: utf-8 -*-
"""Native ArcGIS Pro chart definitions as plain CIM JSON.  No arcpy.

Pro stores a layer's charts inside its layer definition (``charts`` in a
``.lyrx``).  Writing that JSON directly is deterministic and cannot crash,
which the ``arcpy.charts`` API is not: building a chart against a feature-class
data source and calling ``addToLayer`` raised an access violation in testing -
inside Pro that would take the whole session down - and even when it "worked"
it stored a series without its field names.  The structure below was copied
from a chart Pro itself saved and completed with the ``fields`` entry.

``mapSelectionHandling: "Highlight"`` is what links the chart to the map:
selecting on the chart highlights the features and vice versa.
"""

from __future__ import annotations

import uuid

__all__ = ["line_chart", "inject_chart"]

_TEXT_DARK = {"type": "CIMRGBColor", "values": [68, 68, 68, 100]}


def _text(size, weight="Normal"):
    return {"type": "CIMChartTextProperties", "fontFillColor": dict(_TEXT_DARK),
            "fontFamilyName": "Segoe UI", "fontSize": size, "fontWeight": weight,
            "textCase": "Normal", "visible": True}


def _axis(title=None, value_format=None):
    ax = {
        "type": "CIMChartAxis", "visible": True, "showTitle": True,
        "useAutomaticTitle": title is None,
        "calculateAutomaticMinimum": True, "calculateAutomaticMaximum": True,
        "minimum": None, "maximum": None,
        "titleText": _text(12), "labelText": _text(10),
        "labelCharacterLimit": 11, "zoomEndPosition": 1,
        "useAutomaticLabelAngle": True, "useAutomaticInterval": True,
    }
    if title is not None:
        ax["title"] = title
    if value_format:
        ax["valueFormat"] = value_format
    return ax


def _rgb(values):
    return {"type": "CIMRGBColor", "values": list(values)}


def line_chart(x_field, series, title, x_title=None, y_title=None):
    """A CIM line chart.

    *series* is a list of ``(field, label, (r, g, b))`` tuples, one line each.
    """
    cim_series = []
    for i, (field, label, rgb) in enumerate(series):
        colour = _rgb([*rgb, 100])
        cim_series.append({
            "type": "CIMChartLineSeries",
            "name": label, "uniqueName": f"Series{i}",
            "fields": [x_field, field],
            "verticalAxis": 1, "colorType": "SingleColor", "visible": True,
            "lineSymbolProperties": {"type": "CIMChartLineSymbolProperties",
                                     "visible": True, "width": 2, "style": "Solid",
                                     "color": dict(colour)},
            "markerSymbolProperties": {"type": "CIMChartMarkerSymbolProperties",
                                       "visible": False, "width": 2, "height": 2,
                                       "style": "Circle", "color": dict(colour)},
            "timeAggregationType": "EqualIntervalsFromStartTime",
            "timeIntervalUnits": "esriTimeUnitsMonths",
            "calculateAutomaticTimeInterval": True,
            "trimIncompleteTimeInterval": True,
            "nullPolicy": "Null", "verticalOrientation": True,
        })

    return {
        "type": "CIMChart",
        "name": f"chart{uuid.uuid4()}",
        "series": cim_series,
        "generalProperties": {
            "type": "CIMChartGeneralProperties",
            "title": title, "showTitle": True, "useAutomaticTitle": False,
            "showSubTitle": True, "showFooter": True,
            "titleText": _text(16), "subTitleText": _text(12), "footerText": _text(12),
            "backgroundSymbolProperties": {"type": "CIMChartFillSymbolProperties",
                                           "color": _rgb([255, 255, 255, 100]),
                                           "visible": True},
            "gridLineSymbolProperties": {"type": "CIMChartLineSymbolProperties",
                                         "visible": True, "width": 1, "style": "Solid",
                                         "color": _rgb([225, 225, 225, 100])},
        },
        "legend": {"type": "CIMChartLegend", "visible": True, "showTitle": False,
                   "alignment": "Right", "legendText": _text(10),
                   "legendTitle": _text(10.8)},
        "axes": [_axis(x_title), _axis(y_title, "N2")],
        "mapSelectionHandling": "Highlight",
        "chartType": "Basic",
    }


def inject_chart(lyrx_path, chart):
    """Add *chart* to the first layer definition of a saved ``.lyrx``."""
    import json

    with open(lyrx_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    defs = doc.get("layerDefinitions") or []
    if not defs:
        raise ValueError("Layer file has no layer definitions.")
    defs[0]["charts"] = [chart]
    with open(lyrx_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return lyrx_path
