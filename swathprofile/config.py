# -*- coding: utf-8 -*-
"""Run configuration for a swath extraction, and the pre-run cost estimate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .errors import ConfigError

__all__ = ["SwathConfig", "estimate_run", "METHOD_OFFSET", "METHOD_ZONAL"]

METHOD_OFFSET = "offset"
METHOD_ZONAL = "zonal"

_METHOD_ALIASES = {
    "offset": METHOD_OFFSET,
    "parallel offset lines": METHOD_OFFSET,
    "parallel": METHOD_OFFSET,
    "offset lines (classic, keeps profile curves)": METHOD_OFFSET,
    "offset lines": METHOD_OFFSET,
    "zonal": METHOD_ZONAL,
    "perpendicular zonal slices": METHOD_ZONAL,
    "perpendicular": METHOD_ZONAL,
    "all cells (exact envelope)": METHOD_ZONAL,
    "all cells": METHOD_ZONAL,
}


@dataclass
class SwathConfig:
    """Everything the engine needs, independent of how it was collected.

    Kept separate from the toolbox layer so the engine can be driven from a
    script or a test without constructing arcpy Parameter objects.
    """

    swath_width: float
    step: float
    n_lines: int = 51
    method: str = METHOD_OFFSET
    interpolation: str = "bilinear"
    z_factor: float = 1.0

    # geometry
    multipart: str = "each_part"          # each_part | largest_part | error
    tangent_smooth_window: int = 0
    fold_policy: str = "mask"             # mask | warn | ignore

    # statistics
    smooth_window: int = 0
    smooth_dilate: bool = False
    quantile_method: str = "weibull"
    pct_low: float = 5.0
    pct_high: float = 95.0
    flat_hi: str = "nan"                  # nan | half | reference
    thi_clip_factor: bool = True
    thi_clip_range: bool = False
    min_valid: int = 1
    legacy_stats: bool = False
    # How the individual profile lines are coloured in saved charts: single |
    # side | ramp.  Carried into the bundle so the chart window opens showing
    # what the run's exported figures show.
    chart_profile_colour: str = "single"

    # transects generated from a trace (0 = not used)
    transect_spacing: float = 0.0
    transect_length: float = 0.0
    transect_side: str = "both"           # both | left | right
    transect_start: float = 0.0

    # performance
    max_block_cells: int = 40_000_000
    edge_nearest: bool = True
    max_stations: int = 200_000

    # coordinate systems
    geo_transform: str = ""

    # bookkeeping
    label_field: str = ""
    thi_raster: str = ""
    tool_version: str = ""
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        self.method = _METHOD_ALIASES.get(
            str(self.method).strip().lower(), str(self.method).strip().lower())
        if self.method not in (METHOD_OFFSET, METHOD_ZONAL):
            raise ConfigError(
                f"Unknown sampling method {self.method!r}; expected "
                "'Offset lines' or 'All cells'.")
        if self.swath_width <= 0:
            raise ConfigError(f"Swath width must be positive; got {self.swath_width}.")
        if self.step <= 0:
            raise ConfigError(f"Step size must be positive; got {self.step}.")
        if self.n_lines < 1:
            raise ConfigError(f"Number of profiles must be at least 1; got {self.n_lines}.")
        if self.fold_policy not in ("mask", "warn", "ignore"):
            raise ConfigError(f"Unknown fold policy {self.fold_policy!r}.")
        if self.multipart not in ("each_part", "largest_part", "error"):
            raise ConfigError(f"Unknown multipart handling {self.multipart!r}.")
        if self.flat_hi not in ("nan", "half", "reference"):
            raise ConfigError(f"Unknown flat-HI policy {self.flat_hi!r}.")
        if not (0.0 < self.pct_low < self.pct_high < 100.0):
            raise ConfigError(
                f"Percentiles must satisfy 0 < low < high < 100; got "
                f"{self.pct_low} and {self.pct_high}.")
        if self.transect_side not in ("both", "left", "right"):
            raise ConfigError(f"Unknown transect side {self.transect_side!r}.")
        if self.transect_spacing < 0 or self.transect_length < 0:
            raise ConfigError("Transect spacing and length must not be negative.")

    @property
    def is_offset(self) -> bool:
        return self.method == METHOD_OFFSET

    @property
    def uses_transects(self) -> bool:
        return self.transect_spacing > 0 and self.transect_length > 0

    @property
    def half_width(self) -> float:
        return self.swath_width / 2.0

    @property
    def cross_track_spacing(self) -> float:
        """Distance between adjacent offset lines."""
        if self.n_lines < 2:
            return 0.0
        return self.swath_width / (self.n_lines - 1)

    def to_json(self) -> str:
        """Provenance blob for the summary table."""
        d = asdict(self)
        d.pop("extras", None)
        return json.dumps(d, sort_keys=True, default=str)


#: Measured on the development machine (0.5 m LiDAR, file geodatabase):
#: offset mode ~3 us per sample including raster I/O, zonal ~1 us per cell.
_SECONDS_PER_SAMPLE = 3.0e-6
_SECONDS_PER_CELL = 1.0e-6
_FIXED_SECONDS = 3.0


def estimate_run(total_length: float, step: float, n_lines: int, width: float,
                 cell_size: float, method: str = METHOD_OFFSET,
                 n_rasters: int = 1) -> dict:
    """Rough cost of a run, for the dialog's pre-run message.

    Returns ``stations``, ``work`` (samples for offset mode, cells for zonal),
    ``seconds`` and a ready-made ``text``.  Deliberately simple - it exists to
    stop someone launching a ten-minute run by accident, not to be precise.
    """
    method = _METHOD_ALIASES.get(str(method).strip().lower(), method)
    length = max(float(total_length), 0.0)
    step = max(float(step), 1e-9)
    stations = int(length / step) + 1
    n_rasters = max(int(n_rasters), 1)
    if method == METHOD_ZONAL:
        cell = max(float(cell_size), 1e-9)
        work = int(length * float(width) / (cell * cell)) * n_rasters
        seconds = _FIXED_SECONDS + work * _SECONDS_PER_CELL
        what = "cells"
    else:
        work = stations * max(int(n_lines), 1) * n_rasters
        seconds = _FIXED_SECONDS + work * _SECONDS_PER_SAMPLE
        what = "samples"

    if work >= 1e6:
        size = f"{work / 1e6:.1f} million {what}"
    else:
        size = f"{work:,} {what}"
    if seconds < 90:
        when = f"about {max(int(round(seconds)), 1)} s"
    else:
        when = f"about {seconds / 60:.0f} min"
    return {"stations": stations, "work": work, "seconds": seconds,
            "text": f"{stations:,} stations, {size}, {when}."}
