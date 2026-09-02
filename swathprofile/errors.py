# -*- coding: utf-8 -*-
"""Exception types for the swath profile toolbox.

Every error raised deliberately by this package derives from :class:`SwathError`,
so the toolbox layer can catch that one type and turn it into an arcpy error
message instead of a Python traceback.
"""


class SwathError(Exception):
    """Base class for all deliberate swath-profile errors."""


class CrsError(SwathError):
    """Coordinate reference system is missing, geographic, or unusable."""


class GeometryError(SwathError):
    """A baseline geometry cannot be turned into a usable centreline."""


class RasterError(SwathError):
    """A raster is unreadable, multiband, or otherwise unusable."""


class MemoryBudgetError(SwathError):
    """A single station's read window exceeds the configured memory budget."""


class ConfigError(SwathError):
    """Invalid combination of parameters."""
