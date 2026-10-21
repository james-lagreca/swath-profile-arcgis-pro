# -*- coding: utf-8 -*-
"""Swath profile extraction for ArcGIS Pro.

A port of the Perez-Pena et al. (2017) SwathProfiler method
(Computers & Geosciences 104, 135-150) to arcpy/numpy, by way of the
OpenQGIS/SwathProfile4QGIS Python re-implementation.

The modules split along one line: anything that imports arcpy is I/O, and
everything else is testable with plain numpy.

arcpy-free
    :mod:`geometry`, :mod:`sampling`, :mod:`stats`, :mod:`smoothing`,
    :mod:`plotting`, :mod:`bundle`, :mod:`errors`, :mod:`mask`, :mod:`transforms`
arcpy-dependent
    :mod:`raster_io`, :mod:`features`, :mod:`zonal`, :mod:`outputs`, :mod:`engine`
    (arcpy is imported inside their functions, so the engine also runs on an
    in-memory raster - see ``tests/fake_raster.py``)
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
