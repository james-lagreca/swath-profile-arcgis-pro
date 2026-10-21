# arcpy-free subset: geometry, statistics, smoothing, transforms, the sampling
# mask, and the engine on an in-memory raster.  The same files CI runs.
# Runs under any Python 3.11 with numpy + scipy + matplotlib; no ArcGIS needed.
$py = "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
& $py -m pytest "$PSScriptRoot\..\tests\test_core.py" "$PSScriptRoot\..\tests\test_transforms.py" "$PSScriptRoot\..\tests\test_mask.py" -q
