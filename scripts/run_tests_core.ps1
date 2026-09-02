# arcpy-free subset: geometry, statistics, smoothing.
# Runs under any Python 3.11 with numpy + scipy; no ArcGIS needed.
$py = "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
& $py -m pytest "$PSScriptRoot\..\tests\test_core.py" -q
