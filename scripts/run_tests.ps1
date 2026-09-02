# Full test suite. Must use the ArcGIS Pro interpreter (arcpy-dependent tests).
$py = "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
& $py -m pytest "$PSScriptRoot\..\tests" -q
