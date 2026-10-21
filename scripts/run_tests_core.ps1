# arcpy-free subset: geometry, statistics, smoothing, transforms, the sampling
# mask, the engine on an in-memory raster and the browser chart.  The same
# files CI runs.  Runs under any Python 3.11 with numpy + scipy + matplotlib;
# no ArcGIS needed.  The browser test uses Chrome or Edge if installed, and the
# JavaScript tests need Node.
$py = "C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
$tests = Join-Path $PSScriptRoot "..\tests"
& $py -m pytest (Join-Path $tests "test_core.py") (Join-Path $tests "test_transforms.py") `
    (Join-Path $tests "test_mask.py") (Join-Path $tests "test_chartwindow.py") `
    (Join-Path $tests "test_webchart.py") `
    (Join-Path $tests "test_webchart_render.py") -q
if (Get-Command node -ErrorAction SilentlyContinue) {
    & node --test (Join-Path $tests "js\webchart_core.test.js")
}
