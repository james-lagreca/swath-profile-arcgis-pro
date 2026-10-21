@echo off
rem Open the Swath Profile viewer on its own (no ArcGIS Pro needed).
rem Drag a .swath.npz file onto this, or run it and pick one.
rem On a computer that allows only approved programs this is blocked too;
rem open the run's .html browser chart in swath_results instead.
set "PYW=C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\pythonw.exe"
if not exist "%PYW%" (
    echo Could not find %PYW%
    echo Edit SwathViewer.cmd to point at your ArcGIS Pro Python environment.
    pause
    exit /b 1
)
start "" "%PYW%" "%~dp0viewer\swath_viewer.py" %1
