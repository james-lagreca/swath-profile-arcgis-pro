# Changelog

## 1.0.0 — 2026-09-02

First public release.

### What it does
- Swath profiles along drawn baselines, existing line layers, or transects
  placed automatically along a fault trace (with their own swath width).
- Per-station minimum, maximum, mean, median, quartiles, configurable
  percentiles (default 5/95), the profile along the baseline itself, local
  relief, hypsometric integral (HI) and terrain hypsometric index (THi / THi*),
  after Pérez-Peña et al. (2017).
- Two sampling methods: offset lines (the published method, keeps the profile
  family) and all cells (exact envelope from every raster cell).
- Lines and rasters in any coordinate system; computed in a projected working
  system with samples transformed into each raster's own.
- Interactive chart window that stays open across runs: cross-section at a
  clicked station, measurements (Δd, Δz, gradient), distance re-zero and flip,
  linear detrend / relative-to-minimum views, overlay or stacked (ridgeline)
  display of several profiles, vertical exaggeration control, publication
  export presets with editable PDF/SVG text.
- Map outputs: transparent swath band, station points carrying every statistic
  with a native Pro chart linked to map selection, labelled transects, optional
  offset lines, Z-enabled 3D profile lines, summary table with provenance, CSV.
- Hover help on every parameter; pre-run time estimate; progress bar.

### Numerical fixes relative to the reference implementations
Seventeen defects are documented and regression-tested in
`docs/ALGORITHM.md`, including NaN-corrupted quartiles, a THi relief factor that
inverts at low relief, a scale-dependent flat-station guard, silently dropped
multipart baselines, and a zonal mode that zero-filled its quartiles. A
*Legacy statistics* switch reproduces the original numbers on request.

### Engineering notes
- Vectorised, tiled raster sampling with bitwise-identical results at any
  block size; empirically verified `RasterToNumPyArray` conventions.
- Zonal mode is a cross-section sweep in numpy (no Spatial Analyst), the same
  construction as the offset lines so both modes agree on curves.
- Writers avoid ArcGIS Pro's geoprocessing framework (tables and points via
  `arcpy.da`, layer files as JSON), because in some projects every
  geoprocessing call costs tens of seconds.
