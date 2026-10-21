# Changelog

## Unreleased

### Added
- **Colour the profile lines by which side of the baseline they came from.**
  Offset-lines sampling only. Choose *Side of line* (one colour each side) or
  *Side + distance* (the same two colours fading in towards the baseline, so how
  far out a profile sits is readable too) in the chart window under *Profile
  lines*, or with the new *Colour Profile Lines By* parameter for the exported
  figures. The default is the historic single grey, so existing figures are
  unchanged.
  - Left and right are relative to the direction the baseline was digitised,
    which is invisible on a map, so the legend adds the compass point each side
    faces: `Left (W)`, `Right (E)`. On a baseline that curves too much for one
    bearing to be honest the compass is left off. Flipping the display turns the
    compass through 180 degrees, so a side keeps the label it deserves.
  - The *Offset Lines* output gains a `SIDE` field holding `Left`, `Right` or
    `Centre`, and for the first time gets a layer file: a unique-value renderer
    on that field, in the same colours as the chart. The colours are copied from
    the chart palette when the layer file is written, so the two cannot drift.
  - The cross-section panel colours its samples to match.
  - The line's own profile (offset zero) is drawn in the baseline's near-black
    rather than either side's colour, using the same centre test the statistics
    use, so the chart and `Z_CENTRE` always agree about which line that is.
- **Sampling mask: measure one surface only.** A new *Sampling mask* group
  takes a polygon drawn on the map and/or a polygon layer (a selection is
  honoured; everything is merged), with *Keep only inside* (default) or
  *Exclude inside*. Use it for an alluvial fan whose swath clips the bedrock
  beside it.
  - Masked samples and cells are treated exactly like NoData, so nothing outside
    the mask enters any statistic or the smoothing, in either sampling mode and
    for the baseline's own profile.
  - Offset lines: decided on each sample's position, so every stacked raster sees
    the same samples. All cells: a strict cut on each cell's centre, with the
    mask moved into each raster's own coordinate system.
  - The swath band and the offset lines on the map are cut to what was sampled.
    The summary gains `MASK_KEPT` (percent of each band kept), and the chart
    window and saved charts name the mask.
  - Profiles, stretches and raster blocks the mask removes completely are not
    read. No geoprocessing calls are added.

### Fixed
- **All cells mode ignored *Minimum Valid Samples per Station*.** Only the
  offset-lines path applied it. It now nulls stations with too few valid cells
  in both modes, before relief, HI and THi are computed.
- **All cells results could differ in the last decimal places with *Max Raster
  Block*.** Cell centres were computed from each block's own corner, which on a
  fractional grid (0.3 m LiDAR, say) rounds differently for about one cell in
  five depending on how the read was tiled. They are now computed from the
  raster's origin, like every other index calculation, and a test proves the
  results are bitwise identical at any block size.
- **The chart window's smoothing dropped the run's statistics settings.**
  Choosing a smoothing preset re-computed the statistics without *Minimum Valid
  Samples*, the flat-HI rule or the legacy switch, so stations the run had left
  out reappeared. It now uses the run's settings.
- **Drawing a baseline worked only every other time the tool was opened.** The
  sketch parameter was given a POLYLINE template created in the `memory`
  workspace. That workspace lives as long as the ArcGIS Pro process, so the
  first dialog open created the template and bound the parameter to that
  dataset, which disables the sketch tool; every later open raised
  `ERROR 000258: Output ... already exists`, had it swallowed, and left the
  parameter unbound, which is the state that lets you draw. Running the tool
  set `overwriteOutput` to true for the rest of the session, so creation began
  succeeding again and the sketch broke again. The geometry type is now set
  with the parameter's own `filter.list` and no template is created. Dialog
  construction no longer makes a geoprocessing call or touches `arcpy.mp`,
  which also helps in slow projects. Two related defects went with it: the
  template was created with no coordinate system whenever the active map could
  not be read, and a feature set whose coordinate system is unknown cannot be
  edited; and `execute` left `overwriteOutput` switched on for the rest of the
  session, which is what made the failure alternate rather than repeat.
  `updateMessages` now also rejects a sketch that is not a line, rather than
  reading a polygon as its own boundary or failing part-way through a run.

### Changed
- The offset-lines template feature class is keyed `offsets_side` rather than
  `offsets`, because the cached `_swath_tmpl_offsets` in a results geodatabase
  written by 1.0.0 has no `SIDE` field. Existing geodatabases keep that one
  unused table; it can be deleted.
- Parameter positions changed: *Colour Profile Lines By* and the three
  *Sampling mask* parameters were inserted, so scripts that pass the tool's
  parameters by position rather than by name need updating from *Sampling
  Method* onward.
- Saved charts state each profile's own swath width and line count in the
  footer (transects can differ from the main baseline), and a profile with no
  valid data in any raster no longer gets an empty chart.

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
