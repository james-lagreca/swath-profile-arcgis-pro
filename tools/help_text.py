# -*- coding: utf-8 -*-
"""Parameter help for the Swath Profile tool.

This is the single source for the text ArcGIS Pro shows when you hover a
parameter (via the generated ``SwathProfile.SwathProfileTool.pyt.xml``) and
for the parameter table in the README.  Each entry is
``name: (one-line summary, [paragraphs], [bullets])``.  Run
``scripts/build_help.py`` after changing parameters or text.
"""

TOOL_SUMMARY = (
    "Extracts swath profiles along one or more baselines: for every station along "
    "the line, the minimum, maximum, mean, median, quartiles and chosen percentiles "
    "of a raster across a band of the chosen width, plus local relief, the "
    "hypsometric integral (HI) and the terrain hypsometric index (THi). Draw a line "
    "on the map or use a line layer, or give a fault trace and let the tool place "
    "perpendicular transects along it. Results open in an interactive chart window "
    "that refreshes on every run. After Perez-Pena et al. (2017)."
)

TOOL_USAGE = [
    "Lines and rasters may be in any coordinate system; the swath is computed in a "
    "projected working system (the first projected raster's, else the baseline's, "
    "else a UTM zone) and samples are transformed into each raster's own system.",
    "Leave the chart window open between runs: each run adds its profiles to the "
    "window's list so you can compare them. Every run also writes a browser chart "
    "(an .html file in swath_results) that needs no Python, for computers where "
    "the chart window is blocked and for colleagues without ArcGIS.",
    "For high-resolution DEMs or wide bands, choose the All cells sampling method: "
    "offset lines skip most of the cells in the band and the tool will tell you "
    "what fraction.",
    "To measure one surface only - a fan, not the bedrock beside it - draw or choose "
    "a Sampling Mask. Everything outside it (or inside it, in Exclude mode) is left "
    "out of the statistics, and the band and offset lines on the map are cut to "
    "match.",
]

PARAM_HELP = {
    "draw_lines": (
        "Draw one or more baselines directly on the map.",
        ["Click the sketch tool beside this parameter, then click along the map to "
         "draw a line and double-click to finish. Each sketched line becomes a "
         "profile. Sketched lines are labelled 'sketch 1', 'sketch 2', ... and get "
         "OIDs from 1000000 upward so they never collide with a layer's features."],
        []),
    "in_lines": (
        "Existing line features to use as baselines.",
        ["Any polyline layer, in any coordinate system. If features are selected on "
         "the map, only the selection is processed. Multipart features are profiled "
         "part by part (see Multipart Handling)."],
        []),
    "label_field": (
        "Text field of the baseline layer whose value names each profile.",
        ["Carried into every output as LINE_NAME and shown as the profile's name in "
         "the chart window. Leave empty to use the feature's OID."],
        []),
    "in_rasters": (
        "Raster(s) to profile. The first is the primary (DEM).",
        ["Each raster gets its own panel in the chart. The first raster defines the "
         "default step size (1.5 x its cell size) and the working coordinate system "
         "when it is projected. Rasters must be single band; pick a band explicitly "
         "for multiband data, e.g. dem.tif\\Band_1."],
        []),
    "swath_width": (
        "Total width of the band, centred on the baseline.",
        ["This is the full width, not a half-width. A 2000 m swath samples 1000 m on "
         "each side of the line. Wider bands average out more drainage noise; on a "
         "curved baseline a band cannot follow bends tighter than its half-width "
         "without folding onto itself (see Tight-Bend Handling)."],
        []),
    "out_table": (
        "The per-station statistics table (long format: one row per baseline, "
        "raster and station).",
        ["Fields include DIST, Z_MIN, Z_MAX, Z_MEAN, Z_MEDIAN, Z_Q1, Z_Q3, Z_PLO, "
         "Z_PHI (the chosen percentiles), RELIEF, HI, THI, THI_RAW, THI_STAR, "
         "N_VALID, KAPPA and FOLDED. Defaults to swath_results.gdb in the project "
         "folder, named from the baseline layer and a timestamp; existing outputs "
         "are never overwritten. Outputs deliberately avoid the project's own "
         "geodatabase: writing into a geodatabase with unsaved edits waits until "
         "you save them."],
        []),
    "transect_source": (
        "A fault trace (or any line) along which perpendicular transects are "
        "generated and profiled.",
        ["Use this for a systematic set of swaths across a structure. Transects are "
         "placed every Transect Spacing along the trace, each Transect Length long, "
         "numbered T001, T002, ... and written to the Transects output. They are "
         "profiled in addition to any drawn or layer baselines."],
        []),
    "transect_spacing": (
        "Distance between successive transects along the trace.",
        ["Bands of adjacent transects overlap when the spacing is less than the "
         "swath width; that is allowed and the tool says so."],
        []),
    "transect_length": (
        "Total length of each transect.",
        ["Make it longer than the feature you want to see across strike. The band "
         "is Swath Width wide along the whole transect."],
        []),
    "transect_width": (
        "Swath width for the transects, when it should differ from the main "
        "Swath Width.",
        ["Leave blank to use the same width as drawn or layer baselines. Useful "
         "when transects across a fault want a narrow band while a long drawn "
         "baseline wants a wide one; the two are computed separately and shown "
         "together."],
        []),
    "transect_side": (
        "Whether transects straddle the trace or extend to one side only.",
        ["'Both sides' centres each transect on the trace. 'Left of trace' and "
         "'Right of trace' are relative to the digitising direction of the trace."],
        []),
    "transect_start": (
        "Distance along the trace at which the first transect is placed.",
        ["Use it to align transects with a reference point on the trace."],
        []),
    "mask_draw": (
        "Draw a polygon on the map to limit where the swath samples.",
        ["Use it to measure one surface only: outline an alluvial fan so the bedrock "
         "a corner of the band runs onto is left out, or, with Mask Mode 'Exclude "
         "inside', draw round just the part to leave out. Click the sketch tool "
         "beside this parameter, click round the area and double-click to finish. "
         "Several polygons may be drawn, and a Sampling Mask Layer may be used as "
         "well; everything is merged into one mask.",
         "Samples and cells the mask removes count as missing, exactly like NoData, "
         "so they never enter any statistic or the smoothing. The swath band and "
         "the offset lines on the map are cut to what was actually sampled, and the "
         "run reports how much of each band the mask kept (MASK_KEPT in the "
         "summary). Masking changes the relief a station sees, so it changes HI and "
         "THi too: that is the point of measuring one surface."],
        []),
    "mask_layer": (
        "Existing polygon features to use as the sampling mask.",
        ["For example mapped fan or terrace units. Any polygon layer, in any "
         "coordinate system; if features are selected on the map, only the "
         "selection is used. Overlapping polygons are merged, and holes in a "
         "polygon are honoured."],
        []),
    "mask_mode": (
        "Whether the mask keeps what is inside it or removes it.",
        ["Offset lines: a sample is kept or removed by its own position. One just "
         "inside the boundary may still be interpolated from cells up to about one "
         "cell beyond it, which is far finer than a drawn boundary. All cells: a "
         "cell counts only if its centre is inside, a strict cell-level cut."],
        ["Keep only inside - sample only within the polygons (default).",
         "Exclude inside - sample everywhere except within the polygons."]),
    "method": (
        "How the band is sampled.",
        ["Both methods give the same statistics on a straight baseline; they differ "
         "in what is sampled and what the chart can show. In either mode the "
         "profile along the baseline itself is also drawn (the dark 'Baseline' "
         "line) and written as Z_CENTRE."],
        ["Offset lines (classic, keeps profile curves) - N parallel lines across the "
         "band, sampled at each station. Matches the published SwathProfiler method "
         "and produces the individual profile curves drawn in grey. Skips cells "
         "between the lines: on a fine DEM or a wide band most cells never "
         "contribute, and the tool reports the fraction.",
         "All cells (exact envelope) - every raster cell in the band contributes to "
         "its station. The min/max envelope and percentiles are exact. No "
         "individual curves. Recommended for LiDAR and for bands wider than a few "
         "hundred cells."]),
    "n_lines": (
        "Number of parallel profile lines across the band (offset lines only).",
        ["Default 51 (25 each side plus the centreline). Use an odd number so a line "
         "falls exactly on the baseline. The cross-track spacing is Swath Width "
         "divided by (N - 1); if that is coarser than the cell size, cells between "
         "the lines are never sampled."],
        []),
    "step": (
        "Spacing of stations along the baseline.",
        ["Defaults to 1.5 x the primary raster's cell size, as in the original "
         "SwathProfiler. Smaller steps add stations but no new information below "
         "the cell size."],
        []),
    "interpolation": (
        "How a sample between cell centres is read (offset lines only).",
        [],
        ["Bilinear - weighted average of the four surrounding cells. Exact for a "
         "planar surface; the default.",
         "Nearest - the value of the nearest cell. Preserves categorical or "
         "integer data."]),
    "z_factor": (
        "Multiplier applied to raster values.",
        ["Use it when elevations are in different units from the map, e.g. feet on a "
         "metre grid. Relief scales with it; HI and THi are dimensionless and do "
         "not."],
        []),
    "multipart": (
        "What to do with multipart baseline features.",
        [],
        ["Each part separately - every part becomes its own profile, identified by "
         "PART_ID. Parts are never joined, since that would invent a connecting "
         "segment.",
         "Largest part - only the longest part is profiled.",
         "Error - stop and report multipart features."]),
    "tangent_smooth": (
        "Smooth the cross-section direction along the baseline.",
        ["On by default. A hand-drawn or digitised line has a kink at every vertex, "
         "and a wide band folds onto itself at any corner tighter than its "
         "half-width. Stabilising smooths the direction of the cross-sections over "
         "about half a swath width so corners are rounded; the baseline itself is "
         "never moved. Turn off to reproduce strictly per-vertex normals."],
        []),
    "fold_policy": (
        "What to do where the band folds onto itself on the inside of a bend.",
        ["Beyond the centre of curvature the cross-section lines cross and would "
         "count the same ground for several stations."],
        ["Mask folded samples - exclude them; N_VALID is lower there. Default.",
         "Warn only - keep them and report the count.",
         "Ignore - keep them silently."]),
    "smoothing": (
        "Savitzky-Golay smoothing applied to the profile lines before statistics.",
        ["Presets None / Low / Medium / High / Ultra (windows of 5, 11, 21, 41 "
         "stations). Smoothing the lines and then taking min and max narrows the "
         "envelope slightly compared with smoothing the output curves; this matches "
         "the original tool. The chart window can change smoothing afterwards "
         "without re-running."],
        []),
    "pct_low": (
        "Lower percentile reported as Z_PLO.",
        ["Default 5. Together with the upper percentile it gives a robust envelope "
         "that ignores isolated extreme cells, useful on LiDAR."],
        []),
    "pct_high": (
        "Upper percentile reported as Z_PHI.",
        ["Default 95."],
        []),
    "thi_raster": (
        "Which raster drives the THi panel when several rasters are profiled.",
        ["HI and THi are computed for every raster but are only meaningful for "
         "elevation. Defaults to the first raster."],
        []),
    "legacy_stats": (
        "Reproduce the original SwathProfiler statistics exactly, including two "
        "known defects.",
        ["Off by default. When on, a flat cross-section gets HI = 0 instead of "
         "Null, and the THi relief factor is not clamped, so it can invert the index "
         "at very low relief. Use only to match previously published numbers. See "
         "docs/ALGORITHM.md, D3 and D4."],
        []),
    "flat_hi": (
        "Value of the hypsometric integral where the cross-section is flat.",
        ["A flat cross-section has no hypsometry, so Null is the honest answer."],
        ["Null - undefined (default).",
         "0.5 - the neutral value.",
         "Reference (0.0) - what the original tool reports."]),
    "min_valid": (
        "Minimum number of valid samples for a station to be reported.",
        ["Stations with fewer valid samples (NoData, outside the raster, masked "
         "folds, or removed by the sampling mask) get Null statistics. Default 1. "
         "Applies to both sampling methods; in All cells mode it counts cells.",
         "With a sampling mask, raise it to leave out stations at the edge of the "
         "mask that keep only a sliver of the band, whose statistics describe a much "
         "narrower swath than the rest."],
        []),
    "out_band": (
        "Polygon of the band actually sampled, one feature per baseline.",
        ["Added to the map with a transparent fill so you can see what was "
         "measured. It self-intersects wherever the band folds; that is deliberate.",
         "With a sampling mask it is cut to the area the mask kept, and a profile "
         "the mask removed entirely has no band."],
        []),
    "out_stations": (
        "One point per station carrying all the statistics as attributes.",
        ["This is what links the chart to the map: a native line chart is attached "
         "to this layer, and selecting on the chart highlights stations on the map "
         "and vice versa. Attributes are prefixed by raster name, e.g. "
         "MEADOWS_MEAN, MEADOWS_THI, so the layer can be symbolised by any "
         "statistic."],
        []),
    "out_transects": (
        "The generated transects (when a trace is given), numbered and labelled.",
        ["Fields TRANSECT_ID, LINE_NAME (T001...), DIST_ALONG, AZIMUTH and "
         "LENGTH. Labelled on the map with the transect name."],
        []),
    "out_offset_lines": (
        "The parallel profile lines as features (offset lines only).",
        ["One feature per line with its signed OFFSET (positive to the left of the "
         "digitising direction) and the number of valid samples along it.",
         "SIDE holds Left, Right or Centre, and the layer is symbolised on it in "
         "the same colours the chart uses, so the two can be read together.",
         "With a sampling mask, each line is cut to the stretch the mask kept, and "
         "a line it removed entirely is left out."],
        []),
    "out_profiles_3d": (
        "Z-enabled lines of the minimum, mean and maximum profiles.",
        ["For 3D scenes and CAD export. Lines are split into parts at NoData gaps "
         "rather than interpolated across them."],
        []),
    "out_summary": (
        "One row per baseline and raster with run-level values and provenance.",
        ["Includes lengths, station counts, the percentile pair, maximum relief, "
         "mean HI and THi, minimum radius of curvature, NoData fraction, elapsed "
         "time and a JSON blob of every parameter so a result can be reproduced "
         "later.",
         "MASK_KEPT is the percentage of the swath band's area the sampling mask "
         "kept (Null without a mask). PCT_NODATA counts samples removed by the mask "
         "as missing, as it already does folded ones."],
        []),
    "out_csv_folder": (
        "Folder for one CSV per baseline with all statistics side by side.",
        ["Wide format across rasters. NoData is written as an empty cell, never "
         "'nan', so spreadsheets read the columns as numbers."],
        []),
    "chart_formats": (
        "Also write the chart as image files without opening the viewer.",
        ["PNG, PDF and SVG use the chosen Chart Preset: one file per profile, "
         "plus a stacked 'ridgeline' figure of all profiles when there are "
         "several (transects along a trace). The native Pro chart is always "
         "provided on the Station Points layer."],
        []),
    "chart_preset": (
        "Size and typography for saved chart files.",
        [],
        ["Screen - as in the viewer.",
         "Journal single column - 85 mm wide, 8 pt.",
         "Journal double column - 170 mm wide, 8 pt.",
         "Poster - 300 mm wide, 14 pt.",
         "PDF and SVG keep text editable."]),
    "chart_profile_colour": (
        "Colour the individual profile lines by which side of the baseline they "
        "came from.",
        ["Offset lines sampling only: All cells keeps no per-line samples to "
         "colour, and the setting is ignored. The same choice is available live "
         "in the chart window, under Profile lines.",
         "Positive offsets are to the left of the direction the baseline was "
         "digitised. Because that direction is invisible on a map, the legend "
         "adds the compass point each side faces, for example 'Left (W)'; on a "
         "baseline that curves too much for one bearing to be honest the "
         "compass is left off. The Offset Lines output carries the same "
         "classification in its SIDE field, symbolised to match."],
        ["Single colour - faint grey, as in earlier versions.",
         "Side of line - one colour each side, the centre line in near-black.",
         "Side + distance - the same two colours, fading in towards the "
         "baseline, so how far out a profile sits is readable too."]),
    "out_chart_folder": (
        "Folder for the saved chart files.",
        ["Defaults to the swath_results folder beside the statistics table, where "
         "each run's saved result and browser chart also go."],
        []),
    "add_to_map": (
        "Add the outputs (band, station points, transects, table) to the map when "
        "the tool finishes.",
        ["On by default. Turn off for a quick look when you only want the chart "
         "window: ArcGIS Pro's own step of adding layers and redrawing the map "
         "after a tool finishes is what shows as 'updating', and in a heavy map it "
         "can take longer than the tool itself. The outputs and their layer files "
         "are still written and can be added later."],
        []),
    "open_viewer": (
        "Where to show the interactive chart: the chart window or a page in your "
        "web browser.",
        ["The chart window is a separate Python program. It stays open, adds each "
         "run's profiles to its list, and can be reopened later with "
         "SwathViewer.cmd on any saved .swath.npz result.",
         "On a computer where security software lets only approved programs run, "
         "the chart window can be blocked; each attempt then usually shows a notice "
         "and is logged for IT. The tool checks that the window really started, says "
         "so when it did not, opens the browser chart instead and remembers, so it "
         "does not try again on that computer. To use the window there, ask IT to "
         "approve the toolbox's viewer folder, then choose Chart window once.",
         "The browser chart is written on every run beside the saved result, as "
         "<table name>.html in swath_results. It needs no Python and no internet, "
         "and can be sent to someone without ArcGIS: profile list, the swath "
         "envelope with series toggles and side colours, the THi panel, a readout "
         "on hover, a cross-section where you click, zoom, and PNG or SVG export."],
        ["Automatic - the chart window, or the browser chart on a computer where "
         "the window cannot run (default).",
         "Chart window - always try the window, for example after IT has approved "
         "it; still falls back to the browser chart.",
         "Browser page - open the browser chart and never start the window.",
         "Off - show nothing; the browser chart is still written."]),
    "geo_transform": (
        "Datum transformation between the baseline and raster coordinate systems.",
        ["Choose one whenever the list offers any. Without a transformation, arcpy "
         "may apply none and shift the baseline by 100 m or more with no other "
         "symptom."],
        []),
    "max_block_mb": (
        "Memory budget for each block of raster read.",
        ["Default 512 MB. Lower it on a machine with little memory; the tool then "
         "reads the raster in more, smaller pieces. Results are identical."],
        []),
    "bundle_path": (
        "The live result file the chart window watches.",
        ["Overwritten on every run; a permanent copy of each run is also kept under "
         "the project's swath_results folder. Change this only to keep several "
         "viewer windows independent."],
        []),
}
