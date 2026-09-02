# Algorithm and deviations from the reference implementations

## Lineage

| | |
|---|---|
| Method | Pérez‑Peña, J.V., Al‑Awabdeh, M., Azañón, J.M., Galve, J.P., Booth‑Rea, G. & Notti, D. (2017). *SwathProfiler and NProfiler: Two new ArcGIS Add‑ins for the automatic extraction of swath and normalized river profiles.* Computers & Geosciences **104**, 135–150. |
| Original code | `geolovic/swathprofiler_arcgis` — VB.NET / ArcObjects ArcMap add‑in (ArcMap 10.x only) |
| Python re‑implementation | `OpenQGIS/SwathProfile4QGIS` — QGIS 3.16+ plugin, `swath_core.py` |
| This tool | arcpy / numpy port for ArcGIS Pro 3.x |

ArcObjects does not exist in ArcGIS Pro, so the VB.NET original cannot be ported
directly; it serves as the definition of the method. The QGIS plugin is the
Python statement of the same algorithm and is the direct source for this port.

## Definitions

For each station `j` along the baseline, over the `N` samples across the band:

```
relief_j = max_j − min_j
HI_j     = (mean_j − min_j) / relief_j                    hypsometric integral
factor_j = 0.2 · ln(relief_j / max_relief) + 1
THi_j    = ((HI_j − 0.5) · factor_j + 0.5) · 100
THi*_j   = as THi, with HI first rescaled to clip((HI − 0.2)/0.6, 0, 1)
```

`max_relief` is the maximum of `relief` over the whole profile, so THi is a
*relative* measure: it expresses how the hypsometry at a station compares with
the rest of that profile, damped where the local relief is small.

Quartiles use the **position‑based (Weibull) convention** of the original:
interpolate at virtual index `q·(n+1)`, clamped at both ends. This is *not*
numpy's default (`linear`, type 7), and the difference is real — the tool
deliberately preserves the reference convention so results stay comparable with
published SwathProfiler output.

## Sampling modes

**Offset lines (mode A, default).** `N` parallel lines are laid across the band
and the raster sampled at `M` stations along each. Only this mode produces the
individual profile curves, and it is the mode the published method describes.

**Zonal slices (mode B).** Every raster cell within `width/2` of the baseline
contributes to the station nearest to it. For a straight baseline that is
exactly the classic `step × width` rectangle per station; on a bend the band is
partitioned between stations with no overlap and no gaps, where rectangles would
double‑count the inside and leave wedges on the outside. The first and last
stations are clipped to the half‑step they own, so the band does not extend
beyond the line's ends. Nothing between the profile lines is missed, but there
are no individual curves.

Implementation note: assignment is a Euclidean distance transform on each raster
block (O(cells)), followed by a refinement that projects each cell onto the local
tangent. The refinement is not optional. Station positions are quantised to cell
centres for the transform, and far from the line that is enough to pick the
wrong station — at distance *u* the true distances to stations *j* and *j+k*
differ by only (k·step)²/2u, which half a cell of quantisation noise exceeds for
*k* up to √(u·cell)/step (≈14 stations at u = 150 m on a 0.5 m grid). The
earlier per‑station rectangle test was correct but O(bbox cells × stations),
which on an oblique 2 km band over 0.5 m LiDAR meant ~10¹¹ operations.

One definitional difference between the modes shows in the extremes: a zonal
slice has a finite along‑track length (`step`) where an offset line has none,
so where the surface also slopes *along* the baseline, zonal `Z_MAX`/`Z_MIN`
reach up to `along‑slope × step/2` beyond the offset‑line values. Means are
unaffected (the slice is symmetric about the station).

Two discrete effects are worth knowing when reading `N_VALID` in zonal mode.
Cells form a lattice, so the number caught by a slice of width `step` alternates
between ⌊step/lattice⌋ and ⌈step/lattice⌉ rows along the profile (2 vs 3 rows
for a 0.75 m step on a 0.5 m grid at 45°); the statistics are unaffected, but
the count is not constant. And a whole row can sit exactly on the boundary
between two stations. Ties are resolved with a deterministic half‑open interval
and a bias larger than the float drift of `along` across a row, so the row goes
wholly to one station — resolving them by rounding splits the row between the
stations by floating‑point noise, one side of the band to each, which biases
both means.

### Why not `ZonalStatisticsAsTable` for mode B

Two independent disqualifying reasons, plus a licence cost:

1. It cannot produce a Weibull‑position quartile — it offers MIN/MAX/RANGE/
   MEAN/STD/SUM/MEDIAN, not an arbitrary `q·(n+1)` order statistic. The QGIS
   plugin works around this by **zero‑filling Q1, Q3 and THi in zonal mode**, so
   half the chart is fabricated (see D17).
2. It rasterises zones and does not support **overlapping** polygons — and on the
   inside of any bend, adjacent slices genuinely overlap. Silent last‑wins
   clipping would corrupt exactly the curved baselines that matter most.
3. It requires Spatial Analyst. The numpy implementation requires no extension.

## Deviation: normal offsets, not `offsetCurve`

arcpy has no offset‑curve operation, so offsets are constructed from unit
normals: `P[i,j] = C[j] + o_i · n[j]`.

This is not merely a substitute — it is more consistent than what the reference
does. QGIS builds each offset line with `QgsGeometry.offsetCurve(d)` and then
samples it at `n_points` evenly spaced along *that curve's own* length. On a bend
the outer curve is longer than the centreline, so its `j`‑th sample sits at a
different along‑track position than the centreline's `j`‑th sample — yet the
statistics `vstack` all of them into one array indexed by a single `distances`
vector. **The reference's data model already assumes perpendicular alignment; its
implementation does not provide it.** GEOS join styles and dropped
self‑intersection loops make the offset curve's length, and therefore the sample
spacing, neither documented nor reproducible.

**Consequence:** results match the QGIS plugin exactly on straight baselines and
differ measurably on curved ones. That is expected, and is not a port bug.

### Fold detection

For a parallel (normal‑offset) curve the local area element scales by the
Jacobian `J = 1 − o·κ`. Where `J ≤ 0` the offset point has crossed the centre of
curvature and the band has folded onto itself, so that sample would double‑count
inside‑of‑bend terrain. The test is exact and per‑sample:

```
folded[i, j] = (1 − offsets[i] · κ[j]) ≤ 0
```

Under the default `fold_policy = mask` those samples become NoData, reducing
`N_VALID` rather than corrupting the statistics. `KAPPA` and `FOLDED` are written
to the output table, and the minimum radius of curvature is reported.

Curvature uses `np.gradient(..., edge_order=2)`. With the default first‑order
one‑sided difference the second derivative at the first and last station comes
out at **half** its true value, which would let a genuinely folded profile end
slip through the check.

## Deviation: tangent smoothing, not centreline smoothing

Optional stabilising smooths the *tangent direction*, never the centreline.
Smoothing the centreline would change the reported along‑track distances and
displace the profile off the feature the user digitised. Smoothing the tangent
only stabilises the cross‑section azimuth, which is the actual problem with
hand‑digitised baselines.

## Coordinate systems

Swath width, step and along‑track distance are ground lengths, so the geometry
is always built in a projected **working CRS**. It is chosen in this order:

1. the first raster that is projected — so the common case (a projected DEM)
   involves no reprojection at all;
2. otherwise the baseline's CRS, if projected;
3. otherwise a UTM zone selected from the baseline's centre, with a warning.

Baselines are read into the working CRS by the cursor (`spatial_reference=`).
A raster in a different CRS is *not* reprojected — that would resample it. Instead
the sample points are transformed into the raster's own CRS and the raster is
sampled there. For All‑cells mode the station centres, the (stabilised)
along‑track direction and the half‑width vector are transformed, and the sweep
runs in the raster's CRS with across‑track position measured as a fraction of
the local half‑width vector — so a lat/lon raster takes the same fast path as a
projected one. (An earlier version fell back to a per‑station quadrilateral test
here, which on a national DEM took twenty minutes.)

Point transformation goes through one WKT `MULTIPOINT` round trip per grid,
which is ~20× faster than building Point objects. arcpy multipoints silently
**merge coincident points**, which would break the one‑to‑one mapping, so only
the unique coordinates are sent and the count is asserted on return.

The datum transformation, if any, is the user's choice under *Advanced*; the
tool warns when one is available but unselected, because the failure mode — a
100–200 m horizontal shift — has no other visible symptom.

## Raster index conventions

Verified empirically against ArcGIS Pro 3.4, not taken from the documentation:

* `RasterToNumPyArray` returns **row 0 as the topmost row**, while its
  `lower_left_corner` argument is in lower‑left world coordinates. Two opposite
  conventions in one call.
* `lower_left_corner` is snapped by **rounding to the nearest cell**, not
  flooring (a 0.4‑cell offset reads the floored block; 0.6 reads the next one).
  The tool therefore always passes an exactly cell‑aligned corner and asserts the
  returned shape.
* A window extending past the raster edge **does not raise** — it returns the
  requested shape padded with the NoData sentinel, which must be masked.
* `nodata_to_value=np.nan` raises `ValueError` on integer rasters, which take a
  separate sentinel path.
* `noDataValue` can be **`None` even when the raster contains NoData cells**
  (observed for both S32 and Float32 rasters written from numpy). `nodata_to_value`
  is therefore always passed explicitly rather than trusting that property.

Index arithmetic is anchored on the top‑left cell edge:

```
fc = (x − x_min_edge)/cell_w − 0.5          fr = (y_max_edge − y)/cell_h − 0.5
```

computed against the **source raster's** origin and then shifted by the integer
block offset. Both steps are exact, so results do not depend on how the read was
tiled. Anchoring on the block's own origin instead makes tiled and untiled reads
differ in the last ULP, because `x − x_min_edge` subtracts a different large
number (NZTM eastings are ~1.5 × 10⁶, northings ~5 × 10⁶).

Nearest‑neighbour uses `floor(f + 0.5)`, not `np.rint`, which is
round‑half‑to‑even and disagrees with GDAL/ArcGIS at exact half‑cell boundaries.

Coordinates are always `float64`. Southern‑hemisphere UTM and NZTM northings
reach 5–10 × 10⁶ m, which `float32` quantises to about 1 m.

## Defects fixed

Each is regression‑tested. `Legacy (ArcGIS-compatible) statistics` restores D3
and D4 for reproducing previously published numbers.

| # | Reference behaviour | This tool |
|---|---|---|
| D1 | Quartiles use `np.sort`, which is **not** NaN‑aware, while still dividing by the full `n`. Any NoData mis‑ranks Q1/Q3; Q3 can land in the NaN region and return NaN. | `np.nanquantile(..., method='weibull')`, which reduces `n` per station. Verified equivalent to the reference formula to **2.8 × 10⁻¹⁴** over n = 2…60. *Measured impact: with the top 10 of 50 samples as NoData, true Q1 = 10.25; the reference returns 12.75.* |
| D2 | `np.nanmin`/`nanmean` leak `RuntimeWarning` on all‑NoData stations. | Suppressed deliberately; `N_VALID` exposed as a column; `min_valid` parameter. |
| D3 | HI guards a zero denominator with `1e-5`, returning **HI = 0** for flat ground — dragging THi down 50 points at every lake and valley floor. The constant is also unit‑dependent (m vs ft). | `HI = Null` where relief ≤ `max(1e-9, 1e-6·max_relief)` — scale‑relative. Configurable: Null / 0.5 / Reference. |
| D4 | `factor` goes **negative** below `relief/max_relief = e⁻⁵ ≈ 0.0067`, *inverting* THi about 50, and → −∞ as the ratio → 0. The `1e-10` floor guards only the `relief ≤ 0` branch, so small positive relief passes through; even the floored value gives `factor = −3.6`, i.e. THi ∈ [−130, +230]. | `factor` clamped at 0, so negligible‑relief stations read the neutral THi = 50. Both `THI` (clamped) and `THI_RAW` (verbatim) are written. |
| D5 | `1e-10` ratio floor fabricates an extreme value. | Null where relief = 0. |
| D6 | `max_relief == 0` divides by zero. | All THi Null, with a warning. |
| D7 | `n_lines == 1` gives relief ≡ 0, so HI ≡ 0 silently. | Validation warning; HI/THi Null, not 0. |
| D8 | `abs(v) < 1e10` as a validity test — catches float sentinels, misses −9999 and −32768 entirely. | Dtype‑exact sentinel masking plus `isfinite`. |
| D9 | `linspace(−W/2, W/2, 50)` is symmetric but **never samples the centreline**. | 50 kept as the default for fidelity; informational message on even counts. |
| D10 | Cross‑track under‑sampling is silent. At W = 2000 m and N = 50 the spacing is 40.8 m — on a 10 m DEM about 75 % of the band's cells never contribute. | Warning quantifying the fraction skipped, with a suggested N. |
| D11 | Savitzky–Golay crashes when `window > n`; `np.interp` extrapolates *constant* across leading/trailing NaN runs. | Both guarded; opt‑in `dilate` widens the NaN mask by `window//2` so values computed partly from interpolated data are not presented as measurements. |
| D12 | Silently falls back to a box moving average when scipy is missing, changing the algorithm with no indication. | Hard import; fails loudly. |
| D13 | `smoothed()` smooths the input lines then re‑reduces, which is **not** the same as smoothing the output curves — `min(smooth(y)) ≥ smooth(min(y))`, so the band narrows. | Reference behaviour kept as the default, and documented. |
| D14 | A baseline shorter than one step silently yields a 2‑station profile. | Error naming the OID and both lengths. |
| D15 | No median. | `Z_MEDIAN` added, same convention as Q1/Q3. |
| D16 | Multipart lines: `parts[0]` is used and **the rest silently discarded**. | Each part profiled separately with its own `PART_ID`; never joined, which would invent a connecting segment. |
| D17 | Zonal mode zero‑fills Q1/Q3/THi. | Genuine percentiles from the full cell population. |

## Percentiles

Alongside the reference quartiles, a configurable percentile pair (default
5/95) is reported as `Z_PLO`/`Z_PHI`, computed with the same Weibull convention
so all five order statistics are mutually consistent. On LiDAR the raw min/max
are set by single cells (a tree, a car, a drop‑out); the percentile band is the
robust envelope.

## Display transforms (viewer)

All in `swathprofile/transforms.py`, applied to a `SwathStats` copy, never to
the stored result.

- **Linear detrend**: a straight line is fitted to the mean profile against
  distance (`np.polyfit`, degree 1) and the *same* trend is subtracted from every
  elevation series and from the raw samples. The envelope keeps its shape;
  only the regional slope goes. Relief, HI and THi are shift‑invariant and are
  not touched.
- **Relative to minimum**: every series minus the minimum profile.
- **Re‑zero / flip**: `d' = ±(d − origin)`. Flipping also negates the
  cross‑track offsets so a cross‑section mirrors to match the viewing
  direction. Arrays are never reordered, so station indices stay stable.
- **Cross‑section**: column `j` of the raw `(N × M)` samples against the signed
  offsets. Available in offset‑line mode only; All‑cells keeps no per‑line
  samples.
- **Measure**: Δd, Δz, gradient, angle between two chart points read from the
  chosen series.

## Transects

`geometry.transects_along` densely resamples the trace, applies the same
direction stabilising as a baseline, and places a transect every `spacing`
along it from `start`, each `length` long, centred on the trace or extending to
one side. Transect direction is the trace's left normal, so `AZIMUTH` is the
bearing from the right end to the left end, clockwise from north. On a
circular trace the transects are exactly radial (tested). Transects are then
ordinary baselines: OIDs from 2 000 000, names `T001…`.

## Known limitations

* Curved‑baseline results differ from the QGIS plugin by design (above).
* No live map‑linked scan line. Driving the Pro map from the external viewer
  process would need a CIM/IPC channel that does not exist for Python; that
  feature genuinely requires the .NET Pro SDK.
* The band polygon self‑intersects where the swath folds. This is deliberate —
  the map should show the region actually sampled, not a tidied buffer.
* `arcpy.charts` cannot draw a filled envelope, so the native chart is a
  convenience only; matplotlib is the publication path.
