// Tests for the pure functions of the browser chart (swathprofile/templates/webchart.html).
//
// The page keeps everything that needs no DOM between the SWATH-CORE markers;
// this file runs exactly that block in a fresh Node context, so what is tested
// is what ships.  No dependencies: `node --test tests/js/webchart_core.test.js`.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const TEMPLATE = path.join(__dirname, "..", "..", "swathprofile", "templates", "webchart.html");

function loadCore() {
  const html = fs.readFileSync(TEMPLATE, "utf8");
  const start = html.indexOf("/* SWATH-CORE-START");
  const end = html.indexOf("/* SWATH-CORE-END */");
  assert.ok(start >= 0 && end > start, "the template carries the core markers");
  const sandbox = {
    module: { exports: {} },
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
  };
  vm.createContext(sandbox);
  vm.runInContext(html.slice(start, end), sandbox);
  return sandbox.module.exports;
}

const C = loadCore();
const id = (v) => v;
const arr = (a) => Array.from(a);

test("float32 arrays written by numpy decode exactly, NaN included", () => {
  // numpy: base64.b64encode(np.array([1.5, nan, -2.25, 1e6, 0.1], "<f4").tobytes())
  const got = arr(C.decodeF32("AADAPwAAwH8AABDAACR0Sc3MzD0="));
  assert.equal(got.length, 5);
  assert.equal(got[0], 1.5);
  assert.ok(Number.isNaN(got[1]));
  assert.equal(got[2], -2.25);
  assert.equal(got[3], 1e6);
  assert.equal(got[4], Math.fround(0.1));
  assert.equal(C.decodeF32(null), null);
  assert.equal(C.decodeF32(undefined), null);
});

test("float64 coordinates written by numpy decode exactly", () => {
  // numpy: base64.b64encode(np.array([1500000.123456789, -0.5], "<f8").tobytes())
  const got = arr(C.decodeF64("N92aH2DjNkEAAAAAAADgvw=="));
  assert.deepEqual(got, [1500000.123456789, -0.5]);
  assert.equal(C.decodeF64(null), null);
});

test("at most `cap` rows, spread evenly, always with the edges and the centre", () => {
  assert.deepEqual(arr(C.pickRows(5, 101, 2)), [0, 1, 2, 3, 4]);
  const rows = arr(C.pickRows(1001, 101, 500));
  assert.equal(rows.length, 101);
  assert.equal(rows[0], 0);
  assert.equal(rows[rows.length - 1], 1000);
  assert.ok(rows.includes(500));
  assert.deepEqual(rows, [...rows].sort((a, b) => a - b));
  // A centre that is not on the even grid is added, not substituted.
  const odd = arr(C.pickRows(1000, 101, 499));
  assert.ok(odd.includes(499) && odd.length === 102);
});

test("decimation keeps each pixel column's first, lowest, highest and last point", () => {
  // Ten stations in one pixel column, then ten in the next.
  const xs = Array.from({ length: 20 }, (_, i) => i < 10 ? 0.1 * i : 1 + 0.1 * (i - 10));
  const ys = [5, 9, 1, 7, 3, 8, 2, 6, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const sx = (v) => v;                                   // 1 unit = 1 pixel
  const idx = arr(C.decimate(xs, ys, 0, 0, 19, sx));
  assert.deepEqual(idx, [0, 1, 2, 9, 10, 19]);           // col 0: first 0, high 1, low 2, last 9
  // NoData lifts the pen and is never bridged.
  const gap = arr(C.decimate([0, 1, 2, 3], [1, NaN, 2, 3], 0, 0, 3, sx));
  assert.deepEqual(gap, [0, -1, 2, 3]);
  assert.equal(C.linePathIdx([0, 1, 2, 3], [1, NaN, 2, 3], 0, gap, sx, sx),
               "M0.0 1.0M2.0 2.0L3.0 3.0");
  // The decimated line has exactly the extremes of the full one.
  const n = 50000, big = Float32Array.from({ length: n }, (_, i) => Math.sin(i / 37) + (i % 101) / 50);
  const bx = Float64Array.from({ length: n }, (_, i) => i);
  const sel = arr(C.decimate(bx, big, 0, 0, n - 1, (v) => v / 100));   // 500 pixel columns
  assert.ok(sel.length <= 4 * 501);
  const kept = sel.map((i) => big[i]);
  assert.equal(Math.max(...kept), Math.max(...big));
  assert.equal(Math.min(...kept), Math.min(...big));
});

test("ticks are round numbers inside the range", () => {
  assert.deepEqual(arr(C.niceTicks(0, 2400, 8)), [0, 500, 1000, 1500, 2000]);
  assert.deepEqual(arr(C.niceTicks(-156, 156, 4)), [-100, 0, 100]);
  assert.deepEqual(arr(C.niceTicks(233.12, 233.74, 6)), [233.2, 233.4, 233.6]);
  assert.deepEqual(arr(C.niceTicks(5, 5, 4)), [5]);
  assert.deepEqual(arr(C.niceTicks(NaN, 5, 4)), []);
  assert.deepEqual(arr(C.niceTicks(10, 0, 2)), [0, 5, 10]);   // reversed input
});

test("tick labels show exactly the step's decimals", () => {
  assert.equal(C.stepDecimals(500), 0);
  assert.equal(C.stepDecimals(0.2), 1);
  assert.equal(C.stepDecimals(0.25), 2);
  assert.equal(C.stepDecimals(0.05), 2);
  assert.equal(C.stepDecimals(2.5), 1);
  assert.equal(C.formatTick(-1e-15, 0.2), "0.0");
  assert.equal(C.formatTick(233.4, 0.2), "233.4");
  assert.equal(C.formatValue(NaN), "-");
  assert.equal(C.formatValue(1234.567), "1234.6");
  assert.equal(C.formatValue(12.3456), "12.346");
});

test("nearest station and index ranges", () => {
  const xs = [0, 10, 20, 30];
  assert.equal(C.nearestIndex(xs, 14), 1);
  assert.equal(C.nearestIndex(xs, 15), 1);       // a tie goes to the lower index
  assert.equal(C.nearestIndex(xs, 16), 2);
  assert.equal(C.nearestIndex(xs, -5), 0);
  assert.equal(C.nearestIndex(xs, 99), 3);
  assert.equal(C.nearestIndex([], 1), -1);
  assert.deepEqual(arr(C.indexRange([0, 10, 20, 30, 40], 12, 28)), [1, 3]);
  assert.deepEqual(arr(C.indexRange([0, 10, 20, 30, 40], -1e9, 1e9)), [0, 4]);
  assert.deepEqual(arr(C.indexRange([0, 10, 20, 30, 40], 20, 20)), [1, 2]);
});

test("lines lift the pen at NoData, bands split into gap-free runs", () => {
  assert.deepEqual(arr(C.finiteRuns([[1, NaN, 3, 4, NaN, 6]], 0, 5)).map(arr),
                   [[0, 1], [2, 4], [5, 6]]);
  assert.equal(C.linePath([0, 1, 2, 3], [0, NaN, 2, 3], 0, 3, id, id),
               "M0.0 0.0M2.0 2.0L3.0 3.0");
  assert.equal(C.linePath([0, 1], [5, 6, 7, 8], 0, 1, id, id, 2), "M0.0 7.0L1.0 8.0");
  assert.equal(C.bandPath([0, 1, 2], [0, 0, 0], [1, 1, 1], 0, 2, id, id),
               "M0.0 1.0L1.0 1.0L2.0 1.0L2.0 0.0L1.0 0.0L0.0 0.0Z");
  // A single finite station cannot make a polygon.
  assert.equal(C.bandPath([0, 1, 2], [0, NaN, 0], [1, NaN, 1], 0, 2, id, id), "");
});

test("extents ignore NoData", () => {
  assert.deepEqual(arr(C.extent([[3, NaN, -1], [7, 2, NaN]], 0, 2)), [-1, 7]);
  assert.deepEqual(arr(C.extent([[3, NaN, -1]], 1, 1)), [Infinity, -Infinity]);
  // 2 lines x 3 stations, row-major
  const m = [1, 2, NaN, 9, -4, 5];
  assert.deepEqual(arr(C.matrixExtent(m, 2, 3, 0, 2)), [-4, 9]);
  assert.deepEqual(arr(C.matrixExtent(m, 2, 3, 2, 2)), [5, 5]);
});

test("a cross-section takes one station's column, sorted by offset", () => {
  // 3 lines x 2 stations, row-major; offsets deliberately unsorted.
  const values = [10, 11, 20, 21, 30, 31];
  const pts = arr(C.crossSection(values, [5, -5, 0], 2, 1, [1, -1, 0]));
  assert.deepEqual(pts.map((q) => q.offset), [-5, 0, 5]);
  assert.deepEqual(pts.map((q) => q.z), [21, 31, 11]);
  assert.deepEqual(pts.map((q) => q.side), [-1, 0, 1]);
  assert.deepEqual(pts.map((q) => q.row), [1, 2, 0]);
});

test("kilometres beyond 5000 units, padding of degenerate ranges", () => {
  assert.equal(C.distanceScale(5000).factor, 1);
  assert.equal(C.distanceScale(5001).factor, 1000);
  assert.equal(C.distanceScale(5001).unit, "km");
  assert.deepEqual(arr(C.padRange(0, 10, 0.1)), [-1, 11]);
  assert.deepEqual(arr(C.padRange(100, 100)), [99, 101]);
  assert.deepEqual(arr(C.padRange(NaN, 1)), [0, 1]);
});
