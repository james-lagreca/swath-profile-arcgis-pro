# -*- coding: utf-8 -*-
"""Sampling mask: which samples and raster cells lie inside a set of polygons.

Pure numpy - no arcpy - so it is tested in CI.  :func:`features.read_mask`
reads the user's polygons, unions them and hands the rings over in the working
coordinate system; the engine asks this module which offset-line samples to
drop, and the zonal sweep asks which raster cells to drop.

The rule
--------
One crossing rule answers both questions.  A ray is cast east from the point,
and the number of polygon edges it crosses decides inside (odd) from outside
(even).  An edge counts when the point's ``y`` lies in the half-open range
``[y_low, y_high)`` of the edge and the crossing lies *strictly* east of the
point.  In words: a point exactly on the boundary is classified as if nudged an
infinitesimal step east, and north at a vertex's height.  Two polygons that
share an edge therefore never both claim a point on it, and the answer does not
depend on the direction the rings were digitised in.

Even-odd makes holes free (a hole is just more crossings), but it also means two
*separate* polygons would cancel where they overlap.  The rings handed to this
module come from a union, so they never overlap; :class:`SampleMask` does not
union anything itself.

Points and cells agree bitwise
------------------------------
:meth:`SampleMask.inside` tests arbitrary points;
:meth:`SampleMask.inside_cells` tests every cell centre of a raster block with a
scanline fill.  Both compute the crossing ``x`` with the same expression from
the same operands and compare it the same way, so a point placed at a cell
centre gets exactly the answer the cell gets.  The cell centres come from
:meth:`GridSpec.cell_centres`, which is anchored on the raster's own origin, so
a cell's answer does not depend on which block it was read in.

There is deliberately no prefilter on the polygon's x-range: the crossing ``x``
can round one ulp beyond the vertices' x-range, and a prefilter would then give
a point a different answer from the cell it sits on.  The y-range prefilter is
exact, because it compares the same numbers the crossing rule does.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MODES", "SampleMask", "densify"]

MODES = ("keep", "exclude")

#: Points tested per chunk.  Sorting each chunk by y dominates the cost; the
#: chunk only bounds memory.
_CHUNK = 4_000_000


def _open_ring(ring) -> np.ndarray:
    """(K, 2) float64 vertices with any repeated closing vertex removed."""
    v = np.asarray(ring, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(f"A ring must be a (K, 2) array of vertices; got shape {v.shape}.")
    if len(v) > 1 and v[0, 0] == v[-1, 0] and v[0, 1] == v[-1, 1]:
        v = v[:-1]
    return v


def _ranges(lo, hi):
    """Concatenation of ``arange(lo[i], hi[i])`` for every i, without a loop."""
    counts = hi - lo
    first = np.cumsum(counts) - counts
    return np.arange(int(counts.sum()), dtype=np.int64) + np.repeat(lo - first, counts)


def densify(rings, max_seg: float):
    """Insert vertices so that no edge is longer than *max_seg*.

    The original vertices are kept exactly, and each edge of length ``L`` gains
    ``ceil(L / max_seg) - 1`` evenly spaced points.  Used before moving a mask
    into another coordinate system, where a straight edge becomes a curve.
    Returns open rings (no repeated closing vertex).
    """
    max_seg = float(max_seg)
    if not np.isfinite(max_seg) or max_seg <= 0.0:
        raise ValueError(f"max_seg must be a positive distance; got {max_seg}.")
    out = []
    for ring in rings:
        v = _open_ring(ring)
        if len(v) < 2:
            out.append(v.copy())
            continue
        nxt = np.roll(v, -1, axis=0)
        seg = np.hypot(nxt[:, 0] - v[:, 0], nxt[:, 1] - v[:, 1])
        pieces = np.maximum(np.ceil(seg / max_seg), 1.0).astype(np.int64)
        edge = np.repeat(np.arange(len(v)), pieces)
        first = np.repeat(np.cumsum(pieces) - pieces, pieces)
        t = (np.arange(int(pieces.sum())) - first) / np.repeat(pieces, pieces)
        # t == 0 reproduces each original vertex exactly: v + d * 0.0 == v.
        out.append(v[edge] + (nxt[edge] - v[edge]) * t[:, None])
    return out


class SampleMask:
    """Polygon rings in one coordinate system, and what to do with them.

    *rings* is a sequence of (K, 2) vertex arrays - outer rings and holes alike,
    because the rule is even-odd - none of which overlaps another polygon.
    *mode* is ``"keep"`` (drop everything outside) or ``"exclude"`` (drop
    everything inside).  Rings with fewer than three distinct vertices are
    ignored; a non-finite vertex is an error, because it means part of the
    polygon could not be placed in this coordinate system.
    """

    def __init__(self, rings, mode: str = "keep"):
        if mode not in MODES:
            raise ValueError(f"Mask mode must be one of {MODES}; got {mode!r}.")
        clean = []
        for ring in rings:
            v = _open_ring(ring)
            if not np.isfinite(v).all():
                raise ValueError(
                    "A mask polygon has a vertex that is not a finite coordinate, so "
                    "it could not be placed in this coordinate system.")
            if len(v) >= 3:
                clean.append(v)
        self.mode = mode
        self.rings = tuple(clean)
        if clean:
            a = np.concatenate(clean)
            b = np.concatenate([np.roll(v, -1, axis=0) for v in clean])
            up = a[:, 1] <= b[:, 1]
            low = np.where(up[:, None], a, b)
            high = np.where(up[:, None], b, a)
            sloped = low[:, 1] < high[:, 1]          # horizontal edges never cross
            low, high = low[sloped], high[sloped]
            self._ylo = np.ascontiguousarray(low[:, 1])
            self._yhi = np.ascontiguousarray(high[:, 1])
            self._x0 = np.ascontiguousarray(low[:, 0])     # x at the low end
            self._inv = np.ascontiguousarray(
                (high[:, 0] - low[:, 0]) / (high[:, 1] - low[:, 1]))
            self.bbox = (float(a[:, 0].min()), float(a[:, 1].min()),
                         float(a[:, 0].max()), float(a[:, 1].max()))
            self.perimeter = float(np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]).sum())
        else:
            empty = np.zeros(0, dtype=np.float64)
            self._ylo = self._yhi = self._x0 = self._inv = empty
            self.bbox = None
            self.perimeter = 0.0

    def __repr__(self):
        return (f"SampleMask(mode={self.mode!r}, rings={len(self.rings)}, "
                f"vertices={self.n_vertices}, edges={self._ylo.size})")

    @property
    def is_empty(self) -> bool:
        """No edges at all: nothing is inside."""
        return self._ylo.size == 0

    @property
    def n_vertices(self) -> int:
        return int(sum(len(r) for r in self.rings))

    # -- the crossing rule ----------------------------------------------------

    def inside(self, x, y) -> np.ndarray:
        """Boolean array, True where the point (x, y) is inside the polygons.

        Points with a non-finite coordinate are never inside.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(f"x and y shapes differ: {x.shape} vs {y.shape}.")
        flat = np.zeros(x.size, dtype=bool)
        if self.is_empty or x.size == 0:
            return flat.reshape(x.shape)
        xf, yf = x.ravel(), y.ravel()
        with np.errstate(invalid="ignore"):
            cand = np.flatnonzero(np.isfinite(xf) & np.isfinite(yf) &
                                  (yf >= self._ylo.min()) & (yf < self._yhi.max()))
        for a in range(0, cand.size, _CHUNK):
            idx = cand[a:a + _CHUNK]
            idx = idx[np.argsort(yf[idx], kind="stable")]
            ys, xs = yf[idx], xf[idx]
            # Points s..t-1 satisfy y_low <= y < y_high for edge e.
            lo = np.searchsorted(ys, self._ylo, side="left")
            hi = np.searchsorted(ys, self._yhi, side="left")
            par = np.zeros(idx.size, dtype=bool)
            for e in np.flatnonzero(hi > lo):
                s, t = lo[e], hi[e]
                par[s:t] ^= (self._x0[e] + (ys[s:t] - self._ylo[e]) * self._inv[e]) > xs[s:t]
            flat[idx] = par
        return flat.reshape(x.shape)

    def inside_cells(self, grid) -> np.ndarray:
        """(n_rows, n_cols) boolean array: True where the cell centre is inside.

        A scanline fill of the block described by *grid* (a
        :class:`~swathprofile.sampling.GridSpec`): every edge contributes one
        crossing per row centre it spans, and each crossing flips the cells
        west of it.  No per-cell coordinate array is built, which matters for
        blocks of tens of millions of cells.
        """
        n_rows, n_cols = int(grid.n_rows), int(grid.n_cols)
        if self.is_empty or n_rows == 0 or n_cols == 0:
            return np.zeros((n_rows, n_cols), dtype=bool)
        xs, ys = grid.cell_centres()
        ys_asc = ys[::-1]                  # north-up: y falls as the row index rises
        lo = np.searchsorted(ys_asc, self._ylo, side="left")
        hi = np.searchsorted(ys_asc, self._yhi, side="left")
        live = np.flatnonzero(hi > lo)
        if live.size == 0:
            return np.zeros((n_rows, n_cols), dtype=bool)
        e = np.repeat(live, (hi - lo)[live])
        i = _ranges(lo[live], hi[live])
        xint = self._x0[e] + (ys_asc[i] - self._ylo[e]) * self._inv[e]
        # Cells [0, k) have centres strictly west of the crossing, the same
        # strict comparison inside() makes.
        k = np.searchsorted(xs, xint, side="left").astype(np.int64)
        r = (n_rows - 1) - i
        width = n_cols + 1
        u, counts = np.unique(r * width + k, return_counts=True)
        tog = np.zeros(n_rows * width, dtype=np.uint8)
        tog[u] = (counts & 1).astype(np.uint8)
        tog = tog.reshape(n_rows, width)
        tog[:, 0] ^= (np.bincount(r, minlength=n_rows) & 1).astype(np.uint8)
        return np.bitwise_xor.accumulate(tog, axis=1)[:, :n_cols].astype(bool)

    # -- what gets dropped ----------------------------------------------------

    def drops(self, x, y) -> np.ndarray:
        """True where a sample at (x, y) is removed by the mask.

        A sample whose position is not finite is never reported as dropped: it
        has no value to remove, and counting it would distort the coverage.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ins = self.inside(x, y)
        finite = np.isfinite(x) & np.isfinite(y)
        return finite & (~ins if self.mode == "keep" else ins)

    def drops_cells(self, grid) -> np.ndarray:
        """True where a raster cell of *grid* is removed by the mask."""
        ins = self.inside_cells(grid)
        return ~ins if self.mode == "keep" else ins

    def drops_all_in_box(self, x0, y0, x1, y1, pad: float = 0.0) -> bool:
        """True when everything in the box is certainly dropped.

        Only ever true in keep mode: when the mask is empty, or when the box,
        widened by *pad* on every side, misses the mask's bounding box.  Used to
        skip raster reads that could not contribute anything.  Exclude mode
        always answers False: proving a box lies wholly inside the polygons is
        not worth the work just to save a read.
        """
        if self.mode != "keep":
            return False
        if self.is_empty:
            return True
        pad = abs(float(pad))
        bx0, by0, bx1, by1 = self.bbox
        return bool(x1 + pad < bx0 or x0 - pad > bx1 or y1 + pad < by0 or y0 - pad > by1)

    def transformed(self, fn, max_seg=None) -> "SampleMask":
        """The same mask with every vertex passed through ``fn((K, 2)) -> (K, 2)``.

        With *max_seg*, edges are densified first, so that a straight edge in
        this coordinate system stays close to its true path in the other.  All
        rings go through *fn* in one call.
        """
        rings = densify(self.rings, max_seg) if max_seg else [r.copy() for r in self.rings]
        if not rings:
            return SampleMask([], self.mode)
        lengths = [len(r) for r in rings]
        moved = np.asarray(fn(np.concatenate(rings)), dtype=np.float64)
        if moved.shape != (sum(lengths), 2):
            raise ValueError("The coordinate transformation returned the wrong number "
                             "of mask vertices.")
        if not np.isfinite(moved).all():
            raise ValueError("Part of the sampling mask could not be placed in the "
                             "raster's coordinate system.")
        return SampleMask(np.split(moved, np.cumsum(lengths)[:-1]), self.mode)
