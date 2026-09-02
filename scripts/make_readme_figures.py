# -*- coding: utf-8 -*-
"""Render the README figures from synthetic terrain.  No ArcGIS needed.

Builds a hillslope with a fault scarp, samples it with the package's own
numpy sampler, and draws (1) a single swath chart with the cross-section panel
and (2) a stacked ridgeline of transects across the scarp.  Run with any
Python that has numpy, scipy and matplotlib::

    python scripts/make_readme_figures.py

Outputs ``docs/img/swath_chart.png`` and ``docs/img/stacked.png``.  Replace
them with real screenshots whenever you like; this script just guarantees the
README has images at first publish and that they can be regenerated.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from swathprofile import geometry as geom          # noqa: E402
from swathprofile import plotting                  # noqa: E402
from swathprofile.sampling import GridSpec, sample_block   # noqa: E402
from swathprofile.stats import compute_stats       # noqa: E402
from swathprofile.transforms import cross_section  # noqa: E402

OUT = os.path.join(ROOT, "docs", "img")
CELL = 2.0
NX, NY = 1500, 1000                     # 3 km x 2 km


def terrain():
    """Piedmont sloping north, a NE-trending fault scarp, hills and a channel."""
    x = (np.arange(NX) + 0.5) * CELL
    y = (np.arange(NY) + 0.5) * CELL
    X, Y = np.meshgrid(x, y)
    y_top = NY * CELL
    Yg = y_top - Y                                  # row 0 is the top
    z = 300.0 + 0.03 * Yg                           # regional slope
    fault = Yg - (0.35 * X + 500.0)                 # distance from a NE trace
    z += 6.0 * (1.0 + np.tanh(fault / 25.0)) / 2.0  # 6 m scarp, upthrown to N
    z += 12.0 * np.sin(X / 260.0) * np.cos(Yg / 340.0)   # gentle hills
    z -= 9.0 * np.exp(-((X - 1650.0 - 0.15 * Yg) ** 2) / (2 * 60.0 ** 2))  # channel
    rng = np.random.default_rng(3)
    z += rng.normal(scale=0.15, size=z.shape)       # LiDAR-scale noise
    grid = GridSpec(0.0, y_top, CELL, CELL, NY, NX)
    return z, grid


def swath(block, grid, verts, width, step, n_lines):
    m = geom.station_count(geom.polyline_length(verts), step)
    c, s = geom.resample_polyline(verts, m)
    t, nrm = geom.tangents_normals(c, 21)
    p, offsets = geom.offset_grid(c, nrm, width, n_lines)
    values = sample_block(block, grid, p[..., 0], p[..., 1], method="bilinear")
    st = compute_stats(values, s, offsets=offsets, name="lidar_dem")
    st.z_centre = st.values[n_lines // 2].copy()
    return st, c


def main():
    os.makedirs(OUT, exist_ok=True)
    plotting.use_agg()
    block, grid = terrain()

    # A single swath across the scarp, with the cross-section panel.
    verts = np.array([[1200.0, 200.0], [1500.0, 1800.0]])
    st, _ = swath(block, grid, verts, width=300.0, step=3.0, n_lines=51)
    xs = cross_section(st, int(0.42 * st.n_stations))
    fig, _ = plotting.build_figure(
        {"lidar_dem": st}, title="Swath across a fault scarp - 300 m band",
        cross_section=xs, footer="synthetic terrain; regenerate with "
        "scripts/make_readme_figures.py")
    plotting.save_figure(fig, os.path.join(OUT, "swath_chart.png"), dpi=130)

    # Transects along the trace, stacked.
    trace = np.array([[300.0, 0.35 * 300.0 + 500.0], [2700.0, 0.35 * 2700.0 + 500.0]])
    tr = geom.transects_along(trace, spacing=300.0, length=700.0, smooth_window=21)
    entries = []
    for k, ends in enumerate(tr["ends"]):
        s_k, _ = swath(block, grid, np.asarray(ends), width=120.0, step=3.0, n_lines=41)
        entries.append((f"T{k + 1:03d}", {"lidar_dem": s_k}, None))
    fig, _ = plotting.build_ridgeline_figure(
        entries, series="z_mean", align="min", colour_by="thi",
        title="Transects across the scarp, stacked along the trace",
        footer="synthetic terrain; each profile above its own minimum; "
               "colour = mean THi")
    plotting.save_figure(fig, os.path.join(OUT, "stacked.png"), dpi=130)
    print("wrote", os.path.join(OUT, "swath_chart.png"))
    print("wrote", os.path.join(OUT, "stacked.png"))


if __name__ == "__main__":
    main()
