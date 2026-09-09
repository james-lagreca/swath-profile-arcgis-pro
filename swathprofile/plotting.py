# -*- coding: utf-8 -*-
"""Swath figure construction.  Shared by the static exporter and the viewer.

The palette follows the QGIS plugin so figures stay visually continuous with
published SwathProfiler output.

.. warning::
   This module must never import a GUI backend.  Inside an ArcGIS Pro
   geoprocessing tool the code runs in Pro's own process, and importing an
   interactive backend there can deadlock or crash Pro's UI thread.  The viewer
   selects TkAgg *before* importing this module, in a separate process.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np

__all__ = ["PALETTE", "SERIES", "DEFAULT_SHOW", "PRESETS", "OVERLAY_COLOURS",
           "PROFILE_COLOUR_MODES", "side_labels", "profile_colours",
           "use_agg", "preset_context", "build_figure", "build_overlay_figure",
           "build_ridgeline_figure", "draw_cross_section", "save_figure"]

PALETTE = {
    "profiles": "#C8C8C8",
    # The two sides of the baseline.  A cool/warm pair rather than the usual
    # red/green, so it survives both common forms of colour blindness, and both
    # members are kept clear of the headline series: the blue is lighter than
    # the mean's #01008a, and the sienna is darker than the max line's pale
    # #f8ba81, so neither disappears into a statistic where they overlap.
    "side_left": "#2C6FA8",
    "side_right": "#B4632A",
    "min": "#4dd48a",
    "max": "#f8ba81",
    "mean": "#01008a",
    "median": "#555555",
    "centre": "#111111",
    "quartile": "#008a8c",
    "pband": "#7f5fa8",
    "relief": "#cc3333",
    "thi": "#8B4513",
    "fill": "#666666",
    "grid": "#e0e0e0",
    "parchment": "#f5f5f7",
    "axis": "#cccccc",
    "marker": "#d62728",
}

#: Distinct colours for overlaying several profiles.
OVERLAY_COLOURS = ["#01008a", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                   "#8c564b", "#17becf", "#7f7f7f"]

#: Toggleable series, in draw order.
SERIES = ("profiles", "min", "max", "pband", "q1", "q3", "mean", "median",
          "centre", "relief")

DEFAULT_SHOW = {
    "profiles": True, "min": True, "max": True, "pband": False, "q1": True,
    "q3": True, "mean": True, "median": False, "centre": True, "relief": False,
}

#: Publication presets: figure width in mm (None = screen default) and base
#: font size in points.  PDF/SVG text stays editable (Type 42 / no paths).
PRESETS = {
    "Screen": {"width_mm": None, "font": 9},
    "Journal single column": {"width_mm": 85.0, "font": 8},
    "Journal double column": {"width_mm": 170.0, "font": 8},
    "Poster": {"width_mm": 300.0, "font": 14},
}


#: How the individual offset profiles are coloured.  "single" is the historic
#: flat grey; "side" splits them by which side of the baseline they came from;
#: "ramp" keeps the two side hues but fades them towards the baseline, so the
#: distance out across the swath is readable too.
PROFILE_COLOUR_MODES = ("single", "side", "ramp")

#: Below this circular resultant the baseline bends too much for one compass
#: bearing to describe "the left side" honestly, so the label is left off.
_BEARING_MIN_RESULTANT = 0.97


def side_labels(stats, flip: bool = False):
    """Legend labels for the two sides of the baseline.

    Left and right are relative to the direction the line was digitised, which
    is invisible on a map, so a compass point is appended when the baseline is
    straight enough for one to mean anything: ``("Left (W)", "Right (E)")``.
    On a curved baseline the bare words are returned instead of a bearing that
    would be wrong at both ends.

    The two words are *not* swapped when the display is flipped.  Flipping
    reverses the direction you are looking along the line, so the ground on
    your left changes but the left of the chart does not: ``profile_colours``
    negates the offsets to match, and the left colour keeps the word "Left".
    What does change is the compass point, which turns through 180 degrees - a
    line drawn due north reads "Left (W)" and, flipped, "Left (E)".
    """
    from . import geometry as _geom

    left, right = "Left of line", "Right of line"
    bearing = getattr(stats, "normal_bearing", None)
    resultant = getattr(stats, "normal_resultant", None)
    if bearing is not None and np.isfinite(bearing) \
            and (resultant is None or resultant >= _BEARING_MIN_RESULTANT):
        b = float(bearing) + (180.0 if flip else 0.0)
        lp, rp = _geom.compass_point(b), _geom.compass_point(b + 180.0)
        if lp and rp:
            left, right = f"Left ({lp})", f"Right ({rp})"
    return left, right


def profile_colours(offsets, mode: str = "side", flip: bool = False):
    """Per-profile colours for the individual offset lines.

    Returns a list of colours, one per row of ``stats.values``, in the same
    order.  The centre line (offset 0, present whenever the profile count is
    odd) is the baseline itself and takes the baseline's near-black rather than
    either side's colour.
    """
    from . import geometry as _geom

    o = np.asarray(offsets, dtype=np.float64)
    if flip:
        o = -o
    side = _geom.side_of(o)
    if mode == "ramp":
        span = float(np.nanmax(np.abs(o))) if o.size else 0.0
        if not np.isfinite(span) or span <= 0.0:
            span = 1.0
        out = []
        for value, s in zip(o, side):
            if s == _geom.CENTRE:
                out.append(PALETTE["centre"])
                continue
            base = PALETTE["side_left"] if s == _geom.LEFT else PALETTE["side_right"]
            out.append(_fade(base, abs(float(value)) / span))
        return out
    return [PALETTE["centre"] if s == _geom.CENTRE else
            (PALETTE["side_left"] if s == _geom.LEFT else PALETTE["side_right"])
            for s in side]


def _fade(colour, strength: float):
    """Blend *colour* towards a light neutral; strength 1 keeps it unchanged."""
    from matplotlib.colors import to_rgb

    t = float(np.clip(strength, 0.0, 1.0)) * 0.75 + 0.25
    r, g, b = to_rgb(colour)
    return (r * t + (1 - t) * 0.88, g * t + (1 - t) * 0.88, b * t + (1 - t) * 0.88)


def use_agg():
    """Select the non-interactive backend.  Call before importing pyplot."""
    import matplotlib

    matplotlib.use("Agg")


@contextlib.contextmanager
def preset_context(preset=None):
    """rc settings for a publication preset; a no-op for None / 'Screen'."""
    import matplotlib

    spec = PRESETS.get(preset or "Screen", PRESETS["Screen"])
    font = spec["font"]
    rc = {
        "font.size": font, "axes.titlesize": font + 1, "axes.labelsize": font,
        "xtick.labelsize": font - 1, "ytick.labelsize": font - 1,
        "legend.fontsize": font - 1,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    with matplotlib.rc_context(rc):
        yield spec


def _figsize_for(preset, n_panels, show_thi, show_xs=False):
    spec = PRESETS.get(preset or "Screen", PRESETS["Screen"])
    if spec["width_mm"] is None:
        w = 12.0 + (4.0 if show_xs else 0.0)
    else:
        w = spec["width_mm"] / 25.4
    h_per_panel = 2.6 if spec["width_mm"] is None else max(0.45 * w, 1.6)
    h = h_per_panel * n_panels + (0.4 * h_per_panel if show_thi else 0.0) + 0.8
    return (w, h)


def _style_axes(ax):
    ax.tick_params(direction="in", length=3, width=0.6, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(PALETTE["axis"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.4, alpha=0.7)


def _x_scale(stats):
    span = float(stats.distances.max() - stats.distances.min()) if stats.n_stations else 0.0
    if span > 5000.0:
        return 1000.0, "Distance along baseline (km)"
    return 1.0, "Distance along baseline (map units)"


def _draw_panel(ax, stats, show, y_label, x_scale=1.0, pct=(5, 95),
                distances=None, profile_colour="single", flip=False):
    d = (stats.distances if distances is None else distances) / x_scale
    ax.set_facecolor("white")

    if show.get("profiles") and stats.values is not None and stats.n_total > 1:
        # Individual lines are context, not data to read off: keep them faint.
        # All-cells (zonal) mode has no per-line samples, so stats.values is
        # None there and this whole block, colouring included, is skipped.
        offsets = getattr(stats, "offsets", None)
        usable = (profile_colour in ("side", "ramp") and offsets is not None
                  and len(offsets) == len(stats.values))
        colours = (profile_colours(offsets, profile_colour, flip) if usable
                   else [PALETTE["profiles"]] * len(stats.values))
        for row, colour in zip(stats.values, colours):
            ax.plot(d, row, color=colour, linewidth=0.4, alpha=0.6, zorder=1)
        if usable:
            # The faint lines are too thin to read in a legend, so label them
            # with proxies at a visible weight.
            left, right = side_labels(stats, flip)
            ax.plot([], [], color=PALETTE["side_left"], lw=1.2, label=left)
            ax.plot([], [], color=PALETTE["side_right"], lw=1.2, label=right)

    if show.get("min") and show.get("max"):
        ax.fill_between(d, stats.z_min, stats.z_max, color=PALETTE["fill"],
                        alpha=0.12, linewidth=0, zorder=2)
    if show.get("pband") and stats.z_plo is not None and stats.z_phi is not None:
        ax.fill_between(d, stats.z_plo, stats.z_phi, color=PALETTE["pband"],
                        alpha=0.15, linewidth=0, zorder=2,
                        label=f"P{pct[0]:g}–P{pct[1]:g}")
    if show.get("q1") and show.get("q3"):
        ax.fill_between(d, stats.z_q1, stats.z_q3, color=PALETTE["fill"],
                        alpha=0.10, linewidth=0, zorder=2)

    if show.get("min"):
        ax.plot(d, stats.z_min, color=PALETTE["min"], lw=1.0, label="Min", zorder=3)
    if show.get("max"):
        ax.plot(d, stats.z_max, color=PALETTE["max"], lw=1.0, label="Max", zorder=3)
    if show.get("q1"):
        ax.plot(d, stats.z_q1, color=PALETTE["quartile"], lw=0.9, label="Q1", zorder=3)
    if show.get("q3"):
        ax.plot(d, stats.z_q3, color=PALETTE["quartile"], lw=0.9, ls=(0, (4, 2)),
                label="Q3", zorder=3)
    if show.get("median"):
        ax.plot(d, stats.z_median, color=PALETTE["median"], lw=1.0, ls=(0, (2, 2)),
                label="Median", zorder=4)
    if show.get("mean"):
        ax.plot(d, stats.z_mean, color=PALETTE["mean"], lw=1.4, label="Mean", zorder=5)
    if show.get("centre") and getattr(stats, "z_centre", None) is not None:
        # The profile along the baseline itself - the classic single-line
        # profile, for reading against the band it sits in.
        ax.plot(d, stats.z_centre, color=PALETTE["centre"], lw=1.1,
                label="Baseline", zorder=6)

    ax.set_ylabel(y_label, labelpad=4)
    _style_axes(ax)

    if show.get("relief"):
        rax = ax.twinx()
        rax.plot(d, stats.relief, color=PALETTE["relief"], lw=0.9, zorder=2)
        rax.set_ylabel("Local relief", color=PALETTE["relief"])
        rax.tick_params(axis="y", colors=PALETTE["relief"], direction="in",
                        length=3, width=0.6)
        for side in ("top", "bottom", "left"):
            rax.spines[side].set_visible(False)
        rax.spines["right"].set_color(PALETTE["relief"])
        return rax
    return None


def _draw_thi(ax, stats, enhanced, x_scale=1.0, distances=None, color=None,
              label=None):
    d = (stats.distances if distances is None else distances) / x_scale
    y = stats.thi_star if enhanced else stats.thi
    ax.set_facecolor(PALETTE["parchment"])
    ax.plot(d, y, color=color or PALETTE["thi"], lw=0.9, zorder=3, label=label)
    ax.axhline(50, color="#999999", lw=0.6, ls=(0, (4, 3)), zorder=1)
    for ref in (25, 75):
        ax.axhline(ref, color=PALETTE["grid"], lw=0.4, ls=(0, (1, 3)), zorder=1)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("THi*" if enhanced else "THi", labelpad=4)
    _style_axes(ax)


def draw_cross_section(ax, xs, title=None, profile_colour="single"):
    """Draw an across-track cross-section (from ``transforms.cross_section``).

    With side colouring on, the sample markers take the same two colours as the
    profile lines, so the halves of the cross-section match the halves of the
    main chart.  The offsets in *xs* are already flipped if the display is, so
    no further flip is applied here.
    """
    ax.clear()
    ax.set_facecolor("white")
    if xs is None:
        ax.text(0.5, 0.5, "No per-line samples\n(All-cells mode)",
                ha="center", va="center", transform=ax.transAxes, color="#777777")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    o, z = xs["offsets"], xs["z"]
    order = np.argsort(o)
    ax.plot(o[order], z[order], color=PALETTE["mean"], lw=1.0, zorder=3)
    marker_colour = (profile_colours(o, profile_colour)
                     if profile_colour in ("side", "ramp") else PALETTE["mean"])
    ax.scatter(o, z, s=6, color=marker_colour, zorder=4)
    ax.axhline(xs["z_mean"], color=PALETTE["mean"], lw=0.6, ls=(0, (4, 2)),
               alpha=0.7)
    ax.axhline(xs["z_min"], color=PALETTE["min"], lw=0.6, ls=(0, (2, 2)))
    ax.axhline(xs["z_max"], color=PALETTE["max"], lw=0.6, ls=(0, (2, 2)))
    ax.axvline(0.0, color="#999999", lw=0.5)
    ax.set_xlabel("Across-track offset (map units)")
    ax.set_ylabel("Elevation")
    ax.set_title(title or f"Cross-section at {xs['distance']:,.1f}", fontsize="medium")
    _style_axes(ax)


def build_figure(by_raster, *, title="", show=None, show_thi=True,
                 thi_raster=None, thi_enhanced=False, footer="",
                 figsize=None, dpi=110, figure=None, preset=None,
                 pct=(5, 95), distances=None, cross_section=None,
                 profile_colour="single", flip=False):
    """Build a stacked multi-panel swath figure.

    One panel per raster on a shared X axis, plus an optional THi sub-panel.
    Pass ``figure`` to draw into an existing Figure (the viewer does this);
    otherwise a new one is created.  ``distances`` overrides the x values
    (used for re-zeroed / flipped display).  ``cross_section`` adds a
    right-hand panel from ``transforms.cross_section``.
    """
    from matplotlib.figure import Figure

    show = dict(DEFAULT_SHOW, **(show or {}))
    names = list(by_raster.keys())
    n = len(names)
    if n == 0:
        raise ValueError("Nothing to plot.")

    first = by_raster[names[0]]
    d_first = first.distances if distances is None else distances
    span = float(np.nanmax(d_first) - np.nanmin(d_first)) if first.n_stations else 0.0
    x_scale, x_label = (1000.0, "Distance along baseline (km)") if span > 5000.0 \
        else (1.0, "Distance along baseline (map units)")
    show_xs = cross_section is not None

    with preset_context(preset):
        if figsize is None:
            figsize = _figsize_for(preset, n, show_thi, show_xs)
        fig = figure if figure is not None else Figure(figsize=figsize, dpi=dpi)
        fig.clear()
        fig.patch.set_facecolor("white")

        ratios = [3] * n + ([1] if show_thi else [])
        if show_xs:
            outer = fig.add_gridspec(1, 2, width_ratios=[3, 1.2], wspace=0.25)
            gs = outer[0].subgridspec(len(ratios), 1, height_ratios=ratios,
                                      hspace=0.10)
            xs_ax = fig.add_subplot(outer[1])
        else:
            gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.10)
            xs_ax = None

        axes, ax0 = [], None
        for i, name in enumerate(names):
            ax = fig.add_subplot(gs[i], sharex=ax0) if ax0 is not None \
                else fig.add_subplot(gs[i])
            ax0 = ax0 or ax
            _draw_panel(ax, by_raster[name], show, name, x_scale, pct, distances,
                        profile_colour, flip)
            if i == 0 and title:
                ax.set_title(title, pad=8)
            if i < len(ratios) - 1:
                ax.tick_params(labelbottom=False)
            axes.append(ax)

        if show_thi:
            src = thi_raster if thi_raster in by_raster else names[0]
            thi_ax = fig.add_subplot(gs[-1], sharex=ax0)
            _draw_thi(thi_ax, by_raster[src], thi_enhanced, x_scale, distances)
            axes.append(thi_ax)

        axes[-1].set_xlabel(x_label)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right", frameon=False,
                           ncol=min(len(handles), 6))

        if xs_ax is not None:
            draw_cross_section(xs_ax, cross_section, profile_colour=profile_colour)

        if footer:
            fig.text(0.01, 0.005, footer, fontsize="x-small", color="#888888",
                     va="bottom")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                fig.tight_layout(rect=(0, 0.02, 1, 1))
            except Exception:      # pragma: no cover - advisory only
                pass
    return fig, axes


def build_overlay_figure(entries, *, raster=None, show_thi=True,
                         thi_enhanced=False, envelope=True, title="",
                         footer="", figure=None, preset=None, dpi=110):
    """Overlay several profiles on one axis.

    *entries* is a list of ``(label, by_raster_dict, distances_or_None)``.  The
    mean of each is drawn in a distinct colour with a faint min-max fill; THi
    curves are overlaid below in the same colours.
    """
    from matplotlib.figure import Figure

    if not entries:
        raise ValueError("Nothing to overlay.")

    with preset_context(preset):
        fig = figure if figure is not None else Figure(
            figsize=_figsize_for(preset, 1, show_thi), dpi=dpi)
        fig.clear()
        fig.patch.set_facecolor("white")
        ratios = [3, 1] if show_thi else [1]
        gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.10)
        ax = fig.add_subplot(gs[0])
        ax.set_facecolor("white")
        thi_ax = fig.add_subplot(gs[1], sharex=ax) if show_thi else None

        span = 0.0
        for label, by_raster, distances in entries:
            name = raster if raster in by_raster else next(iter(by_raster))
            s = by_raster[name]
            d = s.distances if distances is None else distances
            span = max(span, float(np.nanmax(d) - np.nanmin(d)))
        x_scale = 1000.0 if span > 5000.0 else 1.0

        for k, (label, by_raster, distances) in enumerate(entries):
            colour = OVERLAY_COLOURS[k % len(OVERLAY_COLOURS)]
            name = raster if raster in by_raster else next(iter(by_raster))
            s = by_raster[name]
            d = (s.distances if distances is None else distances) / x_scale
            if envelope:
                ax.fill_between(d, s.z_min, s.z_max, color=colour, alpha=0.08,
                                linewidth=0, zorder=1)
            ax.plot(d, s.z_mean, color=colour, lw=1.3, label=label, zorder=3)
            if thi_ax is not None:
                _draw_thi(thi_ax, s, thi_enhanced, x_scale, distances,
                          color=colour, label=label)

        ax.set_ylabel(raster or "Elevation")
        _style_axes(ax)
        if title:
            ax.set_title(title, pad=8)
        ax.legend(loc="upper right", frameon=False)
        (thi_ax or ax).set_xlabel(
            "Distance along baseline (km)" if x_scale > 1 else
            "Distance along baseline (map units)")
        if thi_ax is not None:
            ax.tick_params(labelbottom=False)
        if footer:
            fig.text(0.01, 0.005, footer, fontsize="x-small", color="#888888",
                     va="bottom")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                fig.tight_layout(rect=(0, 0.02, 1, 1))
            except Exception:      # pragma: no cover
                pass
    axes = [ax] + ([thi_ax] if thi_ax is not None else [])
    return fig, axes


def build_ridgeline_figure(entries, *, raster=None, series="z_mean", align="min",
                           envelope=True, spacing=None, colour_by="none",
                           first_on_top=True, title="", footer="", figure=None,
                           preset=None, dpi=110):
    """Stacked profiles, one ridge per baseline, in list order up the page.

    *entries* is a list of ``(label, by_raster_dict, distances_or_None)``, in
    the order they should appear - for transects generated along a trace that
    is their order along the trace.  Each profile is drawn relative to its own
    minimum (``align='min'``), its own mean, or in absolute elevation
    (``align='none'``), offset vertically by *spacing* (default 0.6 x the median
    profile range), and later ridges are drawn in front of earlier ones.

    ``series`` names the SwathStats attribute drawn as the ridge line
    (``z_mean``, ``z_centre``, ``z_min``, ``z_max``); with ``envelope`` the
    min-max band is shaded behind it.  ``colour_by`` is ``'none'``, ``'thi'``
    (each ridge coloured by its mean THi) or ``'elevation'`` (by its mean
    elevation).
    """
    import matplotlib
    from matplotlib.figure import Figure

    if not entries:
        raise ValueError("Nothing to stack.")

    items = []
    for label, by_raster, distances in entries:
        name = raster if raster in by_raster else next(iter(by_raster))
        s = by_raster[name]
        d = np.asarray(s.distances if distances is None else distances, dtype=np.float64)
        y = getattr(s, series, None)
        if y is None:
            y = s.z_mean
        y = np.asarray(y, dtype=np.float64)
        lo = np.asarray(s.z_min, dtype=np.float64)
        hi = np.asarray(s.z_max, dtype=np.float64)
        if align == "min":
            ref = np.nanmin(lo)
        elif align == "mean":
            ref = np.nanmean(y)
        else:
            ref = 0.0
        if not np.isfinite(ref):
            ref = 0.0
        thi = float(np.nanmean(s.thi)) if np.isfinite(s.thi).any() else np.nan
        items.append({"label": label, "d": d, "y": y - ref, "lo": lo - ref,
                      "hi": hi - ref, "thi": thi,
                      "elev": float(np.nanmean(y)) if np.isfinite(y).any() else np.nan})

    n = len(items)
    ranges = []
    for it in items:
        top = it["hi"] if envelope else it["y"]
        bot = it["lo"] if envelope else it["y"]
        r = float(np.nanmax(top) - np.nanmin(bot))
        if np.isfinite(r) and r > 0:
            ranges.append(r)
    if spacing is None:
        spacing = 0.6 * float(np.median(ranges)) if ranges else 1.0
    spacing = float(spacing) if spacing and spacing > 0 else 1.0

    span = max(float(np.nanmax(it["d"]) - np.nanmin(it["d"])) for it in items)
    x_scale = 1000.0 if span > 5000.0 else 1.0

    offsets = [((n - 1 - k) if first_on_top else k) * spacing for k in range(n)]

    if colour_by == "thi":
        cmap = matplotlib.colormaps["YlOrRd"]
        vals = [it["thi"] for it in items]
        norm = matplotlib.colors.Normalize(0.0, 100.0)
    elif colour_by == "elevation":
        cmap = matplotlib.colormaps["viridis"]
        vals = [it["elev"] for it in items]
        finite = [v for v in vals if np.isfinite(v)]
        norm = matplotlib.colors.Normalize(min(finite), max(finite)) if finite \
            else matplotlib.colors.Normalize(0, 1)
    else:
        cmap, vals, norm = None, None, None

    with preset_context(preset):
        if figure is None:
            spec = PRESETS.get(preset or "Screen", PRESETS["Screen"])
            w = 12.0 if spec["width_mm"] is None else spec["width_mm"] / 25.4
            h = max(4.0, min(0.55 * n + 2.0, 3.0 * w))
            figure = Figure(figsize=(w, h), dpi=dpi)
        fig = figure
        fig.clear()
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        # Draw from the top ridge down so each lower ridge sits in front.
        order = sorted(range(n), key=lambda k: -offsets[k])
        for z, k in enumerate(order):
            it, off = items[k], offsets[k]
            if cmap is not None and np.isfinite(vals[k]):
                colour = cmap(norm(vals[k]))
            else:
                colour = "#c9d6ea"
            x = it["d"] / x_scale
            if envelope:
                ax.fill_between(x, off + it["lo"], off + it["hi"], color=colour,
                                alpha=0.35, linewidth=0, zorder=2 * z + 1)
            ax.fill_between(x, off, off + it["y"], color=colour, alpha=0.9,
                            linewidth=0, zorder=2 * z + 1)
            ax.plot(x, off + it["y"], color="#222222", lw=0.7, zorder=2 * z + 2)
            ax.axhline(off, color="#bbbbbb", lw=0.4, zorder=0)

        ax.set_yticks(offsets)
        ax.set_yticklabels([it["label"] for it in items])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Distance along profile (km)" if x_scale > 1
                      else "Distance along profile (map units)")
        rel = {"min": "above each profile's minimum", "mean": "about each profile's mean",
               "none": "absolute elevation"}.get(align, "")
        ax.set_ylabel(f"{spacing:g} units between baselines; {rel}", fontsize="small")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.4, alpha=0.7)
        if title:
            ax.set_title(title, pad=8)
        if cmap is not None:
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("Mean THi" if colour_by == "thi" else "Mean elevation")
        if footer:
            fig.text(0.01, 0.005, footer, fontsize="x-small", color="#888888",
                     va="bottom")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                fig.tight_layout(rect=(0, 0.02, 1, 1))
            except Exception:      # pragma: no cover
                pass
    return fig, [ax]


def save_figure(fig, path, dpi=200):
    """Write a figure, choosing the writer from the extension."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if fig.canvas is None or not hasattr(fig.canvas, "print_figure"):
        FigureCanvasAgg(fig)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path
