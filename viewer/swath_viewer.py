# -*- coding: utf-8 -*-
"""Interactive swath profile viewer.

Runs as a **separate process**, launched by the geoprocessing tool.  That is not
a stylistic choice: ArcGIS Pro owns its process's UI thread, so starting a Tk
main loop inside a running tool freezes or crashes Pro, and Qt is not present in
the arcgispro-py3 environment, so QtAgg is not an option either.

The window stays open across runs.  It polls its bundle file and, when the tool
writes a new result, loads it and adds the new profiles to the list - so the
loop "draw a new line, Run" refreshes this window rather than opening another.
While alive it writes a heartbeat file so the tool knows not to start a second
copy.

All profile mathematics (detrending, re-zeroing, cross-sections, measurement)
lives in ``swathprofile.transforms``; this file is only the Tk shell.

Usage::

    pythonw.exe swath_viewer.py [bundle.swath.npz]
"""

from __future__ import annotations

import csv
import datetime
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# High-DPI: without this Tk renders blurry and undersized on scaled displays.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import (          # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from matplotlib.figure import Figure                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402

from swathprofile import plotting, transforms             # noqa: E402
from swathprofile.bundle import load_bundle               # noqa: E402
from swathprofile.smoothing import SMOOTH_PRESETS         # noqa: E402
from swathprofile.stats import compute_stats              # noqa: E402

SERIES = [("profiles", "Profile lines"), ("min", "Minimum"), ("max", "Maximum"),
          ("pband", "Percentile band"), ("q1", "Q1"), ("q3", "Q3"),
          ("mean", "Mean"), ("median", "Median"), ("centre", "Baseline profile"),
          ("relief", "Relief (right axis)")]
DETREND_MODES = ["None", "Linear detrend", "Relative to minimum"]
#: How the individual profile lines are coloured.  "Side of line" answers
#: "which side of the baseline did this profile come from"; "Side + distance"
#: keeps the two side colours but fades them in towards the baseline, so how
#: far out across the swath a profile sits is readable as well.
PROFILE_COLOUR = {"Single colour": "single", "Side of line": "side",
                  "Side + distance": "ramp"}


def _mask_words(mode) -> str:
    return {"keep": "keep only inside", "exclude": "exclude inside"}.get(mode, str(mode))


def _stats_settings(config) -> dict:
    """The run's statistics settings, as compute_stats keyword arguments.

    Re-smoothing in the viewer re-reduces the raw sample lines, and it must use
    the settings the tool used: without Minimum Valid Samples, for one, the
    slivers a run left out at the edge of a sampling mask would come back the
    moment a smoothing preset is chosen.  Older bundles lack some keys and get
    compute_stats' own defaults, which are the tool's defaults.
    """
    cfg = config or {}
    out = {}
    for key, name, cast in (("min_valid", "min_valid", int),
                            ("legacy_stats", "legacy", bool),
                            ("flat_hi", "flat_hi", str),
                            ("thi_clip_factor", "thi_clip_factor", bool),
                            ("thi_clip_range", "thi_clip_range", bool),
                            ("quantile_method", "quantile_method", str),
                            ("smooth_dilate", "smooth_dilate", bool)):
        if cfg.get(key) is not None:
            out[name] = cast(cfg[key])
    return out
STACK_SERIES = {"Mean": "z_mean", "Baseline": "z_centre", "Min": "z_min",
                "Max": "z_max", "Median": "z_median"}
STACK_ALIGN = {"Own minimum": "min", "Own mean": "mean", "Absolute": "none"}
STACK_COLOUR = {"None": "none", "Mean THi": "thi", "Elevation": "elevation"}
MEASURE_SERIES = ["Mean", "Baseline", "Min", "Max", "Q1", "Q3", "Median"]
_SERIES_ATTR = {"Mean": "z_mean", "Baseline": "z_centre", "Min": "z_min",
                "Max": "z_max", "Q1": "z_q1", "Q3": "z_q3", "Median": "z_median"}

POLL_MS = 1500
HEARTBEAT_MS = 2000


class Entry:
    """One baseline part from one loaded bundle, plus its view state."""

    def __init__(self, part, meta, loaded_at, source):
        self.part = part
        self.meta = meta
        self.loaded_at = loaded_at
        self.source = source
        self.origin = 0.0
        self.flip = False
        self.pinned = None            # station index
        self.measurements = []        # list of dicts from transforms.measure

    @property
    def label(self):
        t = self.loaded_at.strftime("%H:%M")
        name = self.part.get("name") or f"OID {self.part['oid']}.{self.part['part_id']}"
        return f"{t}  {name}"

    @property
    def title(self):
        return self.part.get("name") or f"OID {self.part['oid']} part {self.part['part_id']}"

    @property
    def by_raster(self):
        return self.part["by_raster"]

    @property
    def thi_raster(self):
        return self.meta.get("thi_raster", "")

    @property
    def pct(self):
        return tuple(self.meta.get("pct", (5.0, 95.0)))


class SwathViewer(tk.Tk):

    def __init__(self, bundle_path=None):
        super().__init__()
        self.title("Swath Profile")
        self.geometry("1480x900")
        self.minsize(960, 620)
        try:
            ttk.Style(self).theme_use("vista")
        except Exception:
            pass

        self.entries: list[Entry] = []
        self.watch_path = None
        self.watch_mtime = None
        self._cache = {}
        self._axes = []
        self._main_axes = []
        self._crosshair = []
        self._x_scale = 1.0
        self._display_d = None
        self.raster_vars = {}
        self.mode = None                  # None | "origin" | "measure"
        self._measure_first = None

        self._build_menu()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._quit)

        if bundle_path and os.path.exists(bundle_path):
            self.watch(bundle_path)
            self.load(bundle_path)
        self.after(POLL_MS, self._poll)
        self.after(HEARTBEAT_MS, self._heartbeat)

    # -- construction ---------------------------------------------------------

    def _build_menu(self):
        bar = tk.Menu(self)
        f = tk.Menu(bar, tearoff=False)
        f.add_command(label="Open result(s)...", command=self.open_dialog,
                      accelerator="Ctrl+O")
        f.add_command(label="Clear list", command=self.clear)
        f.add_command(label="Show results folder", command=self.show_folder)
        f.add_separator()
        f.add_command(label="Export chart image...", command=self.export_image,
                      accelerator="Ctrl+S")
        f.add_command(label="Export CSV...", command=self.export_csv)
        f.add_separator()
        f.add_command(label="Exit", command=self._quit)
        bar.add_cascade(label="File", menu=f)

        v = tk.Menu(bar, tearoff=False)
        self.xs_var = tk.BooleanVar(value=True)
        v.add_checkbutton(label="Cross-section panel", variable=self.xs_var,
                          command=self.redraw)
        self.equal_var = tk.BooleanVar(value=False)
        v.add_checkbutton(label="Equal aspect (VE = 1)", variable=self.equal_var,
                          command=self.redraw)
        v.add_separator()
        v.add_command(label="Reset zoom", command=self.reset_view,
                      accelerator="Double-click / Home")
        bar.add_cascade(label="View", menu=v)

        t = tk.Menu(bar, tearoff=False)
        t.add_command(label="Measure (two clicks)", command=lambda: self.set_mode("measure"),
                      accelerator="M")
        t.add_command(label="Set distance origin (one click)",
                      command=lambda: self.set_mode("origin"), accelerator="O")
        t.add_command(label="Clear measurements", command=self.clear_measurements)
        t.add_command(label="Cancel tool", command=lambda: self.set_mode(None),
                      accelerator="Esc")
        bar.add_cascade(label="Tools", menu=t)

        h = tk.Menu(bar, tearoff=False)
        h.add_command(label="About", command=self._about)
        bar.add_cascade(label="Help", menu=h)
        self.config(menu=bar)

        self.bind("<Control-o>", lambda e: self.open_dialog())
        self.bind("<Control-s>", lambda e: self.export_image())
        self.bind("<Home>", lambda e: self.reset_view())
        self.bind("<Escape>", lambda e: self.set_mode(None))
        self.bind("<Key-m>", lambda e: self.set_mode("measure"))
        self.bind("<Key-o>", lambda e: self.set_mode("origin"))
        self.bind("<Left>", lambda e: self.step_pinned(-1))
        self.bind("<Right>", lambda e: self.step_pinned(+1))

    def _build_layout(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        side_outer = ttk.Frame(paned)
        paned.add(side_outer, weight=0)
        main = ttk.Frame(paned)
        paned.add(main, weight=1)

        # Scrollable side panel.
        canvas = tk.Canvas(side_outer, width=250, highlightthickness=0)
        vsb = ttk.Scrollbar(side_outer, orient=tk.VERTICAL, command=canvas.yview)
        side = ttk.Frame(canvas, padding=(8, 8))
        side.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=side, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        def heading(text, pad=(10, 0)):
            ttk.Label(side, text=text, font=("", 10, "bold")).pack(anchor="w", pady=pad)

        # -- Profiles ---------------------------------------------------------
        heading("Profiles", pad=(0, 0))
        box = ttk.Frame(side)
        box.pack(fill=tk.X, pady=(2, 4))
        self.listbox = tk.Listbox(box, height=7, exportselection=False,
                                  activestyle="none", selectmode=tk.EXTENDED)
        sb = ttk.Scrollbar(box, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.redraw())
        row = ttk.Frame(side)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Open...", command=self.open_dialog).pack(side=tk.LEFT)
        ttk.Button(row, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=4)
        ttk.Label(side, text="Select several to overlay or stack",
                  foreground="#777777").pack(anchor="w")

        # -- Stacked (ridgeline) view of several profiles ---------------------
        heading("Several profiles")
        self.stack_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Stack (ridgeline) instead of overlay",
                        variable=self.stack_var, command=self.redraw).pack(anchor="w")
        row = ttk.Frame(side)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Series").pack(side=tk.LEFT)
        self.stack_series = ttk.Combobox(row, state="readonly", width=9,
                                         values=list(STACK_SERIES.keys()))
        self.stack_series.current(0)
        self.stack_series.pack(side=tk.LEFT, padx=4)
        self.stack_series.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        row = ttk.Frame(side)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Align").pack(side=tk.LEFT)
        self.stack_align = ttk.Combobox(row, state="readonly", width=12,
                                        values=list(STACK_ALIGN.keys()))
        self.stack_align.current(0)
        self.stack_align.pack(side=tk.LEFT, padx=4)
        self.stack_align.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        row = ttk.Frame(side)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Colour").pack(side=tk.LEFT)
        self.stack_colour = ttk.Combobox(row, state="readonly", width=10,
                                         values=list(STACK_COLOUR.keys()))
        self.stack_colour.current(0)
        self.stack_colour.pack(side=tk.LEFT, padx=4)
        self.stack_colour.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        self.stack_env = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Shade min-max band", variable=self.stack_env,
                        command=self.redraw).pack(anchor="w")
        row = ttk.Frame(side)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Spacing").pack(side=tk.LEFT)
        self.stack_spacing = tk.StringVar(value="")
        sp = ttk.Entry(row, textvariable=self.stack_spacing, width=8)
        sp.pack(side=tk.LEFT, padx=4)
        sp.bind("<Return>", lambda e: self.redraw())
        ttk.Label(row, text="(blank = auto)", foreground="#777777").pack(side=tk.LEFT)

        # -- Series -----------------------------------------------------------
        heading("Series")
        self.show_vars = {}
        self.show_widgets = {}
        for key, label in SERIES:
            var = tk.BooleanVar(value=plotting.DEFAULT_SHOW.get(key, False))
            self.show_vars[key] = var
            cb = ttk.Checkbutton(side, text=label, variable=var,
                                 command=self.redraw)
            cb.pack(anchor="w")
            self.show_widgets[key] = cb
            if key == "profiles":
                # Colouring belongs to the profile lines, so it sits with them.
                row = ttk.Frame(side)
                row.pack(anchor="w", fill=tk.X, padx=(18, 0))
                self.profile_colour_label = ttk.Label(row, text="Colour by")
                self.profile_colour_label.pack(side=tk.LEFT)
                self.profile_colour_box = ttk.Combobox(
                    row, state="readonly", width=12,
                    values=list(PROFILE_COLOUR.keys()))
                self.profile_colour_box.current(0)
                self.profile_colour_box.pack(side=tk.LEFT, padx=4)
                self.profile_colour_box.bind("<<ComboboxSelected>>",
                                             lambda e: self.redraw())

        # -- Processing -------------------------------------------------------
        heading("Smoothing")
        self.smooth_box = ttk.Combobox(side, state="readonly", width=14,
                                       values=list(SMOOTH_PRESETS.keys()))
        self.smooth_box.current(0)
        self.smooth_box.pack(anchor="w", pady=2)
        self.smooth_box.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        heading("Detrend")
        self.detrend_box = ttk.Combobox(side, state="readonly", width=20,
                                        values=DETREND_MODES)
        self.detrend_box.current(0)
        self.detrend_box.pack(anchor="w", pady=2)
        self.detrend_box.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        # -- Distance axis ----------------------------------------------------
        heading("Distance axis")
        row = ttk.Frame(side)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Origin").pack(side=tk.LEFT)
        self.origin_var = tk.StringVar(value="0")
        ent = ttk.Entry(row, textvariable=self.origin_var, width=9)
        ent.pack(side=tk.LEFT, padx=4)
        ent.bind("<Return>", lambda e: self.apply_origin_entry())
        ttk.Button(row, text="Pick", width=5,
                   command=lambda: self.set_mode("origin")).pack(side=tk.LEFT)
        self.flip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Flip direction", variable=self.flip_var,
                        command=self.apply_flip).pack(anchor="w")

        # -- Measure ----------------------------------------------------------
        heading("Measure")
        row = ttk.Frame(side)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Measure", command=lambda: self.set_mode("measure")).pack(
            side=tk.LEFT)
        self.measure_series = ttk.Combobox(row, state="readonly", width=7,
                                           values=MEASURE_SERIES)
        self.measure_series.current(0)
        self.measure_series.pack(side=tk.LEFT, padx=4)
        ttk.Button(side, text="Clear measurements",
                   command=self.clear_measurements).pack(anchor="w", pady=2)

        # -- View -------------------------------------------------------------
        heading("Vertical exaggeration")
        row = ttk.Frame(side)
        row.pack(fill=tk.X)
        self.ve_var = tk.StringVar(value="")
        ttk.Label(row, text="Set VE").pack(side=tk.LEFT)
        ve_entry = ttk.Entry(row, textvariable=self.ve_var, width=6)
        ve_entry.pack(side=tk.LEFT, padx=4)
        ve_entry.bind("<Return>", lambda e: self.redraw())
        ttk.Button(row, text="Auto", width=5,
                   command=self.clear_ve).pack(side=tk.LEFT)
        ttk.Checkbutton(side, text="Show cross-section panel", variable=self.xs_var,
                        command=self.redraw).pack(anchor="w", pady=(4, 0))

        heading("THi panel")
        self.thi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Show THi", variable=self.thi_var,
                        command=self.redraw).pack(anchor="w")
        self.star_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Enhanced THi*", variable=self.star_var,
                        command=self.redraw).pack(anchor="w")

        heading("Raster panels")
        self.raster_frame = ttk.Frame(side)
        self.raster_frame.pack(anchor="w", fill=tk.X)

        ttk.Separator(side).pack(fill=tk.X, pady=10)
        self.info_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.info_var, wraplength=230,
                  foreground="#555555").pack(anchor="w")

        # -- Chart ------------------------------------------------------------
        self.figure = Figure(figsize=(11, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(main)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, bottom, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT)
        self.status = tk.StringVar(value="Open a result, or run the Swath Profile "
                                         "tool in ArcGIS Pro.")
        ttk.Label(bottom, textvariable=self.status, anchor="e",
                  foreground="#333333").pack(side=tk.RIGHT, padx=8, fill=tk.X,
                                             expand=True)

        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # -- loading --------------------------------------------------------------

    def watch(self, path):
        self.watch_path = path
        try:
            self.watch_mtime = os.path.getmtime(path)
        except OSError:
            self.watch_mtime = None

    def load(self, path, select_new=True):
        try:
            b = load_bundle(path)
        except Exception as exc:
            messagebox.showerror("Could not open bundle", f"{path}\n\n{exc}")
            return
        now = datetime.datetime.now()
        first_new = len(self.entries)
        if first_new == 0:
            # First result of the session: open showing what this run's saved
            # charts show.  Never on a later load, which would silently change
            # how the profiles already on screen are drawn.
            self._seed_profile_colour(b.meta.get("config", {}))
        for part in b.parts:
            self.entries.append(Entry(part, b.meta, now, path))
            self.listbox.insert(tk.END, self.entries[-1].label)
        self._rebuild_raster_toggles()
        if select_new and len(self.entries) > first_new:
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(first_new)
            self.listbox.see(first_new)
        self.status.set(f"Loaded {len(b.parts)} profile(s) from {os.path.basename(path)}")
        self.redraw()

    def open_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Open swath result(s) - select several to compare",
            filetypes=[("Swath result", "*.swath.npz"), ("NumPy archive", "*.npz")])
        for path in paths:
            self.load(path, select_new=(path == paths[-1]))

    def show_folder(self):
        folder = None
        e = self.entry
        if e is not None:
            folder = os.path.dirname(e.source)
            parent = os.path.dirname(folder)
            candidate = os.path.join(parent, "swath_results")
            if os.path.isdir(candidate):
                folder = candidate
        elif self.watch_path:
            folder = os.path.dirname(self.watch_path)
        if folder and os.path.isdir(folder):
            try:
                os.startfile(folder)
            except Exception as exc:
                messagebox.showerror("Could not open folder", str(exc))

    def clear(self):
        self.entries.clear()
        self.listbox.delete(0, tk.END)
        self._cache.clear()
        self.figure.clear()
        self.canvas.draw_idle()
        self.info_var.set("")

    def _poll(self):
        """Reload when the tool writes a new result to the watched bundle."""
        if self.watch_path:
            try:
                m = os.path.getmtime(self.watch_path)
            except OSError:
                m = None
            if m is not None and m != self.watch_mtime:
                self.watch_mtime = m
                self.after(300, lambda: self.load(self.watch_path))
                self.lift()
                self.attributes("-topmost", True)
                self.after(200, lambda: self.attributes("-topmost", False))
        self.after(POLL_MS, self._poll)

    def _heartbeat(self):
        if self.watch_path:
            try:
                with open(self.watch_path + ".lock", "w") as fh:
                    fh.write(f"{os.getpid()} {time.time():.0f}\n")
            except OSError:
                pass
        self.after(HEARTBEAT_MS, self._heartbeat)

    def _quit(self):
        if self.watch_path:
            try:
                os.remove(self.watch_path + ".lock")
            except OSError:
                pass
        self.destroy()

    # -- data -----------------------------------------------------------------

    @property
    def selected(self):
        sel = self.listbox.curselection()
        return [self.entries[i] for i in sel if i < len(self.entries)]

    @property
    def entry(self):
        s = self.selected
        return s[0] if s else None

    def _rebuild_raster_toggles(self):
        names = []
        for e in self.entries:
            for n in e.by_raster:
                if n not in names:
                    names.append(n)
        for child in self.raster_frame.winfo_children():
            child.destroy()
        for n in names:
            var = self.raster_vars.get(n) or tk.BooleanVar(value=True)
            self.raster_vars[n] = var
            ttk.Checkbutton(self.raster_frame, text=n, variable=var,
                            command=self.redraw).pack(anchor="w")

    def stats_for(self, e: Entry):
        """Statistics for *e* at the current smoothing and detrend settings.

        Smoothing is re-applied to the raw sample lines and the statistics
        re-reduced (as the tool does); detrending then shifts every series by
        the same trend so the envelope keeps its shape.
        """
        window = SMOOTH_PRESETS.get(self.smooth_box.get(), 0)
        mode = self.detrend_box.get()
        key = (id(e), window, mode)
        if key not in self._cache:
            out = {}
            for name, s in e.by_raster.items():
                st = s
                if window and s.values is not None:
                    st = compute_stats(s.values, s.distances, offsets=s.offsets,
                                       name=name, smooth_window=window,
                                       pct_low=e.pct[0], pct_high=e.pct[1],
                                       **_stats_settings(e.meta.get("config")))
                if mode == "Linear detrend":
                    st = transforms.detrend_linear(st)
                elif mode == "Relative to minimum":
                    st = transforms.relative_to_min(st)
                out[name] = st
            self._cache[key] = out
        stats = self._cache[key]
        return {n: s for n, s in stats.items()
                if self.raster_vars.get(n) is None or self.raster_vars[n].get()}

    def current_stats(self):
        e = self.entry
        return {} if e is None else self.stats_for(e)

    def display_distances(self, e: Entry, stats):
        first = next(iter(stats.values()))
        d, _ = transforms.reorient(first.distances, None, e.origin, e.flip)
        return d

    def profile_colour(self) -> str:
        """The chosen colouring mode for the individual profile lines."""
        box = getattr(self, "profile_colour_box", None)
        return PROFILE_COLOUR.get(box.get(), "single") if box else "single"

    def _seed_profile_colour(self, config):
        """Match the combobox to the run's Colour Profile Lines By setting.

        Bundles written before this option existed have no such key, so they
        open on the historic single colour.
        """
        box = getattr(self, "profile_colour_box", None)
        if box is None:
            return
        mode = (config or {}).get("chart_profile_colour", "single")
        for label, value in PROFILE_COLOUR.items():
            if value == mode:
                box.set(label)
                return

    def _sync_profile_controls(self, stats):
        """Grey out the profile-line controls when there are no per-line samples.

        All-cells sampling keeps only the per-station statistics, so there is
        nothing to draw or colour; saying so beats a checkbox that does nothing.
        """
        box = getattr(self, "profile_colour_box", None)
        if box is None:
            return
        first = next(iter(stats.values()), None)
        available = first is not None and first.values is not None
        box.configure(state="readonly" if available else "disabled")
        self.profile_colour_label.configure(
            text="Colour by" if available else "Colour by (all-cells: none)")
        cb = self.show_widgets.get("profiles")
        if cb is not None:
            cb.configure(state="normal" if available else "disabled")
        return available

    def _side_note(self, stats, flip) -> str:
        """One line naming which compass direction each side of the band faces."""
        if self.profile_colour() == "single":
            return ""
        first = next(iter(stats.values()), None)
        if first is None or first.values is None:
            return ""
        left, right = plotting.side_labels(first, flip)
        if "(" not in left:
            return "\nSides: left/right of the digitising direction"
        return f"\nSides: {left}, {right}"

    # -- drawing --------------------------------------------------------------

    def redraw(self):
        sel = self.selected
        if not sel:
            self.figure.clear()
            self.canvas.draw_idle()
            return
        if len(sel) > 1:
            if self.stack_var.get():
                self._draw_stacked(sel)
            else:
                self._draw_overlay(sel)
        else:
            self._draw_single(sel[0])
        self._apply_ve()
        self._crosshair = []
        self.canvas.draw_idle()

    def _stack_items(self, entries):
        items = []
        for e in entries:
            stats = self.stats_for(e)
            if stats:
                items.append((e.title, stats, self.display_distances(e, stats)))
        return items

    def _stack_options(self):
        try:
            spacing = float(self.stack_spacing.get()) if self.stack_spacing.get().strip() \
                else None
        except ValueError:
            spacing = None
        raster = next((n for n, v in self.raster_vars.items() if v.get()), None)
        return dict(raster=raster,
                    series=STACK_SERIES.get(self.stack_series.get(), "z_mean"),
                    align=STACK_ALIGN.get(self.stack_align.get(), "min"),
                    colour_by=STACK_COLOUR.get(self.stack_colour.get(), "none"),
                    envelope=bool(self.stack_env.get()), spacing=spacing)

    def _draw_stacked(self, entries):
        items = self._stack_items(entries)
        if not items:
            self.figure.clear()
            return
        _, axes = plotting.build_ridgeline_figure(
            items, title=f"{len(items)} profiles stacked", figure=self.figure,
            footer=self.detrend_box.get().lower() if self.detrend_box.get() != "None" else "",
            **self._stack_options())
        self._axes = axes
        self._main_axes = axes
        self._display_d = items[0][2]
        span = max(float(np.nanmax(d) - np.nanmin(d)) for _, _, d in items)
        self._x_scale = 1000.0 if span > 5000.0 else 1.0
        self.info_var.set("Stacked: " + "\n".join(t for t, _, _ in items))

    def _draw_single(self, e: Entry):
        stats = self.stats_for(e)
        if not stats:
            self.figure.clear()
            return
        cfg = dict(e.meta.get("config", {}))
        # Per-part settings win over the run's (transects may use their own width).
        for key in ("width", "n_lines", "method"):
            if e.part.get(key) not in (None, "", 0):
                cfg["swath_width" if key == "width" else key] = e.part[key]
        show = {k: v.get() for k, v in self.show_vars.items()}
        self._sync_profile_controls(stats)
        d = self.display_distances(e, stats)
        self._display_d = d
        span = float(np.nanmax(d) - np.nanmin(d))
        self._x_scale = 1000.0 if span > 5000.0 else 1.0

        xs = None
        if self.xs_var.get():
            src = e.thi_raster if e.thi_raster in stats else next(iter(stats))
            j = e.pinned if e.pinned is not None else stats[src].n_stations // 2
            xs = transforms.cross_section(stats[src], j, flip=e.flip)
            if xs is not None:
                xs["distance"] = float(d[xs["station"]])
            else:
                xs = {"unavailable": True}
        footer = (f"width {cfg.get('swath_width', '?')} | step {cfg.get('step', '?')}"
                  f" | {cfg.get('n_lines', '?')} profiles | {cfg.get('method', '?')}"
                  + (f" | mask: {_mask_words(cfg.get('mask_mode'))}"
                     if cfg.get("mask_mode") else "")
                  + (f" | {self.detrend_box.get().lower()}"
                     if self.detrend_box.get() != "None" else ""))
        _, axes = plotting.build_figure(
            stats, title=e.title, show=show, show_thi=self.thi_var.get(),
            thi_raster=e.thi_raster, thi_enhanced=self.star_var.get(),
            footer=footer, figure=self.figure, pct=e.pct, distances=d,
            cross_section=(None if xs is None else
                           (None if xs.get("unavailable") else xs)),
            profile_colour=self.profile_colour(), flip=e.flip)
        self._axes = axes
        n_rasters = len(stats)
        self._main_axes = axes[:n_rasters]
        if xs is not None and xs.get("unavailable"):
            # build_figure got None for cross_section; draw the notice panel.
            pass
        self._draw_overlays_single(e, stats, d)

        first = next(iter(stats.values()))
        elapsed = e.meta.get("elapsed")
        self.info_var.set(
            f"{first.n_stations} stations, {first.n_total} samples across\n"
            f"Width {cfg.get('swath_width', '?')}  Step {cfg.get('step', '?')}\n"
            f"Method: {cfg.get('method', '?')}   "
            f"P{e.pct[0]:g}/P{e.pct[1]:g}\n"
            f"CRS: {e.meta.get('crs', '')}"
            + self._side_note(stats, e.flip)
            + self._mask_note(e, cfg)
            + (f"\nRun time {elapsed:.1f} s" if elapsed else ""))

    @staticmethod
    def _mask_note(e, cfg) -> str:
        """One info line naming the sampling mask and how much of the band it kept."""
        if not cfg.get("mask_mode"):
            return ""
        kept = e.part.get("mask_kept")
        share = f", {kept:.0f}% of band kept" if kept is not None else ""
        return f"\nMask: {_mask_words(cfg.get('mask_mode'))}{share}"

    def _draw_overlays_single(self, e, stats, d):
        """Pinned-station marker and measurement annotations, after build_figure."""
        xs_scale = self._x_scale
        for ax in self._main_axes:
            if e.pinned is not None and 0 <= e.pinned < len(d):
                ax.axvline(d[e.pinned] / xs_scale, color=plotting.PALETTE["marker"],
                           lw=0.8, ls=(0, (2, 2)), zorder=9)
            if e.origin:
                ax.axvline(0.0, color="#444444", lw=0.6, zorder=8)
            for m in e.measurements:
                (x1, y1), (x2, y2) = m["p1"], m["p2"]
                ax.plot([x1 / xs_scale, x2 / xs_scale], [y1, y2],
                        color=plotting.PALETTE["marker"], lw=1.2, marker="o",
                        ms=3, zorder=10)
                ax.annotate(m["text"], ((x1 + x2) / 2 / xs_scale, max(y1, y2)),
                            textcoords="offset points", xytext=(0, 6), ha="center",
                            fontsize=8, color=plotting.PALETTE["marker"],
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec=plotting.PALETTE["marker"], lw=0.5))

    def _draw_overlay(self, entries):
        raster = None
        for n, var in self.raster_vars.items():
            if var.get():
                raster = n
                break
        items = []
        for e in entries:
            stats = self.stats_for(e)
            if not stats:
                continue
            d = self.display_distances(e, stats)
            items.append((e.title, stats, d))
        if not items:
            self.figure.clear()
            return
        _, axes = plotting.build_overlay_figure(
            items, raster=raster, show_thi=self.thi_var.get(),
            thi_enhanced=self.star_var.get(),
            title=f"{len(items)} profiles overlaid",
            footer=self.detrend_box.get().lower() if self.detrend_box.get() != "None" else "",
            figure=self.figure)
        self._axes = axes
        self._main_axes = axes[:1]
        self._display_d = items[0][2]
        span = max(float(np.nanmax(d) - np.nanmin(d)) for _, _, d in items)
        self._x_scale = 1000.0 if span > 5000.0 else 1.0
        self.info_var.set("Overlay: " + "\n".join(t for t, _, _ in items))

    def _apply_ve(self):
        ve = None
        if self.equal_var.get():
            ve = 1.0
        else:
            try:
                ve = float(self.ve_var.get()) if self.ve_var.get().strip() else None
            except ValueError:
                ve = None
        for ax in self._main_axes:
            if ve and ve > 0:
                ax.set_aspect(ve / self._x_scale, adjustable="datalim")
            else:
                ax.set_aspect("auto")

    def reset_view(self):
        for ax in self._axes:
            ax.relim()
            ax.autoscale()
        self.canvas.draw_idle()

    def clear_ve(self):
        self.ve_var.set("")
        self.equal_var.set(False)
        self.redraw()

    # -- interaction ----------------------------------------------------------

    def set_mode(self, mode):
        self.mode = mode
        self._measure_first = None
        hints = {
            None: "Ready.",
            "measure": "Measure: click the first point, then the second. Esc cancels.",
            "origin": "Origin: click where distance should be zero. Esc cancels.",
        }
        self.status.set(hints.get(mode, ""))
        try:
            self.canvas.get_tk_widget().configure(
                cursor="crosshair" if mode else "")
        except Exception:
            pass

    def _station_at(self, xdata):
        """Nearest station to a chart x (in the chart's units)."""
        d = self._display_d
        if d is None or xdata is None:
            return None
        target = xdata * self._x_scale
        return int(np.nanargmin(np.abs(d - target)))

    def _series_value(self, stats, j):
        name = self.measure_series.get() or "Mean"
        src = next(iter(stats))
        arr = getattr(stats[src], _SERIES_ATTR.get(name, "z_mean"), None)
        if arr is None:                       # e.g. no centreline in an old bundle
            arr = stats[src].z_mean
        return float(arr[j])

    def _on_click(self, event):
        if event.dblclick:
            self.reset_view()
            return
        if event.inaxes is None or event.xdata is None or event.button != 1:
            return
        if event.inaxes not in self._main_axes:
            return
        e = self.entry
        if e is None or len(self.selected) > 1:
            return
        j = self._station_at(event.xdata)
        if j is None:
            return
        stats = self.stats_for(e)

        if self.mode == "origin":
            d_raw = next(iter(stats.values())).distances
            e.origin = float(d_raw[j])
            self.origin_var.set(f"{e.origin:.1f}")
            self.set_mode(None)
            self.redraw()
            return

        if self.mode == "measure":
            point = (float(self._display_d[j]), self._series_value(stats, j))
            if self._measure_first is None:
                self._measure_first = point
                self.status.set(f"Measure: first point at {point[0]:,.1f}, "
                                f"{point[1]:,.2f}. Click the second.")
                return
            m = transforms.measure(self._measure_first, point)
            grad = (f"{m['angle_deg']:.1f}° · {m['percent']:.1f} %"
                    if np.isfinite(m["percent"]) else "vertical")
            text = (f"Δd {m['dd']:,.1f} · Δz {m['dz']:,.2f} · {grad}")
            e.measurements.append({"p1": self._measure_first, "p2": point,
                                   "text": text, **m,
                                   "series": self.measure_series.get()})
            self._measure_first = None
            self.set_mode(None)
            self.status.set(text)
            self.redraw()
            return

        # Plain click: pin the station for the cross-section panel.
        e.pinned = j
        self.redraw()

    def step_pinned(self, delta):
        e = self.entry
        if e is None:
            return
        stats = self.stats_for(e)
        if not stats:
            return
        n = next(iter(stats.values())).n_stations
        cur = e.pinned if e.pinned is not None else n // 2
        e.pinned = int(np.clip(cur + delta, 0, n - 1))
        self.redraw()

    def apply_origin_entry(self):
        e = self.entry
        if e is None:
            return
        try:
            e.origin = float(self.origin_var.get())
        except ValueError:
            return
        self.redraw()

    def apply_flip(self):
        e = self.entry
        if e is None:
            return
        e.flip = bool(self.flip_var.get())
        self.redraw()

    def clear_measurements(self):
        e = self.entry
        if e is not None:
            e.measurements.clear()
            self.redraw()

    def _on_motion(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        if event.inaxes not in self._axes:
            return
        e = self.entry
        stats = self.current_stats()
        if e is None or not stats or len(self.selected) > 1:
            return
        j = self._station_at(event.xdata)
        if j is None:
            return
        first = next(iter(stats.values()))
        bits = [f"dist {self._display_d[j]:,.1f}"]
        for name, s in stats.items():
            if np.isfinite(s.z_mean[j]):
                bits.append(f"{name}: {s.z_mean[j]:,.1f} "
                            f"[{s.z_min[j]:,.1f} - {s.z_max[j]:,.1f}]")
        src = e.thi_raster if e.thi_raster in stats else next(iter(stats))
        thi = stats[src].thi_star[j] if self.star_var.get() else stats[src].thi[j]
        if np.isfinite(thi):
            bits.append(f"THi {thi:.1f}")
        if self._main_axes:
            ve = transforms.vertical_exaggeration(self._main_axes[0])
            if np.isfinite(ve):
                bits.append(f"VE ≈ {ve:.1f}×")
        if self.mode:
            bits.insert(0, self.mode.upper())
        self.status.set("   |   ".join(bits))

        for line in self._crosshair:
            try:
                line.remove()
            except Exception:
                pass
        self._crosshair = [ax.axvline(event.xdata, color="#888888", lw=0.6,
                                      ls=(0, (3, 3)), zorder=10)
                           for ax in self._axes if ax in self._main_axes or
                           ax is self._axes[-1]]
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        factor = 0.85 if event.button == "up" else 1.0 / 0.85
        for ax in self._main_axes + ([self._axes[-1]] if self._axes else []):
            lo, hi = ax.get_xlim()
            ax.set_xlim(event.xdata - (event.xdata - lo) * factor,
                        event.xdata + (hi - event.xdata) * factor)
        self.canvas.draw_idle()

    # -- export ---------------------------------------------------------------

    def export_image(self):
        e = self.entry
        if e is None:
            return
        dlg = ExportDialog(self)
        self.wait_window(dlg)
        if not dlg.result:
            return
        opts = dlg.result
        path = filedialog.asksaveasfilename(
            defaultextension="." + opts["format"].lower(),
            filetypes=[(opts["format"], "*." + opts["format"].lower())],
            initialfile=f"swath_{e.title.replace(' ', '_')}.{opts['format'].lower()}")
        if not path:
            return
        try:
            sel = self.selected
            if len(sel) > 1 and self.stack_var.get():
                items = self._stack_items(sel)
                fig, _ = plotting.build_ridgeline_figure(
                    items, title=f"{len(items)} profiles stacked",
                    preset=opts["preset"], **self._stack_options())
            elif len(sel) > 1:
                items = self._stack_items(sel)
                raster = next((n for n, v in self.raster_vars.items() if v.get()), None)
                fig, _ = plotting.build_overlay_figure(
                    items, raster=raster, show_thi=opts["thi"],
                    thi_enhanced=self.star_var.get(), preset=opts["preset"],
                    title=f"{len(items)} profiles overlaid")
            else:
                stats = self.stats_for(e)
                d = self.display_distances(e, stats)
                xs = None
                if opts["xs"]:
                    src = e.thi_raster if e.thi_raster in stats else next(iter(stats))
                    j = e.pinned if e.pinned is not None else stats[src].n_stations // 2
                    xs = transforms.cross_section(stats[src], j, flip=e.flip)
                    if xs is not None:
                        xs["distance"] = float(d[xs["station"]])
                show = {k: v.get() for k, v in self.show_vars.items()}
                fig, _ = plotting.build_figure(
                    stats, title=e.title, show=show, show_thi=opts["thi"],
                    thi_raster=e.thi_raster, thi_enhanced=self.star_var.get(),
                    preset=opts["preset"], pct=e.pct, distances=d,
                    cross_section=xs, profile_colour=self.profile_colour(),
                    flip=e.flip)
            plotting.save_figure(fig, path, dpi=opts["dpi"])
            self.status.set(f"Saved {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_csv(self):
        e = self.entry
        stats = self.current_stats()
        if e is None or not stats:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile=f"swath_{e.title.replace(' ', '_')}.csv")
        if not path:
            return
        first = next(iter(stats.values()))
        d = self.display_distances(e, stats)
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                header = ["STATION", "DIST", "DIST_RAW"]
                for name in stats:
                    safe = "".join(c if c.isalnum() else "_" for c in name)[:24]
                    header += [f"{safe}_{k}" for k in
                               ("MIN", "MAX", "MEAN", "MEDIAN", "Q1", "Q3", "PLO",
                                "PHI", "CENTRE", "RELIEF", "HI", "THI")]
                w.writerow(header)
                for j in range(first.n_stations):
                    row = [j, f"{d[j]:.4f}", f"{first.distances[j]:.4f}"]
                    for s in stats.values():
                        for arr in (s.z_min, s.z_max, s.z_mean, s.z_median,
                                    s.z_q1, s.z_q3, s.z_plo, s.z_phi, s.z_centre,
                                    s.relief, s.hi, s.thi):
                            v = np.nan if arr is None else arr[j]
                            # Empty, never "nan": Excel reads that as text.
                            row.append("" if not np.isfinite(v) else f"{float(v):.4f}")
                    w.writerow(row)
            if e.measurements:
                mpath = os.path.splitext(path)[0] + "_measurements.csv"
                with open(mpath, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["SERIES", "D1", "Z1", "D2", "Z2", "DD", "DZ",
                                "GRADIENT", "PERCENT", "ANGLE_DEG"])
                    for m in e.measurements:
                        w.writerow([m["series"], *m["p1"], *m["p2"], m["dd"], m["dz"],
                                    m["gradient"], m["percent"], m["angle_deg"]])
            self.status.set(f"Saved {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def _about(self):
        messagebox.showinfo(
            "Swath Profile viewer",
            "Swath profiles after Perez-Pena et al. (2017).\n\n"
            "Click the chart to pin a station and see its cross-section; "
            "M to measure between two points; O to set the distance origin; "
            "select several profiles to overlay them. Scroll to zoom, "
            "double-click or Home to reset. Re-run the ArcGIS Pro tool and this "
            "window refreshes.")


class ExportDialog(tk.Toplevel):
    """Preset / format / dpi chooser for image export."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Export chart")
        self.resizable(False, False)
        self.result = None
        self.transient(parent)
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Preset").grid(row=0, column=0, sticky="w")
        self.preset = ttk.Combobox(frm, state="readonly", width=24,
                                   values=list(plotting.PRESETS.keys()))
        self.preset.current(0)
        self.preset.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(frm, text="Format").grid(row=1, column=0, sticky="w")
        self.fmt = ttk.Combobox(frm, state="readonly", width=8,
                                values=["PNG", "PDF", "SVG"])
        self.fmt.current(0)
        self.fmt.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(frm, text="DPI (PNG)").grid(row=2, column=0, sticky="w")
        self.dpi = tk.StringVar(value="300")
        ttk.Entry(frm, textvariable=self.dpi, width=8).grid(row=2, column=1,
                                                             sticky="w", pady=2)

        self.xs = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Include cross-section panel",
                        variable=self.xs).grid(row=3, column=0, columnspan=2,
                                               sticky="w")
        self.thi = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Include THi panel",
                        variable=self.thi).grid(row=4, column=0, columnspan=2,
                                                sticky="w")

        row = ttk.Frame(frm)
        row.grid(row=5, column=0, columnspan=2, pady=(10, 0), sticky="e")
        ttk.Button(row, text="Export", command=self._ok).pack(side=tk.RIGHT)
        ttk.Button(row, text="Cancel", command=self.destroy).pack(side=tk.RIGHT,
                                                                  padx=6)
        self.grab_set()

    def _ok(self):
        try:
            dpi = int(self.dpi.get())
        except ValueError:
            dpi = 300
        self.result = {"preset": self.preset.get(), "format": self.fmt.get(),
                       "dpi": dpi, "xs": bool(self.xs.get()),
                       "thi": bool(self.thi.get())}
        self.destroy()


def main(argv):
    path = argv[1] if len(argv) > 1 else None
    app = SwathViewer(path)
    if path is None:
        app.after(200, app.open_dialog)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
