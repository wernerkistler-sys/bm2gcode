import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import copy
import dataclasses
import hashlib
import json
import os
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading

# Diverging colormap that is dark at zero (like "hot") and bright at both extremes.
# Negative half: black → blue → cyan → white  (cold mirror of hot)
# Positive half: black → red → yellow → white  (standard hot)
_cold_cmap = mcolors.LinearSegmentedColormap.from_list(
    '_cold', [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)])
_DIFF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'diff_hot',
    np.vstack([
        _cold_cmap(np.linspace(1, 0, 128)),   # white/cyan → black  (most-negative → zero)
        plt.cm.hot(np.linspace(0, 1, 128)),    # black → yellow/white (zero → most-positive)
    ])
)

# Distinct, saturated facecolor shown through wherever a ToolPath.alpha
# array marks a pixel as physically cut through (a real hole) -- the
# ordinary grayscale "Step Result" colormap never produces a saturated hue,
# so this always reads clearly as "this is a hole," not a depth value.
_HOLE_FACECOLOR = "#d9534f"
# Overhang overlay (View → Show overhangs): air enclosed by the source, air
# open down to full depth, and overhangs a computed Undercut step reaches.
_OVERHANG_CLOSED_COLOR  = "#ff2bd6"
_OVERHANG_OPEN_COLOR    = "#ffc400"
_OVERHANG_REACHED_COLOR = "#2ea8ff"

from constants import (
    TOOL_PROFILES, RASTER_DIRECTIONS, POCKET_PATTERNS, POCKET_ZIGZAG, WOUND_DIRECTIONS,
    CUT_DIRECTIONS, UNDERCUT_OPEN_MODES, ALGORITHMS,
    ALGORITHM_FAMILY, DEFAULTS, REG_HOLE_PATTERNS, DIAG_DIRECTIONS, LAYER_MIRRORS,
    STEP_PEAK_BYTES_PER_PX, STEP_PEAK_BYTES_PER_PX_DEFAULT,
    DEFAULT_PREAMBLE, DEFAULT_POSTAMBLE, BLEND_MODES,
    GATING_MODES, GATING_DEPTH_MAP_CHANGE,
    DISPLAY_ORIGINAL, DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
    DISPLAY_STEP_3D, DISPLAY_STEP_ACCESSIBLE, DISPLAY_STEP_UNACCESSIBLE,
    DISPLAY_PROC_MASK,
    DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT, DISPLAY_DIFF_DEPTH_MAP,
    DISPLAY_MODES, DISPLAY_3D_MODES, DISPLAY_STEP_MODES,
)
from widgets import (ProgressDialog, LabeledEntry, LabeledRangeEntry,
                     SettingsDialog, DressUpDialog, ExportToolpathDialog,
                     ImportSTLDialog, ToolLibraryDialog, ApplyMaterialDialog,
                     _fmt_time)
from algorithms import (compute_raster_toolpath, compute_vcarve_toolpath,
                        compute_contour_toolpath, compute_vcarve_mask,
                        resolve_vcarve_threshold_norm, compute_raster_gate_mask,
                        compute_cutout_toolpath, compute_cutout_silhouette_mask,
                        _compute_cutout_rings, _compute_cutout_from_rings,
                        compute_pocket_toolpath, compute_pocket_mask, _build_kernel,
                        compute_vcarve_synth_achievable_depth,
                        compute_registration_holes_toolpath, registration_hole_centers,
                        registration_flip_axis, registration_auto_spacing,
                        depth_map_change, depth_map_change_reference)
from tool import Tool, UNDERCUT_PROFILES
from erosion_cache import ErosionCache
from undercut import compute_undercut_toolpath
from tool_library import (ToolLibrary, cutting_for, step_values_from, sync_state,
                          update_cutting_from_step, tool_from_step, format_value,
                          STEPDOWN_FIELDS, SYNC_GEOMETRY_DIFFERS, SYNC_MODIFIED,
                          SYNC_MISSING, SYNC_NO_DATA)
from dressup import RampSettings, BackoffSettings, ArcSettings, TabSettings
import layers as layers_mod
import machine_defaults
import stl_mesh
import tool_profile

# Tool profiles each algorithm accepts (see _refresh_layout, which forces the
# step's profile into this list, and the step's Library combobox, which only
# offers library tools with one of these profiles).
_VCARVE_PROFILES  = ["V-Bit", "Ball", "Custom"]
_HYBRID_PROFILES  = ["V-Bit"]
_RASTER_PROFILES  = ["Ball", "End Mill", "V-Bit", "Custom"]
_CUTOUT_PROFILES  = ["End Mill"]
_UNDERCUT_PROFILES = list(UNDERCUT_PROFILES)

# Step fields holding an undercut tool's extra geometry (see tool.UNDERCUT_FIELDS).
_UNDERCUT_KEYS = ('neck_diameter', 'neck_length', 'shank_diameter',
                  'head_height', 'dovetail_angle')


class _StepInputError(ValueError):
    """Bad step input whose message is shown to the user as it is."""


class _MissingProfileCurve(_StepInputError):
    """A step uses the Custom profile but carries no curve."""


def _allowed_profiles(algorithm):
    family = ALGORITHM_FAMILY[algorithm]
    if algorithm == "V-Carving (Hybrid)":
        return _HYBRID_PROFILES
    if family == "V-Carving":
        return _VCARVE_PROFILES
    if family in ("Cutout", "Registration Holes"):
        return _CUTOUT_PROFILES
    if family == "Undercut":
        return _UNDERCUT_PROFILES
    return _RASTER_PROFILES


# Status text shown next to a linked step's Library combobox.
_SYNC_STATUS_TEXT = {
    SYNC_GEOMETRY_DIFFERS: "● geometry differs",
    SYNC_MODIFIED:         "● modified",
    SYNC_MISSING:          "● not in library",
}


class GCodeGeneratorApp:

    def __init__(self, root):
        self.root = root
        # Path of the .prj file the project was opened from or last saved
        # to; None for an untitled project.
        self._project_path = None
        self._title_shown = None
        self.root.geometry("1240x780")
        self.root.minsize(940, 560)

        # User-wide default machine settings, which seed the settings of the
        # startup project and of every new project.
        self._machine_defaults_path = machine_defaults.default_path()
        self._machine_defaults, defaults_warning = machine_defaults.load(
            self._machine_defaults_path)
        self._settings = dict(self._machine_defaults)

        # User-wide tool library (outside any project) and the project's
        # material, which picks each linked step's cutting data.
        self._tool_library_path = ToolLibrary.default_path()
        self.tool_library, library_warning = ToolLibrary.load_or_seed(
            self._tool_library_path)
        self._material_id, self._material_name = self._default_material()

        self.current_image   = None   # 2-D float32 [0,1] grayscale array -- cached
                                       # composite of self.layers, see _recompute_composite()
        self.n_grey_levels   = 256
        self.layers = []   # list[layers_mod.Layer]; layers[0] = bottom/base source
        self._next_layer_id = 1   # monotonically increasing, for stable Layer ids
        self._last_width_mm  = float(DEFAULTS["width_mm"])          # see _on_canvas_size_changed
        self._last_height_mm = float(DEFAULTS["height_mm"])
        self._last_min_px_per_mm = float(DEFAULTS["min_px_per_mm"])  # see _on_min_px_per_mm_changed
        self._stl_placement_pending = None   # after_idle id, see _schedule_stl_placement
        self.steps     = []   # list of step-widget dicts, built by _build_sidebar
        self.toolpaths = []   # list of ToolPath|None, parallel to self.steps
        # tool_signature -> (center_depth_contour, accessible), bounded by
        # the machine's memory (see erosion_cache.py)
        self._erosion_cache = ErosionCache()
        self._overhang_cache = {}  # see _overhang_masks

        # Zoom / pan state (2D)
        self._pan_origin     = None
        self._view_xlim      = None
        self._view_ylim      = None

        # 3D view state
        self._3d_elev        = 30.0   # elevation angle (degrees)
        self._3d_azim        = -60.0  # azimuth angle (degrees)
        self._3d_zoom        = 1.0    # zoom factor (larger = more zoomed out)
        self._3d_pan_offset  = np.array([0.0, 0.0, 0.0])  # XYZ pan offset
        self._3d_mouse_start = None   # (x_px, y_px, elev, azim) or pan variant
        self._prev_display_mode = None  # last mode seen by _on_display_mode_changed

        # Currently rendered 2D array (used for hover coordinate readout)
        self._displayed_array = None

        self._build_menu()
        self._build_layout()

        # Baseline snapshot of the default, "empty" project — lets us detect
        # changes even if the user never opens or saves a project file.
        self._saved_snapshot = self._snapshot_key()
        self._update_title()
        self._poll_title()
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        if library_warning:
            self.root.after(200, lambda: messagebox.showwarning(
                "Tool Library", library_warning))
        if defaults_warning:
            self.root.after(200, lambda: messagebox.showwarning(
                "Settings", defaults_warning))

    # ------------------------------------------------------------------ #
    #  Menu
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        self.display_mode_var   = tk.StringVar(value=DISPLAY_ORIGINAL)
        self.display_step_var   = tk.IntVar(value=1)
        self.show_cut_moves_var = tk.BooleanVar(value=True)
        self.show_rapids_var    = tk.BooleanVar(value=True)
        self.show_tabs_var      = tk.BooleanVar(value=True)
        self.show_reg_holes_var = tk.BooleanVar(value=True)
        self.show_shading_var   = tk.BooleanVar(value=False)
        self.show_overhangs_var = tk.BooleanVar(value=False)

        menubar   = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        # ── Project ─────────────────────────────────────────────────
        file_menu.add_command(label="New Project",   accelerator="Ctrl+N",
                              command=self._new_project)
        file_menu.add_command(label="Open Project…", accelerator="Ctrl+Shift+O",
                              command=self._load_project)
        file_menu.add_command(label="Save Project",  accelerator="Ctrl+S",
                              command=self._save_project)
        file_menu.add_command(label="Save Project As…", accelerator="Ctrl+Shift+S",
                              command=self._save_project_as)
        file_menu.add_separator()
        # ── Source ──────────────────────────────────────────────────
        file_menu.add_command(label="Open Source…",  accelerator="Ctrl+O",
                              command=self._open_source_file)
        file_menu.add_separator()
        # ── Toolpath ────────────────────────────────────────────────
        file_menu.add_command(label="Export Toolpath…",
                              command=self._export_toolpaths)
        file_menu.add_separator()
        # ── Settings ────────────────────────────────────────────────
        file_menu.add_command(label="Settings…",     command=self._open_settings)
        file_menu.add_separator()
        # ── Exit ────────────────────────────────────────────────────
        file_menu.add_command(label="Exit",          accelerator="Ctrl+Q",
                              command=self._on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        compute_menu = tk.Menu(menubar, tearoff=0)
        compute_menu.add_command(
            label="Compute Current Step",
            accelerator="F5",
            command=lambda: self._compute_step(
                self.steps[self._steps_nb.index("current")]))
        compute_menu.add_command(
            label="Compute All Steps",
            accelerator="F6",
            command=self._compute_all_steps)
        menubar.add_cascade(label="Compute", menu=compute_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Tool Library…", accelerator="Ctrl+T",
                               command=self._open_tool_library)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        self._view_menu = tk.Menu(menubar, tearoff=0)

        _display_hotkeys = [
            (DISPLAY_STEP_3D,           "0"),
            (DISPLAY_ORIGINAL,          "1"),
            (DISPLAY_STEP_ACCESSIBLE,   "2"),
            (DISPLAY_STEP_RESULT,       "3"),
            (DISPLAY_STEP_RES,          "4"),
            (DISPLAY_STEP_UNACCESSIBLE, "5"),
            (DISPLAY_DIFF_RES_UNCUT,     "6"),
            (DISPLAY_DIFF_UNCUT_UNCUT,  "7"),
            (DISPLAY_DIFF_DEPTH_MAP,    "8"),
            (DISPLAY_PROC_MASK,         "9"),
        ]
        self._display_menu_indices = {}
        for mode, key in _display_hotkeys:
            self._view_menu.add_radiobutton(
                label=mode,
                accelerator=key,
                variable=self.display_mode_var,
                value=mode,
                command=self._on_display_mode_changed,
            )
            self._display_menu_indices[mode] = self._view_menu.index(tk.END)

        self._view_menu.add_separator()
        self._view_menu.add_checkbutton(label="Show cut moves",
                                        accelerator="C",
                                        variable=self.show_cut_moves_var,
                                        command=self._on_display_mode_changed)
        self._view_menu.add_checkbutton(label="Show rapids",
                                        accelerator="R",
                                        variable=self.show_rapids_var,
                                        command=self._on_display_mode_changed)
        self._view_menu.add_checkbutton(label="Show tabs",
                                        accelerator="T",
                                        variable=self.show_tabs_var,
                                        command=self._on_display_mode_changed)
        self._view_menu.add_checkbutton(label="Show registration holes",
                                        accelerator="H",
                                        variable=self.show_reg_holes_var,
                                        command=self._on_display_mode_changed)
        self._view_menu.add_checkbutton(label="Show shading",
                                        accelerator="S",
                                        variable=self.show_shading_var,
                                        command=self._on_display_mode_changed)
        self._view_menu.add_checkbutton(label="Show overhangs",
                                        accelerator="U",
                                        variable=self.show_overhangs_var,
                                        command=self._on_display_mode_changed)
        self._view_menu.add_separator()
        self._view_menu.add_command(label="Reset Zoom  (double-click)",
                                    command=self._reset_zoom_and_render)

        menubar.add_cascade(label="View", menu=self._view_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-n>", lambda e: self._new_project())
        self.root.bind("<Control-o>", lambda e: self._open_source_file())
        self.root.bind("<Control-s>", lambda e: self._save_project())
        self.root.bind("<Control-S>", lambda e: self._save_project_as())
        self.root.bind("<Control-O>", lambda e: self._load_project())
        self.root.bind("<Control-q>", lambda e: self._on_exit())
        self.root.bind("<Control-t>", lambda e: self._open_tool_library())
        self.root.bind("<F5>", lambda _: self._compute_step(
            self.steps[self._steps_nb.index("current")]))
        self.root.bind("<F6>", lambda _: self._compute_all_steps())

        def _make_mode_handler(mode):
            def _handler(e):
                fw = self.root.focus_get()
                if isinstance(fw, (tk.Entry, ttk.Entry, tk.Text)):
                    return  # don't steal keypresses from text-entry widgets
                self.display_mode_var.set(mode)
                self._on_display_mode_changed()
            return _handler

        for mode, key in _display_hotkeys:
            self.root.bind(f"<Key-{key}>", _make_mode_handler(mode))

        def _make_toggle_handler(var):
            def _handler(e):
                fw = self.root.focus_get()
                if isinstance(fw, (tk.Entry, ttk.Entry, tk.Text)):
                    return
                var.set(not var.get())
                self._on_display_mode_changed()
            return _handler

        self.root.bind("<Key-c>", _make_toggle_handler(self.show_cut_moves_var))
        self.root.bind("<Key-r>", _make_toggle_handler(self.show_rapids_var))
        self.root.bind("<Key-t>", _make_toggle_handler(self.show_tabs_var))
        self.root.bind("<Key-h>", _make_toggle_handler(self.show_reg_holes_var))
        self.root.bind("<Key-s>", _make_toggle_handler(self.show_shading_var))
        self.root.bind("<Key-u>", _make_toggle_handler(self.show_overhangs_var))

    # ------------------------------------------------------------------ #
    #  Top-level layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        # Status bar (pack first so it anchors to bottom reliably)
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="No image loaded.")
        ttk.Label(status_bar, textvariable=self.status_var,
                  anchor=tk.W, padding=(6, 2)
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.coord_var = tk.StringVar(value="")
        ttk.Label(status_bar, textvariable=self.coord_var,
                  anchor=tk.E, padding=(6, 2)
                  ).pack(side=tk.RIGHT)

        # Horizontal pane: sidebar | right column
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5,
                              sashrelief=tk.RAISED, bg="#bbbbbb")
        pane.pack(fill=tk.BOTH, expand=True)

        # ── Right column: image canvas ────────────────────────────────
        right = ttk.Frame(pane)
        pane.add(right, minsize=420, stretch="always")
        self._build_right_panel(right)

        # ── Sidebar ────────────────────────────────────────────────────
        sidebar_outer = ttk.Frame(pane, width=315)
        sidebar_outer.pack_propagate(False)
        pane.add(sidebar_outer, minsize=280, stretch="never")
        self._build_scrollable_sidebar(sidebar_outer)

    # ------------------------------------------------------------------ #
    #  Scrollable sidebar
    # ------------------------------------------------------------------ #
    def _build_scrollable_sidebar(self, outer):
        canvas    = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner  = ttk.Frame(canvas, padding=(10, 10, 4, 12))
        win_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Re-laying out every sidebar widget to a new width is slow on
        # Windows, so dragging the pane sash only reflows the content once
        # the width has settled.
        reflow_after_id = None

        def _on_canvas_configure(event):
            nonlocal reflow_after_id
            if reflow_after_id is not None:
                canvas.after_cancel(reflow_after_id)

            def _apply():
                nonlocal reflow_after_id
                reflow_after_id = None
                canvas.itemconfig(win_id, width=event.width)
            reflow_after_id = canvas.after(120, _apply)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _mw(event):
            mpl = getattr(self, 'mpl_canvas', None)
            if mpl is not None and event.widget is mpl.get_tk_widget():
                return
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _mw)

        self._build_sidebar(inner)

    def _build_sidebar(self, p):

        # ── 0. Canvas + Sources (dynamic notebook) ─────────────────────
        self._build_layers_notebook(p)

        # ── 1. Tool path steps (dynamic notebook) ──────────────────────
        self._steps_nb = ttk.Notebook(p)
        self._steps_nb.pack(fill=tk.X, pady=(0, 4))
        self._steps_nb.bind("<<NotebookTabChanged>>", self._on_step_tab_changed)

        # ── 2. G-code Info ────────────────────────────────────────────
        # Must be created before _add_step() so that info labels can attach.
        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self._info_frame = ttk.Frame(p)
        self._info_frame.pack(fill=tk.X)

        self._add_step()   # start with one step

    # ------------------------------------------------------------------ #
    #  Canvas + Sources notebook
    # ------------------------------------------------------------------ #
    def _build_layers_notebook(self, p):
        """One ttk.Notebook mirroring the Steps notebook's pattern: tab 0 is
        the permanent Canvas tab (physical size / precision / etc, never
        closed or reordered), tabs 1..N are one per imported source
        (self.layers), index-aligned, in stack order (bottom-of-stack first,
        same order as self.layers -- no reversal, no separate list widget)."""
        self._layers_nb = ttk.Notebook(p)
        self._layers_nb.pack(fill=tk.X, pady=(0, 8))
        self._layer_tabs = []   # list of dicts, index-aligned with self.layers
        self._build_canvas_tab()
        # ttk.Notebook sizes itself to the tallest pane it has ever held, not
        # the one currently showing (Tk always reports the max reqheight
        # across every managed tab) -- so the Canvas tab, whose button row is
        # shorter than a Source tab's Duplicate/Add Source/Remove row, would
        # otherwise carry a dead strip at the bottom the moment any Source
        # tab exists, and the whole notebook (and everything packed below it)
        # would briefly resize whenever _refresh_layer_tabs() tears down and
        # rebuilds every Source tab. Pin -height explicitly to whichever tab
        # is actually selected so the notebook always matches just that
        # tab's content.
        self._layers_nb.bind("<<NotebookTabChanged>>", self._on_layers_tab_changed)
        self._on_layers_tab_changed()

    def _build_canvas_tab(self):
        tab = ttk.Frame(self._layers_nb, padding=(8, 6))
        self._layers_nb.add(tab, text="Canvas")

        # Claim bottom space before content so the button stays pinned to
        # the bottom, mirroring _make_layer_tab's Source tabs.
        bf = ttk.Frame(tab)
        bf.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        ttk.Button(bf, text="Add Source",
                   command=self._add_source_at_canvas).pack(fill=tk.X)

        self.width_mm  = LabeledEntry(tab, "Width",  "mm", DEFAULTS["width_mm"],
                                      minval=1.0, maxval=20000.0, increment=1.0)
        self.height_mm = LabeledEntry(tab, "Height", "mm", DEFAULTS["height_mm"],
                                      minval=1.0, maxval=20000.0, increment=1.0)
        self.width_mm.pack(fill=tk.X, pady=2)
        self.height_mm.pack(fill=tk.X, pady=2)

        self._aspect_lock_var = tk.BooleanVar(value=True)
        self._image_aspect    = None   # pixel W/H of the first imported source
        self._updating_aspect_pair = False   # reentrancy guard, see _recalc_height/_recalc_width
        ttk.Checkbutton(self.height_mm, text="Lock",
                        variable=self._aspect_lock_var,
                        command=self._on_aspect_lock_toggle
                        ).pack(side=tk.RIGHT, padx=(0, 4))

        self.min_px_per_mm = LabeledEntry(
            tab, "Min. px/mm", "px/mm", DEFAULTS["min_px_per_mm"],
            minval=0.02, maxval=100.0, increment=0.5)
        self.min_px_per_mm.pack(fill=tk.X, pady=2)

        self.max_megapixels = LabeledEntry(
            tab, "Max Megapixels", "MP", DEFAULTS["max_megapixels"],
            minval=0.1, maxval=500.0, increment=1.0)
        self.max_megapixels.pack(fill=tk.X, pady=2)

        eff_row = ttk.Frame(tab)
        eff_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(eff_row, text="Effective resolution").pack(side=tk.LEFT)
        self.effective_resolution_var = tk.StringVar(value="—")
        ttk.Label(eff_row, textvariable=self.effective_resolution_var,
                  foreground="#666666").pack(side=tk.RIGHT)

        self.max_depth = LabeledEntry(tab, "Max cut depth", "mm", DEFAULTS["max_depth"],
                                      minval=0.01, maxval=200.0, increment=1.0)
        self.max_depth.pack(fill=tk.X, pady=2)

        material_row = ttk.Frame(tab)
        material_row.pack(fill=tk.X, pady=2)
        ttk.Label(material_row, text="Material").pack(side=tk.LEFT)
        self.material_var = tk.StringVar()
        self._material_cb = ttk.Combobox(material_row, textvariable=self.material_var,
                                         state="readonly", width=22)
        self._material_cb.pack(side=tk.RIGHT)
        self._material_cb.bind("<<ComboboxSelected>>", self._on_material_selected)
        self._refresh_material_combo()

        self.margin_mm = LabeledEntry(tab, "Stock margin", "mm", DEFAULTS["margin_mm"],
                                      minval=0.0, maxval=5000.0, increment=1.0)
        self.margin_mm.pack(fill=tk.X, pady=2)
        self.margin_depth_pct = LabeledEntry(tab, "Margin depth", "%", DEFAULTS["margin_depth_pct"],
                                             minval=0.0, maxval=100.0, increment=5.0)
        self.margin_depth_pct.pack(fill=tk.X, pady=2)
        self.background_depth_pct = LabeledEntry(
            tab, "Background depth", "%", DEFAULTS["canvas_background_depth_pct"],
            minval=0.0, maxval=100.0, increment=5.0)
        self.background_depth_pct.pack(fill=tk.X, pady=2)

        self.width_mm.committed_var.trace_add("write", self._on_width_changed)
        self.height_mm.committed_var.trace_add("write", self._on_height_changed)
        self.min_px_per_mm.committed_var.trace_add("write", self._on_min_px_per_mm_changed)
        self.max_megapixels.committed_var.trace_add("write", self._on_max_megapixels_changed)
        self.max_depth.committed_var.trace_add("write", lambda *_: self._render())
        self.margin_mm.committed_var.trace_add("write", lambda *_: self._render())
        self.margin_depth_pct.committed_var.trace_add("write", self._on_depth_map_content_changed)
        self.background_depth_pct.committed_var.trace_add("write", self._on_depth_map_content_changed)
        self._on_aspect_lock_toggle()   # apply initial locked state

        polarity_row = ttk.Frame(tab)
        polarity_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(polarity_row, text="z=0 colour").pack(side=tk.LEFT)
        self.surface = tk.StringVar(value=DEFAULTS["surface"])
        ttk.Combobox(polarity_row, textvariable=self.surface,
                     values=["white", "black"],
                     state="readonly", width=22).pack(side=tk.RIGHT)
        self.surface.trace_add("write", self._on_depth_map_content_changed)

        origin_row = ttk.Frame(tab)
        origin_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(origin_row, text="G-code origin").pack(side=tk.LEFT)
        self.origin_var = tk.StringVar(value=DEFAULTS["origin"])
        ttk.Combobox(origin_row, textvariable=self.origin_var,
                     values=["Bottom-Left", "Bottom-Right",
                             "Top-Left", "Top-Right", "Center"],
                     state="readonly", width=22).pack(side=tk.RIGHT)
        self.origin_var.trace_add("write", self._on_origin_changed)

    def _contain_footprint_mm(self, native_aspect, width_mm, height_mm, scale_pct):
        """Physical (footprint_w_mm, footprint_h_mm) a source occupies on
        the canvas at the given Scale-%, aspect-preserving 'contain': at
        scale_pct=100 the source is inscribed in the canvas rectangle along
        whichever axis its own aspect ratio (native_aspect = width/height,
        in any consistent unit -- source pixels or source mm, only the
        ratio matters) binds. This is the mm-only geometry shared by both
        Scale-%'s meaning (see layers._affine_params) and by how a
        bitmap's own native resolution is recovered (_native_px_per_mm) --
        it depends only on the canvas's physical size and Scale-%, never on
        canvas pixel density."""
        canvas_aspect = width_mm / max(height_mm, 1e-9)
        s = max(scale_pct, 1e-6) / 100.0
        if native_aspect >= canvas_aspect:
            footprint_w_mm = width_mm * s
            return footprint_w_mm, footprint_w_mm / native_aspect
        footprint_h_mm = height_mm * s
        return footprint_h_mm * native_aspect, footprint_h_mm

    def _native_px_per_mm(self, layer, width_mm, height_mm):
        """A bitmap layer's own native resolution, expressed in canvas-
        physical px/mm -- how finely its native pixels are actually
        packed onto the stock, given where Scale-% currently places it."""
        H_l, W_l = layer.raw.shape
        footprint_w_mm, _ = self._contain_footprint_mm(
            W_l / H_l, width_mm, height_mm, layer.scale_pct)
        return W_l / max(footprint_w_mm, 1e-9)

    def _auto_px_per_mm(self, width_mm, height_mm, layers=None):
        """Canvas pixel density (px/mm), auto-chosen between the Min.
        px/mm floor and the Max Megapixels ceiling: with no bitmap source
        present the floor is used as-is; otherwise the finest bitmap's own
        native px/mm (_native_px_per_mm) drives it, still bounded by the
        floor/ceiling. If the floor alone already exceeds the ceiling for
        this physical size, the floor wins uncapped -- that combination can
        only arise from editing Width/Height/Min. px/mm/Max Megapixels,
        each of which gates the change through _confirm_canvas_shape before
        committing, so by the time this runs it's already something the
        user explicitly chose to allow. *layers* defaults to self.layers."""
        try:
            floor_density = float(self.min_px_per_mm.get())
        except ValueError:
            floor_density = float(DEFAULTS["min_px_per_mm"])
        if floor_density <= 1e-9:
            floor_density = float(DEFAULTS["min_px_per_mm"])

        bitmap_layers = [l for l in (self.layers if layers is None else layers)
                         if l.source_kind == "image"]
        if not bitmap_layers:
            candidate = floor_density
        else:
            candidate = max(self._native_px_per_mm(l, width_mm, height_mm)
                             for l in bitmap_layers)

        try:
            max_mp = float(self.max_megapixels.get())
        except ValueError:
            max_mp = float(DEFAULTS["max_megapixels"])
        if max_mp <= 1e-9:
            max_mp = float(DEFAULTS["max_megapixels"])
        cap_density = float(np.sqrt(max_mp * 1e6 / max(width_mm * height_mm, 1e-9)))

        if floor_density > cap_density:
            return max(candidate, floor_density)
        return min(max(candidate, floor_density), cap_density)

    def _canvas_grid(self):
        """(W, H, density): the canvas pixel shape and its px/mm, derived
        live from the physical size and the auto-chosen density
        (_auto_px_per_mm) -- the single source of truth for how big the
        depth map is."""
        try:
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
        except ValueError:
            width_mm, height_mm = float(DEFAULTS["width_mm"]), float(DEFAULTS["height_mm"])
        density = self._auto_px_per_mm(width_mm, height_mm)
        W = max(2, round(width_mm * density))
        H = max(2, round(height_mm * density))
        return W, H, density

    def _canvas_shape_px(self):
        """Canvas pixel shape (W, H), see _canvas_grid."""
        W, H, _ = self._canvas_grid()
        return W, H

    def _on_canvas_shape_changed(self, *_):
        """Shared tail once canvas pixel shape and/or every source's
        placement has already been reconciled with whatever just changed
        (Width/Height via _on_canvas_size_changed, Min. px/mm/Max
        Megapixels via their own handlers) -- all of them already gated the
        change itself on _confirm_canvas_shape, so by the time we get here
        a huge canvas is something the user explicitly chose to proceed
        with; the status line still notes it. Then treat this as a
        structural change like add/remove -- invalidate caches and
        recomposite every source onto the new grid so the change is
        visible immediately."""
        W, H, _ = self._canvas_grid()
        try:
            max_mp = float(self.max_megapixels.get())
        except ValueError:
            max_mp = float(DEFAULTS["max_megapixels"])
        if W * H > max_mp * 1e6:
            self.status_var.set(
                f"Canvas is {W}x{H} px (~{W * H / 1e6:.0f} MP) -- "
                "this may be slow.")
        self._on_layers_structure_changed()

    def _confirm_canvas_shape(self, width_mm, height_mm, density, max_mp=None):
        """Ask before committing to a Width/Height/Min.-px/mm/Max-Megapixels
        combination whose Min. px/mm floor alone -- ignoring any bitmap --
        would already exceed the Max Megapixels ceiling for this physical
        size. W*H drives every per-pixel pass (compositing, rasterizing,
        toolpath generation), so a multi-hundred-megapixel canvas doesn't
        error out, it just makes the app sit there looking hung for however
        long that pass takes. A status-line warning printed *after* the
        shape already changed doesn't help, since by the time it's on
        screen the expensive work triggered by that same change is already
        queued -- the gate has to sit in front of the change, not after
        it."""
        if max_mp is None:
            try:
                max_mp = float(self.max_megapixels.get())
            except ValueError:
                max_mp = float(DEFAULTS["max_megapixels"])
        if density <= 1e-9:
            density = float(DEFAULTS["min_px_per_mm"])
        W = max(2, round(width_mm * density))
        H = max(2, round(height_mm * density))
        if W * H <= max_mp * 1e6:
            return True
        return messagebox.askyesno(
            "Large canvas",
            f"This would create a {W}x{H} px canvas (~{W * H / 1e6:.0f} MP).\n\n"
            "Compositing and generating toolpaths at this resolution can "
            "take a long time and may make the app unresponsive.\n\n"
            "Proceed anyway?")

    def _on_canvas_size_changed(self, *_):
        """Width/Height changed. Scale-% is canvas-size-relative (see
        layers._affine_params), so every source's physical footprint
        already survives a Width/Height edit on its own -- nothing to
        rescale there. Offset X/Y, however, is an absolute mm distance from
        the canvas center, and the center's own position just moved, so
        left alone the whole composition would drift relative to the
        board. Rescale every source's Offset X/Y by the ratio each canvas
        dimension changed by, so the user's relative layout survives
        untouched; only the absolute physical size of the whole carving
        changes."""
        try:
            new_width_mm  = float(self.width_mm.get())
            new_height_mm = float(self.height_mm.get())
        except ValueError:
            return
        if new_width_mm > 1e-9 and new_height_mm > 1e-9:
            try:
                floor_density = float(self.min_px_per_mm.get())
            except ValueError:
                floor_density = float(DEFAULTS["min_px_per_mm"])
            if not self._confirm_canvas_shape(new_width_mm, new_height_mm, floor_density):
                self.width_mm.set(str(self._last_width_mm))
                self.height_mm.set(str(self._last_height_mm))
                return
            old_width_mm, old_height_mm = self._last_width_mm, self._last_height_mm
            if (old_width_mm > 1e-9 and old_height_mm > 1e-9
                    and (new_width_mm != old_width_mm or new_height_mm != old_height_mm)):
                width_ratio  = new_width_mm / old_width_mm
                height_ratio = new_height_mm / old_height_mm
                for layer in self.layers:
                    layer.offset_x_mm *= width_ratio
                    layer.offset_y_mm *= height_ratio
            self._last_width_mm, self._last_height_mm = new_width_mm, new_height_mm
        self._on_canvas_shape_changed()

    def _on_min_px_per_mm_changed(self, *_):
        """Min. px/mm changed -- since Scale-% is canvas-size-relative
        (layers._affine_params), no source needs rescaling: physical
        footprints never depended on pixel density to begin with. Just
        re-gate against Max Megapixels and let the shared tail recompute
        the (possibly auto-finer) density and re-resample."""
        try:
            new_floor = float(self.min_px_per_mm.get())
        except ValueError:
            return
        if new_floor > 1e-9:
            if not self._confirm_canvas_shape(float(self.width_mm.get()),
                                               float(self.height_mm.get()),
                                               new_floor):
                self.min_px_per_mm.set(str(self._last_min_px_per_mm))
                return
            self._last_min_px_per_mm = new_floor
        self._on_canvas_shape_changed()

    def _on_max_megapixels_changed(self, *_):
        """Max Megapixels changed -- tightening it can newly make the Min.
        px/mm floor alone exceed the budget, so re-gate the same way Min.
        px/mm itself does; loosening it can only ever relax the ceiling, so
        the gate is a no-op in that direction."""
        try:
            new_max_mp = float(self.max_megapixels.get())
        except ValueError:
            return
        if new_max_mp > 1e-9:
            try:
                floor_density = float(self.min_px_per_mm.get())
            except ValueError:
                floor_density = float(DEFAULTS["min_px_per_mm"])
            if not self._confirm_canvas_shape(float(self.width_mm.get()),
                                               float(self.height_mm.get()),
                                               floor_density, max_mp=new_max_mp):
                return
        self._on_canvas_shape_changed()

    # ------------------------------------------------------------------ #
    #  Source tabs
    # ------------------------------------------------------------------ #
    def _make_layer_tab(self, layer):
        tab = ttk.Frame(self._layers_nb, padding=(8, 6))
        self._layers_nb.add(tab, text="")
        entry = {'tab': tab, 'layer': layer}

        # Claim bottom space before content so buttons stay pinned to the
        # bottom, mirroring Steps' _add_step.
        bf = ttk.Frame(tab)
        bf.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(1, weight=1)

        self._build_layer_tab_widgets(tab, entry)

        ttk.Button(bf, text="Duplicate",
                   command=lambda: self._duplicate_layer_tab(entry)
                   ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        ttk.Button(bf, text="Add Source",
                   command=lambda: self._add_source_after(entry)
                   ).grid(row=1, column=0, sticky="ew", padx=(0, 2))
        entry['remove_btn'] = ttk.Button(
            bf, text="Remove this source",
            command=lambda: self._remove_layer_tab(entry))
        entry['remove_btn'].grid(row=1, column=1, sticky="ew", padx=(2, 0))
        return entry

    def _build_layer_tab_widgets(self, tab, entry):
        layer = entry['layer']
        is_base = self.layers.index(layer) == 0

        # The source's file, shown in a read-only combobox styled like the
        # Mirror/Blend rows below; clicking it (or Return/space/Down while it
        # has focus) opens a file dialog instead of a dropdown, to load a
        # different file into this source while keeping every setting below.
        file_row = ttk.Frame(tab)
        file_row.pack(fill=tk.X, pady=2)
        ttk.Label(file_row, text="File").pack(side=tk.LEFT)
        file_cb = ttk.Combobox(file_row, values=(), state="readonly", width=22,
                               cursor="hand2")
        file_cb.set(os.path.basename(layer.path) if layer.path else "(no file)")
        file_cb.pack(side=tk.RIGHT)

        def _open_replace(_event=None):
            self._replace_layer_source(entry)
            return "break"   # suppress the (empty) dropdown list
        for seq in ("<Button-1>", "<Return>", "<space>", "<Down>"):
            file_cb.bind(seq, _open_replace)
        if layer.path:
            file_cb.bind("<Enter>", lambda _e: self.status_var.set(
                f"{layer.path}  —  click to replace this source's file"))
            file_cb.bind("<Leave>", lambda _e: self.status_var.set(""))

        ox = LabeledEntry(tab, "Offset X", "mm", str(layer.offset_x_mm),
                          minval=-100000.0, maxval=100000.0, increment=1.0)
        ox.pack(fill=tk.X, pady=2)
        oy = LabeledEntry(tab, "Offset Y", "mm", str(layer.offset_y_mm),
                          minval=-100000.0, maxval=100000.0, increment=1.0)
        oy.pack(fill=tk.X, pady=2)
        rot = LabeledEntry(tab, "Rotation", "deg", str(layer.rotation_deg),
                           minval=-360.0, maxval=360.0, increment=1.0)
        rot.pack(fill=tk.X, pady=2)
        scale = LabeledEntry(tab, "Scale", "%", str(layer.scale_pct),
                             minval=1.0, maxval=1000.0, increment=1.0)
        scale.pack(fill=tk.X, pady=2)

        # Reflects the placed source across a stock flip axis, for setting
        # up the back side of a two-sided job (see layers._mirror_axis_mm).
        mirror_row = ttk.Frame(tab)
        mirror_row.pack(fill=tk.X, pady=2)
        ttk.Label(mirror_row, text="Mirror").pack(side=tk.LEFT)
        mirror_var = tk.StringVar(value=layer.mirror)
        ttk.Combobox(mirror_row, textvariable=mirror_var,
                     values=LAYER_MIRRORS, state="readonly", width=22
                     ).pack(side=tk.RIGHT)

        blend_row = ttk.Frame(tab)
        blend_row.pack(fill=tk.X, pady=2)
        ttk.Label(blend_row, text="Blend mode").pack(side=tk.LEFT)
        blend_var = tk.StringVar(value=layer.blend_mode)
        if is_base:
            # Forced to Replace at composite time regardless of what's
            # stored (see layers.composite) -- so it isn't user-editable.
            ttk.Label(blend_row, text="Replace (base source)",
                      foreground="#666666").pack(side=tk.RIGHT)
        else:
            ttk.Combobox(blend_row, textvariable=blend_var,
                        values=BLEND_MODES, state="readonly", width=22
                        ).pack(side=tk.RIGHT)

        opacity = LabeledEntry(tab, "Opacity", "%", str(layer.opacity_pct),
                               minval=0.0, maxval=100.0, increment=5.0)
        opacity.pack(fill=tk.X, pady=2)
        # Max below min is allowed: it flips the source's polarity.
        depth_range = LabeledRangeEntry(tab, "Depth range", "%",
                                        str(layer.min_depth_pct), str(layer.max_depth_pct),
                                        minval=0.0, maxval=100.0, increment=5.0)
        depth_range.pack(fill=tk.X, pady=2)

        # Visible is per-STEP (see step['source_visibility']), not a
        # property of the Layer itself -- its checked state reflects
        # whichever step tab is currently selected, and toggling it edits
        # that step's own list, not layer.visible (which is only used as a
        # fallback default -- see _composite_with_visibility). Kept on a
        # dedicated trace (not the combined _apply below) so re-syncing
        # this checkbox on a step-tab switch doesn't re-touch every other
        # field.
        idx = self.layers.index(layer)
        step = self._current_step()
        initial_visible = (step['source_visibility'][idx]
                           if step is not None and idx < len(step['source_visibility'])
                           else layer.visible)
        visible_var = tk.BooleanVar(value=initial_visible)
        ttk.Checkbutton(tab, text="Visible", variable=visible_var
                        ).pack(anchor=tk.W, pady=2)
        entry['visible_var'] = visible_var

        def _apply_visible(*_):
            if getattr(self, '_suspend_visibility_apply', False):
                return
            cur_step = self._current_step()
            if cur_step is not None:
                vis = cur_step['source_visibility']
                i = self.layers.index(layer)
                if i < len(vis):
                    vis[i] = visible_var.get()
            self._on_layer_property_changed()
        visible_var.trace_add("write", _apply_visible)

        def _apply(*_):
            try:
                layer.offset_x_mm   = float(ox.get())
                layer.offset_y_mm   = float(oy.get())
                layer.rotation_deg  = float(rot.get())
                layer.scale_pct     = float(scale.get())
                layer.opacity_pct   = float(opacity.get())
                layer.min_depth_pct = float(depth_range.lo_entry.get())
                layer.max_depth_pct = float(depth_range.hi_entry.get())
            except ValueError:
                return   # mid-edit invalid text -- skip silently
            layer.blend_mode = blend_var.get()   # no-op when is_base
            layer.mirror     = mirror_var.get()
            self._on_layer_property_changed()

        for w in (ox, oy, rot, scale, opacity,
                  depth_range.lo_entry, depth_range.hi_entry):
            w.committed_var.trace_add("write", _apply)
        blend_var.trace_add("write", _apply)
        mirror_var.trace_add("write", _apply)

    def _renumber_layer_tabs(self):
        for i, entry in enumerate(self._layer_tabs, start=1):
            self._layers_nb.tab(entry['tab'], text=f"{i}: {entry['layer'].name}")

    def _update_layer_remove_buttons(self):
        for entry in self._layer_tabs:
            entry['remove_btn'].config(state=tk.NORMAL)

    def _select_layer_tab(self, layer):
        for entry in self._layer_tabs:
            if entry['layer'] is layer:
                self._layers_nb.select(entry['tab'])
                return

    def _on_layers_tab_changed(self, *_):
        """Keep self._layers_nb's -height pinned to whichever tab is
        currently selected (see _build_layers_notebook for why)."""
        try:
            tab_id = self._layers_nb.select()
        except tk.TclError:
            return
        if not tab_id:
            return
        tab = self._layers_nb.nametowidget(tab_id)
        # A freshly built/packed tab hasn't had its requested size computed
        # yet -- pack's geometry pass is deferred to idle time, so reading
        # winfo_reqheight() right after construction (startup, or right
        # after _refresh_layer_tabs() rebuilds tabs) sees a stale ~1px
        # placeholder, not the real content height. Force that pass now.
        tab.update_idletasks()
        self._layers_nb.configure(height=tab.winfo_reqheight())

    def _refresh_layer_tabs(self):
        """(Re)build every source tab from self.layers. Full teardown/
        rebuild rather than Steps' incremental insert/forget: sources
        restructure far less often than steps get added/removed, and
        unlike Steps there's no 'insert after the currently selected tab'
        position to preserve -- append-only plus arbitrary-position
        Duplicate is simplest to get right by just recomputing from
        scratch."""
        selected_layer = None
        if self._layer_tabs:
            try:
                current = self._layers_nb.index("current")
                if current > 0:   # 0 is the Canvas tab
                    selected_layer = self._layer_tabs[current - 1]['layer']
            except tk.TclError:
                pass

        for entry in self._layer_tabs:
            self._layers_nb.forget(entry['tab'])
            entry['tab'].destroy()

        self._layer_tabs = [self._make_layer_tab(layer) for layer in self.layers]
        self._renumber_layer_tabs()
        self._update_layer_remove_buttons()

        if selected_layer is not None:
            self._select_layer_tab(selected_layer)
        # Forgetting the previously-selected Source tab above can make Tk
        # briefly reselect the Canvas tab before it's set back above; make
        # sure the notebook's pinned height reflects whichever tab actually
        # ends up selected, not a stale mid-rebuild snapshot.
        self._on_layers_tab_changed()

    def _remove_layer_tab(self, entry):
        idx = self.layers.index(entry['layer'])
        self.layers.remove(entry['layer'])
        for s in self.steps:
            vis = s.setdefault('source_visibility', [])
            if idx < len(vis):
                del vis[idx]
        self._on_layers_structure_changed()

    def _duplicate_layer_tab(self, entry):
        import copy
        dup = copy.copy(entry['layer'])
        if dup.stl_params is not None:
            dup.stl_params = dict(dup.stl_params)
        dup.layer_id = self._new_layer_id()
        dup.name = f"{entry['layer'].name} copy"
        idx = self.layers.index(entry['layer'])
        self.layers.insert(idx + 1, dup)
        # The duplicate starts visible/hidden in each step exactly like the
        # source it was copied from, matching how its other settings (via
        # copy.copy) are cloned rather than reset to defaults.
        for s in self.steps:
            vis = s.setdefault('source_visibility', [])
            vis.insert(idx + 1, vis[idx] if idx < len(vis) else True)
        self._on_layers_structure_changed()
        self._select_layer_tab(dup)

    def _add_step(self):
        """Insert a new step tab after the currently selected tab."""
        # Capture the currently selected step for cloning its settings.
        src = None
        if self.steps:
            try:
                src = self.steps[self._steps_nb.index("current")]
            except Exception:
                src = self.steps[-1]

        # Insert position: after the selected tab, or 0 on first call.
        if self.steps:
            try:
                insert_at = self._steps_nb.index("current") + 1
            except Exception:
                insert_at = len(self.steps)
        else:
            insert_at = 0

        tab = ttk.Frame(self._steps_nb, padding=(8, 6))
        if insert_at < len(self._steps_nb.tabs()):
            self._steps_nb.insert(insert_at, tab, text=f"Step {insert_at + 1}")
        else:
            self._steps_nb.add(tab, text=f"Step {insert_at + 1}")

        # source_visibility is a list index-aligned with self.layers (True =
        # visible in this step); cloned from src when adding after an
        # existing step, or all-visible when this is the very first step.
        step = {'tab': tab,
                'source_visibility': (list(src['source_visibility']) if src is not None
                                      else [True] * len(self.layers))}
        self.steps.insert(insert_at, step)
        self.toolpaths.insert(insert_at, None)

        # Claim bottom space before content so buttons stay pinned to the bottom.
        bf = ttk.Frame(tab)
        bf.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(1, weight=1)

        self._build_step_widgets(tab, step)

        if src is not None:
            self._copy_step_settings(src, step)

        # Renumber tabs that shifted due to the insertion.
        for i in range(insert_at + 1, len(self.steps)):
            s = self.steps[i]
            self._steps_nb.tab(s['tab'], text=f"Step {i + 1}")
            txt = s['info_var'].get()
            s['info_var'].set(f"Step {i + 1}:" + txt.split(":", 1)[1])

        # 2×2 button grid — all four buttons share equal width and height
        ttk.Button(bf, text="Compute",
                   command=lambda s=step: self._compute_step(s)
                   ).grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(0, 2))
        ttk.Button(bf, text="Dress-up…",
                   command=lambda s=step: self._open_dressup(s)
                   ).grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=(0, 2))
        ttk.Button(bf, text="Add step",
                   command=self._add_step
                   ).grid(row=1, column=0, sticky="ew", padx=(0, 2), pady=(2, 0))
        step['remove_btn'] = ttk.Button(
            bf, text="Remove this step",
            command=lambda s=step: self._remove_step(s))
        step['remove_btn'].grid(row=1, column=1, sticky="ew", padx=(2, 0), pady=(2, 0))

        # Info label (shown in the G-code Info area below the notebook)
        step['info_var'] = tk.StringVar(value=f"Step {insert_at + 1}: not computed")
        lbl = ttk.Label(self._info_frame, textvariable=step['info_var'],
                        wraplength=230, justify=tk.LEFT)
        step['info_label'] = lbl
        # Repack all info labels in step order so the new one appears correctly.
        for s in self.steps:
            s['info_label'].pack_forget()
        for s in self.steps:
            s['info_label'].pack(anchor=tk.W, pady=(0, 4))

        self._update_remove_buttons()
        self.display_mode_var.set(DISPLAY_ORIGINAL)
        self._steps_nb.select(tab)

    def _copy_step_settings(self, src, dst):
        """Copy all user-editable settings from src step dict to dst step dict."""
        dst['algorithm_var'].set(src['algorithm_var'].get())
        self._set_step_profile_nodes(dst, copy.deepcopy(src['profile_nodes']))
        dst['profile_var'].set(src['profile_var'].get())
        for key in ('diameter', 'stepover', 'raster_offset', 'stepdown', 'raster_final_stepdown',
                    'extra_offset',
                    'vbit_angle', 'vcarve_stepdown', 'vcarve_final_stepdown',
                    'cutout_depth', 'cutout_stepdown', 'cutout_final_stepdown', 'cutout_extra_offset',
                    'contour_z_increment', 'contour_lateral_tol', 'feed_rate',
                    'plunge_rate', 'spindle_rpm', 'threshold',
                    'min_fw', 'max_fw', 'pocket_boundary_offset',
                    'reghole_diameter', 'reghole_depth', 'reghole_inset', 'reghole_spacing',
                    'undercut_z_increment'
                    ) + _UNDERCUT_KEYS:
            # Auto first, like _apply_step_dict: an Auto sync fired by the
            # value write must see the copied Auto state, not the default.
            if dst[key].auto_var is not None and src[key].auto_var is not None:
                dst[key].auto_var.set(src[key].auto_var.get())
            dst[key].set(src[key].get())
        dst['z_range'].set(*src['z_range'].get())
        dst['raster_dir_var'].set(src['raster_dir_var'].get())
        for key in ('library_material_id_var', 'library_tool_name_var',
                    'library_tool_id_var'):
            dst[key].set(src[key].get())
        dst['reghole_pattern_var'].set(src['reghole_pattern_var'].get())
        dst['reghole_diag_dir_var'].set(src['reghole_diag_dir_var'].get())
        dst['contour_adaptive_var'].set(src['contour_adaptive_var'].get())
        dst['undercut_open_var'].set(src['undercut_open_var'].get())
        dst['tsp_2opt_var'].set(src['tsp_2opt_var'].get())
        dst['avoid_air_var'].set(src['avoid_air_var'].get())
        dst['gating_mode_var'].set(src['gating_mode_var'].get())
        for key in ('ramp_in_enabled', 'ramp_angle_deg', 'ramp_max_length_mm',
                    'ramp_start_z_mm', 'ramp_max_depth_mm',
                    'backoff_enabled', 'backoff_distance_mm',
                    'arc_rapids_enabled', 'arc_radius_mm'):
            dst[f'{key}_var'].set(src[f'{key}_var'].get())

    def _remove_step(self, step):
        """Remove a step tab and its associated data."""
        idx = self.steps.index(step)
        self._steps_nb.forget(step['tab'])
        step['info_label'].destroy()
        self.steps.pop(idx)
        self.toolpaths.pop(idx)
        # Renumber remaining tabs and info labels
        for i, s in enumerate(self.steps):
            self._steps_nb.tab(s['tab'], text=f"Step {i + 1}")
            if self.toolpaths[i] is None:
                s['info_var'].set(f"Step {i + 1}: not computed")
            else:
                # Preserve existing computed text, just update the step number
                txt = s['info_var'].get()
                s['info_var'].set(f"Step {i + 1}:" + txt.split(":", 1)[1])

        self._update_remove_buttons()

    def _update_remove_buttons(self):
        state = tk.DISABLED if len(self.steps) <= 1 else tk.NORMAL
        for s in self.steps:
            s['remove_btn'].config(state=state)

    def _build_step_widgets(self, frame, step):
        """Populate a step tab with all tool-parameter widgets.
        Widget references are stored in the step dict. Values from DEFAULTS."""

        # ── Tool path dress-up (no visible widgets; edited via the
        #    "Dress-up" button/dialog) ──────────────────────────────────
        step['ramp_in_enabled_var']     = tk.BooleanVar(
            value=bool(int(DEFAULTS['step_ramp_in_enabled'])))
        step['ramp_angle_deg_var']      = tk.DoubleVar(
            value=float(DEFAULTS['step_ramp_angle_deg']))
        step['ramp_max_length_mm_var']  = tk.DoubleVar(
            value=float(DEFAULTS['step_ramp_max_length_mm']))
        step['ramp_start_z_mm_var']     = tk.DoubleVar(
            value=float(DEFAULTS['step_ramp_start_z_mm']))
        step['ramp_max_depth_mm_var']   = tk.DoubleVar(
            value=float(DEFAULTS['step_ramp_max_depth_mm']))
        step['backoff_enabled_var']     = tk.BooleanVar(
            value=bool(int(DEFAULTS['step_backoff_enabled'])))
        step['backoff_distance_mm_var'] = tk.DoubleVar(
            value=float(DEFAULTS['step_backoff_distance_mm']))
        step['arc_rapids_enabled_var']  = tk.BooleanVar(
            value=bool(int(DEFAULTS['step_arc_rapids_enabled'])))
        step['arc_radius_mm_var']       = tk.DoubleVar(
            value=float(DEFAULTS['step_arc_radius_mm']))
        step['arc_feed_mm_min_var']     = tk.DoubleVar(
            value=float(DEFAULTS['step_arc_feed_mm_min']))
        step['tab_enabled_var']         = tk.BooleanVar(
            value=bool(int(DEFAULTS['step_tab_enabled'])))
        step['tab_count_var']           = tk.IntVar(
            value=int(DEFAULTS['step_tab_count']))
        step['tab_width_mm_var']        = tk.DoubleVar(
            value=float(DEFAULTS['step_tab_width_mm']))
        step['tab_height_mm_var']       = tk.DoubleVar(
            value=float(DEFAULTS['step_tab_height_mm']))
        step['tab_phase_pct_var']       = tk.DoubleVar(
            value=float(DEFAULTS['step_tab_phase_pct']))
        step['tab_angle_deg_var']       = tk.DoubleVar(
            value=float(DEFAULTS['step_tab_angle_deg']))
        step['_cutout_rings_cache']     = None

        # ── Operation group ────────────────────────────────────────────
        operation_lf = ttk.LabelFrame(frame, text="Operation", padding=4)
        operation_lf.pack(fill=tk.X, pady=(0, 4))
        step['operation_lf'] = operation_lf

        alg_row = ttk.Frame(operation_lf)
        alg_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(alg_row, text="Algorithm").pack(side=tk.LEFT)
        algorithm_var = tk.StringVar(value=DEFAULTS["step_algorithm"])
        step['algorithm_var'] = algorithm_var
        ttk.Combobox(alg_row, textvariable=algorithm_var,
                     values=ALGORITHMS, state="readonly", width=22
                     ).pack(side=tk.RIGHT)
        step['alg_row'] = alg_row

        # ── Raster-only op controls (hidden when V-Carving or Contour) ─
        raster_frame = ttk.Frame(operation_lf)
        raster_frame.pack(fill=tk.X, after=alg_row)   # initially shown
        step['raster_frame'] = raster_frame

        stepover = LabeledEntry(raster_frame, "Step-over", "%", DEFAULTS["step_stepover"],
                                minval=1.0, maxval=100.0, increment=10.0)
        stepover.pack(fill=tk.X, pady=2)
        step['stepover'] = stepover

        raster_offset = LabeledEntry(raster_frame, "Line offset", "%", DEFAULTS["step_raster_offset"],
                                     minval=0.0, maxval=99.0, increment=10.0)
        raster_offset.pack(fill=tk.X, pady=2)
        step['raster_offset'] = raster_offset

        dir_row = ttk.Frame(raster_frame)
        dir_row.pack(fill=tk.X, pady=2)
        step['raster_dir_row'] = dir_row
        ttk.Label(dir_row, text="Scan direction").pack(side=tk.LEFT)
        dir_var = tk.StringVar(value=DEFAULTS["step_raster_dir"])
        step['raster_dir_var'] = dir_var
        raster_dir_cb = ttk.Combobox(dir_row, textvariable=dir_var,
                                     values=RASTER_DIRECTIONS, state="readonly", width=22)
        raster_dir_cb.pack(side=tk.RIGHT)
        step['raster_dir_cb'] = raster_dir_cb

        cut_dir_row = ttk.Frame(operation_lf)
        cut_dir_row.pack(fill=tk.X, pady=2, after=raster_frame)   # initially shown (default = Raster Scan)
        step['cut_dir_row'] = cut_dir_row
        ttk.Label(cut_dir_row, text="Cut direction").pack(side=tk.LEFT)
        cut_dir_var = tk.StringVar(value=DEFAULTS["step_cut_direction"])
        step['cut_dir_var'] = cut_dir_var
        ttk.Combobox(cut_dir_row, textvariable=cut_dir_var,
                     values=CUT_DIRECTIONS, state="readonly", width=22
                     ).pack(side=tk.RIGHT)

        # ── Contour-only op controls (hidden unless Contour algorithm) ──
        contour_frame = ttk.Frame(operation_lf)
        step['contour_frame'] = contour_frame
        # (not packed initially — shown on demand by _refresh_layout)

        contour_z_increment = LabeledEntry(contour_frame, "Z increment", "mm",
                                        DEFAULTS["step_contour_z_increment"],
                                        minval=0.01, maxval=50.0, increment=0.1)
        contour_z_increment.pack(fill=tk.X, pady=2)
        step['contour_z_increment'] = contour_z_increment

        contour_adaptive_var = tk.BooleanVar(
            value=bool(int(DEFAULTS["step_contour_adaptive"])))
        ttk.Checkbutton(contour_z_increment, text="Adaptive",
                        variable=contour_adaptive_var
                        ).pack(side=tk.RIGHT, padx=(0, 4))
        step['contour_adaptive_var'] = contour_adaptive_var

        contour_lateral_tol = LabeledEntry(contour_frame, "Lateral tolerance", "mm",
                                           DEFAULTS["step_contour_lateral_tol"],
                                           minval=0.0, maxval=50.0, increment=0.05)
        contour_lateral_tol.pack(fill=tk.X, pady=2)
        step['contour_lateral_tol'] = contour_lateral_tol

        # ── Depth band (Raster Scan, Contour, Undercut) ────────────────
        # One widget, packed by _refresh_layout into whichever frame holds
        # the algorithm's depth controls (pack in_= works because every
        # such frame is a descendant of operation_lf).
        z_range = LabeledRangeEntry(operation_lf, "Z range", "%",
                                    DEFAULTS["step_z_min_pct"],
                                    DEFAULTS["step_z_max_pct"],
                                    minval=0.0, maxval=100.0, increment=5.0)
        step['z_range'] = z_range

        # ── Undercut-only op control (hidden unless Undercut algorithm) ──
        # Not part of the library's cutting data, unlike Max step-down.
        undercut_z_increment = LabeledEntry(operation_lf, "Level step-down", "mm",
                                            DEFAULTS["step_undercut_z_increment"],
                                            minval=0.01, maxval=50.0, increment=0.1)
        step['undercut_z_increment'] = undercut_z_increment
        # (not packed initially — shown on demand by _refresh_layout)

        undercut_open_row = ttk.Frame(operation_lf)
        step['undercut_open_row'] = undercut_open_row
        ttk.Label(undercut_open_row, text="Open undercuts").pack(side=tk.LEFT)
        undercut_open_var = tk.StringVar(value=DEFAULTS["step_undercut_open"])
        step['undercut_open_var'] = undercut_open_var
        ttk.Combobox(undercut_open_row, textvariable=undercut_open_var,
                     values=list(UNDERCUT_OPEN_MODES), state="readonly", width=22
                     ).pack(side=tk.RIGHT)
        # (not packed initially — shown on demand by _refresh_layout)

        # ── Registration-Holes-only controls (hidden unless that algorithm)
        reghole_frame = ttk.Frame(operation_lf)
        step['reghole_frame'] = reghole_frame
        # (not packed initially — shown on demand by _refresh_layout)

        reghole_pattern_row = ttk.Frame(reghole_frame)
        reghole_pattern_row.pack(fill=tk.X, pady=2)
        step['reghole_pattern_row'] = reghole_pattern_row
        ttk.Label(reghole_pattern_row, text="Flip axis").pack(side=tk.LEFT)
        reghole_pattern_var = tk.StringVar(value=DEFAULTS["step_reghole_pattern"])
        step['reghole_pattern_var'] = reghole_pattern_var
        ttk.Combobox(reghole_pattern_row, textvariable=reghole_pattern_var,
                     values=REG_HOLE_PATTERNS, state="readonly", width=22
                     ).pack(side=tk.RIGHT)

        # Only a Center origin leaves the Diagonal line's direction open --
        # a corner origin implies it (see registration_flip_axis).
        reghole_diag_dir_row = ttk.Frame(reghole_frame)
        step['reghole_diag_dir_row'] = reghole_diag_dir_row
        ttk.Label(reghole_diag_dir_row, text="Diagonal direction").pack(side=tk.LEFT)
        reghole_diag_dir_var = tk.StringVar(value=DEFAULTS["step_reghole_diag_dir"])
        step['reghole_diag_dir_var'] = reghole_diag_dir_var
        ttk.Combobox(reghole_diag_dir_row, textvariable=reghole_diag_dir_var,
                     values=DIAG_DIRECTIONS, state="readonly", width=22
                     ).pack(side=tk.RIGHT)

        reghole_diameter = LabeledEntry(reghole_frame, "Hole diameter", "mm",
                                        DEFAULTS["step_reghole_diameter"],
                                        minval=0.01, maxval=200.0, increment=0.5)
        reghole_diameter.pack(fill=tk.X, pady=2)
        step['reghole_diameter'] = reghole_diameter

        reghole_depth = LabeledEntry(reghole_frame, "Hole depth", "mm",
                                     DEFAULTS["step_reghole_depth"],
                                     minval=0.01, maxval=500.0, increment=0.5)
        reghole_depth.pack(fill=tk.X, pady=2)
        step['reghole_depth'] = reghole_depth

        reghole_inset = LabeledEntry(reghole_frame, "Edge inset", "mm",
                                     DEFAULTS["step_reghole_inset"], auto_label="Auto",
                                     minval=0.0, maxval=100000.0, increment=1.0)
        reghole_inset.pack(fill=tk.X, pady=2)
        reghole_inset.auto_var.set(bool(int(DEFAULTS["step_reghole_inset_auto"])))
        step['reghole_inset'] = reghole_inset

        reghole_spacing = LabeledEntry(reghole_frame, "Spacing", "mm",
                                       DEFAULTS["step_reghole_spacing"], auto_label="Auto",
                                       minval=0.0, maxval=100000.0, increment=1.0)
        reghole_spacing.auto_var.set(bool(int(DEFAULTS["step_reghole_spacing_auto"])))
        step['reghole_spacing'] = reghole_spacing
        # (packed by _refresh_layout, Diagonal only)

        step['reghole_hint_var'] = tk.StringVar(value="")
        reghole_hint = ttk.Label(reghole_frame, textvariable=step['reghole_hint_var'],
                                 foreground="#666666", wraplength=230, justify=tk.LEFT)
        reghole_hint.pack(fill=tk.X, pady=(2, 0))
        step['reghole_hint'] = reghole_hint

        tsp_frame = ttk.Frame(operation_lf)
        step['tsp_frame'] = tsp_frame

        tsp_2opt_var = tk.BooleanVar(
            value=bool(int(DEFAULTS["step_tsp_2opt"])))
        tsp_2opt_cb = ttk.Checkbutton(tsp_frame, text="Optimize path order (2-opt)",
                                      variable=tsp_2opt_var)
        tsp_2opt_cb.pack(anchor=tk.W, pady=2)
        step['tsp_2opt_var'] = tsp_2opt_var
        step['tsp_2opt_cb'] = tsp_2opt_cb

        avoid_air_var = tk.BooleanVar(
            value=bool(int(DEFAULTS["step_avoid_air"])))
        avoid_air_cb = ttk.Checkbutton(tsp_frame, text="Avoid cutting air",
                                       variable=avoid_air_var)
        avoid_air_cb.pack(anchor=tk.W, pady=2)
        step['avoid_air_var'] = avoid_air_var
        step['avoid_air_cb'] = avoid_air_cb

        gating_frame = ttk.Frame(operation_lf)
        gating_frame.pack(fill=tk.X)
        step['gating_frame'] = gating_frame

        tsp_frame.pack(fill=tk.X)   # always last; packed after gating_frame

        region_row = ttk.Frame(gating_frame)
        region_row.pack(fill=tk.X, pady=(0, 4))
        step['gating_region_row'] = region_row
        ttk.Label(region_row, text="Process region").pack(side=tk.LEFT)
        gating_var = tk.StringVar(value=DEFAULTS["step_gating_mode"])
        step['gating_mode_var'] = gating_var
        gating_cb = ttk.Combobox(region_row, textvariable=gating_var,
                                 values=self._gating_mode_values(step),
                                 state="readonly", width=22)
        gating_cb.configure(postcommand=lambda: gating_cb.configure(
            values=self._gating_mode_values(step)))
        gating_cb.pack(side=tk.RIGHT)
        step['gating_cb'] = gating_cb

        threshold = LabeledEntry(gating_frame, "Z-threshold", "mm", DEFAULTS["step_threshold"],
                                 auto_label="Auto", minval=0.0, maxval=200.0, increment=0.1)
        threshold.pack(fill=tk.X, pady=2)
        threshold.set_auto_visible(False)   # only meaningful for Pocket's Target
                                             # depth -- _refresh_layout shows it
                                             # there and hides it everywhere else
        step['threshold'] = threshold

        min_fw = LabeledEntry(gating_frame, "Min feature width", "mm", DEFAULTS["step_min_fw"],
                              minval=0.0, maxval=200.0, increment=0.1)
        step['min_fw'] = min_fw
        max_fw = LabeledEntry(gating_frame, "Max feature width", "mm", DEFAULTS["step_max_fw"],
                              auto_label="Auto", minval=0.0, maxval=200.0, increment=0.1)
        step['max_fw'] = max_fw
        # min_fw/max_fw packed/unpacked by _refresh_layout below

        # ── Tool group ─────────────────────────────────────────────────
        tool_lf = ttk.LabelFrame(frame, text="Tool", padding=4)
        tool_lf.pack(fill=tk.X, pady=(0, 4))
        step['tool_lf'] = tool_lf

        step['library_tool_id_var']     = tk.StringVar(value="")
        step['library_tool_name_var']   = tk.StringVar(value="")
        step['library_material_id_var'] = tk.StringVar(value="")
        step['library_cb_ids'] = [""]   # tool id per combobox entry ("" = none)
        library_row = ttk.Frame(tool_lf)
        library_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(library_row, text="Library").pack(side=tk.LEFT)
        library_menu_btn = ttk.Menubutton(library_row, text="⋯", width=2)
        library_menu_btn.pack(side=tk.RIGHT, padx=(4, 0))
        library_menu = tk.Menu(library_menu_btn, tearoff=0)
        library_menu.add_command(label="Re-apply from library",
                                 command=lambda: self._reapply_library_tool(step))
        library_menu.add_command(label="Update library tool from this step…",
                                 command=lambda: self._update_library_tool_from_step(step))
        library_menu.add_command(label="Save as new library tool…",
                                 command=lambda: self._save_step_as_library_tool(step))
        library_menu.add_command(label="Unlink",
                                 command=lambda: self._unlink_library_tool(step))
        library_menu.add_separator()
        library_menu.add_command(
            label="Open Tool Library…",
            command=lambda: self._open_tool_library(
                step['library_tool_id_var'].get() or None))
        library_menu_btn['menu'] = library_menu
        step['library_display_var'] = tk.StringVar(value="(none)")
        library_cb = ttk.Combobox(library_row, textvariable=step['library_display_var'],
                                  state="readonly", width=22)
        library_cb.pack(side=tk.RIGHT)
        library_cb.bind("<<ComboboxSelected>>",
                        lambda _e: self._on_library_tool_chosen(step))
        step['library_cb'] = library_cb
        step['tool_status_var'] = tk.StringVar(value="")
        ttk.Label(tool_lf, textvariable=step['tool_status_var'],
                  foreground="#c07000").pack(anchor=tk.E)

        profile_row = ttk.Frame(tool_lf)
        profile_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(profile_row, text="Profile").pack(side=tk.LEFT)
        profile_var = tk.StringVar(value=DEFAULTS["step_profile"])
        step['profile_var'] = profile_var
        profile_cb = ttk.Combobox(profile_row, textvariable=profile_var,
                                  values=TOOL_PROFILES, state="readonly", width=22)
        profile_cb.pack(side=tk.RIGHT)
        step['profile_cb'] = profile_cb

        diam = LabeledEntry(tool_lf, "Diameter", "mm", DEFAULTS["step_diameter"],
                            minval=0.01, maxval=200.0, increment=1.0)
        diam.pack(fill=tk.X, pady=2)
        step['diameter'] = diam

        # A Custom profile's Bezier nodes, copied from its library tool,
        # where the curve is edited.
        step['profile_nodes'] = None

        # ── Undercut-tool geometry (shown for the Undercut algorithm) ────
        undercut_frame = ttk.Frame(tool_lf)
        step['undercut_frame'] = undercut_frame
        # (not packed initially — shown on demand by _refresh_layout)
        for key, label, unit, lo_, hi_, inc in (
                ('neck_diameter',  "Neck diameter",  "mm", 0.01, 200.0, 0.5),
                ('neck_length',    "Neck length",    "mm", 0.0, 500.0, 1.0),
                ('shank_diameter', "Shank diameter", "mm", 0.0, 200.0, 0.5),
                ('head_height',    "Head height",    "mm", 0.01, 100.0, 0.5),
                ('dovetail_angle', "Dovetail angle", "°", 1.0, 89.0, 5.0)):
            le = LabeledEntry(undercut_frame, label, unit, DEFAULTS[f"step_{key}"],
                              minval=lo_, maxval=hi_, increment=inc)
            le.pack(fill=tk.X, pady=2)
            step[key] = le

        # ── Raster step-down controls (shown for Raster Scan only) ───────
        stepdown = LabeledEntry(tool_lf, "Max step-down", "mm", DEFAULTS["step_stepdown"],
                                minval=0.01, maxval=50.0, increment=0.1)
        stepdown.pack(fill=tk.X, pady=2)   # initially shown (default = Raster Scan)
        step['stepdown'] = stepdown

        raster_final_stepdown = LabeledEntry(tool_lf, "Final step-down", "mm",
                                             DEFAULTS["step_raster_final_stepdown"],
                                             minval=0.0, maxval=50.0, increment=0.1)
        raster_final_stepdown.pack(fill=tk.X, pady=2)
        step['raster_final_stepdown'] = raster_final_stepdown

        extra_offset = LabeledEntry(tool_lf, "Extra offset", "mm",
                                    DEFAULTS["step_extra_offset"],
                                    minval=0.0, maxval=50.0, increment=0.1)
        extra_offset.pack(fill=tk.X, pady=2)
        step['extra_offset'] = extra_offset

        # Pocket's own lateral-only boundary push -- unlike Extra offset
        # above (lateral+vertical stock-to-leave), a pocket's cut depth is
        # already fixed independently by Target depth, so it gets its own
        # field, same as Cutout's Boundary offset.
        pocket_boundary_offset = LabeledEntry(tool_lf, "Boundary offset", "mm",
                                              DEFAULTS["step_pocket_boundary_offset"],
                                              minval=0.0, maxval=50.0, increment=0.1)
        step['pocket_boundary_offset'] = pocket_boundary_offset
        # (not packed initially — shown on demand by _refresh_layout, Pocket only)

        # ── V-bit angle (shown whenever V-Bit profile is active) ──────
        vbit_frame = ttk.Frame(tool_lf)
        step['vbit_frame'] = vbit_frame
        # (not packed initially — shown on demand)

        vbit_angle = LabeledEntry(vbit_frame, "V-bit angle", "°", DEFAULTS["step_vbit_angle"],
                                  minval=1.0, maxval=179.0, increment=10.0)
        vbit_angle.pack(fill=tk.X, pady=2)
        step['vbit_angle'] = vbit_angle

        # ── V-carving-only controls (hidden when Raster Scan) ─────────
        vcarve_frame = ttk.Frame(tool_lf)
        step['vcarve_frame'] = vcarve_frame
        # (not packed initially — shown on demand by _refresh_layout)

        vcarve_stepdown = LabeledEntry(vcarve_frame, "Max step-down", "mm", DEFAULTS["step_vcarve_stepdown"],
                                       minval=0.0, maxval=50.0, increment=0.1)
        vcarve_stepdown.pack(fill=tk.X, pady=2)
        step['vcarve_stepdown'] = vcarve_stepdown

        vcarve_final_stepdown = LabeledEntry(vcarve_frame, "Final step-down", "mm", DEFAULTS["step_vcarve_final_stepdown"],
                                             minval=0.0, maxval=50.0, increment=0.1)
        vcarve_final_stepdown.pack(fill=tk.X, pady=2)
        step['vcarve_final_stepdown'] = vcarve_final_stepdown

        # ── Cutout-only controls (hidden unless Algorithm == Cutout) ──
        cutout_frame = ttk.Frame(tool_lf)
        step['cutout_frame'] = cutout_frame
        # (not packed initially — shown on demand by _refresh_layout)

        cutout_depth = LabeledEntry(cutout_frame, "Cut depth", "mm",
                                    DEFAULTS["step_cutout_depth"],
                                    minval=0.01, maxval=500.0, increment=0.5)
        cutout_depth.pack(fill=tk.X, pady=2)
        step['cutout_depth'] = cutout_depth

        cutout_stepdown = LabeledEntry(cutout_frame, "Max step-down", "mm",
                                       DEFAULTS["step_cutout_stepdown"],
                                       minval=0.0, maxval=50.0, increment=0.1)
        cutout_stepdown.pack(fill=tk.X, pady=2)
        step['cutout_stepdown'] = cutout_stepdown

        cutout_final_stepdown = LabeledEntry(cutout_frame, "Final step-down", "mm",
                                             DEFAULTS["step_cutout_final_stepdown"],
                                             minval=0.0, maxval=50.0, increment=0.1)
        cutout_final_stepdown.pack(fill=tk.X, pady=2)
        step['cutout_final_stepdown'] = cutout_final_stepdown

        cutout_extra_offset = LabeledEntry(cutout_frame, "Boundary offset", "mm",
                                           DEFAULTS["step_cutout_extra_offset"],
                                           minval=0.0, maxval=50.0, increment=0.1)
        cutout_extra_offset.pack(fill=tk.X, pady=2)
        step['cutout_extra_offset'] = cutout_extra_offset

        feed = LabeledEntry(tool_lf, "Feed rate", "mm/min", DEFAULTS["step_feed_rate"],
                            minval=1.0, maxval=30000.0, increment=50.0)
        feed.pack(fill=tk.X, pady=2)
        step['feed_rate'] = feed

        plunge = LabeledEntry(tool_lf, "Plunge rate", "mm/min", DEFAULTS["step_plunge_rate"],
                              minval=0.0, maxval=30000.0, increment=50.0)
        plunge.pack(fill=tk.X, pady=2)
        step['plunge_rate'] = plunge

        rpm = LabeledEntry(tool_lf, "Spindle speed", "RPM", DEFAULTS["step_spindle_rpm"],
                           minval=0.0, maxval=100000.0, increment=1000.0)
        rpm.pack(fill=tk.X, pady=2)
        step['spindle_rpm'] = rpm

        # ── Wire up algorithm / profile toggle ────────────────────────
        def _place_z_range():
            """Pack the Z range row with the algorithm's own depth controls,
            or hide it for algorithms without one."""
            alg = algorithm_var.get()
            family = ALGORITHM_FAMILY[alg]
            if alg == "Contour":
                step['z_range'].pack(in_=step['contour_frame'], fill=tk.X, pady=2,
                                     after=step['contour_z_increment'])
            elif family == "Undercut":
                step['z_range'].pack(in_=step['operation_lf'], fill=tk.X, pady=2,
                                     after=step['undercut_z_increment'])
            elif family == "Raster Scan":
                step['z_range'].pack(in_=step['gating_frame'], fill=tk.X, pady=2,
                                     after=step['threshold'])
            else:
                step['z_range'].pack_forget()
            # Siblings stack in creation order, so a frame created after
            # z_range (gating_frame) would hide it when it's packed inside.
            step['z_range'].lift()

        def _refresh_layout(*_):
            alg     = algorithm_var.get()
            profile = step['profile_var'].get()
            # V-Carving Clean-up's display name starts with "V-Carving" but
            # its pipeline (and widget layout) is Pocket's -- dispatch on
            # the family it actually belongs to, not the display string.
            family  = ALGORITHM_FAMILY[alg]

            # Cutout reuses the same field to split "object" from
            # "background" rather than gate on depth (its own "0 = auto,
            # split at the depth-map midpoint" sentinel, same convention as
            # Z-threshold's own "0 = auto" meaning for V-Carving) -- and
            # Pocket (and V-Carving Clean-up, which shares Pocket's Target
            # depth field) reuses it for Target depth, taken literally
            # (including 0, a full-stock flattening pass) with its own Auto
            # checkbox instead -- different enough concepts that each gets
            # its own label.
            step['threshold'].set_label(
                "Split level" if alg == "Cutout" else
                "Target depth" if family == "Pocket" else
                "Z-threshold")
            step['threshold'].set_auto_visible(family == "Pocket")
            if family != "Pocket" and step['threshold'].auto_var is not None:
                step['threshold'].auto_var.set(False)

            if family != "Registration Holes":
                step['reghole_frame'].pack_forget()
            if family != "Undercut":
                step['undercut_frame'].pack_forget()
                step['undercut_z_increment'].pack_forget()
                step['undercut_open_row'].pack_forget()
                if not step['raster_offset'].winfo_manager():
                    step['raster_offset'].pack(fill=tk.X, pady=2, after=step['stepover'])
                if not step['raster_dir_row'].winfo_manager():
                    step['raster_dir_row'].pack(fill=tk.X, pady=2, after=step['raster_offset'])

            if family == "V-Carving":
                step['raster_frame'].pack_forget()
                step['contour_frame'].pack_forget()
                step['stepdown'].pack_forget()
                step['raster_final_stepdown'].pack_forget()
                step['vbit_frame'].pack(fill=tk.X, after=step['diameter'])
                step['vcarve_frame'].pack(fill=tk.X, after=step['vbit_frame'])
                step['min_fw'].pack(fill=tk.X, pady=2)
                step['max_fw'].pack(fill=tk.X, pady=2)
                step['gating_frame'].pack(fill=tk.X)
                step['gating_region_row'].pack(fill=tk.X, pady=(0, 4), before=step['threshold'])
                step['tsp_frame'].pack(fill=tk.X, after=step['gating_frame'])
                if alg == "V-Carving (Hybrid)":
                    # Only the hybrid mode's deep-layer waterline contours
                    # respect cut direction — the medial-axis skeleton used
                    # by the other V-Carving modes has no inside/outside.
                    step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['tsp_frame'])
                else:
                    step['cut_dir_row'].pack_forget()
                allowed = _allowed_profiles(alg)
                step['profile_cb'].config(state="readonly", values=allowed)
                if profile not in allowed:
                    step['profile_var'].set("V-Bit")
                step['vbit_angle'].set_state(
                    "normal" if profile == "V-Bit" else "disabled")
                step['extra_offset'].pack(fill=tk.X, pady=2, after=step['vcarve_frame'])
                step['pocket_boundary_offset'].pack_forget()
                step['cutout_frame'].pack_forget()
            elif alg == "Contour":
                step['raster_frame'].pack_forget()
                step['vcarve_frame'].pack_forget()
                step['raster_final_stepdown'].pack_forget()
                step['min_fw'].pack_forget()
                step['max_fw'].pack_forget()
                step['gating_frame'].pack_forget()
                step['tsp_frame'].pack(fill=tk.X)
                step['profile_cb'].config(state="readonly", values=_RASTER_PROFILES)
                if profile not in _RASTER_PROFILES:
                    step['profile_var'].set("Ball")
                if profile == "V-Bit":
                    step['vbit_frame'].pack(fill=tk.X, after=step['diameter'])
                    step['stepdown'].pack(fill=tk.X, pady=2, after=step['vbit_frame'])
                else:
                    step['vbit_frame'].pack_forget()
                    step['stepdown'].pack(fill=tk.X, pady=2, after=step['diameter'])
                step['extra_offset'].pack(fill=tk.X, pady=2, after=step['stepdown'])
                step['pocket_boundary_offset'].pack_forget()
                step['contour_frame'].pack(fill=tk.X, after=step['alg_row'])
                step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['contour_frame'])
                step['vbit_angle'].set_state("normal")
                step['cutout_frame'].pack_forget()
            elif alg == "Cutout":
                step['raster_frame'].pack_forget()
                step['vcarve_frame'].pack_forget()
                step['contour_frame'].pack_forget()
                step['vbit_frame'].pack_forget()
                step['stepdown'].pack_forget()
                step['raster_final_stepdown'].pack_forget()
                step['extra_offset'].pack_forget()   # Cutout uses cutout_extra_offset instead
                step['pocket_boundary_offset'].pack_forget()
                step['min_fw'].pack_forget()
                step['max_fw'].pack_forget()
                step['gating_frame'].pack(fill=tk.X)
                step['gating_region_row'].pack_forget()   # Cutout always reads
                                                           # prior_accessible/previous_result
                                                           # unconditionally, mirroring Contour --
                                                           # only the Split level field is used.
                step['tsp_frame'].pack(fill=tk.X, after=step['gating_frame'])
                step['profile_cb'].config(state="readonly", values=_CUTOUT_PROFILES)
                if profile not in _CUTOUT_PROFILES:
                    step['profile_var'].set("End Mill")
                step['vbit_angle'].set_state("disabled")
                step['cutout_frame'].pack(fill=tk.X, after=step['diameter'])
                step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['alg_row'])
            elif family == "Registration Holes":
                # Stock-anchored like Cutout, but with no depth map input at
                # all -- no threshold, Process Region, path ordering or
                # air pruning. Max step-down is the helix pitch / peck depth.
                for key in ('raster_frame', 'vcarve_frame', 'contour_frame', 'vbit_frame',
                            'cutout_frame', 'raster_final_stepdown', 'extra_offset',
                            'pocket_boundary_offset', 'min_fw', 'max_fw',
                            'gating_frame', 'tsp_frame'):
                    step[key].pack_forget()
                step['profile_cb'].config(state="readonly", values=_CUTOUT_PROFILES)
                if profile not in _CUTOUT_PROFILES:
                    step['profile_var'].set("End Mill")
                step['vbit_angle'].set_state("disabled")
                step['stepdown'].pack(fill=tk.X, pady=2, after=step['diameter'])
                step['reghole_frame'].pack(fill=tk.X, after=step['alg_row'])
                step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['reghole_frame'])
                diagonal = step['reghole_pattern_var'].get() == "Diagonal"
                if diagonal and self.origin_var.get() == "Center":
                    step['reghole_diag_dir_row'].pack(fill=tk.X, pady=2,
                                                      after=step['reghole_pattern_row'])
                else:
                    step['reghole_diag_dir_row'].pack_forget()
                if diagonal:
                    step['reghole_spacing'].pack(fill=tk.X, pady=2,
                                                 after=step['reghole_inset'])
                else:
                    step['reghole_spacing'].pack_forget()
                self._sync_reghole_fields(step)
            elif family == "Undercut":
                # Uniform depth levels (Level step-down) and ring spacing
                # (Step-over); no Process Region, path-order or air-cut
                # options -- see undercut.py.
                for key in ('vcarve_frame', 'contour_frame', 'cutout_frame', 'vbit_frame',
                            'stepdown', 'raster_final_stepdown', 'pocket_boundary_offset',
                            'min_fw', 'max_fw', 'gating_frame', 'tsp_frame',
                            'raster_offset', 'raster_dir_row'):
                    step[key].pack_forget()
                step['profile_cb'].config(state="readonly", values=_UNDERCUT_PROFILES)
                if profile not in _UNDERCUT_PROFILES:
                    step['profile_var'].set(_UNDERCUT_PROFILES[0])
                    profile = _UNDERCUT_PROFILES[0]
                step['undercut_frame'].pack(fill=tk.X, after=step['diameter'])
                if profile in ("T-Slot", "Dovetail"):
                    step['head_height'].pack(fill=tk.X, pady=2, after=step['shank_diameter'])
                else:
                    step['head_height'].pack_forget()
                if profile == "Dovetail":
                    step['dovetail_angle'].pack(fill=tk.X, pady=2, after=step['head_height'])
                else:
                    step['dovetail_angle'].pack_forget()
                step['extra_offset'].pack(fill=tk.X, pady=2, after=step['undercut_frame'])
                step['undercut_open_row'].pack(fill=tk.X, pady=2, after=step['alg_row'])
                step['undercut_z_increment'].pack(fill=tk.X, pady=2,
                                                  after=step['undercut_open_row'])
                step['raster_frame'].pack(fill=tk.X, after=step['undercut_z_increment'])
                step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['raster_frame'])
            elif family == "Pocket":   # Pocket, V-Carving Clean-up
                step['vcarve_frame'].pack_forget()
                step['contour_frame'].pack_forget()
                step['cutout_frame'].pack_forget()
                step['min_fw'].pack_forget()
                step['max_fw'].pack_forget()
                # No Process Region selector for Pocket/Clean-up -- like
                # Cutout, only the (relabeled) threshold field is used,
                # always against the full depth map.
                step['gating_frame'].pack(fill=tk.X, after=step['alg_row'])
                step['gating_region_row'].pack_forget()
                step['profile_cb'].config(state="readonly", values=_RASTER_PROFILES)
                if profile not in _RASTER_PROFILES:
                    step['profile_var'].set("Ball")
                if profile == "V-Bit":
                    step['vbit_frame'].pack(fill=tk.X, after=step['diameter'])
                    step['stepdown'].pack(fill=tk.X, pady=2, after=step['vbit_frame'])
                else:
                    step['vbit_frame'].pack_forget()
                    step['stepdown'].pack(fill=tk.X, pady=2, after=step['diameter'])
                step['raster_final_stepdown'].pack(fill=tk.X, pady=2,
                                                   after=step['stepdown'])
                step['extra_offset'].pack_forget()   # Pocket uses pocket_boundary_offset instead
                step['pocket_boundary_offset'].pack(fill=tk.X, pady=2,
                                          after=step['raster_final_stepdown'])
                step['raster_dir_cb'].config(values=POCKET_PATTERNS)
                if step['raster_dir_var'].get() not in POCKET_PATTERNS:
                    step['raster_dir_var'].set(POCKET_ZIGZAG[0])
                step['raster_frame'].pack(fill=tk.X, after=step['gating_frame'])
                step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['raster_frame'])
                step['tsp_frame'].pack(fill=tk.X, after=step['cut_dir_row'])
                step['vbit_angle'].set_state("normal")
                step['pocket_boundary_offset'].set_state("normal")
            else:  # Raster Scan
                step['vcarve_frame'].pack_forget()
                step['contour_frame'].pack_forget()
                step['min_fw'].pack_forget()
                step['max_fw'].pack_forget()
                step['gating_frame'].pack(fill=tk.X)
                step['gating_region_row'].pack(fill=tk.X, pady=(0, 4), before=step['threshold'])
                step['tsp_frame'].pack(fill=tk.X, after=step['gating_frame'])
                step['cutout_frame'].pack_forget()
                step['profile_cb'].config(state="readonly", values=_RASTER_PROFILES)
                if profile not in _RASTER_PROFILES:
                    step['profile_var'].set("Ball")
                if profile == "V-Bit":
                    step['vbit_frame'].pack(fill=tk.X, after=step['diameter'])
                    step['stepdown'].pack(fill=tk.X, pady=2, after=step['vbit_frame'])
                else:
                    step['vbit_frame'].pack_forget()
                    step['stepdown'].pack(fill=tk.X, pady=2, after=step['diameter'])
                step['raster_final_stepdown'].pack(fill=tk.X, pady=2,
                                                   after=step['stepdown'])
                step['extra_offset'].pack(fill=tk.X, pady=2,
                                          after=step['raster_final_stepdown'])
                step['pocket_boundary_offset'].pack_forget()
                step['raster_dir_cb'].config(values=RASTER_DIRECTIONS)
                if step['raster_dir_var'].get() not in RASTER_DIRECTIONS:
                    step['raster_dir_var'].set(DEFAULTS["step_raster_dir"])
                step['raster_frame'].pack(fill=tk.X, after=step['alg_row'])
                step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['raster_frame'])
                step['vbit_angle'].set_state("normal")

            _place_z_range()

            step['diameter'].set_state(
                "disabled" if step['profile_var'].get() == "Custom" else "normal")

            locked = ((family != "V-Carving") or alg == "V-Carving (Hybrid)") \
                     and step['cut_dir_var'].get() != "Either"
            # Pocket's own per-pass ordering (see compute_pocket_toolpath's
            # Stage 6) only ever consults use_2opt for a Coil or Spiral scan
            # direction -- Zig-Zag and Onion Skin each use their own
            # dedicated ordering (_order_zigzag_pass /
            # _order_fixed_orientation_pass) that never reads it, so the
            # checkbox has nothing to do there regardless of Cut direction.
            pocket_2opt_moot = (family == "Pocket"
                                 and step['raster_dir_var'].get() not in WOUND_DIRECTIONS)
            step['tsp_2opt_cb'].config(
                state="disabled" if (locked or pocket_2opt_moot) else "normal")
            # Contour orders its rings along the contour graph, never by 2-opt.
            if alg == "Contour":
                step['tsp_2opt_cb'].pack_forget()
            elif not step['tsp_2opt_cb'].winfo_manager():
                step['tsp_2opt_cb'].pack(anchor=tk.W, pady=2, before=step['avoid_air_cb'])

            if hasattr(self, '_view_menu'):
                self._on_display_mode_changed()

        step['refresh_layout'] = _refresh_layout
        # The widgets packed above already form the default algorithm's
        # layout, except for the Z range row, which goes in only here.
        _place_z_range()
        profile_var.trace_add("write", _refresh_layout)
        algorithm_var.trace_add("write", _refresh_layout)
        reghole_pattern_var.trace_add("write", _refresh_layout)
        reghole_diag_dir_var.trace_add("write", _refresh_layout)

        def _on_reghole_param_changed(*_):
            self._sync_reghole_fields(step)
            if hasattr(self, '_view_menu'):
                self._render()

        for le in (reghole_diameter, reghole_inset, reghole_spacing):
            le.committed_var.trace_add("write", _on_reghole_param_changed)
        reghole_inset.auto_var.trace_add("write", _on_reghole_param_changed)
        reghole_spacing.auto_var.trace_add("write", _on_reghole_param_changed)
        for le in (self.width_mm, self.height_mm, self.margin_mm):
            le.committed_var.trace_add("write", lambda *_: self._sync_reghole_fields(step))
        cut_dir_var.trace_add("write", _refresh_layout)
        dir_var.trace_add("write", _refresh_layout)

        def _on_mask_param_changed(*_):
            if not hasattr(self, '_view_menu'):
                return
            if self.display_mode_var.get() != DISPLAY_PROC_MASK:
                return
            idx = self.display_step_var.get() - 1
            if 0 <= idx < len(self.steps) and self.steps[idx] is step:
                self._render()

        def _update_auto_max_fw(*_):
            if not max_fw.auto_var.get():
                return
            try:
                import math
                profile   = step['profile_var'].get()
                diameter  = float(step['diameter'].get())
                angle     = float(step['vbit_angle'].get()) if profile != "Ball" else 90.0
                max_depth = float(self.max_depth.get())
            except (ValueError, AttributeError):
                return
            if profile == "Custom":
                curve = self._step_profile_curve(step)
                if curve is None:
                    return
                r = tool_profile.radius_at_depth(curve, max_depth)
            elif profile == "V-Bit":
                r = max_depth * math.tan(math.radians(angle / 2.0))
            elif profile == "Ball":
                R = diameter / 2.0
                d = min(max_depth, R)
                r = math.sqrt(max(0.0, 2.0 * R * d - d * d))
            else:
                return
            max_fw.set(f"{2.0 * r:.3f}")

        max_fw.auto_var.trace_add("write", _update_auto_max_fw)
        step['profile_var'].trace_add("write", _update_auto_max_fw)
        step['vbit_angle'].committed_var.trace_add("write", _update_auto_max_fw)
        step['diameter'].committed_var.trace_add("write", _update_auto_max_fw)
        self.max_depth.committed_var.trace_add("write", _update_auto_max_fw)

        threshold.auto_var.trace_add("write", lambda *_: self._sync_auto_target_depth(step))
        self.max_depth.committed_var.trace_add(
            "write", lambda *_: self._sync_auto_target_depth(step))

        threshold.committed_var.trace_add("write", _on_mask_param_changed)
        min_fw.committed_var.trace_add("write", _on_mask_param_changed)
        max_fw.committed_var.trace_add("write", _on_mask_param_changed)
        gating_var.trace_add("write", _on_mask_param_changed)
        cutout_extra_offset.committed_var.trace_add("write", _on_mask_param_changed)
        pocket_boundary_offset.committed_var.trace_add("write", _on_mask_param_changed)

        def _on_tool_value_changed(*_):
            self._refresh_library_link(step)

        for key in ('diameter', 'vbit_angle', 'feed_rate', 'plunge_rate',
                    'spindle_rpm') + STEPDOWN_FIELDS + _UNDERCUT_KEYS:
            step[key].committed_var.trace_add("write", _on_tool_value_changed)
        for var in (step['profile_var'], algorithm_var,
                    step['library_tool_id_var'], step['library_tool_name_var']):
            var.trace_add("write", _on_tool_value_changed)
        self._refresh_library_link(step)

    def _reset_step_widgets(self, step):
        """Reset a step's widgets to DEFAULTS."""
        step['algorithm_var'].set(DEFAULTS["step_algorithm"])
        self._set_step_profile_nodes(step, None)
        step['profile_var'].set(DEFAULTS["step_profile"])
        step['profile_cb'].config(state="readonly")
        step['diameter'].set(DEFAULTS["step_diameter"])
        step['stepover'].set(DEFAULTS["step_stepover"])
        step['raster_offset'].set(DEFAULTS["step_raster_offset"])
        step['stepdown'].set(DEFAULTS["step_stepdown"])
        step['extra_offset'].set(DEFAULTS["step_extra_offset"])
        step['feed_rate'].set(DEFAULTS["step_feed_rate"])
        step['plunge_rate'].set(DEFAULTS["step_plunge_rate"])
        step['spindle_rpm'].set(DEFAULTS["step_spindle_rpm"])
        self._set_library_link(step, "", "", "")
        step['threshold'].auto_var.set(False)
        step['threshold'].set(DEFAULTS["step_threshold"])
        step['raster_dir_var'].set(DEFAULTS["step_raster_dir"])
        step['cut_dir_var'].set(DEFAULTS["step_cut_direction"])
        step['vbit_angle'].set(DEFAULTS["step_vbit_angle"])
        step['min_fw'].set(DEFAULTS["step_min_fw"])
        step['max_fw'].auto_var.set(False)
        step['max_fw'].set(DEFAULTS["step_max_fw"])
        step['pocket_boundary_offset'].set(DEFAULTS["step_pocket_boundary_offset"])
        step['vcarve_stepdown'].set(DEFAULTS["step_vcarve_stepdown"])
        step['vcarve_final_stepdown'].set(DEFAULTS["step_vcarve_final_stepdown"])
        step['cutout_depth'].set(DEFAULTS["step_cutout_depth"])
        step['cutout_stepdown'].set(DEFAULTS["step_cutout_stepdown"])
        step['cutout_final_stepdown'].set(DEFAULTS["step_cutout_final_stepdown"])
        step['cutout_extra_offset'].set(DEFAULTS["step_cutout_extra_offset"])
        step['reghole_pattern_var'].set(DEFAULTS["step_reghole_pattern"])
        step['reghole_diag_dir_var'].set(DEFAULTS["step_reghole_diag_dir"])
        step['reghole_diameter'].set(DEFAULTS["step_reghole_diameter"])
        step['reghole_depth'].set(DEFAULTS["step_reghole_depth"])
        step['reghole_inset'].auto_var.set(bool(int(DEFAULTS["step_reghole_inset_auto"])))
        step['reghole_inset'].set(DEFAULTS["step_reghole_inset"])
        step['reghole_spacing'].auto_var.set(bool(int(DEFAULTS["step_reghole_spacing_auto"])))
        step['reghole_spacing'].set(DEFAULTS["step_reghole_spacing"])
        step['contour_z_increment'].set(DEFAULTS["step_contour_z_increment"])
        step['contour_adaptive_var'].set(bool(int(DEFAULTS["step_contour_adaptive"])))
        step['contour_lateral_tol'].set(DEFAULTS["step_contour_lateral_tol"])
        step['undercut_z_increment'].set(DEFAULTS["step_undercut_z_increment"])
        step['undercut_open_var'].set(DEFAULTS["step_undercut_open"])
        step['z_range'].set(DEFAULTS["step_z_min_pct"], DEFAULTS["step_z_max_pct"])
        step['tsp_2opt_var'].set(bool(int(DEFAULTS["step_tsp_2opt"])))
        step['avoid_air_var'].set(bool(int(DEFAULTS["step_avoid_air"])))
        step['gating_mode_var'].set(DEFAULTS["step_gating_mode"])

    # ------------------------------------------------------------------ #
    #  Right panel: image canvas
    # ------------------------------------------------------------------ #
    def _build_right_panel(self, parent):
        scale_bar = ttk.Frame(parent)
        scale_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
        self._build_toolpath_scale(scale_bar)

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._build_image_canvas(canvas_frame)

    def _build_toolpath_scale(self, parent):
        """Horizontal scale to preview how far the tool has progressed along the toolpath."""
        self.toolpath_scale_var = tk.DoubleVar(value=100.0)
        self._toolpath_scale_render_id = None

        ttk.Label(parent, text="Toolpath progress").pack(side=tk.LEFT, padx=(0, 6))
        self.toolpath_scale_label = ttk.Label(parent, text="100%", width=5, anchor=tk.E)
        self.toolpath_scale_label.pack(side=tk.RIGHT, padx=(6, 0))
        scale = ttk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL,
                          variable=self.toolpath_scale_var,
                          command=self._on_toolpath_scale_changed)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # The wheel steps the scale anywhere over the bar, one percent per
        # notch (wheel up advances); "break" keeps the sidebar's bind_all
        # wheel handler from also scrolling.
        for w in (parent, scale, *parent.winfo_children()):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                w.bind(seq, self._on_toolpath_scale_wheel)

    def _on_toolpath_scale_wheel(self, event):
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            step = -1.0
        elif getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            step = 1.0
        else:
            return "break"
        old = self.toolpath_scale_var.get()
        new = min(100.0, max(0.0, round(old) + step))
        if new != old:
            self.toolpath_scale_var.set(new)
            self._on_toolpath_scale_changed()
        return "break"

    def _on_toolpath_scale_changed(self, _value=None):
        self.toolpath_scale_label.config(text=f"{round(self.toolpath_scale_var.get())}%")
        # A full redraw can take longer than the interval between drag
        # motion events, so the render runs once Tk is idle: motion events
        # queued meanwhile only move the slider, and one render then shows
        # the latest position instead of each intermediate one in turn.
        if self._toolpath_scale_render_id is None:
            self._toolpath_scale_render_id = self.toolpath_scale_label.after_idle(
                self._render_toolpath_scale)

    def _render_toolpath_scale(self):
        self._toolpath_scale_render_id = None
        self._render()

    def _build_image_canvas(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        self.fig.patch.set_facecolor("#2b2b2b")
        self.ax.set_facecolor("#2b2b2b")
        self.ax.set_visible(False)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.mpl_canvas.draw()

        # Matplotlib redraws the whole figure on every <Configure>, which
        # makes dragging the pane sash or the window border lag. Resize the
        # figure only once the widget's size has settled.
        tk_widget = self.mpl_canvas.get_tk_widget()
        tk_widget.unbind("<Configure>")
        self._mpl_resize_after_id = None

        def _on_mpl_configure(event):
            if self._mpl_resize_after_id is not None:
                tk_widget.after_cancel(self._mpl_resize_after_id)

            def _apply():
                self._mpl_resize_after_id = None
                self.mpl_canvas.resize(event)
            self._mpl_resize_after_id = tk_widget.after(120, _apply)
        tk_widget.bind("<Configure>", _on_mpl_configure)

        # ── Zoom (scroll wheel) and pan (middle-button drag) ──────────
        self.mpl_canvas.mpl_connect("scroll_event",         self._on_scroll)
        self.mpl_canvas.mpl_connect("button_press_event",   self._on_mouse_press)
        self.mpl_canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.mpl_canvas.mpl_connect("motion_notify_event",  self._on_mouse_move)
        self.mpl_canvas.mpl_connect("axes_leave_event",     self._on_axes_leave)

        self.placeholder = ttk.Label(
            parent,
            text="Open a source via File → Open Source…",
            font=("Helvetica", 13),
            foreground="#888888",
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # ------------------------------------------------------------------ #
    #  File / source actions
    # ------------------------------------------------------------------ #
    _SOURCE_FILETYPES = [
        ("All supported files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp *.stl"),
        ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif *.webp"),
        ("STL files", "*.stl"),
        ("All files", "*.*"),
    ]

    def _new_layer_id(self):
        lid = f"layer{self._next_layer_id}"
        self._next_layer_id += 1
        return lid

    def _load_image_layer(self, path):
        """Load *path* and wrap it as a new (unattached) Layer at its native
        resolution, or show an error dialog and return None on failure."""
        try:
            raw = mpimg.imread(path)
            gray, alpha = layers_mod.to_grayscale(raw)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not load image:\n{exc}")
            return None
        return layers_mod.Layer(name=os.path.basename(path), raw=gray, alpha=alpha,
                                source_kind="image", path=path,
                                layer_id=self._new_layer_id())

    def _stl_raster_px(self, mesh_w_mm, mesh_h_mm, width_mm, height_mm, scale_pct=100.0,
                       density_layers=None):
        """Target (width_px, height_px) for an STL source's raster raw, at
        the canvas density (_auto_px_per_mm over *density_layers*, by
        default self.layers) and the source's own canvas-relative Scale-%
        (see layers._affine_params / _contain_footprint_mm). raw is only
        the source's live preview while its placement is edited: every
        step is computed against the mesh rasterized on the canvas grid
        itself (_place_stl_sources), and the mesh is placed by its mm size
        (layers.native_extent), so raw's pixel count never changes the
        result."""
        footprint_w_mm, footprint_h_mm = self._contain_footprint_mm(
            mesh_w_mm / mesh_h_mm, width_mm, height_mm, scale_pct)
        density = self._auto_px_per_mm(width_mm, height_mm, density_layers)
        width_px  = max(2, round(footprint_w_mm * density))
        height_px = max(2, round(footprint_h_mm * density))
        return width_px, height_px

    def _stl_raster_dims(self, xy_bounds, z_bounds, rot_x, rot_y, rot_z):
        """(mesh_w_mm, mesh_h_mm) for an STL source described by
        *xy_bounds*/*z_bounds* at rotation (0,0,0), rotated by rot_x/rot_y/
        rot_z first."""
        (x_min, x_max, y_min, y_max), _ = stl_mesh.rotate_bounds(
            xy_bounds, z_bounds, rot_x, rot_y, rot_z)
        mesh_w_mm = max(x_max - x_min, 1e-6)
        mesh_h_mm = max(y_max - y_min, 1e-6)
        return mesh_w_mm, mesh_h_mm

    @staticmethod
    def _rasterize_stl(rotated, width_px, height_px):
        """(image, alpha): the rotated mesh as a raster of the given size
        tiling its own XY bounds."""
        hm = stl_mesh.generate_heightmap(rotated, width_px, height_px)
        return stl_mesh.normalize_and_orient(hm, stl_mesh.mesh_z_bounds(rotated))

    @staticmethod
    def _attach_mesh(layer, rotated):
        """Hold the STL layer's rotated mesh on it (see _rotated_mesh)."""
        sp = layer.stl_params or {}
        layer.mesh = rotated
        layer.mesh_key = (layer.path, sp.get('rot_x', 0), sp.get('rot_y', 0),
                          sp.get('rot_z', 0))

    def _build_stl_layer(self, path, triangles, dlg, xy_bounds, z_bounds,
                         density_layers=None):
        """Rasterize *triangles* (already validated/loaded) per *dlg*'s
        orientation and wrap the result as a new (unattached) Layer, or
        show an error dialog and return None on failure. The new layer's
        Scale-% is set so its footprint reproduces the mesh's true physical
        size by default (matching the "populate from mesh bounding box"
        intent) -- computed directly from the mesh's own mm bounding box
        and the canvas's current physical Width/Height, independent of
        canvas pixel density, so it's correct even before rasterizing.
        *density_layers* are the sources the raster's density is chosen
        for (_stl_raster_px)."""
        mesh_w_mm, mesh_h_mm = self._stl_raster_dims(
            xy_bounds, z_bounds, dlg.rot_x, dlg.rot_y, dlg.rot_z)
        try:
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
        except ValueError:
            width_mm, height_mm = float(DEFAULTS["width_mm"]), float(DEFAULTS["height_mm"])
        if mesh_w_mm / mesh_h_mm >= width_mm / max(height_mm, 1e-9):
            scale_pct = 100.0 * mesh_w_mm / max(width_mm, 1e-9)
        else:
            scale_pct = 100.0 * mesh_h_mm / max(height_mm, 1e-9)
        width_px, height_px = self._stl_raster_px(
            mesh_w_mm, mesh_h_mm, width_mm, height_mm, scale_pct, density_layers)
        try:
            rotated = stl_mesh.apply_rotation(triangles, dlg.rot_x, dlg.rot_y, dlg.rot_z)
            image, alpha = self._rasterize_stl(rotated, width_px, height_px)
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))
            return None
        stl_params = {'width_px': width_px, 'height_px': height_px,
                      'mesh_w_mm': mesh_w_mm, 'mesh_h_mm': mesh_h_mm,
                      'rot_x': dlg.rot_x, 'rot_y': dlg.rot_y, 'rot_z': dlg.rot_z}
        layer = layers_mod.Layer(name=os.path.basename(path), raw=image, alpha=alpha,
                                 source_kind="stl", path=path, stl_params=stl_params,
                                 scale_pct=scale_pct,
                                 layer_id=self._new_layer_id())
        self._attach_mesh(layer, rotated)
        return layer

    def _establish_base_image_layer(self, path):
        """Load *path* as a fresh base/canvas-defining source at its native
        resolution and set self.layers = [layer] -- it gets resampled onto
        the canvas grid like every other source during compositing (see
        _recompute_composite). Returns True on success, False if loading
        failed -- caller is responsible for calling
        _on_layers_structure_changed() afterward."""
        layer = self._load_image_layer(path)
        if layer is None:
            return False
        h, w = layer.raw.shape
        # self.layers is assigned last, after any height_mm recalc: while
        # it cascades through _on_canvas_size_changed (see there), that
        # must not rescale THIS brand-new, not-yet-placed layer -- it
        # should only ever apply to sources that existed before the resize.
        if self._aspect_lock_var.get():
            self._image_aspect = w / h
            self._recalc_height()
        # Recalc first (when aspect-locked) so the canvas already matches
        # the image's own aspect ratio -- the layer's Scale-% default
        # (100%, i.e. exactly fills the canvas -- see
        # layers._affine_params) then lands on scale_x == scale_y, rather
        # than being letterboxed against a stale height.
        self.layers = [layer]
        # Fresh single-source project: every step now has exactly one
        # (visible) source, regardless of whatever the old stack's
        # per-step visibility lists happened to contain.
        for s in self.steps:
            s['source_visibility'] = [True]
        return True

    def _current_step(self):
        """The step dict whose tab is currently selected in the Steps
        notebook, or None if there's no valid selection (shouldn't
        normally happen -- _add_step() guarantees at least one step)."""
        if not self.steps:
            return None
        try:
            idx = self._steps_nb.index("current")
        except (tk.TclError, AttributeError):
            return None
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None

    def _background_depth(self):
        """Canvas background depth as a fraction of full depth (the field's
        default while it doesn't parse)."""
        try:
            background_depth_pct = float(self.background_depth_pct.get())
        except ValueError:
            background_depth_pct = float(DEFAULTS["canvas_background_depth_pct"])
        return background_depth_pct / 100.0

    def _composite_with_visibility(self, visibility):
        """Composite self.layers onto the current canvas shape, gating
        source visibility via *visibility* (a list index-aligned with
        self.layers, or None to fall back to each Layer's own .visible).
        Returns (composite_raw, n_grey_levels), or (None, 256) if there
        are no sources at all."""
        if not self.layers:
            return None, 256
        W, H = self._canvas_shape_px()
        try:
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
        except ValueError:
            width_mm, height_mm = float(DEFAULTS["width_mm"]), float(DEFAULTS["height_mm"])
        px_per_mm_x = W / width_mm if width_mm > 1e-9 else 1.0
        px_per_mm_y = H / height_mm if height_mm > 1e-9 else 1.0
        composite_raw = layers_mod.composite(
            self.layers, (H, W), px_per_mm_x, px_per_mm_y, self.surface.get(),
            background_depth=self._background_depth(),
            visibility_override=visibility,
            origin=self.origin_var.get())
        n_grey_levels = layers_mod.compute_n_grey_levels(self.layers, composite_raw)
        return composite_raw, n_grey_levels

    def _recompute_composite(self):
        """Recompute self.current_image (and self.n_grey_levels) from
        self.layers, gated by whichever step tab is currently selected (see
        _current_step) so the live preview always matches what that step
        would actually carve. self.current_image keeps its old role -- the
        single flattened depth map everything downstream of
        _get_depth_map() reads -- it's just a cached compositor output now
        instead of a direct load. Also refreshes the Effective resolution
        readout, which therefore always shows the grid the composite was
        just built on."""
        if not self.layers:
            self.current_image = None
            self.n_grey_levels = 256
            self.effective_resolution_var.set("—")
            return

        base = self.layers[0]
        W, H, density = self._canvas_grid()
        self.effective_resolution_var.set(f"{W}x{H} px  ({density:.3g} px/mm)")
        step = self._current_step()
        visibility = step['source_visibility'] if step is not None else None
        base_visible = visibility[0] if visibility else base.visible

        # Bit-identical fast path for the overwhelmingly common case (one
        # source, default settings, already at the canvas's own pixel
        # shape, visible in the currently-selected step): skip the depth
        # round-trip entirely rather than merely approximating it. A base
        # source with real alpha must NOT take this shortcut -- background
        # would never show through wherever it's partially transparent.
        if (len(self.layers) == 1 and base_visible and base.opacity_pct >= 100.0
                and base.blend_mode == "Replace"
                and base.min_depth_pct == 0.0 and base.max_depth_pct == 100.0
                and base.alpha is None and layers_mod.raw_fills_canvas(base, (H, W))):
            self.current_image = base.raw
            self.n_grey_levels = layers_mod.compute_n_grey_levels(self.layers, self.current_image)
            self._sync_auto_target_depth(step)
            return

        self.current_image, self.n_grey_levels = self._composite_with_visibility(visibility)
        self._sync_auto_target_depth(step)

    def _on_layer_property_changed(self):
        """Live-edit callback: a source's placement/blend/visibility
        changed, but the stack's membership/order didn't. Mirrors
        _on_depth_map_content_changed's cache-invalidation scope. No tab
        refresh needed -- the edited fields already live on the source's
        own tab, so there's nothing else to resync."""
        self._recompute_composite()
        self._erosion_cache.clear()
        # rings_key (see _compute_step_work) covers tool/geometry/threshold
        # params but not the depth map's own pixel content -- a Scale/Offset/
        # Rotation/Opacity/blend/min-max-depth edit changes the composite
        # without touching any of those, so the cached rings must be
        # invalidated explicitly or a subsequent Compute would silently
        # reuse Phase-A results from the stale depth map.
        for s in self.steps:
            s['_cutout_rings_cache'] = None
        # A visibility edit can make the Depth Map Change view (un)available,
        # so the View menu's states are refreshed along with the render.
        if hasattr(self, '_view_menu'):
            self._on_display_mode_changed()
        else:
            self._render()

    def _on_layers_structure_changed(self):
        """Shared tail for anything that changes which sources exist, their
        order, or the canvas's pixel shape: source add/remove/duplicate,
        Canvas Width/Height/Precision edits, project load."""
        # Defensive-only: every call site that actually inserts/removes a
        # source is responsible for making the matching index-aligned edit
        # to every step's source_visibility (see e.g. _remove_layer_tab) --
        # this just pads/truncates away any length mismatch that slips
        # through some other path (project load, a missed call site).
        n = len(self.layers)
        for s in self.steps:
            vis = s.setdefault('source_visibility', [])
            if len(vis) < n:
                vis.extend([True] * (n - len(vis)))
            elif len(vis) > n:
                del vis[n:]

        self._recompute_composite()
        self._erosion_cache.clear()
        self.toolpaths = [None] * len(self.steps)
        for i, s in enumerate(self.steps):
            s['info_var'].set(f"Step {i + 1}: not computed")
            s['_cutout_rings_cache'] = None
        self.display_mode_var.set(DISPLAY_ORIGINAL)
        self._reset_zoom()
        if self.layers:
            self.placeholder.place_forget()
        else:
            self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self._refresh_layer_tabs()
        self._render()
        self._schedule_stl_placement()

    def _open_source_file(self):
        """File -> Open Source...: replaces the whole project with just
        this one source, establishing it as the base/canvas-defining
        source (same as the pre-unification 'Open Image'/'Import STL')."""
        if not self._confirm_discard("opening a new source"):
            return
        path = filedialog.askopenfilename(title="Open Source",
                                          filetypes=self._SOURCE_FILETYPES)
        if not path:
            return
        if path.lower().endswith(".stl"):
            self._import_stl_as_base(path)
        else:
            if not self._establish_base_image_layer(path):
                return
            h, w = self.layers[0].raw.shape
            self._on_layers_structure_changed()
            self.status_var.set(
                f"Loaded: {path}  |  {w} x {h} px  |  {self.n_grey_levels} gray levels")
        if self.layers:
            self._select_layer_tab(self.layers[0])

    def _add_source_at_canvas(self):
        """Canvas tab's 'Add Source' button: establishes the base source
        when none exists yet (same as File -> Open Source...), otherwise
        inserts the new source at the bottom of the stack -- the one
        stack position no per-tab 'Add Source' button can reach."""
        path = filedialog.askopenfilename(title="Add Source",
                                          filetypes=self._SOURCE_FILETYPES)
        if not path:
            return
        if path.lower().endswith(".stl"):
            self._add_stl_source(path, 0)
        else:
            self._add_image_source(path, 0)

    def _add_source_after(self, entry):
        """Per-tab 'Add Source' button: inserts a new source right after
        the tab it's on, mirroring Steps' 'Add step'."""
        path = filedialog.askopenfilename(title="Add Source",
                                          filetypes=self._SOURCE_FILETYPES)
        if not path:
            return
        insert_at = self.layers.index(entry['layer']) + 1
        if path.lower().endswith(".stl"):
            self._add_stl_source(path, insert_at)
        else:
            self._add_image_source(path, insert_at)

    def _replace_layer_source(self, entry):
        """Source tab's 'Replace…' button: loads a different image or STL
        file into this source, keeping its placement, blend, depth range,
        stack position, per-step visibility and the canvas size. An STL is
        oriented via the Import STL dialog (preset to the source's current
        rotation) and rasterized at the source's current Scale."""
        layer = entry['layer']
        initialdir = os.path.dirname(layer.path) if layer.path else None
        path = filedialog.askopenfilename(title="Replace Source",
                                          filetypes=self._SOURCE_FILETYPES,
                                          initialdir=initialdir)
        if not path:
            return
        if path.lower().endswith(".stl"):
            try:
                triangles = stl_mesh.load_stl(path)
            except Exception as exc:
                messagebox.showerror("Load Error", str(exc))
                return
            xy_bounds = stl_mesh.mesh_xy_bounds(triangles)
            z_bounds  = stl_mesh.mesh_z_bounds(triangles)
            sp = layer.stl_params if layer.source_kind == "stl" and layer.stl_params else {}
            rotation = (sp.get('rot_x', 0), sp.get('rot_y', 0), sp.get('rot_z', 0))
            dlg = ImportSTLDialog(self.root, triangles, xy_bounds, z_bounds,
                                  is_base=False, rotation=rotation)
            if not dlg.confirmed:
                return
            mesh_w_mm, mesh_h_mm = self._stl_raster_dims(
                xy_bounds, z_bounds, dlg.rot_x, dlg.rot_y, dlg.rot_z)
            try:
                width_mm  = float(self.width_mm.get())
                height_mm = float(self.height_mm.get())
            except ValueError:
                width_mm, height_mm = float(DEFAULTS["width_mm"]), float(DEFAULTS["height_mm"])
            width_px, height_px = self._stl_raster_px(
                mesh_w_mm, mesh_h_mm, width_mm, height_mm, layer.scale_pct,
                [l for l in self.layers if l is not layer])
            try:
                rotated = stl_mesh.apply_rotation(triangles, dlg.rot_x, dlg.rot_y, dlg.rot_z)
                raw, alpha = self._rasterize_stl(rotated, width_px, height_px)
            except Exception as exc:
                messagebox.showerror("Import Error", str(exc))
                return
            layer.source_kind = "stl"
            layer.stl_params = {'width_px': width_px, 'height_px': height_px,
                                'mesh_w_mm': mesh_w_mm, 'mesh_h_mm': mesh_h_mm,
                                'rot_x': dlg.rot_x, 'rot_y': dlg.rot_y,
                                'rot_z': dlg.rot_z}
            mesh = rotated
        else:
            loaded = self._load_image_layer(path)
            if loaded is None:
                return
            raw, alpha = loaded.raw, loaded.alpha
            layer.source_kind = "image"
            layer.stl_params = None
            mesh = None
        layer.raw, layer.alpha = raw, alpha
        layer.path = path
        layer.name = os.path.basename(path)
        # Derived from the previous file's mesh; rebuilt on demand.
        layer.placed = None
        layer.dexels = None
        layer.dexels_key = None
        layer.mesh = layer.mesh_key = None
        if mesh is not None:
            self._attach_mesh(layer, mesh)
        self._on_layers_structure_changed()
        self._select_layer_tab(layer)
        self.status_var.set(f"Replaced source: {os.path.basename(path)}")

    def _add_image_source(self, path, insert_at):
        if not self.layers:
            # No canvas established yet -- this action is actually the
            # first source, so it establishes the base rather than a
            # plain insert.
            if not self._establish_base_image_layer(path):
                return
            self._on_layers_structure_changed()
            self._select_layer_tab(self.layers[0])
            self.status_var.set(f"Loaded: {path}")
            return
        layer = self._load_image_layer(path)
        if layer is None:
            return
        insert_at = max(0, min(insert_at, len(self.layers)))
        self.layers.insert(insert_at, layer)
        for s in self.steps:
            s.setdefault('source_visibility', []).insert(insert_at, True)
        self._on_layers_structure_changed()
        self._select_layer_tab(layer)
        self.status_var.set(f"Added image source: {path}")

    def _add_stl_source(self, path, insert_at):
        """Inserts an STL-derived source onto the existing canvas -- unlike
        Open Source (which resets to a fresh single-source project), this
        never touches Physical Image Size, since resizing the canvas to fit
        a newly-inserted decorative source would silently move every other
        source's placement relative to the stock. If no canvas exists yet
        (self.layers is empty), this action IS establishing the base
        source, so it delegates to _finish_stl_import -- the same full
        reset flow Open Source uses -- instead of a plain insert."""
        is_base = not self.layers
        try:
            triangles = stl_mesh.load_stl(path)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        xy_bounds = stl_mesh.mesh_xy_bounds(triangles)
        z_bounds  = stl_mesh.mesh_z_bounds(triangles)
        dlg = ImportSTLDialog(self.root, triangles, xy_bounds, z_bounds, is_base=is_base)
        if not dlg.confirmed:
            return

        if is_base:
            self._finish_stl_import(path, triangles, dlg, xy_bounds, z_bounds)
            self._select_layer_tab(self.layers[0])
            return

        layer = self._build_stl_layer(path, triangles, dlg, xy_bounds, z_bounds)
        if layer is None:
            return
        insert_at = max(0, min(insert_at, len(self.layers)))
        self.layers.insert(insert_at, layer)
        for s in self.steps:
            s.setdefault('source_visibility', []).insert(insert_at, True)
        self._on_layers_structure_changed()
        self._select_layer_tab(layer)
        self.status_var.set(f"Added STL source: {os.path.basename(path)}")

    def _on_aspect_lock_toggle(self):
        if self._aspect_lock_var.get():
            if self.layers:
                self._image_aspect = layers_mod.native_aspect(self.layers[0])
            self._recalc_height()
        else:
            self._image_aspect = None

    def _recalc_height(self):
        """When aspect-locked with a known ratio, set height_mm from the
        current width_mm. Returns whether it did. Guarded by
        _updating_aspect_pair so the height_mm.set() below -- which fires
        _on_height_changed synchronously via its own trace -- doesn't turn
        around and recompute width_mm right back, which would otherwise
        ping-pong between the two fields on every edit."""
        if not self._aspect_lock_var.get() or self._image_aspect is None:
            return False
        try:
            w = float(self.width_mm.get())
            new_height = f"{w / self._image_aspect:.4f}"
        except (ValueError, ZeroDivisionError):
            return False
        self._updating_aspect_pair = True
        try:
            self.height_mm.set(new_height)
        finally:
            self._updating_aspect_pair = False
        return True

    def _recalc_width(self):
        """Mirror of _recalc_height: derives width_mm from the current
        height_mm when aspect-locked, so editing either field keeps both in
        sync with _image_aspect."""
        if not self._aspect_lock_var.get() or self._image_aspect is None:
            return False
        try:
            h = float(self.height_mm.get())
        except ValueError:
            return False
        self._updating_aspect_pair = True
        try:
            self.width_mm.set(f"{h * self._image_aspect:.4f}")
        finally:
            self._updating_aspect_pair = False
        return True

    def _on_width_changed(self, *_):
        if self._updating_aspect_pair:
            return
        self._recalc_height()
        self._on_canvas_size_changed()

    def _on_height_changed(self, *_):
        if self._updating_aspect_pair:
            return
        self._recalc_width()
        self._on_canvas_size_changed()

    def _on_origin_changed(self, *_):
        """The G-code origin anchors the diagonal flip line -- both a
        Registration Holes step's (its layout, hint and preview) and a
        source's diagonal Mirror (the composite itself)."""
        for s in getattr(self, 'steps', []):
            s['refresh_layout']()
        if any(l.mirror.startswith("Diagonal") for l in getattr(self, 'layers', [])):
            self._on_layer_property_changed()
        elif hasattr(self, 'ax'):
            self._render()

    def _on_depth_map_content_changed(self, *_):
        """z=0 colour / margin depth / background depth change the *content*
        of depth_map without changing the erosion cache key, so any cached
        entry is now stale. z=0 colour also feeds directly into source
        compositing (raw<->depth polarity), so the composite itself must be
        recomputed too -- for a single default source this is a no-op via
        the fast path in _recompute_composite(), but blended/inverted
        multi-source stacks genuinely change result when the polarity
        flips."""
        self._recompute_composite()
        self._erosion_cache.clear()
        self._render()

    def _import_stl_as_base(self, path):
        try:
            triangles = stl_mesh.load_stl(path)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        xy_bounds = stl_mesh.mesh_xy_bounds(triangles)
        z_bounds  = stl_mesh.mesh_z_bounds(triangles)
        dlg = ImportSTLDialog(self.root, triangles, xy_bounds, z_bounds, is_base=True)
        if not dlg.confirmed:
            return

        self._finish_stl_import(path, triangles, dlg, xy_bounds, z_bounds)

    def _finish_stl_import(self, path, triangles, dlg, xy_bounds, z_bounds):
        # Width/Height must land at their FINAL values before
        # _build_stl_layer runs: its default Scale-% reproduces the mesh's
        # true physical size by comparing the mesh's own bounding box
        # against the *current* Width/Height (see there), so if this import
        # is about to change Width/Height (populate_physical) or Height
        # alone (aspect-locked recalc), that has to happen first, or the
        # comparison is made against the stale, about-to-be-replaced size
        # and Scale-% ends up wrong (only 100% by coincidence).
        #
        # self.layers is still assigned last (see the matching note in
        # _establish_base_image_layer): the width_mm/height_mm.set() calls
        # below cascade through _on_canvas_size_changed, which must not
        # rescale THIS brand-new, not-yet-placed layer -- _build_stl_layer
        # only returns an unattached Layer, it doesn't touch self.layers.
        if dlg.populate_physical:
            rotated_xy, rotated_z = stl_mesh.rotate_bounds(
                xy_bounds, z_bounds, dlg.rot_x, dlg.rot_y, dlg.rot_z)
            x_min, x_max, y_min, y_max = rotated_xy
            z_min, z_max = rotated_z
            width_mm, height_mm, max_depth_mm = stl_mesh.physical_size_from_bounds(
                x_max - x_min, y_max - y_min, z_max - z_min)

            # The raster is always generated proportional to the mesh's own
            # bounding box (see _stl_raster_dims), so its pixel aspect ratio
            # already matches the mesh -- aspect lock can simply be enabled.
            # (Not routed through _on_aspect_lock_toggle(): that reads
            # self.layers[0], which is still the *previous* project's layer
            # at this point.)
            self._aspect_lock_var.set(True)
            self._image_aspect = (x_max - x_min) / max(y_max - y_min, 1e-9)

            self.width_mm.set(f"{width_mm:.4f}")
            self.height_mm.set(f"{height_mm:.4f}")
            self.max_depth.set(f"{max_depth_mm:.4f}")
        elif self._aspect_lock_var.get():
            mesh_w_mm, mesh_h_mm = self._stl_raster_dims(
                xy_bounds, z_bounds, dlg.rot_x, dlg.rot_y, dlg.rot_z)
            self._image_aspect = mesh_w_mm / mesh_h_mm
            self._recalc_height()

        # The previous project's sources are about to be replaced, so the
        # raster's density is chosen for this source alone.
        layer = self._build_stl_layer(path, triangles, dlg, xy_bounds, z_bounds,
                                      density_layers=[])
        if layer is None:
            return
        h, w = layer.raw.shape

        self.layers = [layer]
        # Fresh single-source project: every step now has exactly one
        # (visible) source, regardless of whatever the old stack's
        # per-step visibility lists happened to contain.
        for s in self.steps:
            s['source_visibility'] = [True]
        self._on_layers_structure_changed()   # recomputes composite (n_grey_levels forced
                                               # to 65536 for STL layers), resets toolpaths/view
        self.status_var.set(
            f"Imported STL: {os.path.basename(path)}  |  {w} x {h} px  |  "
            f"{len(triangles)} triangles")

    def _open_settings(self):
        dlg = SettingsDialog(self.root, self._settings, self._machine_defaults,
                             self._machine_defaults_path)
        if dlg.confirmed:
            self._refresh_stats_and_view()

    def _open_dressup(self, step):
        idx = self.steps.index(step)
        settings = {
            'ramp_in_enabled':     step['ramp_in_enabled_var'].get(),
            'ramp_angle_deg':      step['ramp_angle_deg_var'].get(),
            'ramp_max_length_mm':  step['ramp_max_length_mm_var'].get(),
            'ramp_start_z_mm':     step['ramp_start_z_mm_var'].get(),
            'ramp_max_depth_mm':   step['ramp_max_depth_mm_var'].get(),
            'backoff_enabled':     step['backoff_enabled_var'].get(),
            'backoff_distance_mm': step['backoff_distance_mm_var'].get(),
            'arc_rapids_enabled':  step['arc_rapids_enabled_var'].get(),
            'arc_radius_mm':       step['arc_radius_mm_var'].get(),
            'arc_feed_mm_min':     step['arc_feed_mm_min_var'].get(),
            'tab_enabled':         step['tab_enabled_var'].get(),
            'tab_count':           step['tab_count_var'].get(),
            'tab_width_mm':        step['tab_width_mm_var'].get(),
            'tab_height_mm':       step['tab_height_mm_var'].get(),
            'tab_phase_pct':       step['tab_phase_pct_var'].get(),
            'tab_angle_deg':       step['tab_angle_deg_var'].get(),
        }
        tab_keys = ('tab_enabled', 'tab_count', 'tab_width_mm',
                    'tab_height_mm', 'tab_phase_pct', 'tab_angle_deg')
        old_tab_values = tuple(settings[k] for k in tab_keys)
        dlg = DressUpDialog(self.root, settings, self._settings['z_safe'],
                            title=f"Tool Path Dress-Up — Step {idx + 1}",
                            algorithm=step['algorithm_var'].get())
        if dlg.confirmed:
            step['ramp_in_enabled_var'].set(settings['ramp_in_enabled'])
            step['ramp_angle_deg_var'].set(settings['ramp_angle_deg'])
            step['ramp_max_length_mm_var'].set(settings['ramp_max_length_mm'])
            step['ramp_start_z_mm_var'].set(settings['ramp_start_z_mm'])
            step['ramp_max_depth_mm_var'].set(settings['ramp_max_depth_mm'])
            step['backoff_enabled_var'].set(settings['backoff_enabled'])
            step['backoff_distance_mm_var'].set(settings['backoff_distance_mm'])
            step['arc_rapids_enabled_var'].set(settings['arc_rapids_enabled'])
            step['arc_radius_mm_var'].set(settings['arc_radius_mm'])
            step['arc_feed_mm_min_var'].set(settings['arc_feed_mm_min'])
            step['tab_enabled_var'].set(settings['tab_enabled'])
            step['tab_count_var'].set(settings['tab_count'])
            step['tab_width_mm_var'].set(settings['tab_width_mm'])
            step['tab_height_mm_var'].set(settings['tab_height_mm'])
            step['tab_phase_pct_var'].set(settings['tab_phase_pct'])
            step['tab_angle_deg_var'].set(settings['tab_angle_deg'])

            # Tabs change the cut geometry itself (unlike ramp/backoff/arc,
            # which only affect G-code emission) -- if they changed on an
            # already-computed Cutout step, recompute automatically rather
            # than leaving the preview stale. _compute_step_work's rings
            # cache means this normally re-runs only Phase B (fast), not the
            # full mask/contour extraction.
            new_tab_values = tuple(settings[k] for k in tab_keys)
            already_computed = 0 <= idx < len(self.toolpaths) and self.toolpaths[idx] is not None
            if (step['algorithm_var'].get() == "Cutout"
                    and new_tab_values != old_tab_values and already_computed):
                self._compute_step(step)
            else:
                self._refresh_stats_and_view()

    def _refresh_stats_and_view(self):
        """Re-derive the displayed stats (motion count / est. time) and
        refresh the 3D preview for every already-computed step, using each
        step's own dress-up settings. Ramp-in/back-off/arc-rounding only
        affect G-code emission/rendering, not the underlying toolpath
        geometry, so this refresh is cheap and doesn't require recomputing
        any step."""
        for idx, (step, tp) in enumerate(zip(self.steps, self.toolpaths)):
            if tp is None:
                continue
            self._show_step_stats(idx, step, tp)
        self._on_display_mode_changed()

    @staticmethod
    def _store_cutting_metadata(tp, params):
        """Record the step's feeds, spindle speed and library tool on its
        computed toolpath, so stats and export use the values it was
        computed with."""
        tp.metadata['feed_rate']   = params['feed_rate']
        tp.metadata['plunge_rate'] = params['plunge_rate']
        tp.metadata['spindle_rpm'] = params['spindle_rpm']
        tp.metadata['library_tool_id']   = params['library_tool_id']
        tp.metadata['library_tool_name'] = params['library_tool_name']

    def _show_step_stats(self, idx, step, tp):
        """Set a computed step's info line (motion count / est. time),
        using its own dress-up settings and the feeds stored on *tp*."""
        feed_rate = tp.metadata.get('feed_rate', 800.0)
        stats = tp.get_stats(
            feed_rate       = feed_rate,
            feed_rapid      = self._settings['feed_rapid'],
            z_safe          = self._settings['z_safe'],
            accel_mm_s2     = self._settings['accel_mm_s2'],
            ramp            = self._ramp_settings(step),
            backoff         = self._backoff_settings(step),
            arc             = self._arc_settings(step),
            plunge_rate     = tp.metadata.get('plunge_rate', 0.0),
            spindle_rpm     = tp.metadata.get('spindle_rpm', 0.0),
            spindle_dwell_s = self._settings['spindle_dwell_s'],
        )
        text = (f"Step {idx + 1}: {stats['motion_count']} motion lines\n"
                f"Est. time: {_fmt_time(stats['est_minutes'])}"
                f" @ {feed_rate:.0f} mm/min")
        for warning in tp.metadata.get('warnings', ()):
            text += f"\n⚠ {warning}"
        step['info_var'].set(text)

    @staticmethod
    def _sideways_only(step):
        """True for steps whose cuts sit under overhangs, where ramp-in and
        back-off (which move sideways while rising) could hit the part."""
        return ALGORITHM_FAMILY[step['algorithm_var'].get()] == "Undercut"

    def _ramp_settings(self, step):
        return RampSettings(enabled=(step['ramp_in_enabled_var'].get()
                                     and not self._sideways_only(step)),
                             angle_deg=step['ramp_angle_deg_var'].get(),
                             max_length_mm=step['ramp_max_length_mm_var'].get(),
                             start_z_mm=step['ramp_start_z_mm_var'].get(),
                             max_depth_mm=step['ramp_max_depth_mm_var'].get())

    def _backoff_settings(self, step):
        return BackoffSettings(enabled=(step['backoff_enabled_var'].get()
                                        and not self._sideways_only(step)),
                                distance_mm=step['backoff_distance_mm_var'].get())

    def _arc_settings(self, step):
        return ArcSettings(enabled=step['arc_rapids_enabled_var'].get(),
                            radius_mm=step['arc_radius_mm_var'].get(),
                            feed_mm_min=step['arc_feed_mm_min_var'].get())

    def _tab_settings(self, step):
        return TabSettings(
            enabled=step['tab_enabled_var'].get(),
            count=step['tab_count_var'].get(),
            width_mm=step['tab_width_mm_var'].get(),
            height_mm=step['tab_height_mm_var'].get(),
            phase_pct=step['tab_phase_pct_var'].get(),
            tab_angle_deg=step['tab_angle_deg_var'].get())

    # ------------------------------------------------------------------ #
    #  Tool library
    # ------------------------------------------------------------------ #
    def _default_material(self):
        """(id, name) of the library's first material, or (None, "")."""
        if self.tool_library.materials:
            m = self.tool_library.materials[0]
            return m.id, m.name
        return None, ""

    def _material_display_name(self):
        if self._material_id is None:
            return "(none)"
        m = self.tool_library.find_material(self._material_id)
        if m is not None:
            return m.name
        return f"{self._material_name or self._material_id} (not in library)"

    def _refresh_material_combo(self):
        m = self.tool_library.find_material(self._material_id)
        if m is not None:
            self._material_name = m.name
        self._material_cb.config(values=[m.name for m in self.tool_library.materials])
        self.material_var.set(self._material_display_name())

    def _set_project_material(self, material_id, material_name):
        """Set the project material without offering to re-apply cutting
        data (used by New/Open Project)."""
        self._material_id   = material_id
        self._material_name = material_name
        self._refresh_material_combo()
        self._refresh_all_library_links()

    def _on_material_selected(self, _event=None):
        name = self.material_var.get()
        material = next((m for m in self.tool_library.materials if m.name == name), None)
        if material is None or material.id == self._material_id:
            self._refresh_material_combo()
            return
        self._set_project_material(material.id, material.name)

        linked = []
        for i, step in enumerate(self.steps):
            tool_id = step['library_tool_id_var'].get()
            if not tool_id:
                continue
            tool = self.tool_library.find_tool(tool_id)
            if tool is None:
                continue
            has_data = material.id in tool.cutting
            label = f"Step {i + 1} — {tool.name}"
            if not has_data:
                label += f"  (no data for {material.name})"
            linked.append((step, tool, label, has_data))
        if not linked:
            return
        dlg = ApplyMaterialDialog(self.root, material.name,
                                  [(label, has_data) for _, _, label, has_data in linked])
        for i in dlg.selected_indices or []:
            step, tool, _, has_data = linked[i]
            if has_data:
                self._apply_library_tool(step, tool)

    def _save_tool_library(self):
        try:
            self.tool_library.save(self._tool_library_path)
        except OSError as exc:
            messagebox.showerror("Tool Library",
                                 f"Could not save the tool library:\n{exc}")

    def _open_tool_library(self, select_tool_id=None):
        dlg = ToolLibraryDialog(self.root, self.tool_library, select_tool_id,
                                spindle_max_rpm=self._settings['spindle_max_rpm'],
                                spindle_min_rpm=self._settings['spindle_min_rpm'],
                                feed_max=self._settings['feed_max'])
        if not dlg.confirmed:
            return
        self.tool_library = dlg.library
        self._save_tool_library()
        self._refresh_material_combo()
        self._refresh_all_library_links()

    def _step_profile_curve(self, step):
        """The radial height curve of the step's Custom profile, or None."""
        nodes = step['profile_nodes']
        return tool_profile.profile_curve(nodes) if nodes else None

    def _step_tool(self, step):
        """The step's Tool. A Custom profile takes its diameter from its
        curve; the V-bit angle is only read for V-bits."""
        profile = step['profile_var'].get()
        if profile == "Custom":
            curve = self._step_profile_curve(step)
            if curve is None:
                raise _MissingProfileCurve("This step uses a Custom profile but has no curve. "
                                 "Choose a Custom tool from the tool library.")
            return Tool(profile, 2.0 * tool_profile.curve_radius(curve), 90.0, curve)
        angle = float(step['vbit_angle'].get()) if profile == "V-Bit" else 90.0
        return Tool(profile, float(step['diameter'].get()), angle)

    def _set_step_profile_nodes(self, step, nodes):
        """Store a Custom profile's Bezier nodes on the step. The Diameter
        field follows the curve."""
        step['profile_nodes'] = nodes
        curve = self._step_profile_curve(step)
        if curve is not None:
            step['diameter'].set(format_value(2.0 * tool_profile.curve_radius(curve)))

    def _step_tool_values(self, step):
        """The step's current values for every field the library maps to."""
        values = {'profile': step['profile_var'].get(),
                  'profile_nodes': step['profile_nodes']}
        for key in ('diameter', 'vbit_angle', 'feed_rate', 'plunge_rate',
                    'spindle_rpm') + STEPDOWN_FIELDS + _UNDERCUT_KEYS:
            values[key] = step[key].get()
        return values

    def _set_library_link(self, step, tool_id, tool_name, material_id):
        step['library_material_id_var'].set(material_id or "")
        step['library_tool_name_var'].set(tool_name or "")
        step['library_tool_id_var'].set(tool_id or "")

    def _refresh_all_library_links(self):
        for step in self.steps:
            self._refresh_library_link(step)

    def _refresh_library_link(self, step):
        """Refresh a step's Library combobox (tools the current algorithm
        accepts, plus the linked tool if it isn't one of them) and its
        sync status against the library and the project material."""
        if 'library_cb' not in step:
            return
        algorithm = step['algorithm_var'].get()
        tool_id   = step['library_tool_id_var'].get()
        tool      = self.tool_library.find_tool(tool_id) if tool_id else None

        candidates = self.tool_library.tools_for_profiles(_allowed_profiles(algorithm))
        ids    = [""] + [t.id for t in candidates]
        labels = ["(none)"] + [t.name for t in candidates]
        if tool_id and tool_id not in ids:
            ids.append(tool_id)
            labels.append(tool.name if tool is not None else
                          f"{step['library_tool_name_var'].get() or tool_id} (not in library)")
        step['library_cb_ids'] = ids
        step['library_cb'].config(values=labels)
        step['library_display_var'].set(labels[ids.index(tool_id)] if tool_id else "(none)")

        if not tool_id:
            step['tool_status_var'].set("")
            return
        cutting = cutting_for(tool, self._material_id)
        state = sync_state(self._step_tool_values(step), tool, cutting,
                           ALGORITHM_FAMILY[algorithm])
        if state == SYNC_NO_DATA:
            text = f"● no data for {self._material_name or 'this material'}"
        else:
            text = _SYNC_STATUS_TEXT.get(state, "")
        step['tool_status_var'].set(text)

    def _on_library_tool_chosen(self, step):
        idx = step['library_cb'].current()
        if idx < 0:
            return
        tool_id = step['library_cb_ids'][idx]
        if not tool_id:
            self._unlink_library_tool(step)
            return
        tool = self.tool_library.find_tool(tool_id)
        if tool is None:
            self._refresh_library_link(step)
            return
        self._apply_library_tool(step, tool)

    def _apply_library_tool(self, step, tool):
        """Copy *tool*'s geometry and its cutting data for the project
        material into the step, and link the step to it."""
        cutting = cutting_for(tool, self._material_id)
        values = step_values_from(tool, cutting)
        # Curve and profile first: their traces (_refresh_layout, Auto max
        # feature width) must see the new profile before the dependent
        # fields land.
        if 'profile_nodes' in values:
            self._set_step_profile_nodes(step, values.pop('profile_nodes'))
        step['profile_var'].set(values.pop('profile'))
        for key, value in values.items():
            step[key].set(value)
        self._set_library_link(step, tool.id, tool.name,
                               self._material_id if cutting is not None else "")
        self._refresh_library_link(step)

    def _linked_tool_or_warn(self, step):
        tool_id = step['library_tool_id_var'].get()
        if not tool_id:
            messagebox.showinfo("Tool Library", "This step is not linked to a library tool.")
            return None
        tool = self.tool_library.find_tool(tool_id)
        if tool is None:
            messagebox.showwarning(
                "Tool Library",
                f"\"{step['library_tool_name_var'].get()}\" is not in the tool library.")
        return tool

    def _reapply_library_tool(self, step):
        tool = self._linked_tool_or_warn(step)
        if tool is not None:
            self._apply_library_tool(step, tool)

    def _update_library_tool_from_step(self, step):
        tool = self._linked_tool_or_warn(step)
        if tool is None:
            return
        if self._material_id is None or self.tool_library.find_material(self._material_id) is None:
            messagebox.showwarning("Tool Library",
                                   "Choose a library material on the Canvas tab first.")
            return
        family = ALGORITHM_FAMILY[step['algorithm_var'].get()]
        values = self._step_tool_values(step)
        message = (f"Update the cutting data of \"{tool.name}\" for "
                   f"{self._material_name} from this step?")
        if sync_state(values, tool, None, family) == SYNC_GEOMETRY_DIFFERS:
            message += ("\n\nThis step's profile, diameter, V angle or custom curve "
                        "differs from the "
                        "library tool. Only the cutting data is updated; the tool's "
                        "geometry is left unchanged. Use \"Save as new library tool…\" "
                        "to keep this step's geometry.")
        if not messagebox.askyesno("Update library tool", message):
            return
        try:
            updated = update_cutting_from_step(tool, self._material_id, values, family)
        except ValueError:
            messagebox.showerror("Tool Library", "The step's cutting values must be numbers.")
            return
        self.tool_library.tools[self.tool_library.tools.index(tool)] = updated
        self._save_tool_library()
        step['library_material_id_var'].set(self._material_id)
        self._refresh_all_library_links()

    def _save_step_as_library_tool(self, step):
        values = self._step_tool_values(step)
        try:
            default_name = f"{format_value(float(values['diameter']))} mm {values['profile']}"
        except ValueError:
            default_name = values['profile']
        name = simpledialog.askstring("Save as new library tool", "Tool name:",
                                      initialvalue=default_name, parent=self.root)
        if name is None or not name.strip():
            return
        material_id = (self._material_id
                       if self.tool_library.find_material(self._material_id) is not None
                       else None)
        family = ALGORITHM_FAMILY[step['algorithm_var'].get()]
        try:
            tool = tool_from_step(name.strip(), values, material_id, family)
        except ValueError:
            messagebox.showerror("Tool Library", "The step's tool values must be numbers.")
            return
        self.tool_library.tools.append(tool)
        self._save_tool_library()
        self._set_library_link(step, tool.id, tool.name, material_id or "")
        self._refresh_all_library_links()

    def _unlink_library_tool(self, step):
        self._set_library_link(step, "", "", "")
        self._refresh_library_link(step)

    # ------------------------------------------------------------------ #
    #  Project save / load
    # ------------------------------------------------------------------ #
    def _step_to_dict(self, step):
        return {
            'algorithm':            step['algorithm_var'].get(),
            'profile':              step['profile_var'].get(),
            'profile_nodes':        copy.deepcopy(step['profile_nodes']),
            'diameter':             step['diameter'].get(),
            'stepover':             step['stepover'].get(),
            'raster_offset':        step['raster_offset'].get(),
            'stepdown':             step['stepdown'].get(),
            'raster_final_stepdown': step['raster_final_stepdown'].get(),
            'extra_offset':         step['extra_offset'].get(),
            'raster_dir':           step['raster_dir_var'].get(),
            'cut_direction':        step['cut_dir_var'].get(),
            'vbit_angle':           step['vbit_angle'].get(),
            'vcarve_stepdown':      step['vcarve_stepdown'].get(),
            'vcarve_final_stepdown': step['vcarve_final_stepdown'].get(),
            'cutout_depth':          step['cutout_depth'].get(),
            'cutout_stepdown':       step['cutout_stepdown'].get(),
            'cutout_final_stepdown': step['cutout_final_stepdown'].get(),
            'cutout_extra_offset':   step['cutout_extra_offset'].get(),
            'reghole_pattern':       step['reghole_pattern_var'].get(),
            'reghole_diag_dir':      step['reghole_diag_dir_var'].get(),
            'reghole_diameter':      step['reghole_diameter'].get(),
            'reghole_depth':         step['reghole_depth'].get(),
            'reghole_inset':         step['reghole_inset'].get(),
            'reghole_inset_auto':    bool(step['reghole_inset'].auto_var.get()),
            'reghole_spacing':       step['reghole_spacing'].get(),
            'reghole_spacing_auto':  bool(step['reghole_spacing'].auto_var.get()),
            'contour_z_increment':  step['contour_z_increment'].get(),
            'contour_adaptive':     step['contour_adaptive_var'].get(),
            'contour_lateral_tol':    step['contour_lateral_tol'].get(),
            'undercut_z_increment': step['undercut_z_increment'].get(),
            'undercut_open':        step['undercut_open_var'].get(),
            'z_min_pct':            step['z_range'].get()[0],
            'z_max_pct':            step['z_range'].get()[1],
            'tsp_2opt':             step['tsp_2opt_var'].get(),
            'avoid_air':            step['avoid_air_var'].get(),
            'feed_rate':            step['feed_rate'].get(),
            'plunge_rate':          step['plunge_rate'].get(),
            'spindle_rpm':          step['spindle_rpm'].get(),
            'library_tool_id':      step['library_tool_id_var'].get(),
            'library_tool_name':    step['library_tool_name_var'].get(),
            'library_material_id':  step['library_material_id_var'].get(),
            'threshold':            step['threshold'].get(),
            'threshold_auto':       bool(step['threshold'].auto_var.get()),
            'min_fw':               step['min_fw'].get(),
            'max_fw':               step['max_fw'].get(),
            'max_fw_auto':          bool(step['max_fw'].auto_var.get()),
            'pocket_boundary_offset': step['pocket_boundary_offset'].get(),
            **{key: step[key].get() for key in _UNDERCUT_KEYS},
            'gating_mode':          step['gating_mode_var'].get(),
            'ramp_in_enabled':      step['ramp_in_enabled_var'].get(),
            'ramp_angle_deg':       step['ramp_angle_deg_var'].get(),
            'ramp_max_length_mm':   step['ramp_max_length_mm_var'].get(),
            'ramp_start_z_mm':      step['ramp_start_z_mm_var'].get(),
            'ramp_max_depth_mm':    step['ramp_max_depth_mm_var'].get(),
            'backoff_enabled':      step['backoff_enabled_var'].get(),
            'backoff_distance_mm':  step['backoff_distance_mm_var'].get(),
            'arc_rapids_enabled':   step['arc_rapids_enabled_var'].get(),
            'arc_radius_mm':        step['arc_radius_mm_var'].get(),
            'arc_feed_mm_min':      step['arc_feed_mm_min_var'].get(),
            'tab_enabled':          step['tab_enabled_var'].get(),
            'tab_count':            step['tab_count_var'].get(),
            'tab_width_mm':         step['tab_width_mm_var'].get(),
            'tab_height_mm':        step['tab_height_mm_var'].get(),
            'tab_phase_pct':        step['tab_phase_pct_var'].get(),
            'tab_angle_deg':        step['tab_angle_deg_var'].get(),
            # Index-aligned with self.layers at save time (not layer_id --
            # see _on_layers_structure_changed's docstring on why position
            # is what's persisted).
            'source_visibility':    list(step.get('source_visibility', [])),
        }

    def _apply_step_dict(self, step, d):
        step['algorithm_var'].set(d.get('algorithm', DEFAULTS['step_algorithm']))
        self._set_step_profile_nodes(step, tool_profile.normalize_nodes(d.get('profile_nodes')))
        step['profile_var'].set(d.get('profile', DEFAULTS['step_profile']))
        step['diameter'].set(d.get('diameter', DEFAULTS['step_diameter']))
        step['stepover'].set(d.get('stepover', DEFAULTS['step_stepover']))
        step['raster_offset'].set(d.get('raster_offset', DEFAULTS['step_raster_offset']))
        step['stepdown'].set(d.get('stepdown', DEFAULTS['step_stepdown']))
        step['raster_final_stepdown'].set(d.get('raster_final_stepdown', DEFAULTS['step_raster_final_stepdown']))
        step['extra_offset'].set(d.get('extra_offset', DEFAULTS['step_extra_offset']))
        step['raster_dir_var'].set(d.get('raster_dir', DEFAULTS['step_raster_dir']))
        step['cut_dir_var'].set(d.get('cut_direction', DEFAULTS['step_cut_direction']))
        step['vbit_angle'].set(d.get('vbit_angle', DEFAULTS['step_vbit_angle']))
        step['vcarve_stepdown'].set(d.get('vcarve_stepdown', DEFAULTS['step_vcarve_stepdown']))
        step['vcarve_final_stepdown'].set(d.get('vcarve_final_stepdown', DEFAULTS['step_vcarve_final_stepdown']))
        step['cutout_depth'].set(d.get('cutout_depth', DEFAULTS['step_cutout_depth']))
        step['cutout_stepdown'].set(d.get('cutout_stepdown', DEFAULTS['step_cutout_stepdown']))
        step['cutout_final_stepdown'].set(d.get('cutout_final_stepdown', DEFAULTS['step_cutout_final_stepdown']))
        step['cutout_extra_offset'].set(d.get('cutout_extra_offset', DEFAULTS['step_cutout_extra_offset']))
        step['reghole_pattern_var'].set(d.get('reghole_pattern', DEFAULTS['step_reghole_pattern']))
        step['reghole_diag_dir_var'].set(d.get('reghole_diag_dir', DEFAULTS['step_reghole_diag_dir']))
        step['reghole_diameter'].set(d.get('reghole_diameter', DEFAULTS['step_reghole_diameter']))
        step['reghole_depth'].set(d.get('reghole_depth', DEFAULTS['step_reghole_depth']))
        step['reghole_inset'].auto_var.set(bool(d.get(
            'reghole_inset_auto', bool(int(DEFAULTS['step_reghole_inset_auto'])))))
        step['reghole_inset'].set(d.get('reghole_inset', DEFAULTS['step_reghole_inset']))
        step['reghole_spacing'].auto_var.set(bool(d.get(
            'reghole_spacing_auto', bool(int(DEFAULTS['step_reghole_spacing_auto'])))))
        step['reghole_spacing'].set(d.get('reghole_spacing', DEFAULTS['step_reghole_spacing']))
        step['contour_z_increment'].set(d.get('contour_z_increment', DEFAULTS['step_contour_z_increment']))
        step['contour_adaptive_var'].set(bool(d.get('contour_adaptive', False)))
        step['contour_lateral_tol'].set(d.get('contour_lateral_tol', DEFAULTS['step_contour_lateral_tol']))
        # Projects without the field took an Undercut step's level spacing
        # from Max step-down.
        step['undercut_z_increment'].set(d.get(
            'undercut_z_increment',
            d.get('stepdown', DEFAULTS['step_undercut_z_increment'])
            if d.get('algorithm') == "Undercut" else DEFAULTS['step_undercut_z_increment']))
        undercut_open = d.get('undercut_open', DEFAULTS['step_undercut_open'])
        step['undercut_open_var'].set(undercut_open if undercut_open in UNDERCUT_OPEN_MODES
                                      else DEFAULTS['step_undercut_open'])
        step['z_range'].set(d.get('z_min_pct', DEFAULTS['step_z_min_pct']),
                            d.get('z_max_pct', DEFAULTS['step_z_max_pct']))
        step['tsp_2opt_var'].set(bool(d.get('tsp_2opt', False)))
        step['avoid_air_var'].set(bool(d.get('avoid_air', False)))
        step['feed_rate'].set(d.get('feed_rate', DEFAULTS['step_feed_rate']))
        step['plunge_rate'].set(d.get('plunge_rate', DEFAULTS['step_plunge_rate']))
        step['spindle_rpm'].set(d.get('spindle_rpm', DEFAULTS['step_spindle_rpm']))
        self._set_library_link(step, d.get('library_tool_id', ""),
                               d.get('library_tool_name', ""),
                               d.get('library_material_id', ""))
        step['threshold'].auto_var.set(bool(d.get('threshold_auto', False)))
        step['threshold'].set(d.get('threshold', DEFAULTS['step_threshold']))
        step['min_fw'].set(d.get('min_fw', DEFAULTS['step_min_fw']))
        step['max_fw'].auto_var.set(bool(d.get('max_fw_auto', False)))
        step['max_fw'].set(d.get('max_fw', DEFAULTS['step_max_fw']))
        step['pocket_boundary_offset'].set(d.get(
            'pocket_boundary_offset', DEFAULTS['step_pocket_boundary_offset']))
        for key in _UNDERCUT_KEYS:
            step[key].set(d.get(key, DEFAULTS[f'step_{key}']))
        step['gating_mode_var'].set(d.get('gating_mode', DEFAULTS['step_gating_mode']))
        step['ramp_in_enabled_var'].set(bool(d.get(
            'ramp_in_enabled', bool(int(DEFAULTS['step_ramp_in_enabled'])))))
        step['ramp_angle_deg_var'].set(d.get(
            'ramp_angle_deg', DEFAULTS['step_ramp_angle_deg']))
        step['ramp_max_length_mm_var'].set(d.get(
            'ramp_max_length_mm', DEFAULTS['step_ramp_max_length_mm']))
        step['ramp_start_z_mm_var'].set(d.get(
            'ramp_start_z_mm', DEFAULTS['step_ramp_start_z_mm']))
        step['ramp_max_depth_mm_var'].set(d.get(
            'ramp_max_depth_mm', DEFAULTS['step_ramp_max_depth_mm']))
        step['backoff_enabled_var'].set(bool(d.get(
            'backoff_enabled', bool(int(DEFAULTS['step_backoff_enabled'])))))
        step['backoff_distance_mm_var'].set(d.get(
            'backoff_distance_mm', DEFAULTS['step_backoff_distance_mm']))
        step['arc_rapids_enabled_var'].set(bool(d.get(
            'arc_rapids_enabled', bool(int(DEFAULTS['step_arc_rapids_enabled'])))))
        step['arc_radius_mm_var'].set(d.get(
            'arc_radius_mm', DEFAULTS['step_arc_radius_mm']))
        step['arc_feed_mm_min_var'].set(d.get(
            'arc_feed_mm_min', DEFAULTS['step_arc_feed_mm_min']))
        step['tab_enabled_var'].set(bool(d.get(
            'tab_enabled', bool(int(DEFAULTS['step_tab_enabled'])))))
        step['tab_count_var'].set(d.get(
            'tab_count', int(DEFAULTS['step_tab_count'])))
        step['tab_width_mm_var'].set(d.get(
            'tab_width_mm', DEFAULTS['step_tab_width_mm']))
        step['tab_height_mm_var'].set(d.get(
            'tab_height_mm', DEFAULTS['step_tab_height_mm']))
        step['tab_phase_pct_var'].set(d.get(
            'tab_phase_pct', DEFAULTS['step_tab_phase_pct']))
        step['tab_angle_deg_var'].set(d.get(
            'tab_angle_deg', DEFAULTS['step_tab_angle_deg']))
        # Index-aligned with self.layers, which isn't loaded yet at this
        # point in _load_project -- any length mismatch (including an
        # older project file with no saved source_visibility at all) gets
        # padded/truncated to match once layers ARE loaded, by
        # _on_layers_structure_changed's defensive sync.
        step['source_visibility'] = [bool(v) for v in d.get('source_visibility', [])]

    def _resolve_rel_path(self, prj_path, saved_path, saved_path_rel):
        """Try *saved_path_rel* relative to prj_path's directory first,
        falling back to the stored absolute *saved_path*. Returns the
        resolved absolute path, or None if neither exists. Shared by the
        legacy single-image/STL load branch and per-layer v2 loading."""
        if saved_path_rel:
            candidate = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(prj_path)), saved_path_rel))
            if os.path.isfile(candidate):
                return candidate
        if saved_path and os.path.isfile(saved_path):
            return saved_path
        return None

    def _make_rel_path(self, abs_path, prj_path):
        """Inverse of _resolve_rel_path's relative half, used at save time."""
        if not abs_path:
            return None
        try:
            return os.path.relpath(abs_path, os.path.dirname(os.path.abspath(prj_path)))
        except ValueError:
            return None  # e.g. different drive on Windows

    def _layer_to_dict(self, layer):
        d = {
            'name':         layer.name,
            'source_kind':  layer.source_kind,
            'path':         layer.path,
            'offset_x_mm':  layer.offset_x_mm,
            'offset_y_mm':  layer.offset_y_mm,
            'rotation_deg': layer.rotation_deg,
            'scale_pct':    layer.scale_pct,
            'blend_mode':   layer.blend_mode,
            'opacity_pct':  layer.opacity_pct,
            'min_depth_pct': layer.min_depth_pct,
            'max_depth_pct': layer.max_depth_pct,
            'visible':      layer.visible,
            'mirror':       layer.mirror,
        }
        if layer.source_kind == 'stl' and layer.stl_params:
            d['stl_params'] = dict(layer.stl_params)
        return d

    @staticmethod
    def _loaded_stl_params(rotated, width_px, height_px, rot_x, rot_y, rot_z):
        """stl_params for an STL source re-read from a project: its saved
        raster size and rotation, and the mesh's mm size taken from the
        mesh itself -- project files saved without it place the mesh by its
        exact proportions too (layers.native_extent)."""
        x_min, x_max, y_min, y_max = stl_mesh.mesh_xy_bounds(rotated)
        return {'width_px': width_px, 'height_px': height_px,
                'mesh_w_mm': max(float(x_max - x_min), 1e-6),
                'mesh_h_mm': max(float(y_max - y_min), 1e-6),
                'rot_x': rot_x, 'rot_y': rot_y, 'rot_z': rot_z}

    def _layer_from_dict(self, d, prj_path):
        """Resolve and (re)load one saved layer. Returns a Layer, or None if
        its source file can't be found or fails to load/rasterize -- the
        caller is responsible for surfacing that as a warning."""
        source_kind = d.get('source_kind', 'image')
        resolved = self._resolve_rel_path(prj_path, d.get('path'), d.get('path_rel'))
        if resolved is None:
            return None
        try:
            if source_kind == 'stl':
                stl_params = d.get('stl_params') or {}
                width_px  = stl_params['width_px']
                height_px = stl_params['height_px']
                rot_x = stl_params.get('rot_x', 0)
                rot_y = stl_params.get('rot_y', 0)
                rot_z = stl_params.get('rot_z', 0)
                triangles = stl_mesh.load_stl(resolved)
                rotated   = stl_mesh.apply_rotation(triangles, rot_x, rot_y, rot_z)
                image, alpha = self._rasterize_stl(rotated, width_px, height_px)
                layer = layers_mod.Layer(
                    name=d.get('name') or os.path.basename(resolved), raw=image, alpha=alpha,
                    source_kind='stl', path=resolved,
                    stl_params=self._loaded_stl_params(rotated, width_px, height_px,
                                                       rot_x, rot_y, rot_z),
                    layer_id=self._new_layer_id())
                self._attach_mesh(layer, rotated)
            else:
                raw = mpimg.imread(resolved)
                gray, alpha = layers_mod.to_grayscale(raw)
                layer = layers_mod.Layer(
                    name=d.get('name') or os.path.basename(resolved), raw=gray, alpha=alpha,
                    source_kind='image', path=resolved, layer_id=self._new_layer_id())
        except Exception:
            return None

        layer.offset_x_mm  = float(d.get('offset_x_mm', 0.0))
        layer.offset_y_mm  = float(d.get('offset_y_mm', 0.0))
        layer.rotation_deg = float(d.get('rotation_deg', 0.0))
        layer.scale_pct    = float(d.get('scale_pct', 100.0))
        layer.blend_mode   = d.get('blend_mode', 'Replace')
        layer.opacity_pct  = float(d.get('opacity_pct', 100.0))
        layer.min_depth_pct = float(d.get('min_depth_pct', 0.0))
        layer.max_depth_pct = float(d.get('max_depth_pct', 100.0))
        layer.visible      = bool(d.get('visible', True))
        layer.mirror       = d.get('mirror', DEFAULTS['layer_mirror'])
        return layer

    def _project_snapshot(self):
        """Return a JSON-able dict describing all persistable project state."""
        W, H = self._canvas_shape_px() if self.layers else (None, None)
        try:
            background_depth_pct = float(self.background_depth_pct.get())
        except ValueError:
            background_depth_pct = float(DEFAULTS["canvas_background_depth_pct"])
        return {
            'layers': [self._layer_to_dict(l) for l in self.layers],
            'canvas': {
                'background_depth_pct': background_depth_pct,
                'width_px':  W,
                'height_px': H,
            },
            'global': {
                'width_mm':        self.width_mm.get(),
                'height_mm':       self.height_mm.get(),
                'min_px_per_mm':   self.min_px_per_mm.get(),
                'max_megapixels':  self.max_megapixels.get(),
                'max_depth':       self.max_depth.get(),
                'margin_mm':       self.margin_mm.get(),
                'margin_depth_pct': self.margin_depth_pct.get(),
                'surface':         self.surface.get(),
                'origin':          self.origin_var.get(),
                'aspect_locked':   self._aspect_lock_var.get(),
                'material_id':     self._material_id,
                'material_name':   self._material_name,
            },
            'machine': {
                'z_safe':      self._settings['z_safe'],
                'feed_rapid':  self._settings['feed_rapid'],
                'accel_mm_s2': self._settings['accel_mm_s2'],
                'spindle_dwell_s': self._settings['spindle_dwell_s'],
                'spindle_max_rpm': self._settings['spindle_max_rpm'],
                'spindle_min_rpm': self._settings['spindle_min_rpm'],
                'feed_max':        self._settings['feed_max'],
                'preamble':    self._settings['preamble'],
                'postamble':   self._settings['postamble'],
            },
            'steps': [self._step_to_dict(s) for s in self.steps],
        }

    def _snapshot_key(self):
        return json.dumps(self._project_snapshot(), sort_keys=True)

    def _is_dirty(self):
        return self._snapshot_key() != self._saved_snapshot

    _APP_TITLE = "CNC Relief Carving"

    def _update_title(self):
        """Show the project file's name in the window title, prefixed with
        '*' while the project has unsaved changes."""
        name = (os.path.basename(self._project_path) if self._project_path
                else "Untitled")
        title = f"{'*' if self._is_dirty() else ''}{name} — {self._APP_TITLE}"
        if title != self._title_shown:
            self._title_shown = title
            self.root.title(title)

    def _poll_title(self):
        """Keep the title's unsaved-changes marker current; edits reach the
        project state through many widgets, so it is re-checked once a
        second rather than traced per field."""
        try:
            self._update_title()
        except tk.TclError:
            return
        self.root.after(1000, self._poll_title)

    def _confirm_discard(self, verb):
        """If there are unsaved changes, ask the user how to proceed.

        Returns True if it's OK to continue with `verb` (changes were saved
        or the user chose to discard them), False if the action should be
        cancelled.
        """
        if not self._is_dirty():
            return True
        response = messagebox.askyesnocancel(
            "Unsaved Changes",
            f"The current project has unsaved changes.\nSave changes before {verb}?")
        if response is None:
            return False
        if response:
            return self._save_project()
        return True

    def _on_exit(self):
        if self._confirm_discard("exiting"):
            self.root.destroy()

    def _new_project(self):
        if not self._confirm_discard("starting a new project"):
            return

        self._aspect_lock_var.set(True)
        self.width_mm.set(DEFAULTS['width_mm'])
        self.height_mm.set(DEFAULTS['height_mm'])
        self.min_px_per_mm.set(DEFAULTS['min_px_per_mm'])
        self.max_megapixels.set(DEFAULTS['max_megapixels'])
        self.max_depth.set(DEFAULTS['max_depth'])
        self.margin_mm.set(DEFAULTS['margin_mm'])
        self.margin_depth_pct.set(DEFAULTS['margin_depth_pct'])
        self.background_depth_pct.set(DEFAULTS['canvas_background_depth_pct'])
        self.surface.set(DEFAULTS['surface'])
        self.origin_var.set(DEFAULTS['origin'])
        self._set_project_material(*self._default_material())

        self._settings.update(self._machine_defaults)

        while len(self.steps) > 1:
            self._remove_step(self.steps[-1])
        self._apply_step_dict(self.steps[0], {})

        self.layers         = []
        self._last_width_mm  = float(DEFAULTS['width_mm'])
        self._last_height_mm = float(DEFAULTS['height_mm'])
        self._last_min_px_per_mm = float(DEFAULTS['min_px_per_mm'])
        self.effective_resolution_var.set("—")
        self.current_image  = None
        self._image_aspect  = None
        self.n_grey_levels  = 256
        self.toolpaths      = [None]
        self._erosion_cache.clear()
        self.steps[0]['info_var'].set("Step 1: not computed")
        self._refresh_layer_tabs()

        self.display_mode_var.set(DISPLAY_ORIGINAL)
        self._reset_zoom()

        if getattr(self.ax, 'name', '') == '3d':
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.fig.patch.set_facecolor("#2b2b2b")
            self.ax.set_facecolor("#2b2b2b")
        self.ax.clear()
        self.ax.set_visible(False)
        self.mpl_canvas.draw()
        self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.status_var.set("New project")
        self._project_path = None
        self._saved_snapshot = self._snapshot_key()
        self._update_title()

    def _save_project(self):
        """File -> Save Project: write to the project's own file, or ask for
        one (as Save Project As) when the project is untitled. Returns True
        if the project was saved."""
        if self._project_path is None:
            return self._save_project_as()
        return self._write_project(self._project_path)

    def _save_project_as(self):
        """File -> Save Project As: ask for a file, write the project to it,
        and make it the project's own file. Returns True if saved."""
        if self._project_path:
            initialdir  = os.path.dirname(self._project_path)
            initialfile = os.path.basename(self._project_path)
        else:
            initialdir, initialfile = None, "Untitled.prj"
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".prj",
            initialdir=initialdir,
            initialfile=initialfile,
            filetypes=[("BM2GCODE project", "*.prj"), ("All files", "*.*")],
        )
        if not path:
            return False
        return self._write_project(path)

    def _write_project(self, path):
        snapshot = self._project_snapshot()
        layers_out = []
        for layer, d in zip(self.layers, snapshot['layers']):
            d = dict(d)
            d['path_rel'] = self._make_rel_path(layer.path, path)
            layers_out.append(d)
        project = {
            'version': 6,
            'layers': layers_out,
            'canvas': snapshot['canvas'],
            'global': snapshot['global'],
            'machine': snapshot['machine'],
            'steps': snapshot['steps'],
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(project, f, indent=2)
            self.status_var.set(f"Project saved: {path}")
            self._project_path = path
            self._saved_snapshot = json.dumps(snapshot, sort_keys=True)
            self._update_title()
            return True
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save project:\n{exc}")
            return False

    def _load_project(self):
        initialdir = (os.path.dirname(self._project_path)
                      if self._project_path else None)
        path = filedialog.askopenfilename(
            title="Open Project",
            initialdir=initialdir,
            filetypes=[("BM2GCODE project", "*.prj"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                project = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read project file:\n{exc}")
            return

        version = project.get('version', 1)
        if version not in (1, 2, 3, 4, 5, 6):
            messagebox.showerror("Load Error", "Unsupported project file version.")
            return

        if not self._confirm_discard("opening another project"):
            return
        self._project_path = path

        # ── Global image-size parameters ──────────────────────────────
        g = project.get('global', {})
        c = project.get('canvas', {})
        self._aspect_lock_var.set(bool(g.get('aspect_locked', True)))
        self.width_mm.set(g.get('width_mm', DEFAULTS['width_mm']))
        self.height_mm.set(g.get('height_mm', DEFAULTS['height_mm']))
        min_px_per_mm = g.get('min_px_per_mm')
        if min_px_per_mm is None:
            # Pre-v4 project: no Min. px/mm field existed yet -- derive an
            # equivalent floor from the old fixed Precision (mm/px, the
            # exact density every source was rasterized/placed against), or
            # if even that predates v3, from the old base-layer-defined
            # canvas resolution, so re-opening reproduces close to the same
            # effective detail before the user adjusts it.
            legacy_precision_mm = g.get('precision_mm')
            try:
                if legacy_precision_mm is not None and float(legacy_precision_mm) > 1e-9:
                    min_px_per_mm = 1.0 / float(legacy_precision_mm)
            except (TypeError, ValueError):
                pass
        if min_px_per_mm is None:
            try:
                width_mm_val = float(g.get('width_mm', DEFAULTS['width_mm']))
            except (TypeError, ValueError):
                width_mm_val = float(DEFAULTS['width_mm'])
            saved_width_px = c.get('width_px')
            if saved_width_px and width_mm_val > 1e-9:
                min_px_per_mm = saved_width_px / width_mm_val
            else:
                min_px_per_mm = DEFAULTS['min_px_per_mm']
        self.min_px_per_mm.set(str(min_px_per_mm))
        self.max_megapixels.set(g.get('max_megapixels', DEFAULTS['max_megapixels']))
        self.max_depth.set(g.get('max_depth', DEFAULTS['max_depth']))
        self.margin_mm.set(g.get('margin_mm', DEFAULTS['margin_mm']))
        self.margin_depth_pct.set(g.get('margin_depth_pct', DEFAULTS['margin_depth_pct']))
        self.surface.set(g.get('surface', DEFAULTS['surface']))
        self.origin_var.set(g.get('origin', DEFAULTS['origin']))
        if g.get('material_id'):
            self._set_project_material(g['material_id'], g.get('material_name', ""))
        else:
            self._set_project_material(*self._default_material())

        # ── Machine settings ──────────────────────────────────────────
        m = project.get('machine', {})
        for k in machine_defaults.KEYS:
            self._settings[k] = m.get(k, self._machine_defaults[k])

        # ── Steps ─────────────────────────────────────────────────────
        step_dicts = project.get('steps', [])
        if not step_dicts:
            messagebox.showwarning("Load Warning", "Project contains no steps.")
            return

        # Older project files stored dress-up settings once, globally, under
        # 'machine'. Carry them over as every step's dress-up settings when
        # a step dict doesn't already have its own (current-format) values.
        _legacy_dressup_keys = ('ramp_in_enabled', 'ramp_angle_deg', 'ramp_max_length_mm',
                                'backoff_enabled', 'backoff_distance_mm',
                                'arc_rapids_enabled', 'arc_radius_mm')
        legacy_dressup = {k: m[k] for k in _legacy_dressup_keys if k in m}
        if legacy_dressup:
            for d in step_dicts:
                for k, v in legacy_dressup.items():
                    d.setdefault(k, v)

        # Trim or extend the steps list to match the saved count.
        while len(self.steps) > len(step_dicts):
            self._remove_step(self.steps[-1])
        while len(self.steps) < len(step_dicts):
            self._add_step()

        # Project files before version 6 name the rectangular coil scan
        # directions "Spiral (...)"; from version 6 on, "Spiral" is the
        # rounded spiral and the coil is "Coil (...)".
        if version < 6:
            for d in step_dicts:
                rdir = d.get('raster_dir', '')
                if rdir in ("Spiral (Outside-In)", "Spiral (Inside-Out)"):
                    d['raster_dir'] = "Coil" + rdir[len("Spiral"):]

        for step, d in zip(self.steps, step_dicts):
            self._apply_step_dict(step, d)

        # ── Layers (version 2+) ───────────────────────────────────────────
        if version >= 2 and 'layers' in project:
            layer_dicts = project.get('layers') or []
            resolved_layers = []
            failed = []
            saved_w_px, saved_h_px = c.get('width_px'), c.get('height_px')
            for d in layer_dicts:
                layer = self._layer_from_dict(d, path)
                if layer is not None:
                    if version < 4 and saved_w_px and saved_h_px:
                        # Pre-v4 project: scale_pct was pixel-space (1
                        # native px = 1 canvas px at 100%, see the old
                        # layers._affine_params), not canvas-size-relative
                        # -- convert in place using the canvas pixel grid
                        # this project was saved at, so every source keeps
                        # its previous on-screen size and position exactly.
                        H_l, W_l = layer.raw.shape
                        fit_ratio = min(saved_w_px / W_l, saved_h_px / H_l)
                        if fit_ratio > 1e-9:
                            layer.scale_pct = layer.scale_pct / fit_ratio
                    resolved_layers.append(layer)
                else:
                    failed.append(d.get('name') or d.get('path') or d.get('path_rel') or "?")
            self.layers = resolved_layers
            self.background_depth_pct.set(
                str(c.get('background_depth_pct', DEFAULTS['canvas_background_depth_pct'])))
            if self._aspect_lock_var.get() and self.layers:
                self._image_aspect = layers_mod.native_aspect(self.layers[0])
                self._recalc_height()
            self._on_layers_structure_changed()
            if failed:
                messagebox.showwarning(
                    "Source Load Warning",
                    "Project loaded, but these sources could not be found "
                    "or re-rasterized:\n" + "\n".join(failed) +
                    "\n\nUse a source tab's Add Source button, or "
                    "File → Open Source…, to reload them manually.")
            if resolved_layers:
                self.status_var.set(f"Project loaded: {path}  |  {len(resolved_layers)} layer(s)")
            else:
                self.status_var.set(f"Project loaded: {path}")
            self._saved_snapshot = self._snapshot_key()
            self._update_title()
            return

        # ── Legacy single image/STL (version 1) ──────────────────────────
        saved_image_path     = project.get('image_path')
        saved_image_path_rel = project.get('image_path_rel')
        saved_stl_source     = project.get('stl_source')

        resolved_image_path = self._resolve_rel_path(path, saved_image_path, saved_image_path_rel)

        if saved_image_path or saved_image_path_rel:
            if resolved_image_path:
                try:
                    raw = mpimg.imread(resolved_image_path)
                    gray, alpha = layers_mod.to_grayscale(raw)
                    layer = layers_mod.Layer(name=os.path.basename(resolved_image_path),
                                             raw=gray, alpha=alpha, source_kind='image',
                                             path=resolved_image_path,
                                             layer_id=self._new_layer_id())
                    self.layers = [layer]
                    if self._aspect_lock_var.get():
                        h, w = layer.raw.shape
                        self._image_aspect = w / h
                        self._recalc_height()
                    self._on_layers_structure_changed()
                    self.status_var.set(
                        f"Project loaded: {path}  |  Image: {resolved_image_path}")
                    self._saved_snapshot = self._snapshot_key()
                    self._update_title()
                    return
                except Exception as exc:
                    messagebox.showwarning(
                        "Image Load Warning",
                        f"Project loaded but the image could not be opened:\n{exc}")
            else:
                messagebox.showwarning(
                    "Image Not Found",
                    "Project loaded, but the saved image was not found:\n"
                    f"{saved_image_path or saved_image_path_rel}\n\n"
                    "Use File → Open Source… to load an image manually.")

        elif saved_stl_source:
            stl_path     = saved_stl_source.get('path')
            stl_path_rel = saved_stl_source.get('path_rel')
            resolved_stl_path = self._resolve_rel_path(path, stl_path, stl_path_rel)

            if resolved_stl_path:
                try:
                    rot_x, rot_y, rot_z = (saved_stl_source.get('rot_x', 0),
                                           saved_stl_source.get('rot_y', 0),
                                           saved_stl_source.get('rot_z', 0))
                    width_px  = saved_stl_source['width_px']
                    height_px = saved_stl_source['height_px']
                    triangles = stl_mesh.load_stl(resolved_stl_path)
                    rotated   = stl_mesh.apply_rotation(triangles, rot_x, rot_y, rot_z)
                    image, alpha = self._rasterize_stl(rotated, width_px, height_px)
                    layer = layers_mod.Layer(
                        name=os.path.basename(resolved_stl_path), raw=image, alpha=alpha,
                        source_kind='stl', path=resolved_stl_path,
                        stl_params=self._loaded_stl_params(rotated, width_px, height_px,
                                                           rot_x, rot_y, rot_z),
                        layer_id=self._new_layer_id())
                    self._attach_mesh(layer, rotated)
                    self.layers = [layer]
                    if self._aspect_lock_var.get():
                        self._image_aspect = layers_mod.native_aspect(layer)
                        self._recalc_height()
                    self._on_layers_structure_changed()
                    self.status_var.set(
                        f"Project loaded: {path}  |  STL: {resolved_stl_path}")
                    self._saved_snapshot = self._snapshot_key()
                    self._update_title()
                    return
                except Exception as exc:
                    messagebox.showwarning(
                        "STL Load Warning",
                        f"Project loaded but the STL file could not be re-rasterized:\n{exc}")
            else:
                messagebox.showwarning(
                    "STL Not Found",
                    "Project loaded, but the saved STL file was not found:\n"
                    f"{stl_path or stl_path_rel}\n\n"
                    "Use File → Open Source… to load it manually.")

        self.status_var.set(f"Project loaded: {path}")
        self._saved_snapshot = self._snapshot_key()
        self._update_title()

    # ------------------------------------------------------------------ #
    #  Compute / save tool paths
    # ------------------------------------------------------------------ #
    def _resolve_cleanup_reference(self, idx, max_depth, eff_width_mm, eff_height_mm):
        """Gather everything V-Carving Clean-up needs, off the nearest
        preceding V-Carving (Hybrid) step, to later erode that step's own
        idealized-V-bit achievable-depth surface by this step's clearing
        tool (see algorithms.compute_vcarve_synth_achievable_depth) --
        which retreats the pocket boundary exactly as far as needed to not
        gouge the V-bit's actual sloped wall, for any clearing tool
        profile. Must be called on the main thread (like
        _gather_step_params) -- _get_depth_map_for_step reads Tkinter
        variables. Raises ValueError if no such step exists.
        """
        for i in range(idx - 1, -1, -1):
            prior = self.steps[i]
            if prior['algorithm_var'].get() == "V-Carving (Hybrid)":
                tool = self._step_tool(prior)
                extra_offset_mm = float(prior['extra_offset'].get())
                depth_map, n_grey_levels = self._get_depth_map_for_step(prior)
                erosion_cache_key = (
                    tool.profile, tool.diameter_mm, tool.vbit_angle_deg,
                    tool_profile.curve_key(tool.profile_curve), max_depth, eff_width_mm, eff_height_mm,
                    extra_offset_mm, tuple(prior.get('source_visibility', ())))
                return {
                    'tool':              tool,
                    'extra_offset_mm':   extra_offset_mm,
                    'depth_map':         depth_map,
                    'n_grey_levels':     n_grey_levels,
                    'erosion_cache_key': erosion_cache_key,
                }
        raise ValueError(
            "V-Carving Clean-up requires a preceding V-Carving (Hybrid) step.")

    def _step_z_range(self, step):
        """The step's Z range as a normalised (lo, hi) pair; raises
        _StepInputError unless 0 % <= min < max <= 100 %."""
        lo_pct, hi_pct = (float(v) for v in step['z_range'].get())
        if not 0.0 <= lo_pct < hi_pct <= 100.0:
            raise _StepInputError(f"Z range min ({lo_pct:g} %) must be below "
                                  f"its max ({hi_pct:g} %).")
        return lo_pct / 100.0, hi_pct / 100.0

    def _gather_step_params(self, step):
        """Read and validate all parameters for one step.

        Returns a dict of ready-to-use values. Raises ValueError on bad input.
        Must be called on the main thread so that tkinter variable reads are safe.
        """
        idx       = self.steps.index(step)
        algorithm = step['algorithm_var'].get()
        feed_rate    = float(step['feed_rate'].get())
        plunge_rate  = float(step['plunge_rate'].get())
        spindle_rpm  = float(step['spindle_rpm'].get())
        width_mm      = float(self.width_mm.get())
        height_mm     = float(self.height_mm.get())
        margin_mm     = float(self.margin_mm.get())
        max_depth    = float(self.max_depth.get())
        eff_width_mm, eff_height_mm = self._stock_size_mm()
        origin       = self.origin_var.get()

        threshold_mm = float(step['threshold'].get())
        if (ALGORITHM_FAMILY[algorithm] == "Pocket"
                and step['threshold'].auto_var is not None
                and step['threshold'].auto_var.get()):
            # Resolved fresh from this step's own current depth map rather
            # than trusting the displayed value, which _sync_auto_target_depth
            # only keeps live while this step's tab is the selected one.
            auto_depth_map, _ = self._get_depth_map_for_step(step)
            threshold_mm = float(np.max(auto_depth_map)) * max_depth

        params = {
            'idx':           idx,
            'algorithm':     algorithm,
            'feed_rate':     feed_rate,
            'plunge_rate':   plunge_rate,
            'spindle_rpm':   spindle_rpm,
            'library_tool_id':   step['library_tool_id_var'].get(),
            'library_tool_name': step['library_tool_name_var'].get(),
            'threshold_mm':  threshold_mm,
            'eff_width_mm':  eff_width_mm,
            'eff_height_mm': eff_height_mm,
            'max_depth':     max_depth,
            'origin':        origin,
            'extra_offset_mm': float(step['extra_offset'].get()),
            'avoid_air':     step['avoid_air_var'].get(),
            # Used only to key the erosion cache (see _compute_step_work) --
            # the actual depth map is precomputed on the main thread and
            # passed in separately, since it needs Tkinter reads.
            'source_visibility': tuple(step.get('source_visibility', ())),
        }

        if ALGORITHM_FAMILY[algorithm] == "V-Carving":
            tool        = self._step_tool(step)
            params.update({
                'tool':                     tool,
                'min_feature_width_mm':     float(step['min_fw'].get()),
                'max_feature_width_mm':     float(step['max_fw'].get()),
                'vcarve_stepdown_mm':       float(step['vcarve_stepdown'].get()),
                'vcarve_final_stepdown_mm': float(step['vcarve_final_stepdown'].get()),
                'use_2opt':                 step['tsp_2opt_var'].get(),
                'skeleton_mode':            ("voronoi"    if algorithm == "V-Carving (Sub-pixel)"
                                             else "inscribed" if algorithm == "V-Carving (Feature Width)"
                                             else "hybrid" if algorithm == "V-Carving (Hybrid)"
                                             else "medial_axis"),
                'cut_direction':            step['cut_dir_var'].get(),
                'gating_mode':              step['gating_mode_var'].get(),
            })
        elif algorithm == "Contour":
            tool = self._step_tool(step)
            params.update({
                'tool':                tool,
                'contour_z_increment_mm': float(step['contour_z_increment'].get()),
                'contour_adaptive':    step['contour_adaptive_var'].get(),
                'contour_lateral_tol': float(step['contour_lateral_tol'].get()),
                'use_2opt':            step['tsp_2opt_var'].get(),
                'contour_max_step_down': float(step['stepdown'].get()),
                'cut_direction':       step['cut_dir_var'].get(),
                'z_range':             self._step_z_range(step),
            })
        elif algorithm == "Cutout":
            tool = Tool("End Mill", float(step['diameter'].get()), 90.0)
            params.update({
                'tool':                     tool,
                'cutout_depth_mm':          float(step['cutout_depth'].get()),
                'cutout_stepdown_mm':       float(step['cutout_stepdown'].get()),
                'cutout_final_stepdown_mm': float(step['cutout_final_stepdown'].get()),
                'cutout_extra_offset_mm':   float(step['cutout_extra_offset'].get()),
                'cut_direction':            step['cut_dir_var'].get(),
                'use_2opt':                 step['tsp_2opt_var'].get(),
                'tab_settings':             self._tab_settings(step),
            })
        elif ALGORITHM_FAMILY[algorithm] == "Pocket":   # Pocket, V-Carving Clean-up
            tool     = self._step_tool(step)
            stepover = float(step['stepover'].get())
            params.update({
                'tool':                     tool,
                'step_mm':                  tool.diameter_mm * stepover / 100.0,
                'raster_offset_pct':        float(step['raster_offset'].get()),
                'pocket_stepdown_mm':       float(step['stepdown'].get()),
                'pocket_final_stepdown_mm': float(step['raster_final_stepdown'].get()),
                'use_2opt':                 step['tsp_2opt_var'].get(),
                'raster_dir':               step['raster_dir_var'].get(),
                'cut_direction':            step['cut_dir_var'].get(),
                'pocket_target_z_mm':       threshold_mm,
                'pocket_boundary_offset_mm': float(step['pocket_boundary_offset'].get()),
            })
            if algorithm == "V-Carving Clean-up":
                # Off-thread numpy work (in _compute_step_work / the
                # Processing Mask preview) erodes this bundle's depth_map by
                # the source step's own tool a second time by *this* step's
                # clearing tool -- see _resolve_cleanup_reference.
                params['cleanup_reference'] = self._resolve_cleanup_reference(
                    idx, max_depth, eff_width_mm, eff_height_mm)
        elif ALGORITHM_FAMILY[algorithm] == "Registration Holes":
            geo = self._reghole_geometry(step)
            params.update({
                'tool':                Tool("End Mill", float(step['diameter'].get()), 90.0),
                'reghole_pattern':     geo['pattern'],
                'reghole_diag_dir':    geo['diag_dir'],
                'reghole_diameter_mm': float(step['reghole_diameter'].get()),
                'reghole_depth_mm':    float(step['reghole_depth'].get()),
                'reghole_stepdown_mm': float(step['stepdown'].get()),
                'reghole_inset_mm':    geo['inset_mm'],
                # None = auto, resolved by the algorithm against the depth
                # map's own pixel-centre extent, like its hole positions.
                'reghole_spacing_mm':  (None if step['reghole_spacing'].auto_var.get()
                                        else geo['spacing_mm']),
                'cut_direction':       step['cut_dir_var'].get(),
            })
        elif ALGORITHM_FAMILY[algorithm] == "Undercut":
            _profile = step['profile_var'].get()
            tool = Tool(_profile, float(step['diameter'].get()),
                        neck_diameter_mm=float(step['neck_diameter'].get()),
                        neck_length_mm=float(step['neck_length'].get()),
                        shank_diameter_mm=float(step['shank_diameter'].get()),
                        head_height_mm=(float(step['head_height'].get())
                                        if _profile in ("T-Slot", "Dovetail") else 0.0),
                        dovetail_angle_deg=(float(step['dovetail_angle'].get())
                                            if _profile == "Dovetail" else 60.0))
            W, H = self._canvas_shape_px()
            pad_y, pad_x = self._margin_pad_px(H, W)
            params.update({
                'tool':              tool,
                'z_increment_mm':    float(step['undercut_z_increment'].get()),
                'z_range':           self._step_z_range(step),
                'stepover_pct':      float(step['stepover'].get()),
                'cut_direction':     step['cut_dir_var'].get(),
                'open_undercuts':    UNDERCUT_OPEN_MODES[step['undercut_open_var'].get()],
                # Read by _undercut_gaps off the main thread; Layer objects
                # are plain data, so a snapshot of the list is enough.
                'layers':            list(self.layers),
                'canvas_shape':      (H, W),
                'px_per_mm_x':       W / width_mm,
                'px_per_mm_y':       H / height_mm,
                'surface':           self.surface.get(),
                'background_depth':  self._background_depth(),
                'margin_pad':        (pad_y, pad_x),
            })
        else:  # Raster
            tool     = self._step_tool(step)
            stepover = float(step['stepover'].get())
            params.update({
                'tool':                     tool,
                'step_mm':                  tool.diameter_mm * stepover / 100.0,
                'raster_offset_pct':        float(step['raster_offset'].get()),
                'raster_stepdown_mm':       float(step['stepdown'].get()),
                'raster_final_stepdown_mm': float(step['raster_final_stepdown'].get()),
                'use_2opt':                 step['tsp_2opt_var'].get(),
                'raster_dir':               step['raster_dir_var'].get(),
                'cut_direction':            step['cut_dir_var'].get(),
                'gating_mode':              step['gating_mode_var'].get(),
                'work_surface_tol':         threshold_mm / max_depth if max_depth else 0.0,
                'z_range':                  self._step_z_range(step),
            })

        return params

    def _compute_step_work(self, step, params, progress_fn, prev_tp, depth_map, n_grey_levels,
                           prior_depth_map=None):
        """Synchronous algorithm computation — call from a background thread.

        depth_map/n_grey_levels are precomputed on the main thread (see
        _get_depth_map_for_step) rather than read here, both because this
        runs off the main thread and because each step must be computed
        against its OWN source_visibility, not whichever step tab happens
        to be selected. prior_depth_map is the depth map of the step
        "Depth map change" compares against (see _change_reference_index),
        precomputed the same way, or None where there is none.

        Returns a toolpath result. Raises ProgressDialog.Cancelled or any
        algorithm exception on failure.
        """
        algorithm    = params['algorithm']
        # Room for this step's own peak, freed from the erosion cache if
        # memory is short.
        self._erosion_cache.make_room(depth_map.size * STEP_PEAK_BYTES_PER_PX.get(
            ALGORITHM_FAMILY[algorithm], STEP_PEAK_BYTES_PER_PX_DEFAULT))

        tool = params['tool']
        erosion_cache_key = (tool.profile, tool.diameter_mm, tool.vbit_angle_deg,
                             tool_profile.curve_key(tool.profile_curve), params['max_depth'], params['eff_width_mm'], params['eff_height_mm'],
                             params['extra_offset_mm'], params['source_visibility'])

        if ALGORITHM_FAMILY[algorithm] == "V-Carving":
            gating_mode = params['gating_mode']
            if gating_mode == GATING_DEPTH_MAP_CHANGE:
                prior_reference       = (depth_map_change_reference(prior_depth_map)
                                         if prior_depth_map is not None else None)
                subtract_unaccessible = False
            elif gating_mode == "Residual from prior step":
                prior_reference       = prev_tp.result    if prev_tp is not None else None
                subtract_unaccessible = False
            elif gating_mode == "Inaccessible by prior tool":
                prior_reference       = prev_tp.accessible if prev_tp is not None else None
                subtract_unaccessible = False
            elif gating_mode == "Reachable residual":
                prior_reference       = prev_tp.result    if prev_tp is not None else None
                subtract_unaccessible = True
            elif gating_mode == "Reachable inaccessible":
                prior_reference       = prev_tp.accessible if prev_tp is not None else None
                subtract_unaccessible = True
            else:
                prior_reference       = None
                subtract_unaccessible = False
            return compute_vcarve_toolpath(
                depth_map             = depth_map,
                n_grey_levels         = n_grey_levels,
                tool                  = params['tool'],
                prior_reference       = prior_reference,
                subtract_unaccessible = subtract_unaccessible,
                previous_result       = prev_tp.result if prev_tp is not None else None,
                max_depth            = params['max_depth'],
                width_mm             = params['eff_width_mm'],
                height_mm            = params['eff_height_mm'],
                origin               = params['origin'],
                feed_rate            = params['feed_rate'],
                threshold_mm         = params['threshold_mm'],
                min_feature_width_mm = params['min_feature_width_mm'],
                max_feature_width_mm = params['max_feature_width_mm'],
                stepdown_mm          = params['vcarve_stepdown_mm'],
                final_stepdown_mm    = params['vcarve_final_stepdown_mm'],
                extra_offset_mm      = params['extra_offset_mm'],
                skeleton_mode        = params['skeleton_mode'],
                use_2opt             = params['use_2opt'],
                cut_direction        = params['cut_direction'],
                avoid_air            = params['avoid_air'],
                z_safe               = self._settings['z_safe'],
                feed_rapid           = self._settings['feed_rapid'],
                progress             = progress_fn,
                erosion_cache        = self._erosion_cache,
                erosion_cache_key    = erosion_cache_key,
            )
        elif ALGORITHM_FAMILY[algorithm] == "Undercut":
            gaps, warnings = self._undercut_gaps(params, depth_map, progress_fn)
            tp = compute_undercut_toolpath(
                depth_map        = depth_map,
                gaps             = gaps,
                tool             = params['tool'],
                z_increment_mm   = params['z_increment_mm'],
                stepover_pct     = params['stepover_pct'],
                max_depth        = params['max_depth'],
                width_mm         = params['eff_width_mm'],
                height_mm        = params['eff_height_mm'],
                origin           = params['origin'],
                feed_rate        = params['feed_rate'],
                previous_result  = prev_tp.result if prev_tp is not None else None,
                z_safe           = self._settings['z_safe'],
                feed_rapid       = self._settings['feed_rapid'],
                cut_direction    = params['cut_direction'],
                extra_offset_mm  = params['extra_offset_mm'],
                open_undercuts   = params['open_undercuts'],
                z_range          = params['z_range'],
                n_grey_levels    = n_grey_levels,
                progress         = progress_fn,
                erosion_cache    = self._erosion_cache,
                erosion_cache_key = erosion_cache_key,
            )
            tp.metadata['warnings'] = warnings + tp.metadata.get('warnings', [])
            return tp
        elif algorithm == "Contour":
            return compute_contour_toolpath(
                depth_map        = depth_map,
                n_grey_levels    = n_grey_levels,
                tool             = params['tool'],
                z_increment_mm   = params['contour_z_increment_mm'],
                max_depth        = params['max_depth'],
                width_mm         = params['eff_width_mm'],
                height_mm        = params['eff_height_mm'],
                origin           = params['origin'],
                feed_rate        = params['feed_rate'],
                prior_accessible = prev_tp.accessible if prev_tp is not None else None,
                previous_result  = prev_tp.result     if prev_tp is not None else None,
                progress         = progress_fn,
                adaptive_levels  = params['contour_adaptive'],
                lateral_tol_mm   = params['contour_lateral_tol'],
                z_safe             = self._settings['z_safe'],
                feed_rapid         = self._settings['feed_rapid'],
                cut_direction      = params['cut_direction'],
                max_step_down_mm   = params['contour_max_step_down'],
                extra_offset_mm    = params['extra_offset_mm'],
                avoid_air          = params['avoid_air'],
                erosion_cache      = self._erosion_cache,
                erosion_cache_key  = erosion_cache_key,
                z_range            = params['z_range'],
            )
        elif algorithm == "Cutout":
            # Split into Phase A (silhouette mask + contour extraction) and
            # Phase B (tabs + multipass + ordering + simulation) so a
            # tab-settings-only change (the common case when tuning the tab
            # dress-up) can skip straight to Phase B -- see _open_dressup and
            # dressup.apply_tabs. rings_key covers every Phase-A input,
            # including the previous step's identity (Phase A copies its
            # result/accessible/alpha through for Phase B to union with).
            rings_key = (
                tool.profile, tool.diameter_mm,
                params['max_depth'], params['eff_width_mm'], params['eff_height_mm'],
                params['origin'], params['threshold_mm'], params['cutout_extra_offset_mm'],
                params['cut_direction'], params['source_visibility'], id(prev_tp),
            )
            # cached[2] (prev_tp) is held onto only so the id(prev_tp) baked
            # into rings_key above stays meaningful -- without a live
            # reference here, CPython could reuse that id() for an unrelated
            # object once the old prev_tp is garbage-collected, producing a
            # false cache hit.
            cached = step.get('_cutout_rings_cache')
            if cached is not None and cached[0] == rings_key:
                rings_cache = cached[1]
            else:
                rings_cache = _compute_cutout_rings(
                    depth_map          = depth_map,
                    tool               = tool,
                    max_depth          = params['max_depth'],
                    width_mm           = params['eff_width_mm'],
                    height_mm          = params['eff_height_mm'],
                    origin             = params['origin'],
                    threshold_mm       = params['threshold_mm'],
                    boundary_offset_mm = params['cutout_extra_offset_mm'],
                    cut_direction      = params['cut_direction'],
                    prior_accessible   = prev_tp.accessible if prev_tp is not None else None,
                    previous_result    = prev_tp.result     if prev_tp is not None else None,
                    previous_alpha     = prev_tp.alpha      if prev_tp is not None else None,
                    progress           = progress_fn,
                )
                step['_cutout_rings_cache'] = (rings_key, rings_cache, prev_tp)
            return _compute_cutout_from_rings(
                rings_cache, tool, params['cutout_depth_mm'], params['max_depth'],
                params['eff_width_mm'], params['eff_height_mm'],
                threshold_mm       = params['threshold_mm'],
                stepdown_mm        = params['cutout_stepdown_mm'],
                final_stepdown_mm  = params['cutout_final_stepdown_mm'],
                boundary_offset_mm = params['cutout_extra_offset_mm'],
                cut_direction      = params['cut_direction'],
                use_2opt           = params['use_2opt'],
                z_safe             = self._settings['z_safe'],
                feed_rapid         = self._settings['feed_rapid'],
                tab_settings       = params['tab_settings'],
                avoid_air          = params['avoid_air'],
                feed_rate          = params['feed_rate'],
                n_grey_levels      = n_grey_levels,
                progress           = progress_fn,
            )
        elif ALGORITHM_FAMILY[algorithm] == "Registration Holes":
            return compute_registration_holes_toolpath(
                tool               = params['tool'],
                width_mm           = params['eff_width_mm'],
                height_mm          = params['eff_height_mm'],
                shape              = depth_map.shape,
                pattern            = params['reghole_pattern'],
                origin             = params['origin'],
                diag_dir           = params['reghole_diag_dir'],
                hole_diameter_mm   = params['reghole_diameter_mm'],
                hole_depth_mm      = params['reghole_depth_mm'],
                stepdown_mm        = params['reghole_stepdown_mm'],
                inset_mm           = params['reghole_inset_mm'],
                spacing_mm         = params['reghole_spacing_mm'],
                cut_direction      = params['cut_direction'],
                max_depth          = params['max_depth'],
                prior_accessible   = prev_tp.accessible if prev_tp is not None else None,
                previous_result    = prev_tp.result     if prev_tp is not None else None,
                previous_alpha     = prev_tp.alpha      if prev_tp is not None else None,
                progress           = progress_fn,
            )
        elif ALGORITHM_FAMILY[algorithm] == "Pocket":   # Pocket, V-Carving Clean-up
            # Pocket's own boundary_offset_mm is a lateral-only mask push
            # applied after the achievable-depth surface is computed (see
            # compute_pocket_toolpath), so -- unlike extra_offset_mm for
            # Raster/Contour/V-Carving -- it plays no part in that surface
            # and must not be baked into its cache key.
            pocket_erosion_cache_key = (
                tool.profile, tool.diameter_mm, tool.vbit_angle_deg,
                params['max_depth'], params['eff_width_mm'], params['eff_height_mm'],
                params['source_visibility'])

            reference_surface = None
            if algorithm == "V-Carving Clean-up":
                # Erode the source V-Carving (Hybrid) step's own idealized-
                # V-bit achievable-depth surface a second time, by this
                # step's own clearing tool, so compute_pocket_toolpath's
                # Stage 1 retreats the boundary exactly as far as needed to
                # not gouge the V-bit's actual sloped wall (see
                # compute_vcarve_synth_achievable_depth). Distinguished
                # from plain Pocket's cache key -- entries there are eroded
                # against the raw depth_map and mean something different.
                ref = params['cleanup_reference']
                reference_surface = compute_vcarve_synth_achievable_depth(
                    ref['depth_map'], ref['tool'], params['max_depth'],
                    params['eff_width_mm'], params['eff_height_mm'],
                    extra_offset_mm = ref['extra_offset_mm'],
                    erosion_cache = self._erosion_cache,
                    erosion_cache_key = ref['erosion_cache_key'],
                    progress = progress_fn,
                )
                pocket_erosion_cache_key = pocket_erosion_cache_key + ('clean_up',)

            return compute_pocket_toolpath(
                depth_map             = depth_map,
                n_grey_levels         = n_grey_levels,
                tool                  = params['tool'],
                step_mm               = params['step_mm'],
                stepdown_mm           = params['pocket_stepdown_mm'],
                final_stepdown_mm     = params['pocket_final_stepdown_mm'],
                boundary_offset_mm    = params['pocket_boundary_offset_mm'],
                reference_surface     = reference_surface,
                max_depth             = params['max_depth'],
                width_mm              = params['eff_width_mm'],
                height_mm             = params['eff_height_mm'],
                raster_dir            = params['raster_dir'],
                target_z_mm           = params['pocket_target_z_mm'],
                origin                = params['origin'],
                previous_result       = prev_tp.result if prev_tp is not None else None,
                raster_offset_pct = params['raster_offset_pct'],
                feed_rate         = params['feed_rate'],
                use_2opt          = params['use_2opt'],
                cut_direction     = params['cut_direction'],
                avoid_air         = params['avoid_air'],
                z_safe            = self._settings['z_safe'],
                feed_rapid        = self._settings['feed_rapid'],
                progress          = progress_fn,
                erosion_cache     = self._erosion_cache,
                erosion_cache_key = pocket_erosion_cache_key,
            )
        else:  # Raster
            gating_mode = params['gating_mode']
            if gating_mode == GATING_DEPTH_MAP_CHANGE:
                prior_reference       = (depth_map_change_reference(prior_depth_map)
                                         if prior_depth_map is not None else None)
                subtract_unaccessible = False
            elif gating_mode == "Residual from prior step":
                prior_reference       = prev_tp.result    if prev_tp is not None else None
                subtract_unaccessible = False
            elif gating_mode == "Inaccessible by prior tool":
                prior_reference       = prev_tp.accessible if prev_tp is not None else None
                subtract_unaccessible = False
            elif gating_mode == "Reachable residual":
                prior_reference       = prev_tp.result    if prev_tp is not None else None
                subtract_unaccessible = True
            elif gating_mode == "Reachable inaccessible":
                prior_reference       = prev_tp.accessible if prev_tp is not None else None
                subtract_unaccessible = True
            else:
                prior_reference       = None
                subtract_unaccessible = False
            H, W      = depth_map.shape
            px_per_mm = max(W / params['eff_width_mm'],
                            H / params['eff_height_mm'])
            rdir      = params['raster_dir']
            step_mm   = params['step_mm']
            return compute_raster_toolpath(
                depth_map             = depth_map,
                n_grey_levels         = n_grey_levels,
                tool                  = params['tool'],
                step_mm               = step_mm,
                stepdown_mm           = params['raster_stepdown_mm'],
                final_stepdown_mm     = params['raster_final_stepdown_mm'],
                extra_offset_mm       = params['extra_offset_mm'],
                max_depth             = params['max_depth'],
                width_mm              = params['eff_width_mm'],
                height_mm             = params['eff_height_mm'],
                raster_dir            = params['raster_dir'],
                origin                = params['origin'],
                work_surface_tol      = params['work_surface_tol'],
                previous_result       = prev_tp.result if prev_tp is not None else None,
                prior_reference       = prior_reference,
                subtract_unaccessible = subtract_unaccessible,
                raster_offset_pct = params['raster_offset_pct'],
                feed_rate         = params['feed_rate'],
                use_2opt          = params['use_2opt'],
                cut_direction     = params['cut_direction'],
                avoid_air         = params['avoid_air'],
                z_safe            = self._settings['z_safe'],
                feed_rapid        = self._settings['feed_rapid'],
                progress          = progress_fn,
                erosion_cache     = self._erosion_cache,
                erosion_cache_key = erosion_cache_key,
                z_range           = params['z_range'],
            )

    def _compute_step(self, step):
        if self.current_image is None:
            messagebox.showinfo("No image", "Please open an image first.")
            return

        self._place_stl_sources()

        idx = self.steps.index(step)
        try:
            params = self._gather_step_params(step)
        except _StepInputError as exc:
            messagebox.showerror("Invalid input", f"Step {idx + 1}: {exc}")
            return
        except ValueError:
            messagebox.showerror("Invalid input",
                                 f"Please check the settings for Step {idx + 1}.")
            return
        depth_map, n_grey_levels = self._get_depth_map_for_step(step)

        # ── Prior-step checks (all before the dialog opens) ─────────────────
        prev_tp   = self.toolpaths[idx - 1] if idx > 0 else None
        algorithm = params['algorithm']

        if algorithm == "Contour":
            # Contour reads the prior step's result only to drop air cuts.
            if params['avoid_air'] and idx > 0 and prev_tp is None:
                if not messagebox.askyesno(
                        "Previous step not computed",
                        f"Step {idx} has not been computed yet.\n\n"
                        "Without its result Avoid air treats the stock as "
                        "uncut and keeps the cuts through material that step "
                        "would clear.\n\nContinue anyway?"):
                    return
        elif ALGORITHM_FAMILY[algorithm] == "Undercut":
            pass  # Undercut cuts the gaps of its own sources regardless of
                  # any prior step -- a missing prev_tp only affects this
                  # step's own result bookkeeping, not which pixels get cut,
                  # so there is nothing to warn about here.
        elif algorithm == "Cutout":
            pass  # Cutout cuts based on its own depth_map threshold regardless
                  # of any prior step -- a missing prev_tp only affects this
                  # step's own result/accessible/alpha bookkeeping, not which
                  # pixels get cut, so there is nothing to warn about here.
        elif ALGORITHM_FAMILY[algorithm] == "Registration Holes":
            pass  # Holes are placed on the stock alone; a prior step only
                  # feeds this step's result/accessible/alpha bookkeeping.
        elif ALGORITHM_FAMILY[algorithm] == "Pocket":   # Pocket, V-Carving Clean-up
            pass  # Pocket/Clean-up cuts based on its own Target depth
                  # against the depth map regardless of any prior step -- a
                  # missing prev_tp only affects this step's own
                  # result/accessible bookkeeping, not which pixels get cut,
                  # so there is nothing to warn about here.
        else:
            gating_mode = params['gating_mode']
            if (gating_mode not in ("Full depth map", GATING_DEPTH_MAP_CHANGE)
                    and idx > 0 and prev_tp is None):
                surf_name = ("accessible surface"
                             if gating_mode in ("Inaccessible by prior tool",
                                                "Reachable inaccessible")
                             else "result surface")
                if not messagebox.askyesno(
                        "Previous step not computed",
                        f"Step {idx} has not been computed yet.\n\n"
                        f"Without its {surf_name} this step will process the "
                        f"full depth map.\n\nContinue anyway?"):
                    return

        # The reference step's depth map needs no computed step; it is
        # built here because _get_depth_map_for_step reads Tk variables.
        prior_depth_map = None
        ref_idx = self._change_reference_index(idx)
        if params.get('gating_mode') == GATING_DEPTH_MAP_CHANGE and ref_idx is not None:
            prior_depth_map, _ = self._get_depth_map_for_step(self.steps[ref_idx])

        dlg = ProgressDialog(self.root,
                             title=f"Computing Step {idx + 1} Tool Path")

        def _worker():
            try:
                result = self._compute_step_work(
                    step, params, dlg.report, prev_tp, depth_map, n_grey_levels,
                    prior_depth_map)
            except ProgressDialog.Cancelled:
                self.root.after(0, lambda: dlg.close("Cancelled."))
                return
            except Exception as exc:
                self.root.after(0, lambda e=exc: dlg.close(f"Error: {e}"))
                self.root.after(0, lambda e=exc: messagebox.showerror(
                    "Computation error", str(e)))
                return

            def _done():
                self.toolpaths[idx] = result
                self._store_cutting_metadata(result, params)
                dlg.close("Completed.")
                self._show_step_stats(idx, step, result)
                self.display_step_var.set(idx + 1)
                if not self.show_cut_moves_var.get():
                    self.show_cut_moves_var.set(True)
                self.toolpath_scale_var.set(100.0)
                self.toolpath_scale_label.config(text="100%")
                self._on_display_mode_changed()

            self.root.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def _compute_all_steps(self):
        if self.current_image is None:
            messagebox.showinfo("No image", "Please open an image first.")
            return
        if not self.steps:
            return
        self._place_stl_sources()

        # Validate every step upfront so the dialog never opens on bad input,
        # and precompute each step's own depth map here on the main thread
        # (each step has its own source_visibility, so these can differ).
        all_params = []
        all_depth_maps = []
        for step in self.steps:
            try:
                all_params.append(self._gather_step_params(step))
            except _StepInputError as exc:
                messagebox.showerror("Invalid input",
                                     f"Step {self.steps.index(step) + 1}: {exc}")
                return
            except ValueError:
                idx = self.steps.index(step)
                messagebox.showerror("Invalid input",
                                     f"Please check the settings for Step {idx + 1}.")
                return
            all_depth_maps.append(self._get_depth_map_for_step(step))
        all_ref_idx = [self._change_reference_index(i) for i in range(len(self.steps))]

        n   = len(self.steps)
        dlg = ProgressDialog(self.root, title="Computing All Step Tool Paths")

        def _worker():
            def step_progress(step_i):
                def fn(pct, msg=""):
                    scaled = step_i * 100.0 / n + pct / n
                    dlg.report(scaled, f"Step {step_i + 1}/{n}: {msg}" if msg else "")
                return fn

            for i, (step, params, (depth_map, n_grey_levels)) in enumerate(
                    zip(self.steps, all_params, all_depth_maps)):
                idx     = params['idx']
                prev_tp = self.toolpaths[idx - 1] if idx > 0 else None
                ref_idx = all_ref_idx[idx]
                prior_depth_map = all_depth_maps[ref_idx][0] if ref_idx is not None else None
                try:
                    dlg.report(i * 100.0 / n, f"Starting Step {i + 1} of {n}…")
                    result = self._compute_step_work(
                        step, params, step_progress(i), prev_tp, depth_map, n_grey_levels,
                        prior_depth_map)
                except ProgressDialog.Cancelled:
                    self.root.after(0, lambda: dlg.close("Cancelled."))
                    return
                except Exception as exc:
                    self.root.after(0, lambda e=exc: dlg.close(f"Error: {e}"))
                    self.root.after(0, lambda e=exc: messagebox.showerror(
                        "Computation error", str(e)))
                    return

                # Store immediately so the next step's prev_tp is up to date.
                self.toolpaths[idx] = result
                self._store_cutting_metadata(result, params)

                def _update_ui(idx=idx, s=step, r=result):
                    self._show_step_stats(idx, s, r)
                self.root.after(0, _update_ui)

            def _all_done():
                dlg.close("All steps completed.")
                self.display_step_var.set(len(self.steps))
                if not self.show_cut_moves_var.get():
                    self.show_cut_moves_var.set(True)
                self.toolpath_scale_var.set(100.0)
                self.toolpath_scale_label.config(text="100%")
                self._on_display_mode_changed()

            self.root.after(0, _all_done)

        threading.Thread(target=_worker, daemon=True).start()

    def _build_steps_info(self):
        info = []
        for i, (step, tp) in enumerate(zip(self.steps, self.toolpaths)):
            profile = (tp.metadata.get('profile', step['profile_var'].get())
                       if tp else step['profile_var'].get())
            diam_mm = tp.metadata.get('radius_mm', 0) * 2 if tp else 0.0
            tool_id   = tp.metadata.get('library_tool_id', "") if tp else ""
            tool_name = tp.metadata.get('library_tool_name', "") if tp else ""
            label = f"Step {i + 1} — {profile} ∅{diam_mm:.4g} mm"
            if tool_id:
                label += f" ({tool_name})"
            info.append({
                'label':       label,
                'computed':    tp is not None,
                'profile':     profile,
                'diameter_mm': diam_mm,
                'profile_key': tool_profile.curve_key(self._step_profile_curve(step))
                               if profile == "Custom" else "",
                'tool_id':     tool_id,
            })
        return info

    def _export_toolpaths(self):
        steps_info = self._build_steps_info()
        dlg = ExportToolpathDialog(self.root, steps_info)
        indices = dlg.selected_indices
        if not indices:
            return

        z_safe      = self._settings['z_safe']
        feed_rapid  = self._settings['feed_rapid']
        accel_mm_s2 = self._settings['accel_mm_s2']
        preamble    = (self._settings['preamble']
                      or DEFAULT_PREAMBLE.format(z_safe=z_safe))
        postamble   = (self._settings['postamble']
                       or DEFAULT_POSTAMBLE.format(z_safe=z_safe))

        # Warn if the selected steps use different tools — tool changes are not
        # supported in a single G-code file. Two linked steps are the same
        # tool exactly when their library ids match; otherwise profile,
        # diameter and (for Custom) the curve decide.
        def _same_tool(a, b):
            if a['tool_id'] and b['tool_id']:
                return a['tool_id'] == b['tool_id']
            return ((a['profile'], a['diameter_mm'], a['profile_key'])
                    == (b['profile'], b['diameter_mm'], b['profile_key']))

        selected = [steps_info[i] for i in indices]
        if any(not _same_tool(a, b)
               for i, a in enumerate(selected) for b in selected[i + 1:]):
            proceed = messagebox.askokcancel(
                "Tool change not supported",
                "The selected steps use different tools. A single G-code file "
                "cannot perform a tool change automatically. Export anyway?",
            )
            if not proceed:
                return

        path = filedialog.asksaveasfilename(
            title="Export Toolpath",
            defaultextension=".nc",
            initialfile=dlg.suggested_stem,
            filetypes=[("G-code", "*.nc *.gcode *.tap"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            bodies = []
            # S word + spin-up dwell only where the speed actually changes:
            # consecutive steps at the same RPM don't wait again.
            last_rpm = None
            for idx in indices:
                tp   = self.toolpaths[idx]
                step = self.steps[idx]
                feed_rate = tp.metadata.get('feed_rate', 800.0)
                rpm = tp.metadata.get('spindle_rpm', 0.0)
                emit_rpm = rpm if (rpm > 0 and rpm != last_rpm) else 0.0
                if rpm > 0:
                    last_rpm = rpm
                bodies.append(tp.to_gcode(feed_rate, feed_rapid, z_safe, accel_mm_s2,
                                           ramp=self._ramp_settings(step),
                                           backoff=self._backoff_settings(step),
                                           arc=self._arc_settings(step),
                                           plunge_rate=tp.metadata.get('plunge_rate', 0.0),
                                           spindle_rpm=emit_rpm,
                                           spindle_dwell_s=self._settings['spindle_dwell_s']))

            gcode = preamble + "\n\n" + "\n\n".join(bodies) + "\n\n" + postamble
            with open(path, "w", encoding="utf-8") as f:
                f.write(gcode)
            self.status_var.set(f"Exported: {path}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    # ------------------------------------------------------------------ #
    #  Display mode
    # ------------------------------------------------------------------ #
    def _on_step_tab_changed(self, _event=None):
        """Sync display_step_var to whichever step tab is now selected, and
        -- since source visibility is per-step -- refresh every source
        tab's Visible checkbox and the live preview to match."""
        tab_id = self._steps_nb.select()
        if not tab_id or not hasattr(self, 'display_mode_var'):
            return
        selected = self._steps_nb.index(tab_id)
        self.display_step_var.set(selected + 1)
        self._sync_layer_visible_checkboxes()
        self._recompute_composite()
        self._on_display_mode_changed()

    def _sync_layer_visible_checkboxes(self):
        """Refresh every source tab's Visible checkbox to reflect the
        currently-selected step's stored source_visibility, without
        triggering each checkbox's own write-back (which would otherwise
        recompute the composite once per source instead of once overall --
        _on_step_tab_changed does that itself right after)."""
        step = self._current_step()
        if step is None:
            return
        vis = step['source_visibility']
        self._suspend_visibility_apply = True
        try:
            for i, entry in enumerate(self._layer_tabs):
                entry['visible_var'].set(vis[i] if i < len(vis) else True)
        finally:
            self._suspend_visibility_apply = False

    def _on_display_mode_changed(self):
        mode = self.display_mode_var.get()
        idx  = self.display_step_var.get() - 1

        # Update combobox to show only modes relevant to this step's algorithm/state
        has_step       = 0 <= idx < len(self.steps)
        has_accessible = self._accessible_for_step(idx) is not None
        step_computed  = (0 <= idx < len(self.toolpaths) and self.toolpaths[idx] is not None)
        has_prev_result = (idx > 0
                           and idx - 1 < len(self.toolpaths)
                           and self.toolpaths[idx - 1] is not None
                           and self.toolpaths[idx - 1].result is not None)
        available_modes = [
            m for m in DISPLAY_MODES
            if not (m == DISPLAY_PROC_MASK  and not has_step)
            and not (m in (DISPLAY_STEP_ACCESSIBLE, DISPLAY_STEP_UNACCESSIBLE)
                     and not has_accessible)
            and not (m in (DISPLAY_STEP_RES, DISPLAY_STEP_RESULT, DISPLAY_STEP_3D) and not step_computed)
            and not (m == DISPLAY_DIFF_RES_UNCUT and not (has_accessible and has_prev_result))
            and not (m == DISPLAY_DIFF_DEPTH_MAP
                     and self._change_reference_index(idx) is None)
            and not (m == DISPLAY_DIFF_UNCUT_UNCUT and not (
                has_accessible
                and idx > 0
                and idx - 1 < len(self.toolpaths)
                and self.toolpaths[idx - 1] is not None
                and self.toolpaths[idx - 1].accessible is not None
            ))
        ]
        for m in DISPLAY_MODES:
            state = "normal" if m in available_modes else "disabled"
            self._view_menu.entryconfig(self._display_menu_indices[m], state=state)
        if mode not in available_modes:
            self.display_mode_var.set(DISPLAY_ORIGINAL)
            mode = DISPLAY_ORIGINAL
        # Reset 3D view state only when actually entering a 3D mode from a
        # non-3D mode — not on every re-render (e.g. toggling rapids/cut
        # moves or recomputing a step) while already viewing in 3D, so the
        # user's camera position is preserved.
        if mode in DISPLAY_3D_MODES:
            if self._prev_display_mode not in DISPLAY_3D_MODES:
                self._3d_elev = 30.0
                self._3d_azim = -60.0
                self._3d_zoom = 1.0
                self._3d_pan_offset = np.array([0.0, 0.0, 0.0])
            # Ensure at least the path is visible — nothing to see without it
            if not self.show_cut_moves_var.get() and not self.show_rapids_var.get():
                self.show_cut_moves_var.set(True)
                self.show_rapids_var.set(True)
        self._prev_display_mode = mode
        self._render()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _elevation_sign(self, mode):
        """Sign that makes a mode's raw array values read as physically-real
        Z in mm, matching _update_coord_display's convention: material
        amount (residual/unaccessible/mask/diff) is positive going up, but
        Original, Step Accessible and Step Result read positive going down
        (deeper cut = lower)."""
        return -1 if mode in (DISPLAY_ORIGINAL, DISPLAY_STEP_RESULT,
                              DISPLAY_STEP_ACCESSIBLE) else 1

    def _apply_shading(self, img, mode, cmap, vmin, vmax, step_idx):
        """RGB hillshade of *img*, tinted with its own cmap/vmin/vmax (so
        colors match the unshaded view exactly) and lit using a physically-
        scaled elevation, with vertical exaggeration auto-scaled from max cut
        depth vs. canvas footprint -- real-world relief carving is typically
        only a few mm deep on a canvas tens of cm wide, so true-to-scale
        shading would look flat.
        """
        try:
            step   = self.steps[step_idx] if 0 <= step_idx < len(self.steps) else None
            params = self._gather_step_params(step) if step is not None else None
            max_depth     = params['max_depth']     if params is not None else float(self.max_depth.get())
            eff_width_mm  = params['eff_width_mm']  if params is not None else float(self.width_mm.get())
            eff_height_mm = params['eff_height_mm'] if params is not None else float(self.height_mm.get())
        except (ValueError, KeyError, ZeroDivisionError, AttributeError):
            max_depth, eff_width_mm, eff_height_mm = 5.0, float(self.width_mm.get()), float(self.height_mm.get())

        H, W = img.shape
        cmap_obj = cmap if isinstance(cmap, mcolors.Colormap) else plt.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        base_rgb = cmap_obj(norm(img))[..., :3]

        # img itself is what to colour, but for DISPLAY_ORIGINAL it's the raw
        # composite (self.current_image), whose polarity flips with the
        # white/black "surface" setting -- unlike every other mode's img,
        # it does NOT follow the canonical 1.0 = deepest-cut convention that
        # _elevation_sign assumes, so elevation must come from the canonical
        # depth map instead (same fix _update_coord_display already applies).
        elevation_img = self._get_depth_map()[0] if mode == DISPLAY_ORIGINAL else img
        elevation_mm = self._elevation_sign(mode) * elevation_img.astype(np.float64) * max_depth
        dx = eff_width_mm  / W
        dy = eff_height_mm / H
        span_mm = min(eff_width_mm, eff_height_mm)
        vert_exag = float(np.clip(span_mm * 0.15 / max_depth, 1.0, 50.0)) if max_depth > 0 else 1.0

        ls = mcolors.LightSource(azdeg=315, altdeg=45)
        return ls.shade_rgb(base_rgb, elevation_mm, blend_mode="soft",
                            vert_exag=vert_exag, dx=dx, dy=dy)

    def _margin_pad_px(self, H, W):
        """(pad_y, pad_x): the stock margin in pixels on each side of an
        H x W canvas, or (0, 0) with no margin."""
        try:
            margin_mm = float(self.margin_mm.get())
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
        except ValueError:
            return 0, 0
        if margin_mm <= 0.0 or width_mm <= 0.0 or height_mm <= 0.0:
            return 0, 0
        return (max(1, round(margin_mm * H / height_mm)),
                max(1, round(margin_mm * W / width_mm)))

    def _undercut_gaps(self, params, depth_map, progress_fn):
        """The undercut gaps of the step's visible STL sources on the step's
        (margin-padded) depth-map grid, and any warnings. Builds each STL
        layer's dexels from its mesh first if missing or stale. Runs off the
        main thread: it reads only *params* and Layer data."""
        stl_layers = self._visible_stl_layers(params)
        if not stl_layers:
            raise ValueError("Undercut needs at least one visible STL source: "
                             "image sources carry no undercut geometry.")
        warnings = self._ensure_dexels(stl_layers, params, progress_fn)
        gaps, gap_warnings = self._composite_gaps_padded(params, depth_map)
        return gaps, warnings + gap_warnings

    @staticmethod
    def _visible_stl_layers(params):
        """The STL layers of params['layers'] visible under
        params['source_visibility']."""
        visibility = params['source_visibility']
        return [layer for i, layer in enumerate(params['layers'])
                if layer.source_kind == "stl"
                and (visibility[i] if i < len(visibility) else layer.visible)]

    @classmethod
    def _rotated_mesh(cls, layer):
        """The STL layer's mesh rotated as its raster was (stl_params
        rot_x/rot_y/rot_z): the one held on the layer (layer.mesh), read
        from its file only if none is held for this path and rotation."""
        sp = layer.stl_params or {}
        key = (layer.path, sp.get('rot_x', 0), sp.get('rot_y', 0), sp.get('rot_z', 0))
        if layer.mesh is None or layer.mesh_key != key:
            cls._attach_mesh(layer, stl_mesh.apply_rotation(
                stl_mesh.load_stl(layer.path), *key[1:]))
        return layer.mesh

    def _schedule_stl_placement(self):
        """Run _place_stl_sources once the current event has been handled,
        so a cascade of changes (a project load, an import setting Width
        and Height) places the STL sources once, against the final canvas
        and source stack. The composite then shows the exact canvas-grid
        rasters, not the resampled preview rasters."""
        if self._stl_placement_pending is None:
            self._stl_placement_pending = self.root.after_idle(self._run_stl_placement)

    def _run_stl_placement(self):
        self._stl_placement_pending = None
        self._place_stl_sources()

    def _place_stl_sources(self):
        """Rasterize every STL source whose placement isn't 1:1 directly on
        the canvas grid at that placement (layers.place_stl_layer), unless
        its placed raster is still current, so a step is computed against
        exact samples of the mesh rather than its raster resampled -- which
        interpolates heights where the surface jumps, and leaves only part
        of each outline pixel covered. Called on the main thread before a
        step's depth map is built and after every structural change
        (_schedule_stl_placement); the live preview keeps resampling while
        a placement is being edited."""
        try:
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
        except ValueError:
            return
        W, H = self._canvas_shape_px()
        if width_mm <= 0.0 or height_mm <= 0.0:
            return
        ppm_x, ppm_y = W / width_mm, H / height_mm
        origin = self.origin_var.get()
        todo = [layer for layer in self.layers
                if layer.source_kind == "stl" and layer.path
                and not layers_mod.raw_fills_canvas(layer, (H, W))
                and layers_mod.placed_raster(layer, (H, W), ppm_x, ppm_y, origin) is None]
        if not todo:
            return
        status = self.status_var.get()
        self.root.config(cursor="watch")
        try:
            for layer in todo:
                self.status_var.set(f"Rasterizing {layer.name} at its placement…")
                self.root.update_idletasks()
                try:
                    layers_mod.place_stl_layer(layer, self._rotated_mesh(layer), (H, W),
                                               ppm_x, ppm_y, origin)
                except (OSError, ValueError) as exc:
                    messagebox.showerror("Rasterize Error",
                                         f"Could not rasterize {layer.name} at its "
                                         f"placement:\n{exc}")
        finally:
            self.root.config(cursor="")
            self.status_var.set(status)
        self._recompute_composite()
        self._erosion_cache.clear()
        self._render()

    @classmethod
    def _ensure_dexels(cls, stl_layers, params, progress_fn):
        """Build the dexels of every layer in *stl_layers* whose dexels are
        missing or stale -- for a layer with a placed raster current on
        params' canvas, that raster's own dexels; returns warnings about
        non-watertight meshes."""
        warnings = []
        for layer in stl_layers:
            placed = layers_mod.placed_raster(
                layer, params['canvas_shape'], params['px_per_mm_x'],
                params['px_per_mm_y'], params['origin'])
            if placed is not None:
                if placed.dexels is not None:
                    continue
                progress_fn(0, f"Reading undercuts of {layer.name}…")
                dexels = layers_mod.add_placed_dexels(layer, cls._rotated_mesh(layer))
            else:
                key = (layer.path, tuple(sorted((layer.stl_params or {}).items())),
                       layer.raw.shape)
                if layer.dexels is not None and layer.dexels_key == key:
                    continue
                progress_fn(0, f"Reading undercuts of {layer.name}…")
                H_l, W_l = layer.raw.shape
                layer.dexels = stl_mesh.generate_dexels(cls._rotated_mesh(layer), W_l, H_l)
                layer.dexels_key = key
                dexels = layer.dexels
            if dexels.n_invalid:
                warnings.append(f"{layer.name}: {dexels.n_invalid} mesh columns "
                                f"are not watertight and are treated as solid.")
        return warnings

    @staticmethod
    def _composite_gaps_padded(params, depth_map):
        """layers.composite_gaps of params' layers on the margin-padded grid
        of *depth_map*, and its warnings. Dexels must already be built."""
        pad_y, pad_x = params['margin_pad']
        H, W = params['canvas_shape']
        composite_depth = depth_map[pad_y:pad_y + H, pad_x:pad_x + W].astype(np.float64)
        gaps, gap_warnings = layers_mod.composite_gaps(
            params['layers'], (H, W), params['px_per_mm_x'], params['px_per_mm_y'],
            params['surface'], composite_depth,
            visibility_override=list(params['source_visibility']),
            origin=params['origin'], background_depth=params['background_depth'])
        return layers_mod.pad_gaps(gaps, pad_y, pad_x), gap_warnings

    def _overhang_masks(self, visibility, depth_map):
        """(closed, open_bottom) overhang masks (see layers.overhang_masks)
        of the STL sources visible under *visibility*, on *depth_map*'s
        margin-padded grid; None without a visible STL source or when the
        canvas fields don't parse. Cached by everything the gaps depend on."""
        try:
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
            W, H = self._canvas_shape_px()
        except (ValueError, ZeroDivisionError):
            return None
        if width_mm <= 0.0 or height_mm <= 0.0:
            return None
        visibility = list(visibility) if visibility is not None else [
            layer.visible for layer in self.layers]
        params = {
            'layers':            list(self.layers),
            'source_visibility': visibility,
            'canvas_shape':      (H, W),
            'px_per_mm_x':       W / width_mm,
            'px_per_mm_y':       H / height_mm,
            'surface':           self.surface.get(),
            'origin':            self.origin_var.get(),
            'background_depth':  self._background_depth(),
            'margin_pad':        self._margin_pad_px(H, W),
        }
        stl_layers = self._visible_stl_layers(params)
        if not stl_layers:
            return None
        key = (tuple(visibility), params['canvas_shape'], params['px_per_mm_x'],
               params['px_per_mm_y'], params['surface'], params['origin'],
               params['margin_pad'], self.n_grey_levels,
               tuple(self._layer_gap_signature(layer) for layer in params['layers']),
               hashlib.blake2b(np.ascontiguousarray(depth_map)).digest())
        cached = self._overhang_cache.get(key)
        if cached is not None:
            return cached
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self._ensure_dexels(stl_layers, params, lambda pct, msg: None)
            gaps, _ = self._composite_gaps_padded(params, depth_map)
        except (OSError, ValueError):
            return None
        finally:
            self.root.config(cursor="")
        if gaps.shape != depth_map.shape:
            return None
        tol = 0.5 / max(self.n_grey_levels - 1, 1)
        masks = layers_mod.overhang_masks(gaps, tol)
        if len(self._overhang_cache) >= 8:
            self._overhang_cache.clear()
        self._overhang_cache[key] = masks
        return masks

    @staticmethod
    def _layer_gap_signature(layer):
        """Everything of *layer* that composite_gaps reads: its raw and
        alpha arrays by identity, every placement/blend/depth field by
        value."""
        return ((id(layer.raw), id(layer.alpha))
                + tuple((f.name, repr(getattr(layer, f.name)))
                        for f in dataclasses.fields(layer)
                        if f.name not in ('raw', 'alpha', 'dexels', 'dexels_key', 'mesh',
                                              'mesh_key')))

    def _overhang_reach(self, step_idx, shape):
        """Union of the undercut_reach masks of the computed Undercut steps
        up to and including *step_idx* that match *shape*, or None."""
        reach = None
        for tp in self.toolpaths[:step_idx + 1]:
            if tp is None or tp.undercut_reach is None or tp.undercut_reach.shape != shape:
                continue
            reach = tp.undercut_reach.copy() if reach is None else reach | tp.undercut_reach
        return reach

    def _overlay_overhangs(self, img, mode, step_idx):
        """Shade the overhangs of the visible STL sources over the Original
        or Step Result view: amber where every gap reaches full depth (open
        bottom, reachable from the back), magenta where a gap is enclosed by
        the source, blue where computed Undercut steps reach. Returns the title
        suffix to append, or ''."""
        if not self.show_overhangs_var.get():
            return ""
        if mode == DISPLAY_ORIGINAL:
            step = self._current_step()
            visibility = step['source_visibility'] if step is not None else None
            depth_map, _ = self._get_depth_map()
        elif 0 <= step_idx < len(self.steps):
            step = self.steps[step_idx]
            visibility = step.get('source_visibility')
            depth_map, _ = self._get_depth_map_for_step(step)
        else:
            return ""
        if depth_map.shape != img.shape[:2]:
            return ""
        masks = self._overhang_masks(visibility, depth_map)
        if masks is None:
            return ""
        closed, open_bottom = masks
        if not (closed.any() or open_bottom.any()):
            return " · no overhangs"
        reach = self._overhang_reach(step_idx, depth_map.shape)
        rgba = np.zeros(depth_map.shape + (4,), np.float32)
        rgba[open_bottom] = mcolors.to_rgba(_OVERHANG_OPEN_COLOR, 0.35)
        rgba[closed]      = mcolors.to_rgba(_OVERHANG_CLOSED_COLOR, 0.5)
        parts = []
        if closed.any():
            parts.append("magenta enclosed")
        if open_bottom.any():
            parts.append("amber open below")
        if reach is not None:
            reached = reach & (closed | open_bottom)
            rgba[reached] = mcolors.to_rgba(_OVERHANG_REACHED_COLOR, 0.55)
            if reached.any():
                parts.append("blue reached")
        self.ax.imshow(rgba, interpolation="none", aspect="equal")
        return " · overhangs: " + " / ".join(parts)

    def _pad_image(self, img):
        """Return *img* padded with the stock margin, or *img* unchanged when margin is 0."""
        H, W = img.shape
        pad_y, pad_x = self._margin_pad_px(H, W)
        if pad_y == 0 and pad_x == 0:
            return img
        try:
            depth_pct = float(self.margin_depth_pct.get()) / 100.0
            if self.surface.get() == "white":
                pad_val = np.float32(1.0 - depth_pct)  # white = z=0; deeper cut → darker
            else:
                pad_val = np.float32(depth_pct)         # black = z=0; deeper cut → brighter
        except (ValueError, ZeroDivisionError):
            return img
        return np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)),
                      mode="constant", constant_values=pad_val)

    def _finish_depth_map(self, composite_raw, n_grey_levels):
        """Shared tail: pad for stock margin and flip to cut-depth polarity.

        Returns ``(depth_map, n_grey_levels)``.
        depth_map: 2-D float32 [0, 1]  — 1.0 = maximum cut depth (intaglio).
        """
        img = self._pad_image(np.clip(composite_raw, 0.0, 1.0))
        if self.surface.get() == "white":
            return 1.0 - img, n_grey_levels   # white = z=0, black = deepest cut
        else:
            return img, n_grey_levels          # black = z=0, white = deepest cut

    def _get_depth_map(self):
        """Depth map for the live preview (self.current_image, already
        gated by whichever step is currently selected -- see
        _recompute_composite)."""
        return self._finish_depth_map(self.current_image, self.n_grey_levels)

    def _get_depth_map_for_step(self, step):
        """Depth map for *step* specifically, built from its own stored
        source_visibility rather than whichever step tab happens to be
        selected right now -- so computing (including Compute All Steps)
        always uses that step's own carving layout regardless of the
        currently-displayed tab. Call from the main thread (like
        _gather_step_params) -- it reads Tkinter variables."""
        composite_raw, n_grey_levels = self._composite_with_visibility(
            step.get('source_visibility'))
        if composite_raw is None:
            W, H = self._canvas_shape_px()
            composite_raw = np.zeros((H, W), dtype=np.float32)
        return self._finish_depth_map(composite_raw, n_grey_levels)

    def _effective_visibility(self, step):
        """Which sources *step* sees, one bool per source -- its own
        source_visibility, falling back to each source's own .visible where
        that list is missing or short, as layers.composite does."""
        vis = step.get('source_visibility')
        return [bool(vis[i]) if vis is not None and i < len(vis) else bool(layer.visible)
                for i, layer in enumerate(self.layers)]

    def _change_reference_index(self, idx):
        """Index of the step "Depth map change" compares step *idx* against:
        the most recent earlier step whose source visibility differs from
        step *idx*'s (and so whose depth map may differ), or None if every
        earlier step sees the same sources."""
        if not 0 <= idx < len(self.steps):
            return None
        own = self._effective_visibility(self.steps[idx])
        for i in range(idx - 1, -1, -1):
            if self._effective_visibility(self.steps[i]) != own:
                return i
        return None

    def _gating_mode_values(self, step):
        """Process Region choices for *step*: GATING_MODES, with "Depth map
        change" only where an earlier step sees different sources (see
        _change_reference_index) -- or where the step already uses it, so
        the setting is never changed behind the user's back."""
        var = step.get('gating_mode_var')
        idx = next((i for i, s in enumerate(self.steps) if s is step), None)
        if ((var is not None and var.get() == GATING_DEPTH_MAP_CHANGE)
                or (idx is not None and self._change_reference_index(idx) is not None)):
            return list(GATING_MODES)
        return [m for m in GATING_MODES if m != GATING_DEPTH_MAP_CHANGE]

    def _sync_auto_target_depth(self, step):
        """If *step* is a Pocket-family step whose Target depth Auto
        checkbox is checked, refresh the displayed value from that step's
        own current depth map (its deepest point, in mm) -- a live-updated
        substitute for the old "0 = auto, use Max cut depth" sentinel.
        Cosmetic only: the authoritative value used for computing is
        resolved fresh in _gather_step_params, so a stale display here
        (e.g. from an edit made on a different step's tab) can't produce a
        wrong toolpath."""
        if step is None:
            return
        threshold = step.get('threshold')
        if threshold is None or threshold.auto_var is None or not threshold.auto_var.get():
            return
        if ALGORITHM_FAMILY.get(step['algorithm_var'].get()) != "Pocket":
            return
        try:
            max_depth = float(self.max_depth.get())
        except ValueError:
            return
        depth_map, _ = self._get_depth_map_for_step(step)
        threshold.set(f"{float(np.max(depth_map)) * max_depth:.3f}")

    def _stock_size_mm(self):
        """(width_mm, height_mm) of the stock, margin included, as the
        toolpaths measure it. The margin is padded in whole pixels
        (_margin_pad_px), so the stock is the canvas plus 2 * pad pixels at
        the canvas's own px per mm -- each side within half a pixel of Stock
        margin -- and the padded depth map keeps exactly the canvas's pixel
        size. Raises ValueError if Width/Height don't parse."""
        width_mm  = float(self.width_mm.get())
        height_mm = float(self.height_mm.get())
        W, H = self._canvas_shape_px()
        pad_y, pad_x = self._margin_pad_px(H, W)
        return width_mm * (W + 2 * pad_x) / W, height_mm * (H + 2 * pad_y) / H

    def _reghole_geometry(self, step):
        """Resolve a Registration Holes step's hole layout from its current
        fields as a dict: stock size W_mm/H_mm (margin included), origin,
        pattern, diag_dir, inset_mm, spacing_mm. Auto inset is half the
        stock margin -- the middle of the margin strip, clear of the artwork
        -- or 10 mm with no margin; auto spacing is
        registration_auto_spacing. Raises ValueError on unparsable input."""
        margin_mm = float(self.margin_mm.get())
        W_mm, H_mm = self._stock_size_mm()
        if step['reghole_inset'].auto_var.get():
            inset_mm = margin_mm / 2.0 if margin_mm > 0.0 else 10.0
        else:
            inset_mm = float(step['reghole_inset'].get())
        if step['reghole_spacing'].auto_var.get():
            spacing_mm = registration_auto_spacing(W_mm, H_mm, inset_mm)
        else:
            spacing_mm = float(step['reghole_spacing'].get())
        return {'W_mm': W_mm, 'H_mm': H_mm, 'origin': self.origin_var.get(),
                'pattern': step['reghole_pattern_var'].get(),
                'diag_dir': step['reghole_diag_dir_var'].get(),
                'inset_mm': inset_mm, 'spacing_mm': spacing_mm}

    def _sync_reghole_fields(self, step):
        """Show a Registration Holes step's resolved Auto inset/spacing in
        their (disabled) fields, and refresh its back-side hint. Cosmetic
        only: _gather_step_params resolves the values afresh."""
        if ALGORITHM_FAMILY.get(step['algorithm_var'].get()) != "Registration Holes":
            return
        try:
            geo = self._reghole_geometry(step)
        except ValueError:
            return
        if step['reghole_inset'].auto_var.get():
            step['reghole_inset'].set(f"{geo['inset_mm']:.3f}")
        if step['reghole_spacing'].auto_var.get():
            step['reghole_spacing'].set(f"{geo['spacing_mm']:.3f}")

        if geo['pattern'] == "Diagonal":
            _, (dx, dy) = registration_flip_axis(
                "Diagonal", geo['W_mm'], geo['H_mm'], geo['origin'], geo['diag_dir'])
            line = "+45°" if dx * dy > 0 else "−45°"
            swap = (" and swapped canvas Width/Height"
                    if abs(geo['W_mm'] - geo['H_mm']) > 1e-9 else "")
            hint = (f"Flip the stock over the {line} line through the G-code "
                    f"origin. Back side: same G-code origin{swap}; don't "
                    f"re-zero X/Y. Mirror: Diagonal ({line}).")
        else:
            which = "vertical" if geo['pattern'] == "Vertical axis" else "horizontal"
            hint = (f"Flip the stock over its {which} centre line. Back side: "
                    f"same stock size and G-code origin; don't re-zero X/Y. "
                    f"Mirror: {geo['pattern']}.")
        step['reghole_hint_var'].set(hint)

    def _is_3d_mode(self):
        return self.display_mode_var.get() in DISPLAY_3D_MODES

    def _accessible_for_step(self, idx):
        """Return the achievable depth surface (morphological opening) for step *idx*, or None.

        If step *idx* has its own accessible array, returns that. Otherwise
        (currently only a V-carve step with an empty skeleton/no moves generated)
        falls back to the prior step's opening.
        """
        if 0 <= idx < len(self.toolpaths):
            tp = self.toolpaths[idx]
            if tp is not None and tp.accessible is not None:
                return tp.accessible
        # Fall back to the prior step's accessible
        prior = idx - 1
        if 0 <= prior < len(self.toolpaths):
            tp = self.toolpaths[prior]
            if tp is not None and tp.accessible is not None:
                return tp.accessible
        return None

    def _toolpath_progress_segments(self, tp):
        """Return tp.segments truncated to the leading x% of path points.

        x is the position of the toolpath-progress scale below the canvas, letting
        the user scrub through how far the tool has travelled along the path.
        """
        segments = tp.segments
        pct = max(0.0, min(1.0, self.toolpath_scale_var.get() / 100.0))
        if pct >= 1.0:
            return segments
        total_points = sum(len(pts) for pts in segments)
        if total_points == 0:
            return segments
        n_keep   = int(round(total_points * pct))
        truncated = []
        count = 0
        for pts in segments:
            if count >= n_keep:
                break
            remaining = n_keep - count
            if len(pts) <= remaining:
                truncated.append(pts)
                count += len(pts)
            else:
                truncated.append(pts[:remaining])
                count = n_keep
                break
        return truncated

    # ------------------------------------------------------------------ #
    #  3D rendering
    # ------------------------------------------------------------------ #
    def _render_3d(self, toolpath, title, step):
        """Render the depth-map surface and G-code tool path in 3D."""
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

        z_safe  = self._settings['z_safe']
        rapids   = []
        plunges  = []
        ramps    = []
        backoffs = []
        arcs     = []
        cuts     = []
        buckets = {
            'rapid_retract':  rapids,
            'rapid_traverse': rapids,
            'plunge':         plunges,
            'ramp':           ramps,
            'backoff':        backoffs,
            'arc':            arcs,
            'cut':            cuts,
        }
        for kind, pts in toolpath.iter_moves(
                segments=self._toolpath_progress_segments(toolpath),
                z_safe=z_safe, ramp=self._ramp_settings(step),
                backoff=self._backoff_settings(step), arc=self._arc_settings(step)):
            if len(pts) < 2:
                continue
            buckets.get(kind, cuts).append(pts)

        # ── Rebuild figure with a 3D axes ─────────────────────────────
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Disable matplotlib's built-in left-button rotation so our custom
        # handler (which keeps self._3d_elev/azim in sync) is the only one.
        try:
            self.ax.mouse_init(rotate_btn=[], zoom_btn=[])
        except Exception:
            pass
        self.fig.patch.set_facecolor("#1e1e1e")
        self.ax.set_facecolor("#1e1e1e")

        try:
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
            max_depth = float(self.max_depth.get())
        except ValueError:
            width_mm = height_mm = max_depth = 1.0

        # Apply 3D pan offset
        ox, oy, oz = self._3d_pan_offset

        # ── Plot tool path segments ────────────────────────────────────
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        def add_segments(segs, color, lw, alpha, label):
            if not segs:
                return
            lines = [np.array([(x, y, z) for x, y, z in s])
                     for s in segs]
            col = Line3DCollection(lines, colors=color, linewidths=lw,
                                   alpha=alpha, label=label)
            self.ax.add_collection3d(col, autolim=False)

        if self.show_rapids_var.get():
            add_segments(rapids,   "#ff0000",   1.0, 1.0, 'Rapids')
            add_segments(plunges,  "#3399ff",   1.0, 1.0, 'Plunges')
            add_segments(ramps,    "#33cfff",   1.0, 1.0, 'Ramp-in')
            add_segments(backoffs, "#ffaa33",   1.5, 1.0, 'Back-off')
            add_segments(arcs,     "#cc66ff",   1.0, 1.0, 'Arc corner')
        if self.show_cut_moves_var.get():
            add_segments(cuts,   "#61b970ff", 1.0, 1.0, 'Cut moves')

        # ── Axis labels & style ────────────────────────────────────────
        self.ax.set_xlabel('X (mm)', color='#aaaaaa', labelpad=6)
        self.ax.set_ylabel('Y (mm)', color='#aaaaaa', labelpad=6)
        self.ax.set_zlabel('Z (mm)', color='#aaaaaa', labelpad=6)
        self.ax.tick_params(colors='#888888', labelsize=7)
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane,
                     self.ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#444444')
        self.ax.grid(False)
        self.ax.set_title(title, color='white', fontsize=10, pad=10)

        try:
            margin_mm = float(self.margin_mm.get())
        except ValueError:
            margin_mm = 0.0
        W_mm = width_mm  + 2.0 * margin_mm
        H_mm = height_mm + 2.0 * margin_mm
        try:
            origin = self.origin_var.get()
        except Exception:
            origin = "Bottom-Left"
        _ox = {"Bottom-Left": 0,      "Bottom-Right": W_mm,
               "Top-Left":    0,      "Top-Right":    W_mm,
               "Center":      W_mm/2}.get(origin, 0)
        _oy = {"Bottom-Left": 0,      "Bottom-Right": 0,
               "Top-Left":    H_mm,   "Top-Right":    H_mm,
               "Center":      H_mm/2}.get(origin, 0)
        cx   = W_mm / 2 - _ox
        cy   = H_mm / 2 - _oy
        cz   = -max_depth / 2
        half = max(W_mm, H_mm, max_depth) / 2 * self._3d_zoom
        self.ax.set_xlim(cx - half + ox, cx + half + ox)
        self.ax.set_ylim(cy - half + oy, cy + half + oy)
        self.ax.set_zlim(cz - half + oz, cz + half + oz)

        self.ax.view_init(elev=self._3d_elev, azim=self._3d_azim)

        # Legend — only if there's at least one labeled artist to show
        if self.show_rapids_var.get() or self.show_cut_moves_var.get():
            self.ax.legend(loc='upper left', fontsize=7,
                           facecolor='#333333', edgecolor='#555555',
                           labelcolor='white')

        self.fig.tight_layout()
        self.mpl_canvas.draw()

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #
    def _render(self):
        if self.current_image is None:
            if getattr(self.ax, 'name', '') == '3d':
                self.fig.clear()
                self.ax = self.fig.add_subplot(111)
                self.fig.patch.set_facecolor("#2b2b2b")
                self.ax.set_facecolor("#2b2b2b")
            self.ax.clear()
            self.ax.set_visible(False)
            self.mpl_canvas.draw()
            return

        mode = self.display_mode_var.get()
        step_idx = self.display_step_var.get() - 1

        # ── 3D mode ───────────────────────────────────────────────────
        if mode == DISPLAY_STEP_3D:
            tp = (self.toolpaths[step_idx]
                  if 0 <= step_idx < len(self.toolpaths) else None)
            if tp:
                self._render_3d(tp, f"Step {step_idx + 1} Tool Path",
                                self.steps[step_idx])
                return

        # ── 2D modes — ensure we have a flat 2D axes ──────────────────
        if getattr(self.ax, 'name', '') == '3d':
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.fig.patch.set_facecolor("#2b2b2b")
            self.ax.set_facecolor("#2b2b2b")

        vmin, vmax = 0.0, 1.0
        img_alpha = None   # per-pixel matplotlib alpha array; set only when
                           # the displayed ToolPath flags physically-removed
                           # (cut-through) pixels -- see DISPLAY_STEP_RESULT

        if mode == DISPLAY_STEP_RES:
            tp = (self.toolpaths[step_idx]
                  if 0 <= step_idx < len(self.toolpaths) else None)
            if tp is not None and tp.result is not None:
                depth_map, _ = self._get_depth_map()
                # Unclipped, like the other diff views (DISPLAY_DIFF_RES_UNCUT
                # etc.) -- clipping negative values to 0 would make an
                # over-cut (result deeper than depth_map calls for) render
                # identically to "exactly finished," silently hiding it.
                # Positive (still needs cutting) renders on the same warm
                # ramp as before; negative (over-cut) renders on the cold
                # half of _DIFF_CMAP instead of disappearing.
                img   = (depth_map - tp.result).astype(np.float32)
                cmap  = _DIFF_CMAP
                vmin, vmax = -1.0, 1.0
                title = f"Step {step_idx + 1} Residual"
            else:
                mode  = DISPLAY_ORIGINAL   # fall through to original

        elif mode == DISPLAY_STEP_RESULT:
            tp = (self.toolpaths[step_idx]
                  if 0 <= step_idx < len(self.toolpaths) else None)
            if tp is not None and tp.result is not None:
                img   = tp.result
                # Invert so that deeper cuts appear darker when z=0 is white
                cmap  = "gray_r" if self.surface.get() == "white" else "gray"
                title = f"Step {step_idx + 1} Result"
                # A cutout kerf clips to the same 1.0 as ordinary full-depth
                # relief carving in `result`, so the two are indistinguishable
                # through this grayscale ramp alone. `alpha` flags pixels that
                # are physically cut through (a real hole) -- render those
                # fully transparent so the axes' hole-indicator facecolor
                # (set below, at the imshow call) shows through instead.
                if tp.alpha is not None:
                    img_alpha = tp.alpha
            else:
                mode  = DISPLAY_ORIGINAL   # fall through to original

        elif mode == DISPLAY_STEP_ACCESSIBLE:
            accessible = self._accessible_for_step(step_idx)
            if accessible is not None:
                img   = accessible
                # Same polarity handling as Step Result, so the two compare directly
                cmap  = "gray_r" if self.surface.get() == "white" else "gray"
                title = f"Step {step_idx + 1} Accessible"
            else:
                mode  = DISPLAY_ORIGINAL   # fall through to original

        elif mode in (DISPLAY_STEP_UNACCESSIBLE, DISPLAY_PROC_MASK):
            depth_map, _ = self._get_depth_map()
            accessible    = self._accessible_for_step(step_idx)
            unaccessible  = (np.clip(depth_map - accessible, 0.0, 1.0)
                             if accessible is not None else depth_map.copy())
            if mode == DISPLAY_STEP_UNACCESSIBLE:
                img   = unaccessible
                cmap  = "hot"
                title = f"Step Unaccessible (Step {step_idx + 1})"
            else:
                step = self.steps[step_idx] if 0 <= step_idx < len(self.steps) else None
                try:
                    params = self._gather_step_params(step) if step is not None else None
                except (ValueError, KeyError, ZeroDivisionError, AttributeError):
                    params = None
                algorithm    = params['algorithm']    if params is not None else "Raster Scan"
                threshold_mm = params['threshold_mm'] if params is not None else 0.0
                max_depth    = params['max_depth']    if params is not None else 5.0

                # Cutout and Pocket (and V-Carving Clean-up, which shares
                # Pocket's pipeline) all cut against the full depth map
                # regardless of any prior step -- none has a Process Region
                # selector (see "Step / Operation / Cutout"/"Step /
                # Operation / Pocket" in README.md), and their own
                # compute_*_toolpath functions never read gating_mode at
                # all. Mirror that here instead of reading whatever stale
                # mode the widget still holds from a different algorithm.
                gating_mode = "Full depth map"
                if (ALGORITHM_FAMILY[algorithm] not in ("Cutout", "Pocket",
                                                        "Registration Holes")
                        and 0 <= step_idx < len(self.steps)
                        and self.steps[step_idx].get('gating_mode_var') is not None):
                    gating_mode = self.steps[step_idx]['gating_mode_var'].get()

                prior_idx = step_idx - 1
                if gating_mode == GATING_DEPTH_MAP_CHANGE:
                    ref_idx = self._change_reference_index(step_idx)
                    if ref_idx is not None:
                        prior_depth_map, _ = self._get_depth_map_for_step(
                            self.steps[ref_idx])
                        ref = np.clip(depth_map_change(depth_map, prior_depth_map), 0.0, 1.0)
                    else:
                        ref = depth_map.copy()
                elif gating_mode == "Inaccessible by prior tool":
                    ref = unaccessible  # depth_map − accessible, computed above
                elif gating_mode == "Residual from prior step":
                    prior_result = (
                        self.toolpaths[prior_idx].result
                        if (0 <= prior_idx < len(self.toolpaths)
                            and self.toolpaths[prior_idx] is not None
                            and self.toolpaths[prior_idx].result is not None)
                        else None
                    )
                    ref = (np.clip(depth_map - prior_result, 0.0, 1.0)
                           if prior_result is not None else depth_map.copy())
                elif gating_mode == "Reachable residual":
                    prior_result = (
                        self.toolpaths[prior_idx].result
                        if (0 <= prior_idx < len(self.toolpaths)
                            and self.toolpaths[prior_idx] is not None
                            and self.toolpaths[prior_idx].result is not None)
                        else None
                    )
                    ref = (np.clip(accessible - prior_result, 0.0, 1.0)
                           if (prior_result is not None and accessible is not None)
                           else depth_map.copy())
                elif gating_mode == "Reachable inaccessible":
                    prior_acc = (
                        self.toolpaths[prior_idx].accessible
                        if (0 <= prior_idx < len(self.toolpaths)
                            and self.toolpaths[prior_idx] is not None
                            and self.toolpaths[prior_idx].accessible is not None)
                        else None
                    )
                    ref = (np.clip(accessible - prior_acc, 0.0, 1.0)
                           if (prior_acc is not None and accessible is not None)
                           else depth_map.copy())
                else:  # "Full depth map"
                    ref = depth_map.copy()

                try:
                    H, W = depth_map.shape
                    eff_width_mm  = (params['eff_width_mm']  if params is not None
                                     else float(self.width_mm.get()))
                    eff_height_mm = (params['eff_height_mm'] if params is not None
                                     else float(self.height_mm.get()))
                    px_per_mm = (W / eff_width_mm + H / eff_height_mm) / 2.0
                except (ValueError, ZeroDivisionError):
                    px_per_mm = 1.0

                # Each algorithm's real processing mask is now built by the
                # exact same function its own compute_*_toolpath calls (see
                # algorithms.py), instead of re-deriving each one's "0 (or
                # unset) means auto" threshold convention a second time here
                # -- two independent implementations of the same rule is
                # exactly how this class of preview/reality mismatch
                # happens (this is what motivated this whole rewrite).
                if params is None:
                    mask = np.zeros_like(depth_map, dtype=bool)
                elif algorithm == "Cutout":
                    r_px = max(1, int(round(
                        (params['tool'].diameter_mm / 2.0 + params['cutout_extra_offset_mm'])
                        * px_per_mm)))
                    mask, _, _ = compute_cutout_silhouette_mask(
                        depth_map, threshold_mm, max_depth, r_px)
                    if mask is None:
                        mask = np.zeros_like(depth_map, dtype=bool)
                elif ALGORITHM_FAMILY[algorithm] == "Registration Holes":
                    # The finished holes, placed exactly as
                    # compute_registration_holes_toolpath places them.
                    H, W = depth_map.shape
                    pxx = W / params['eff_width_mm']
                    pxy = H / params['eff_height_mm']
                    centers = registration_hole_centers(
                        params['reghole_pattern'], (W - 1) / pxx, (H - 1) / pxy,
                        params['origin'], params['reghole_diag_dir'],
                        params['reghole_inset_mm'], params['reghole_spacing_mm'])
                    rr, cc = np.mgrid[0:H, 0:W]
                    r_mm = params['reghole_diameter_mm'] / 2.0
                    mask = np.zeros_like(depth_map, dtype=bool)
                    for hx, hy in centers:
                        mask |= ((cc / pxx - hx) ** 2 + ((H - 1 - rr) / pxy - hy) ** 2
                                 <= r_mm ** 2)
                elif ALGORITHM_FAMILY[algorithm] == "Pocket":   # Pocket, V-Carving Clean-up
                    cache_key = (
                        params['tool'].profile, params['tool'].diameter_mm,
                        params['tool'].vbit_angle_deg, max_depth,
                        params['eff_width_mm'], params['eff_height_mm'],
                        params['source_visibility'])
                    reference_surface = None
                    if algorithm == "V-Carving Clean-up":
                        # Mirror _compute_step_work's own derivation so the
                        # preview matches what Compute Step will actually
                        # produce -- see compute_vcarve_synth_achievable_depth.
                        cref = params['cleanup_reference']
                        reference_surface = compute_vcarve_synth_achievable_depth(
                            cref['depth_map'], cref['tool'], max_depth,
                            params['eff_width_mm'], params['eff_height_mm'],
                            extra_offset_mm=cref['extra_offset_mm'],
                            erosion_cache=self._erosion_cache,
                            erosion_cache_key=cref['erosion_cache_key'])
                        cache_key = cache_key + ('clean_up',)
                    mask, _, _, _, _, _ = compute_pocket_mask(
                        depth_map, params['tool'], px_per_mm, max_depth, threshold_mm,
                        self.n_grey_levels,
                        boundary_offset_mm=params['pocket_boundary_offset_mm'],
                        reference_surface=reference_surface,
                        erosion_cache=self._erosion_cache, erosion_cache_key=cache_key)
                elif ALGORITHM_FAMILY[algorithm] == "V-Carving":
                    threshold_norm = resolve_vcarve_threshold_norm(ref, threshold_mm, max_depth)
                    _, mask = compute_vcarve_mask(
                        ref, threshold_norm, params['min_feature_width_mm'],
                        px_per_mm, params['max_feature_width_mm'])
                else:  # Raster Scan
                    work_surface_tol = threshold_mm / max_depth if max_depth else 0.0
                    _, footprint = _build_kernel(params['tool'], px_per_mm, max_depth)
                    prior_mask = compute_raster_gate_mask(
                        ref, work_surface_tol, None, False, None, footprint)
                    mask = (prior_mask.astype(bool) if prior_mask is not None
                            else np.ones_like(depth_map, dtype=bool))

                img  = mask.astype(np.float32)
                cmap = "Greens"
                title = f"Processing Mask (Step {step_idx + 1})"

        elif mode == DISPLAY_DIFF_RES_UNCUT:
            prev_idx   = step_idx - 1
            accessible = self._accessible_for_step(step_idx)
            prev_tp    = (self.toolpaths[prev_idx]
                          if 0 <= prev_idx < len(self.toolpaths) else None)
            if (accessible is not None and prev_tp is not None
                    and prev_tp.result is not None):
                # (depth_map − prev_tp.result) − (depth_map − accessible) = accessible − prev_tp.result
                # positive → current tool can access material the prior step left behind
                # negative → current tool also cannot reach that region
                img   = (accessible - prev_tp.result).astype(np.float32)
                cmap  = _DIFF_CMAP
                vmin, vmax = -1.0, 1.0
                title = f"Prev Residual − Unaccessible (Step {step_idx + 1})"
            else:
                mode = DISPLAY_ORIGINAL

        elif mode == DISPLAY_DIFF_UNCUT_UNCUT:
            prev_idx   = step_idx - 1
            accessible = self._accessible_for_step(step_idx)
            prev_tp    = (self.toolpaths[prev_idx]
                          if 0 <= prev_idx < len(self.toolpaths) else None)
            if (accessible is not None and prev_tp is not None
                    and prev_tp.accessible is not None):
                # accessible − prev_tp.accessible:
                # positive → current tool can access material the prior tool couldn't
                # negative → current tool is more restricted than the prior tool
                img   = (accessible - prev_tp.accessible).astype(np.float32)
                cmap  = _DIFF_CMAP
                vmin, vmax = -1.0, 1.0
                title = f"Prev Unaccessible − Unaccessible (Step {step_idx + 1})"
            else:
                mode = DISPLAY_ORIGINAL

        elif mode == DISPLAY_DIFF_DEPTH_MAP:
            ref_idx = self._change_reference_index(step_idx)
            if ref_idx is not None:
                # depth_map vs. the depth map of the most recent earlier step
                # that sees different sources (_change_reference_index),
                # each from its own source visibility, with
                # depth_map_change's sideways tolerance (its positive part is
                # what the "Depth map change" Process Region gates on):
                # positive → this step asks for material the reference step's depth map keeps
                # negative → this step's depth map is shallower than the reference step's
                depth_map, _ = self._get_depth_map()
                prior_depth_map, _ = self._get_depth_map_for_step(self.steps[ref_idx])
                img   = depth_map_change(depth_map, prior_depth_map)
                cmap  = _DIFF_CMAP
                vmin, vmax = -1.0, 1.0
                title = f"Depth Map Change (Step {step_idx + 1} vs. Step {ref_idx + 1})"
            else:
                mode = DISPLAY_ORIGINAL

        if mode not in (DISPLAY_STEP_RES, DISPLAY_STEP_RESULT, DISPLAY_STEP_ACCESSIBLE,
                        DISPLAY_STEP_UNACCESSIBLE, DISPLAY_PROC_MASK,
                        DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT,
                        DISPLAY_DIFF_DEPTH_MAP):
            img   = self._pad_image(self.current_image)
            cmap  = "gray"
            title = "Original"

        display_data = img
        if self.show_shading_var.get():
            display_data = self._apply_shading(img, mode, cmap, vmin, vmax, step_idx)

        saved_xlim = self._view_xlim
        saved_ylim = self._view_ylim

        self._displayed_array = img
        self.ax.cla()
        self.ax.set_visible(True)
        has_holes = img_alpha is not None and np.any(img_alpha <= 0.0)
        hole_color = _HOLE_FACECOLOR if has_holes else "#2b2b2b"
        self.ax.set_facecolor(hole_color)
        # ax.axis("off") below disables the Axes' own background patch, so
        # whatever's behind a transparent (alpha=0, "hole") pixel is really
        # the *figure's* patch, not the axes' facecolor set above -- without
        # this, a hole always showed the figure's plain dark background
        # instead of the intended highlight colour.
        self.fig.patch.set_facecolor(hole_color)
        self.ax.imshow(display_data, interpolation="none", aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, alpha=img_alpha)
        if mode in (DISPLAY_ORIGINAL, DISPLAY_STEP_RESULT):
            title += self._overlay_overhangs(img, mode, step_idx)
        self._overlay_toolpath_2d(img, step_idx)
        self._overlay_registration_holes(img)
        self.ax.axis("off")
        if title:
            self.ax.set_title(title, color="white", fontsize=10, pad=6)
        self.fig.tight_layout()

        if saved_xlim is not None and saved_ylim is not None:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)

        self.mpl_canvas.draw()

    def _overlay_px_transform(self, img):
        """(H, W, sx, sy, ox, oy) for drawing stock-mm geometry over the 2D
        view's *img* (which spans the stock, margin included): sx/sy are
        px per mm, ox/oy the G-code origin's offset from the bottom-left
        pixel's centre in mm, as the toolpath algorithms place it. None if
        the canvas fields don't parse."""
        try:
            width_mm, height_mm = self._stock_size_mm()
        except ValueError:
            return None
        if width_mm <= 0 or height_mm <= 0:
            return None
        H, W = img.shape[:2]
        sx, sy = W / width_mm, H / height_mm
        W_mm, H_mm = (W - 1) / sx, (H - 1) / sy
        origin = self.origin_var.get()
        ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
              "Top-Left":    0,       "Top-Right":    W_mm,
              "Center":      W_mm / 2}.get(origin, 0)
        oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
              "Top-Left":    H_mm,    "Top-Right":    H_mm,
              "Center":      H_mm / 2}.get(origin, 0)
        return H, W, sx, sy, ox, oy

    def _overlay_registration_holes(self, img):
        """Outline every Registration Holes step's holes and draw its flip
        axis on the current 2D axes, from the step's current fields -- so
        the placement is visible (and follows edits) before computing."""
        if not self.show_reg_holes_var.get():
            return
        transform = self._overlay_px_transform(img)
        if transform is None:
            return
        H, W, sx, sy, _, _ = transform
        # Same pixel-centre stock extent the algorithm places holes in (see
        # compute_registration_holes_toolpath), so outlines sit exactly on
        # the cut holes.
        W_mm, H_mm = (W - 1) / sx, (H - 1) / sy

        def to_px(x, y):
            return x * sx, (H - 1) - y * sy

        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D
        for step in self.steps:
            if ALGORITHM_FAMILY.get(step['algorithm_var'].get()) != "Registration Holes":
                continue
            try:
                geo = self._reghole_geometry(step)
                r_mm = float(step['reghole_diameter'].get()) / 2.0
            except ValueError:
                continue
            spacing = None if step['reghole_spacing'].auto_var.get() else geo['spacing_mm']
            centers = registration_hole_centers(geo['pattern'], W_mm, H_mm, geo['origin'],
                                                geo['diag_dir'], geo['inset_mm'], spacing)
            for hx, hy in centers:
                self.ax.add_patch(Circle(
                    to_px(hx, hy), radius=max(r_mm * (sx + sy) / 2.0, 1.5),
                    facecolor="none", edgecolor="#ffd54f", linewidth=1.2,
                    linestyle="--", zorder=4))

            # Clip the flip axis to the stock rectangle (Liang-Barsky).
            (px, py), (dx, dy) = registration_flip_axis(
                geo['pattern'], W_mm, H_mm, geo['origin'], geo['diag_dir'])
            t0, t1 = -np.inf, np.inf
            for p, d, hi in ((px, dx, W_mm), (py, dy, H_mm)):
                if abs(d) < 1e-12:
                    if p < -1e-9 or p > hi + 1e-9:
                        t0, t1 = 1.0, 0.0
                    continue
                a, b = (0.0 - p) / d, (hi - p) / d
                t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if t0 < t1:
                (x0, y0), (x1, y1) = (to_px(px + t0 * dx, py + t0 * dy),
                                      to_px(px + t1 * dx, py + t1 * dy))
                self.ax.add_line(Line2D([x0, x1], [y0, y1], color="#ffd54f",
                                        linewidth=0.9, linestyle="-.", alpha=0.7,
                                        zorder=4))

    def _overlay_toolpath_2d(self, img, step_idx):
        """Overlay cut moves, rapids, and tab markers on the current 2D axes."""
        show_cuts   = self.show_cut_moves_var.get()
        show_rapids = self.show_rapids_var.get()
        show_tabs   = self.show_tabs_var.get()
        if not show_cuts and not show_rapids and not show_tabs:
            return
        tp = (self.toolpaths[step_idx]
              if 0 <= step_idx < len(self.toolpaths) else None)
        if tp is None or not tp.segments:
            return
        transform = self._overlay_px_transform(img)
        if transform is None:
            return
        H, W, sx, sy, _ox, _oy = transform

        def to_px(x, y):
            # Inverse of: x_mm = col/sx - _ox,  y_mm = (H-1-row)/sy - _oy
            return (x + _ox) * sx, (H - 1) - (y + _oy) * sy

        from matplotlib.collections import LineCollection

        cut_lines   = []
        rapid_lines = []
        prev_end    = None

        for pts in self._toolpath_progress_segments(tp):
            if not pts:
                continue
            if prev_end is not None and (
                    abs(prev_end[0] - pts[0][0]) > 1e-4
                    or abs(prev_end[1] - pts[0][1]) > 1e-4):
                rapid_lines.append([to_px(*prev_end[:2]), to_px(*pts[0][:2])])
            if len(pts) >= 2:
                cut_lines.append([to_px(x, y) for x, y, _ in pts])
            prev_end = pts[-1]

        if show_cuts and cut_lines:
            self.ax.add_collection(LineCollection(
                cut_lines, colors="#61b970", linewidths=0.8, alpha=0.85,
                zorder=2))
        if show_rapids and rapid_lines:
            self.ax.add_collection(LineCollection(
                rapid_lines, colors="#ff4444", linewidths=0.8, alpha=0.85,
                zorder=2))

        # Tab markers: a Z-only change is otherwise invisible in this
        # top-down view, so mark each tab's position/extent with a filled
        # circle (diameter ~ the tab's actual width along the path) on top
        # of the cut-line overlay.
        if show_tabs:
            tab_regions = (tp.metadata or {}).get('tab_regions')
            if tab_regions:
                from matplotlib.patches import Circle
                px_scale = (sx + sy) / 2.0
                for (cx, cy), (bx, by), (ex, ey) in tab_regions:
                    radius_mm = np.hypot(bx - cx, by - cy)
                    if radius_mm < 1e-6:
                        radius_mm = np.hypot(ex - cx, ey - cy)
                    cpx, cpy = to_px(cx, cy)
                    self.ax.add_patch(Circle(
                        (cpx, cpy), radius=max(radius_mm * px_scale, 1.5),
                        facecolor="#ff8c00", edgecolor="none", alpha=0.55,
                        zorder=3))

    # ------------------------------------------------------------------ #
    #  Zoom & Pan — 2D and 3D
    # ------------------------------------------------------------------ #
    ZOOM_STEP    = 1.15
    ROT_SENS     = 0.4   # degrees per pixel for 3D rotation
    PAN_SENS_3D  = 0.05  # mm per pixel for 3D pan

    def _on_scroll(self, event):
        if self.current_image is None:
            return
        if self._is_3d_mode():
            if event.button == "up":
                self._3d_zoom /= self.ZOOM_STEP
            else:
                self._3d_zoom *= self.ZOOM_STEP
            self._3d_zoom = max(0.01, min(20.0, self._3d_zoom))
            self._render()
        else:
            if event.inaxes is not self.ax:
                return
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            xdata, ydata = event.xdata, event.ydata
            if xdata is None or ydata is None:
                return
            scale = (1.0 / self.ZOOM_STEP if event.button == "up"
                     else self.ZOOM_STEP)
            new_half_x = (cur_xlim[1] - cur_xlim[0]) * scale / 2
            new_half_y = (cur_ylim[1] - cur_ylim[0]) * scale / 2
            rel_x = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
            rel_y = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
            new_xlim = (xdata - rel_x * new_half_x * 2,
                        xdata + (1 - rel_x) * new_half_x * 2)
            new_ylim = (ydata - rel_y * new_half_y * 2,
                        ydata + (1 - rel_y) * new_half_y * 2)
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self._view_xlim = new_xlim
            self._view_ylim = new_ylim
            self.mpl_canvas.draw_idle()

    def _on_mouse_press(self, event):
        if self._is_3d_mode():
            if event.button == 1:   # left button: rotate (or reset on double-click)
                if event.dblclick:
                    self._reset_zoom_and_render()
                else:
                    self._3d_mouse_start = ('rot',
                                            event.x, event.y,
                                            self._3d_elev, self._3d_azim,
                                            self._3d_pan_offset.copy())
            elif event.button == 2:  # middle button: pan
                self._3d_mouse_start = ('pan',
                                        event.x, event.y,
                                        self._3d_elev, self._3d_azim,
                                        self._3d_pan_offset.copy())
        else:
            if event.button == 2 and event.inaxes is self.ax:
                bbox = self.ax.bbox
                self._pan_origin = (event.x, event.y,
                                    self.ax.get_xlim(), self.ax.get_ylim(),
                                    bbox.width, bbox.height)
            elif event.button == 1 and event.dblclick:
                self._reset_zoom_and_render()

    def _on_mouse_release(self, event):
        if self._is_3d_mode():
            if event.button in (1, 2):  # left (rotate) or middle (pan)
                self._3d_mouse_start = None
        else:
            if event.button == 2:
                self._pan_origin = None

    def _on_mouse_move(self, event):
        if self._is_3d_mode():
            if self._3d_mouse_start is None:
                return
            mode_3d = self._3d_mouse_start[0]
            _, x0, y0, elev0, azim0, pan0 = self._3d_mouse_start
            dx = event.x - x0
            dy = event.y - y0

            if mode_3d == 'rot':
                self._3d_azim = azim0 - dx * self.ROT_SENS
                self._3d_elev = max(-90, min(90,
                                   elev0 - dy * self.ROT_SENS))
            else:   # pan
                # Map pixel delta to world-space XY delta, corrected for the
                # current azimuth so panning always follows the visible axes.
                # Derivation: the world "right" direction for screen +X is
                # (-sin(a), cos(a), 0); rotating the naive (-dx, dy) vector
                # by (a + 90°) gives the azimuth-correct world offset.
                try:
                    w_mm = float(self.width_mm.get())
                    h_mm = float(self.height_mm.get())
                except ValueError:
                    w_mm = h_mm = 100.0
                canvas_w = self.mpl_canvas.get_tk_widget().winfo_width()
                scale = (max(w_mm, h_mm) * self._3d_zoom / canvas_w
                         if canvas_w > 0 else 1.0)
                a = np.radians(self._3d_azim)
                c, s = np.cos(a), np.sin(a)
                # Vertical screen drag must flip when elevation crosses zero
                # (camera's effective up-direction in world XY reverses).
                ev = 1.0 if self._3d_elev >= 0 else -1.0
                self._3d_pan_offset = pan0 + np.array([
                    ( dx * s + dy * c * ev) * scale,
                    (-dx * c + dy * s * ev) * scale,
                    0.0])

            self._render()
        else:
            if (event.inaxes is self.ax
                    and event.xdata is not None and event.ydata is not None
                    and self._displayed_array is not None):
                self._update_coord_display(event.xdata, event.ydata)
            else:
                self.coord_var.set("")

            if self._pan_origin is None or event.inaxes is not self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            ox, oy, xlim0, ylim0, bw, bh = self._pan_origin
            # Use pixel-space delta against frozen original limits to avoid
            # the feedback loop where draw_idle() changes event.xdata/ydata.
            dx = (event.x - ox) / bw * (xlim0[1] - xlim0[0])
            dy = (event.y - oy) / bh * (ylim0[1] - ylim0[0])
            new_xlim = (xlim0[0] - dx, xlim0[1] - dx)
            new_ylim = (ylim0[0] - dy, ylim0[1] - dy)
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self._view_xlim = new_xlim
            self._view_ylim = new_ylim
            self.mpl_canvas.draw_idle()

    def _on_axes_leave(self, event):
        self.coord_var.set("")

    def _update_coord_display(self, xdata, ydata):
        img = self._displayed_array
        if img is None:
            self.coord_var.set("")
            return

        H, W = img.shape
        col = int(np.clip(round(xdata), 0, W - 1))
        row = int(np.clip(round(ydata), 0, H - 1))

        try:
            max_depth = float(self.max_depth.get())
        except ValueError:
            self.coord_var.set("")
            return
        transform = self._overlay_px_transform(img)
        if transform is None:
            self.coord_var.set("")
            return
        _, _, px_per_mm_x, px_per_mm_y, _ox, _oy = transform

        x_mm = col / px_per_mm_x - _ox
        y_mm = (H - 1 - row) / px_per_mm_y - _oy

        mode = self.display_mode_var.get()
        if mode == DISPLAY_PROC_MASK:
            val = float(img[row, col])
            z_str = f"Z: {'1' if val > 0.5 else '0'}"
        elif mode in (DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT,
                      DISPLAY_DIFF_DEPTH_MAP):
            val = float(img[row, col])
            z_str = f"ΔZ: {val * max_depth:+.2f} mm"
        else:
            if mode == DISPLAY_ORIGINAL:
                depth_map, _ = self._get_depth_map()
                val = float(depth_map[row, col])
            else:
                val = float(img[row, col])
            sign = self._elevation_sign(mode)
            z_str = f"Z: {sign * val * max_depth:.2f} mm"

        self.coord_var.set(f"X: {x_mm:+.2f}  Y: {y_mm:+.2f}  {z_str}")

    def _reset_zoom(self):
        """Clear saved view limits (next render will auto-fit)."""
        self._view_xlim      = None
        self._view_ylim      = None
        self._pan_origin     = None
        self._3d_elev        = 30.0
        self._3d_azim        = -60.0
        self._3d_zoom        = 1.0
        self._3d_pan_offset  = np.array([0.0, 0.0, 0.0])
        self._3d_mouse_start = None

    def _reset_zoom_and_render(self):
        self._reset_zoom()
        self._render()
