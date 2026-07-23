import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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

from constants import (
    TOOL_PROFILES, RASTER_DIRECTIONS, CUT_DIRECTIONS, ALGORITHMS, DEFAULTS,
    DEFAULT_PREAMBLE, DEFAULT_POSTAMBLE, BLEND_MODES,
    DISPLAY_ORIGINAL, DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
    DISPLAY_STEP_3D, DISPLAY_STEP_UNACCESSIBLE, DISPLAY_PROC_MASK,
    DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT,
    DISPLAY_MODES, DISPLAY_3D_MODES, DISPLAY_STEP_MODES,
)
from widgets import (ProgressDialog, LabeledEntry,
                     SettingsDialog, DressUpDialog, ExportToolpathDialog,
                     ImportSTLDialog, _fmt_time)
from algorithms import (compute_raster_toolpath, compute_vcarve_toolpath,
                        compute_contour_toolpath, compute_vcarve_mask,
                        compute_cutout_toolpath, _compute_cutout_rings,
                        _compute_cutout_from_rings)
from tool import Tool
from dressup import RampSettings, BackoffSettings, ArcSettings, TabSettings
import layers as layers_mod
import stl_mesh



class GCodeGeneratorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("CNC Relief Carving — Parameter Setup")
        self.root.geometry("1240x780")
        self.root.minsize(940, 560)

        self._settings = {
            'z_safe':      float(DEFAULTS['z_safe']),
            'feed_rapid':  float(DEFAULTS['feed_rapid']),
            'accel_mm_s2': float(DEFAULTS['accel_mm_s2']),
            'preamble':    DEFAULTS['preamble'],
            'postamble':   DEFAULTS['postamble'],
        }

        self.current_image   = None   # 2-D float32 [0,1] grayscale array -- cached
                                       # composite of self.layers, see _recompute_composite()
        self.n_grey_levels   = 256
        self.layers = []   # list[layers_mod.Layer]; layers[0] = bottom/base source
        self._next_layer_id = 1   # monotonically increasing, for stable Layer ids
        self._last_precision_mm = float(DEFAULTS["precision_mm"])   # see _on_precision_changed
        self._last_width_mm  = float(DEFAULTS["width_mm"])          # see _on_canvas_size_changed
        self._last_height_mm = float(DEFAULTS["height_mm"])
        self.steps     = []   # list of step-widget dicts, built by _build_sidebar
        self.toolpaths = []   # list of ToolPath|None, parallel to self.steps
        self._erosion_cache = {}   # tool_signature -> (center_depth_contour, accessible)

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
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ------------------------------------------------------------------ #
    #  Menu
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        self.display_mode_var   = tk.StringVar(value=DISPLAY_ORIGINAL)
        self.display_step_var   = tk.IntVar(value=1)
        self.show_cut_moves_var = tk.BooleanVar(value=True)
        self.show_rapids_var    = tk.BooleanVar(value=True)
        self.show_tabs_var      = tk.BooleanVar(value=True)

        menubar   = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        # ── Project ─────────────────────────────────────────────────
        file_menu.add_command(label="New Project",   accelerator="Ctrl+N",
                              command=self._new_project)
        file_menu.add_command(label="Open Project…", accelerator="Ctrl+Shift+O",
                              command=self._load_project)
        file_menu.add_command(label="Save Project…", accelerator="Ctrl+S",
                              command=self._save_project)
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

        self._view_menu = tk.Menu(menubar, tearoff=0)

        _display_hotkeys = [
            (DISPLAY_ORIGINAL,          "1"),
            (DISPLAY_STEP_3D,           "2"),
            (DISPLAY_STEP_RESULT,       "3"),
            (DISPLAY_STEP_RES,          "4"),
            (DISPLAY_STEP_UNACCESSIBLE, "5"),
            (DISPLAY_DIFF_RES_UNCUT,     "6"),
            (DISPLAY_DIFF_UNCUT_UNCUT,  "7"),
            (DISPLAY_PROC_MASK,         "8"),
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
        self._view_menu.add_separator()
        self._view_menu.add_command(label="Reset Zoom  (double-click)",
                                    command=self._reset_zoom_and_render)

        menubar.add_cascade(label="View", menu=self._view_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-n>", lambda e: self._new_project())
        self.root.bind("<Control-o>", lambda e: self._open_source_file())
        self.root.bind("<Control-s>", lambda e: self._save_project())
        self.root.bind("<Control-O>", lambda e: self._load_project())
        self.root.bind("<Control-q>", lambda e: self._on_exit())
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
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

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

        self.precision_mm = LabeledEntry(
            tab, "Precision", "mm", DEFAULTS["precision_mm"],
            minval=0.01, maxval=50.0, increment=0.1)
        self.precision_mm.pack(fill=tk.X, pady=2)

        self.max_depth = LabeledEntry(tab, "Max cut depth", "mm", DEFAULTS["max_depth"],
                                      minval=0.01, maxval=200.0, increment=1.0)
        self.max_depth.pack(fill=tk.X, pady=2)
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
        self.precision_mm.committed_var.trace_add("write", self._on_precision_changed)
        self.max_depth.committed_var.trace_add("write", lambda *_: self._render())
        self.margin_mm.committed_var.trace_add("write", lambda *_: self._render())
        self.margin_depth_pct.committed_var.trace_add("write", self._on_depth_map_content_changed)
        self.background_depth_pct.committed_var.trace_add("write", self._on_depth_map_content_changed)
        self._on_aspect_lock_toggle()   # apply initial locked state

        polarity_row = ttk.Frame(tab)
        polarity_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(polarity_row, text="z=0 colour:").pack(side=tk.LEFT)
        self.surface = tk.StringVar(value=DEFAULTS["surface"])
        ttk.Combobox(polarity_row, textvariable=self.surface,
                     values=["white", "black"],
                     state="readonly", width=22).pack(side=tk.RIGHT)
        self.surface.trace_add("write", self._on_depth_map_content_changed)

        origin_row = ttk.Frame(tab)
        origin_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(origin_row, text="G-code origin:").pack(side=tk.LEFT)
        self.origin_var = tk.StringVar(value=DEFAULTS["origin"])
        ttk.Combobox(origin_row, textvariable=self.origin_var,
                     values=["Bottom-Left", "Bottom-Right",
                             "Top-Left", "Top-Right", "Center"],
                     state="readonly", width=22).pack(side=tk.RIGHT)

    def _canvas_shape_px(self):
        """Canvas pixel shape (W, H), derived from physical size and
        Precision -- the single source of truth for how big the depth map
        is, replacing the old 'whichever source was imported first' rule."""
        try:
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
        except ValueError:
            width_mm, height_mm = float(DEFAULTS["width_mm"]), float(DEFAULTS["height_mm"])
        try:
            precision_mm = float(self.precision_mm.get())
        except ValueError:
            precision_mm = float(DEFAULTS["precision_mm"])
        if precision_mm <= 1e-9:
            precision_mm = float(DEFAULTS["precision_mm"])
        W = max(2, round(width_mm / precision_mm))
        H = max(2, round(height_mm / precision_mm))
        return W, H

    def _fit_scale_pct(self, native_w_px, native_h_px):
        """Scale-% that makes a *native_w_px* x *native_h_px* image fit
        entirely within the current canvas (preserving the image's own
        aspect ratio -- 'contain', not stretch/crop). At scale_pct=100, one
        image pixel maps to exactly one canvas pixel (see
        layers._affine_params), so an image whose native resolution has no
        relation to the canvas's Precision-driven pixel density would
        otherwise appear cropped to a tiny corner or absurdly oversized;
        this makes a freshly added image appear at a sensible size by
        default regardless of its native resolution."""
        if native_w_px <= 0 or native_h_px <= 0:
            return 100.0
        canvas_w_px, canvas_h_px = self._canvas_shape_px()
        scale = min(canvas_w_px / native_w_px, canvas_h_px / native_h_px)
        return max(1.0, scale * 100.0)

    _MAX_CANVAS_PX = 16_000_000

    def _on_canvas_shape_changed(self, *_):
        """Shared tail once canvas pixel shape and/or every source's
        placement has already been reconciled with whatever just changed
        (Width/Height via _on_canvas_size_changed, Precision via
        _on_precision_changed) -- both already gated the change itself on
        _confirm_canvas_shape, so by the time we get here a huge canvas is
        something the user explicitly chose to proceed with. Note it in the
        status line, then treat this as a structural change like add/remove
        -- invalidate caches and recomposite every source onto the new grid."""
        W, H = self._canvas_shape_px()
        if W * H > self._MAX_CANVAS_PX:
            self.status_var.set(
                f"Canvas is {W}x{H} px (~{W * H / 1e6:.0f} MP) -- "
                "this may be slow.")
        self._on_layers_structure_changed()

    def _confirm_canvas_shape(self, width_mm, height_mm, precision_mm):
        """Ask before committing to a Width/Height/Precision combination
        that would make the canvas huge -- W*H drives every per-pixel pass
        (compositing, rasterizing, toolpath generation), so a multi-hundred-
        megapixel canvas doesn't error out, it just makes the app sit there
        looking hung for however long that pass takes. A status-line
        warning printed *after* the shape already changed doesn't help,
        since by the time it's on screen the expensive work triggered by
        that same change is already queued -- the gate has to sit in front
        of the change, not after it."""
        if precision_mm <= 1e-9:
            precision_mm = float(DEFAULTS["precision_mm"])
        W = max(2, round(width_mm / precision_mm))
        H = max(2, round(height_mm / precision_mm))
        if W * H <= self._MAX_CANVAS_PX:
            return True
        return messagebox.askyesno(
            "Large canvas",
            f"This would create a {W}x{H} px canvas (~{W * H / 1e6:.0f} MP).\n\n"
            "Compositing and generating toolpaths at this resolution can "
            "take a long time and may make the app unresponsive.\n\n"
            "Proceed anyway?")

    def _on_canvas_size_changed(self, *_):
        """Width/Height changed. The canvas's pixel *density* is set solely
        by Precision (_canvas_shape_px's W/width_mm ratio is always
        ~1/Precision, independent of width_mm) -- but its *extent* just
        changed, and every source's placement (Scale-%, Offset X/Y) is
        stored either in raw pixel-space terms or in absolute mm, neither
        of which moves with the canvas's own size. Left alone, resizing the
        board would silently shrink/grow every source's size and position
        RELATIVE TO THE CANVAS, discarding a layout the user already tuned
        by hand. Rescale every source's Offset X/Y and Scale-% by the ratio
        each canvas dimension changed by, so the whole composition simply
        scales together with the board -- the user's relative layout
        survives a Width/Height edit untouched; only the absolute physical
        size of the whole carving changes."""
        try:
            new_width_mm  = float(self.width_mm.get())
            new_height_mm = float(self.height_mm.get())
        except ValueError:
            return
        if new_width_mm > 1e-9 and new_height_mm > 1e-9:
            if not self._confirm_canvas_shape(new_width_mm, new_height_mm,
                                               float(self.precision_mm.get())):
                self.width_mm.set(str(self._last_width_mm))
                self.height_mm.set(str(self._last_height_mm))
                return
            old_width_mm, old_height_mm = self._last_width_mm, self._last_height_mm
            if (old_width_mm > 1e-9 and old_height_mm > 1e-9
                    and (new_width_mm != old_width_mm or new_height_mm != old_height_mm)):
                width_ratio  = new_width_mm / old_width_mm
                height_ratio = new_height_mm / old_height_mm
                # Scale-% is a single pixel-space factor shared by both axes
                # (layers._affine_params has one 's', not sx/sy), so when
                # width/height didn't change by the same ratio (aspect lock
                # off), the geometric mean is the closest a uniform factor
                # can get to matching both axes at once.
                scale_ratio = float(np.sqrt(width_ratio * height_ratio))
                for layer in self.layers:
                    layer.offset_x_mm *= width_ratio
                    layer.offset_y_mm *= height_ratio
                    layer.scale_pct = min(1000.0, max(1.0, layer.scale_pct * scale_ratio))
            self._last_width_mm, self._last_height_mm = new_width_mm, new_height_mm
        self._on_canvas_shape_changed()

    def _on_precision_changed(self, *_):
        """Precision changed -- unlike Width/Height, this DOES change canvas
        pixel density (1/Precision), and Scale-% is a pixel-space factor
        (layers._affine_params' scale matrix has no mm/px_per_mm term at
        all -- only the offset does), so a fixed Scale-% would otherwise
        render at a different physical size purely because Precision moved.
        Rescale every source's Scale-% by the same ratio the density
        changed by, so each source's physical footprint on the stock --
        the user's actual design decision -- survives untouched; only the
        internal detail/resolution changes."""
        try:
            new_precision_mm = float(self.precision_mm.get())
        except ValueError:
            return
        if new_precision_mm > 1e-9:
            if not self._confirm_canvas_shape(float(self.width_mm.get()),
                                               float(self.height_mm.get()),
                                               new_precision_mm):
                self.precision_mm.set(str(self._last_precision_mm))
                return
            old_precision_mm = self._last_precision_mm
            if old_precision_mm > 1e-9 and new_precision_mm != old_precision_mm:
                ratio = old_precision_mm / new_precision_mm
                for layer in self.layers:
                    layer.scale_pct = min(1000.0, max(1.0, layer.scale_pct * ratio))
            self._last_precision_mm = new_precision_mm
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

        blend_row = ttk.Frame(tab)
        blend_row.pack(fill=tk.X, pady=2)
        ttk.Label(blend_row, text="Blend mode:").pack(side=tk.LEFT)
        blend_var = tk.StringVar(value=layer.blend_mode)
        if is_base:
            # Forced to Replace at composite time regardless of what's
            # stored (see layers.composite) -- so it isn't user-editable.
            ttk.Label(blend_row, text="Replace (base source)",
                      foreground="#666666").pack(side=tk.RIGHT)
        else:
            ttk.Combobox(blend_row, textvariable=blend_var,
                        values=BLEND_MODES, state="readonly", width=14
                        ).pack(side=tk.RIGHT)

        opacity = LabeledEntry(tab, "Opacity", "%", str(layer.opacity_pct),
                               minval=0.0, maxval=100.0, increment=5.0)
        opacity.pack(fill=tk.X, pady=2)
        min_depth = LabeledEntry(tab, "Min depth", "%", str(layer.min_depth_pct),
                                 minval=0.0, maxval=100.0, increment=5.0)
        min_depth.pack(fill=tk.X, pady=2)
        max_depth = LabeledEntry(tab, "Max depth", "%", str(layer.max_depth_pct),
                                 minval=0.0, maxval=100.0, increment=5.0)
        max_depth.pack(fill=tk.X, pady=2)

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
                layer.min_depth_pct = float(min_depth.get())
                layer.max_depth_pct = float(max_depth.get())
            except ValueError:
                return   # mid-edit invalid text -- skip silently
            layer.blend_mode = blend_var.get()   # no-op when is_base
            self._on_layer_property_changed()

        for entry in (ox, oy, rot, scale, opacity, min_depth, max_depth):
            entry.committed_var.trace_add("write", _apply)
        blend_var.trace_add("write", _apply)

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
        ttk.Button(bf, text="Add Step After",
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
        dst['profile_var'].set(src['profile_var'].get())
        for key in ('diameter', 'stepover', 'raster_offset', 'stepdown', 'raster_final_stepdown',
                    'extra_offset',
                    'vbit_angle', 'vcarve_stepdown', 'vcarve_final_stepdown',
                    'cutout_depth', 'cutout_stepdown', 'cutout_final_stepdown', 'cutout_extra_offset',
                    'contour_z_increment', 'contour_lateral_tol', 'feed_rate', 'threshold',
                    'min_fw', 'max_fw'):
            dst[key].set(src[key].get())
            if dst[key].auto_var is not None and src[key].auto_var is not None:
                dst[key].auto_var.set(src[key].auto_var.get())
        dst['raster_dir_var'].set(src['raster_dir_var'].get())
        dst['contour_adaptive_var'].set(src['contour_adaptive_var'].get())
        dst['tsp_2opt_var'].set(src['tsp_2opt_var'].get())
        dst['gating_mode_var'].set(src['gating_mode_var'].get())
        for key in ('ramp_in_enabled', 'ramp_angle_deg', 'ramp_max_length_mm',
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
        step['backoff_enabled_var']     = tk.BooleanVar(
            value=bool(int(DEFAULTS['step_backoff_enabled'])))
        step['backoff_distance_mm_var'] = tk.DoubleVar(
            value=float(DEFAULTS['step_backoff_distance_mm']))
        step['arc_rapids_enabled_var']  = tk.BooleanVar(
            value=bool(int(DEFAULTS['step_arc_rapids_enabled'])))
        step['arc_radius_mm_var']       = tk.DoubleVar(
            value=float(DEFAULTS['step_arc_radius_mm']))
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
        ttk.Label(alg_row, text="Algorithm:").pack(side=tk.LEFT)
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
        ttk.Label(dir_row, text="Scan direction:").pack(side=tk.LEFT)
        dir_var = tk.StringVar(value=DEFAULTS["step_raster_dir"])
        step['raster_dir_var'] = dir_var
        ttk.Combobox(dir_row, textvariable=dir_var,
                     values=RASTER_DIRECTIONS, state="readonly", width=22
                     ).pack(side=tk.RIGHT)

        cut_dir_row = ttk.Frame(operation_lf)
        cut_dir_row.pack(fill=tk.X, pady=2, after=raster_frame)   # initially shown (default = Raster Scan)
        step['cut_dir_row'] = cut_dir_row
        ttk.Label(cut_dir_row, text="Cut direction:").pack(side=tk.LEFT)
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
        ttk.Checkbutton(contour_frame, text="Adaptive levels (CDF)",
                        variable=contour_adaptive_var).pack(anchor=tk.W, pady=2)
        step['contour_adaptive_var'] = contour_adaptive_var

        contour_lateral_tol = LabeledEntry(contour_frame, "Lateral tolerance", "mm",
                                           DEFAULTS["step_contour_lateral_tol"],
                                           minval=0.0, maxval=50.0, increment=0.05)
        contour_lateral_tol.pack(fill=tk.X, pady=2)
        step['contour_lateral_tol'] = contour_lateral_tol

        tsp_frame = ttk.Frame(operation_lf)
        step['tsp_frame'] = tsp_frame

        tsp_2opt_var = tk.BooleanVar(
            value=bool(int(DEFAULTS["step_tsp_2opt"])))
        tsp_2opt_cb = ttk.Checkbutton(tsp_frame, text="Optimize path order (2-opt)",
                                      variable=tsp_2opt_var)
        tsp_2opt_cb.pack(anchor=tk.W, pady=2)
        step['tsp_2opt_var'] = tsp_2opt_var
        step['tsp_2opt_cb'] = tsp_2opt_cb

        gating_frame = ttk.Frame(operation_lf)
        gating_frame.pack(fill=tk.X)
        step['gating_frame'] = gating_frame

        tsp_frame.pack(fill=tk.X)   # always last; packed after gating_frame

        region_row = ttk.Frame(gating_frame)
        region_row.pack(fill=tk.X, pady=(0, 4))
        step['gating_region_row'] = region_row
        ttk.Label(region_row, text="Process region:").pack(side=tk.LEFT)
        gating_var = tk.StringVar(value=DEFAULTS["step_gating_mode"])
        step['gating_mode_var'] = gating_var
        gating_cb = ttk.Combobox(region_row, textvariable=gating_var,
                                 values=["Full depth map",
                                         "Residual from prior step",
                                         "Inaccessible by prior tool",
                                         "Reachable residual",
                                         "Reachable inaccessible"],
                                 state="readonly", width=22)
        gating_cb.pack(side=tk.RIGHT)
        step['gating_cb'] = gating_cb

        threshold = LabeledEntry(gating_frame, "Z-threshold", "mm", DEFAULTS["step_threshold"],
                                 minval=0.0, maxval=200.0, increment=0.1)
        threshold.pack(fill=tk.X, pady=2)
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

        profile_row = ttk.Frame(tool_lf)
        profile_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(profile_row, text="Profile:").pack(side=tk.LEFT)
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

        # ── Wire up algorithm / profile toggle ────────────────────────
        _VCARVE_PROFILES  = ["V-Bit", "Ball"]
        _RASTER_PROFILES  = ["Ball", "End Mill", "V-Bit"]
        _CUTOUT_PROFILES  = ["End Mill"]

        def _refresh_layout(*_):
            alg     = algorithm_var.get()
            profile = step['profile_var'].get()

            if alg.startswith("V-Carving"):
                step['raster_frame'].pack_forget()
                step['contour_frame'].pack_forget()
                step['stepdown'].pack_forget()
                step['raster_final_stepdown'].pack_forget()
                step['cut_dir_row'].pack_forget()
                step['vbit_frame'].pack(fill=tk.X, after=step['diameter'])
                step['vcarve_frame'].pack(fill=tk.X, after=step['vbit_frame'])
                step['min_fw'].pack(fill=tk.X, pady=2)
                step['max_fw'].pack(fill=tk.X, pady=2)
                step['gating_frame'].pack(fill=tk.X)
                step['gating_region_row'].pack(fill=tk.X, pady=(0, 4), before=step['threshold'])
                step['tsp_frame'].pack(fill=tk.X, after=step['gating_frame'])
                step['profile_cb'].config(state="readonly", values=_VCARVE_PROFILES)
                if profile not in _VCARVE_PROFILES:
                    step['profile_var'].set("V-Bit")
                step['vbit_angle'].set_state(
                    "disabled" if profile == "Ball" else "normal")
                step['extra_offset'].pack(fill=tk.X, pady=2, after=step['vcarve_frame'])
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
                step['min_fw'].pack_forget()
                step['max_fw'].pack_forget()
                step['gating_frame'].pack(fill=tk.X)
                step['gating_region_row'].pack_forget()   # Cutout always reads
                                                           # prior_accessible/previous_result
                                                           # unconditionally, mirroring Contour --
                                                           # only the Z-threshold field is used.
                step['tsp_frame'].pack(fill=tk.X, after=step['gating_frame'])
                step['profile_cb'].config(state="readonly", values=_CUTOUT_PROFILES)
                if profile not in _CUTOUT_PROFILES:
                    step['profile_var'].set("End Mill")
                step['vbit_angle'].set_state("disabled")
                step['cutout_frame'].pack(fill=tk.X, after=step['diameter'])
                step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['alg_row'])
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
                step['raster_frame'].pack(fill=tk.X, after=step['alg_row'])
                step['cut_dir_row'].pack(fill=tk.X, pady=2, after=step['raster_frame'])
                step['vbit_angle'].set_state("normal")

            locked = not alg.startswith("V-Carving") and step['cut_dir_var'].get() != "Either"
            step['tsp_2opt_cb'].config(state="disabled" if locked else "normal")

            if hasattr(self, '_view_menu'):
                self._on_display_mode_changed()

        profile_var.trace_add("write", _refresh_layout)
        algorithm_var.trace_add("write", _refresh_layout)
        cut_dir_var.trace_add("write", _refresh_layout)

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
            if profile == "V-Bit":
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

        threshold.committed_var.trace_add("write", _on_mask_param_changed)
        min_fw.committed_var.trace_add("write", _on_mask_param_changed)
        max_fw.committed_var.trace_add("write", _on_mask_param_changed)
        gating_var.trace_add("write", _on_mask_param_changed)

    def _reset_step_widgets(self, step):
        """Reset a step's widgets to DEFAULTS."""
        step['algorithm_var'].set(DEFAULTS["step_algorithm"])
        step['profile_var'].set(DEFAULTS["step_profile"])
        step['profile_cb'].config(state="readonly")
        step['diameter'].set(DEFAULTS["step_diameter"])
        step['stepover'].set(DEFAULTS["step_stepover"])
        step['raster_offset'].set(DEFAULTS["step_raster_offset"])
        step['stepdown'].set(DEFAULTS["step_stepdown"])
        step['extra_offset'].set(DEFAULTS["step_extra_offset"])
        step['feed_rate'].set(DEFAULTS["step_feed_rate"])
        step['threshold'].set(DEFAULTS["step_threshold"])
        step['raster_dir_var'].set(DEFAULTS["step_raster_dir"])
        step['cut_dir_var'].set(DEFAULTS["step_cut_direction"])
        step['vbit_angle'].set(DEFAULTS["step_vbit_angle"])
        step['min_fw'].set(DEFAULTS["step_min_fw"])
        step['max_fw'].auto_var.set(False)
        step['max_fw'].set(DEFAULTS["step_max_fw"])
        step['vcarve_stepdown'].set(DEFAULTS["step_vcarve_stepdown"])
        step['vcarve_final_stepdown'].set(DEFAULTS["step_vcarve_final_stepdown"])
        step['cutout_depth'].set(DEFAULTS["step_cutout_depth"])
        step['cutout_stepdown'].set(DEFAULTS["step_cutout_stepdown"])
        step['cutout_final_stepdown'].set(DEFAULTS["step_cutout_final_stepdown"])
        step['cutout_extra_offset'].set(DEFAULTS["step_cutout_extra_offset"])
        step['contour_z_increment'].set(DEFAULTS["step_contour_z_increment"])
        step['contour_adaptive_var'].set(bool(int(DEFAULTS["step_contour_adaptive"])))
        step['contour_lateral_tol'].set(DEFAULTS["step_contour_lateral_tol"])
        step['tsp_2opt_var'].set(bool(int(DEFAULTS["step_tsp_2opt"])))
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

        ttk.Label(parent, text="Toolpath progress:").pack(side=tk.LEFT, padx=(0, 6))
        self.toolpath_scale_label = ttk.Label(parent, text="100%", width=5, anchor=tk.E)
        self.toolpath_scale_label.pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Scale(parent, from_=0, to=100, orient=tk.HORIZONTAL,
                  variable=self.toolpath_scale_var,
                  command=self._on_toolpath_scale_changed
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _on_toolpath_scale_changed(self, _value=None):
        self.toolpath_scale_label.config(text=f"{round(self.toolpath_scale_var.get())}%")
        self._render()

    def _build_image_canvas(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        self.fig.patch.set_facecolor("#2b2b2b")
        self.ax.set_facecolor("#2b2b2b")
        self.ax.set_visible(False)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.mpl_canvas.draw()

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

    def _stl_raster_px(self, mesh_w_mm, mesh_h_mm, scale_pct=100.0):
        """Target (width_px, height_px) for rasterizing an STL source of the
        given (rotated) mesh bounding-box size at the canvas's current pixel
        density (1/Precision, independent of Width/Height mm -- see
        _canvas_shape_px), scaled up if the source will be enlarged on the
        canvas (scale_pct > 100) so upscaling at composite time doesn't
        blur mesh detail that could have been rasterized at higher
        resolution to begin with. Bounds-independent so it can be reused
        both at import time (fresh bounds) and to re-evaluate an already-
        rasterized source's staleness (stored mesh_w_mm/mesh_h_mm)."""
        try:
            precision_mm = float(self.precision_mm.get())
        except ValueError:
            precision_mm = float(DEFAULTS["precision_mm"])
        if precision_mm <= 1e-9:
            precision_mm = float(DEFAULTS["precision_mm"])
        # canvas_footprint_px = width_px * (scale_pct/100) (see
        # layers._affine_params), so matching the canvas's own density at
        # the ENLARGED footprint means width_px must grow WITH scale_pct,
        # not shrink -- detail_factor = scale_pct/100, not its reciprocal.
        detail_factor = max(scale_pct, 1.0) / 100.0
        width_px  = max(2, round(mesh_w_mm / precision_mm * detail_factor))
        height_px = max(2, round(mesh_h_mm / precision_mm * detail_factor))
        return width_px, height_px

    def _stl_raster_dims(self, xy_bounds, z_bounds, rot_x, rot_y, rot_z, scale_pct=100.0):
        """Compute (width_px, height_px, mesh_w_mm, mesh_h_mm) for
        rasterizing an STL source described by *xy_bounds*/*z_bounds* at
        rotation (0,0,0), rotated by rot_x/rot_y/rot_z first."""
        (x_min, x_max, y_min, y_max), _ = stl_mesh.rotate_bounds(
            xy_bounds, z_bounds, rot_x, rot_y, rot_z)
        mesh_w_mm = max(x_max - x_min, 1e-6)
        mesh_h_mm = max(y_max - y_min, 1e-6)
        width_px, height_px = self._stl_raster_px(mesh_w_mm, mesh_h_mm, scale_pct)
        return width_px, height_px, mesh_w_mm, mesh_h_mm

    def _build_stl_layer(self, path, triangles, dlg, xy_bounds, z_bounds):
        """Rasterize *triangles* (already validated/loaded) per *dlg*'s
        orientation at the canvas's current density and wrap the result as
        a new (unattached) Layer, or show an error dialog and return None
        on failure."""
        width_px, height_px, mesh_w_mm, mesh_h_mm = self._stl_raster_dims(
            xy_bounds, z_bounds, dlg.rot_x, dlg.rot_y, dlg.rot_z)
        try:
            rotated = stl_mesh.apply_rotation(triangles, dlg.rot_x, dlg.rot_y, dlg.rot_z)
            raw     = stl_mesh.generate_heightmap(rotated, width_px, height_px)
            image, alpha = stl_mesh.normalize_and_orient(raw)
        except Exception as exc:
            messagebox.showerror("Import Error", str(exc))
            return None
        stl_params = {'width_px': width_px, 'height_px': height_px,
                      'mesh_w_mm': mesh_w_mm, 'mesh_h_mm': mesh_h_mm,
                      'rot_x': dlg.rot_x, 'rot_y': dlg.rot_y, 'rot_z': dlg.rot_z}
        return layers_mod.Layer(name=os.path.basename(path), raw=image, alpha=alpha,
                                source_kind="stl", path=path, stl_params=stl_params,
                                layer_id=self._new_layer_id())

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
        # the image's own aspect ratio -- fit-to-canvas then lands on
        # scale_x == scale_y, i.e. the image exactly fills the canvas
        # rather than being letterboxed against a stale height.
        layer.scale_pct = self._fit_scale_pct(w, h)
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
        try:
            background_depth_pct = float(self.background_depth_pct.get())
        except ValueError:
            background_depth_pct = float(DEFAULTS["canvas_background_depth_pct"])

        composite_raw = layers_mod.composite(
            self.layers, (H, W), px_per_mm_x, px_per_mm_y, self.surface.get(),
            background_depth=background_depth_pct / 100.0,
            visibility_override=visibility)
        n_grey_levels = layers_mod.compute_n_grey_levels(self.layers, composite_raw)
        return composite_raw, n_grey_levels

    def _recompute_composite(self):
        """Recompute self.current_image (and self.n_grey_levels) from
        self.layers, gated by whichever step tab is currently selected (see
        _current_step) so the live preview always matches what that step
        would actually carve. self.current_image keeps its old role -- the
        single flattened depth map everything downstream of
        _get_depth_map() reads -- it's just a cached compositor output now
        instead of a direct load."""
        if not self.layers:
            self.current_image = None
            self.n_grey_levels = 256
            return

        base = self.layers[0]
        W, H = self._canvas_shape_px()
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
                and base.is_identity_placement() and base.alpha is None
                and base.raw.shape == (H, W)):
            self.current_image = base.raw
            self.n_grey_levels = layers_mod.compute_n_grey_levels(self.layers, self.current_image)
            return

        self.current_image, self.n_grey_levels = self._composite_with_visibility(visibility)

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
        the tab it's on, mirroring Steps' 'Add Step After'."""
        path = filedialog.askopenfilename(title="Add Source",
                                          filetypes=self._SOURCE_FILETYPES)
        if not path:
            return
        insert_at = self.layers.index(entry['layer']) + 1
        if path.lower().endswith(".stl"):
            self._add_stl_source(path, insert_at)
        else:
            self._add_image_source(path, insert_at)

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
        h, w = layer.raw.shape
        layer.scale_pct = self._fit_scale_pct(w, h)
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
                h, w = self.layers[0].raw.shape
                self._image_aspect = w / h
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
        layer = self._build_stl_layer(path, triangles, dlg, xy_bounds, z_bounds)
        if layer is None:
            return

        # self.layers is assigned last (see the matching note in
        # _establish_base_image_layer): the width_mm/height_mm.set() calls
        # below cascade through _on_canvas_size_changed, which must not
        # rescale THIS brand-new, not-yet-placed layer.
        h, w = layer.raw.shape
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
            self._image_aspect = w / h

            self.width_mm.set(f"{width_mm:.4f}")
            self.height_mm.set(f"{height_mm:.4f}")
            self.max_depth.set(f"{max_depth_mm:.4f}")
        else:
            if self._aspect_lock_var.get():
                self._image_aspect = w / h
                self._recalc_height()

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
        dlg = SettingsDialog(self.root, self._settings)
        if dlg.confirmed:
            self._refresh_stats_and_view()

    def _open_dressup(self, step):
        idx = self.steps.index(step)
        settings = {
            'ramp_in_enabled':     step['ramp_in_enabled_var'].get(),
            'ramp_angle_deg':      step['ramp_angle_deg_var'].get(),
            'ramp_max_length_mm':  step['ramp_max_length_mm_var'].get(),
            'backoff_enabled':     step['backoff_enabled_var'].get(),
            'backoff_distance_mm': step['backoff_distance_mm_var'].get(),
            'arc_rapids_enabled':  step['arc_rapids_enabled_var'].get(),
            'arc_radius_mm':       step['arc_radius_mm_var'].get(),
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
            step['backoff_enabled_var'].set(settings['backoff_enabled'])
            step['backoff_distance_mm_var'].set(settings['backoff_distance_mm'])
            step['arc_rapids_enabled_var'].set(settings['arc_rapids_enabled'])
            step['arc_radius_mm_var'].set(settings['arc_radius_mm'])
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
            feed_rate = tp.metadata.get('feed_rate', 800.0)
            stats = tp.get_stats(
                feed_rate   = feed_rate,
                feed_rapid  = self._settings['feed_rapid'],
                z_safe      = self._settings['z_safe'],
                accel_mm_s2 = self._settings['accel_mm_s2'],
                ramp        = self._ramp_settings(step),
                backoff     = self._backoff_settings(step),
                arc         = self._arc_settings(step),
            )
            step['info_var'].set(
                f"Step {idx + 1}: {stats['motion_count']} motion lines\n"
                f"Est. time: {_fmt_time(stats['est_minutes'])}"
                f" @ {feed_rate:.0f} mm/min")
        self._on_display_mode_changed()

    def _ramp_settings(self, step):
        return RampSettings(enabled=step['ramp_in_enabled_var'].get(),
                             angle_deg=step['ramp_angle_deg_var'].get(),
                             max_length_mm=step['ramp_max_length_mm_var'].get())

    def _backoff_settings(self, step):
        return BackoffSettings(enabled=step['backoff_enabled_var'].get(),
                                distance_mm=step['backoff_distance_mm_var'].get())

    def _arc_settings(self, step):
        return ArcSettings(enabled=step['arc_rapids_enabled_var'].get(),
                            radius_mm=step['arc_radius_mm_var'].get())

    def _tab_settings(self, step):
        return TabSettings(
            enabled=step['tab_enabled_var'].get(),
            count=step['tab_count_var'].get(),
            width_mm=step['tab_width_mm_var'].get(),
            height_mm=step['tab_height_mm_var'].get(),
            phase_pct=step['tab_phase_pct_var'].get(),
            tab_angle_deg=step['tab_angle_deg_var'].get())

    # ------------------------------------------------------------------ #
    #  Project save / load
    # ------------------------------------------------------------------ #
    def _step_to_dict(self, step):
        return {
            'algorithm':            step['algorithm_var'].get(),
            'profile':              step['profile_var'].get(),
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
            'contour_z_increment':  step['contour_z_increment'].get(),
            'contour_adaptive':     step['contour_adaptive_var'].get(),
            'contour_lateral_tol':    step['contour_lateral_tol'].get(),
            'tsp_2opt':             step['tsp_2opt_var'].get(),
            'feed_rate':            step['feed_rate'].get(),
            'threshold':            step['threshold'].get(),
            'min_fw':               step['min_fw'].get(),
            'max_fw':               step['max_fw'].get(),
            'max_fw_auto':          bool(step['max_fw'].auto_var.get()),
            'gating_mode':          step['gating_mode_var'].get(),
            'ramp_in_enabled':      step['ramp_in_enabled_var'].get(),
            'ramp_angle_deg':       step['ramp_angle_deg_var'].get(),
            'ramp_max_length_mm':   step['ramp_max_length_mm_var'].get(),
            'backoff_enabled':      step['backoff_enabled_var'].get(),
            'backoff_distance_mm':  step['backoff_distance_mm_var'].get(),
            'arc_rapids_enabled':   step['arc_rapids_enabled_var'].get(),
            'arc_radius_mm':        step['arc_radius_mm_var'].get(),
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
        step['contour_z_increment'].set(d.get('contour_z_increment', DEFAULTS['step_contour_z_increment']))
        step['contour_adaptive_var'].set(bool(d.get('contour_adaptive', False)))
        step['contour_lateral_tol'].set(d.get('contour_lateral_tol', DEFAULTS['step_contour_lateral_tol']))
        step['tsp_2opt_var'].set(bool(d.get('tsp_2opt', False)))
        step['feed_rate'].set(d.get('feed_rate', DEFAULTS['step_feed_rate']))
        step['threshold'].set(d.get('threshold', DEFAULTS['step_threshold']))
        step['min_fw'].set(d.get('min_fw', DEFAULTS['step_min_fw']))
        step['max_fw'].auto_var.set(bool(d.get('max_fw_auto', False)))
        step['max_fw'].set(d.get('max_fw', DEFAULTS['step_max_fw']))
        step['gating_mode_var'].set(d.get('gating_mode', DEFAULTS['step_gating_mode']))
        step['ramp_in_enabled_var'].set(bool(d.get(
            'ramp_in_enabled', bool(int(DEFAULTS['step_ramp_in_enabled'])))))
        step['ramp_angle_deg_var'].set(d.get(
            'ramp_angle_deg', DEFAULTS['step_ramp_angle_deg']))
        step['ramp_max_length_mm_var'].set(d.get(
            'ramp_max_length_mm', DEFAULTS['step_ramp_max_length_mm']))
        step['backoff_enabled_var'].set(bool(d.get(
            'backoff_enabled', bool(int(DEFAULTS['step_backoff_enabled'])))))
        step['backoff_distance_mm_var'].set(d.get(
            'backoff_distance_mm', DEFAULTS['step_backoff_distance_mm']))
        step['arc_rapids_enabled_var'].set(bool(d.get(
            'arc_rapids_enabled', bool(int(DEFAULTS['step_arc_rapids_enabled'])))))
        step['arc_radius_mm_var'].set(d.get(
            'arc_radius_mm', DEFAULTS['step_arc_radius_mm']))
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
        }
        if layer.source_kind == 'stl' and layer.stl_params:
            d['stl_params'] = dict(layer.stl_params)
        return d

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
                raw       = stl_mesh.generate_heightmap(rotated, width_px, height_px)
                image, alpha = stl_mesh.normalize_and_orient(raw)
                layer = layers_mod.Layer(
                    name=d.get('name') or os.path.basename(resolved), raw=image, alpha=alpha,
                    source_kind='stl', path=resolved,
                    stl_params={'width_px': width_px, 'height_px': height_px,
                                'rot_x': rot_x, 'rot_y': rot_y, 'rot_z': rot_z},
                    layer_id=self._new_layer_id())
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
                'precision_mm':    self.precision_mm.get(),
                'max_depth':       self.max_depth.get(),
                'margin_mm':       self.margin_mm.get(),
                'margin_depth_pct': self.margin_depth_pct.get(),
                'surface':         self.surface.get(),
                'origin':          self.origin_var.get(),
                'aspect_locked':   self._aspect_lock_var.get(),
            },
            'machine': {
                'z_safe':      self._settings['z_safe'],
                'feed_rapid':  self._settings['feed_rapid'],
                'accel_mm_s2': self._settings['accel_mm_s2'],
                'preamble':    self._settings['preamble'],
                'postamble':   self._settings['postamble'],
            },
            'steps': [self._step_to_dict(s) for s in self.steps],
        }

    def _snapshot_key(self):
        return json.dumps(self._project_snapshot(), sort_keys=True)

    def _is_dirty(self):
        return self._snapshot_key() != self._saved_snapshot

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
        self.precision_mm.set(DEFAULTS['precision_mm'])
        self.max_depth.set(DEFAULTS['max_depth'])
        self.margin_mm.set(DEFAULTS['margin_mm'])
        self.margin_depth_pct.set(DEFAULTS['margin_depth_pct'])
        self.background_depth_pct.set(DEFAULTS['canvas_background_depth_pct'])
        self.surface.set(DEFAULTS['surface'])
        self.origin_var.set(DEFAULTS['origin'])

        self._settings['z_safe']      = float(DEFAULTS['z_safe'])
        self._settings['feed_rapid']  = float(DEFAULTS['feed_rapid'])
        self._settings['accel_mm_s2'] = float(DEFAULTS['accel_mm_s2'])
        self._settings['preamble']    = DEFAULTS['preamble']
        self._settings['postamble']   = DEFAULTS['postamble']

        while len(self.steps) > 1:
            self._remove_step(self.steps[-1])
        self._apply_step_dict(self.steps[0], {})

        self.layers         = []
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
        self._saved_snapshot = self._snapshot_key()

    def _save_project(self):
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".prj",
            filetypes=[("BM2GCODE project", "*.prj"), ("All files", "*.*")],
        )
        if not path:
            return False
        snapshot = self._project_snapshot()
        layers_out = []
        for layer, d in zip(self.layers, snapshot['layers']):
            d = dict(d)
            d['path_rel'] = self._make_rel_path(layer.path, path)
            layers_out.append(d)
        project = {
            'version': 3,
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
            self._saved_snapshot = json.dumps(snapshot, sort_keys=True)
            return True
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save project:\n{exc}")
            return False

    def _load_project(self):
        path = filedialog.askopenfilename(
            title="Open Project",
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
        if version not in (1, 2, 3):
            messagebox.showerror("Load Error", "Unsupported project file version.")
            return

        if not self._confirm_discard("opening another project"):
            return

        # ── Global image-size parameters ──────────────────────────────
        g = project.get('global', {})
        c = project.get('canvas', {})
        self._aspect_lock_var.set(bool(g.get('aspect_locked', True)))
        self.width_mm.set(g.get('width_mm', DEFAULTS['width_mm']))
        self.height_mm.set(g.get('height_mm', DEFAULTS['height_mm']))
        precision_mm = g.get('precision_mm')
        if precision_mm is None:
            # Pre-v3 project: derive an equivalent Precision from the old
            # base-layer-defined canvas resolution so re-opening reproduces
            # close to the same effective detail before the user adjusts it.
            try:
                width_mm_val = float(g.get('width_mm', DEFAULTS['width_mm']))
            except (TypeError, ValueError):
                width_mm_val = float(DEFAULTS['width_mm'])
            saved_width_px = c.get('width_px')
            if saved_width_px:
                precision_mm = width_mm_val / saved_width_px
            else:
                precision_mm = DEFAULTS['precision_mm']
        self.precision_mm.set(str(precision_mm))
        self.max_depth.set(g.get('max_depth', DEFAULTS['max_depth']))
        self.margin_mm.set(g.get('margin_mm', DEFAULTS['margin_mm']))
        self.margin_depth_pct.set(g.get('margin_depth_pct', DEFAULTS['margin_depth_pct']))
        self.surface.set(g.get('surface', DEFAULTS['surface']))
        self.origin_var.set(g.get('origin', DEFAULTS['origin']))

        # ── Machine settings ──────────────────────────────────────────
        m = project.get('machine', {})
        self._settings['z_safe']      = m.get('z_safe',      self._settings['z_safe'])
        self._settings['feed_rapid']  = m.get('feed_rapid',  self._settings['feed_rapid'])
        self._settings['accel_mm_s2'] = m.get('accel_mm_s2', self._settings['accel_mm_s2'])
        self._settings['preamble']    = m.get('preamble',    self._settings['preamble'])
        self._settings['postamble']   = m.get('postamble',   self._settings['postamble'])

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

        for step, d in zip(self.steps, step_dicts):
            self._apply_step_dict(step, d)

        # ── Layers (version 2+) ───────────────────────────────────────────
        if version in (2, 3) and 'layers' in project:
            layer_dicts = project.get('layers') or []
            resolved_layers = []
            failed = []
            for d in layer_dicts:
                layer = self._layer_from_dict(d, path)
                if layer is not None:
                    resolved_layers.append(layer)
                else:
                    failed.append(d.get('name') or d.get('path') or d.get('path_rel') or "?")
            self.layers = resolved_layers
            self.background_depth_pct.set(
                str(c.get('background_depth_pct', DEFAULTS['canvas_background_depth_pct'])))
            if self._aspect_lock_var.get() and self.layers:
                h, w = self.layers[0].raw.shape
                self._image_aspect = w / h
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
                    raw       = stl_mesh.generate_heightmap(rotated, width_px, height_px)
                    image, alpha = stl_mesh.normalize_and_orient(raw)
                    layer = layers_mod.Layer(
                        name=os.path.basename(resolved_stl_path), raw=image, alpha=alpha,
                        source_kind='stl', path=resolved_stl_path,
                        stl_params={'width_px': width_px, 'height_px': height_px,
                                    'rot_x': rot_x, 'rot_y': rot_y, 'rot_z': rot_z},
                        layer_id=self._new_layer_id())
                    self.layers = [layer]
                    if self._aspect_lock_var.get():
                        h, w = layer.raw.shape
                        self._image_aspect = w / h
                        self._recalc_height()
                    self._on_layers_structure_changed()
                    self.status_var.set(
                        f"Project loaded: {path}  |  STL: {resolved_stl_path}")
                    self._saved_snapshot = self._snapshot_key()
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

    # ------------------------------------------------------------------ #
    #  Compute / save tool paths
    # ------------------------------------------------------------------ #
    def _gather_step_params(self, step):
        """Read and validate all parameters for one step.

        Returns a dict of ready-to-use values. Raises ValueError on bad input.
        Must be called on the main thread so that tkinter variable reads are safe.
        """
        idx       = self.steps.index(step)
        algorithm = step['algorithm_var'].get()
        feed_rate    = float(step['feed_rate'].get())
        threshold_mm = float(step['threshold'].get())
        width_mm      = float(self.width_mm.get())
        height_mm     = float(self.height_mm.get())
        margin_mm     = float(self.margin_mm.get())
        max_depth    = float(self.max_depth.get())
        eff_width_mm  = width_mm  + 2.0 * margin_mm
        eff_height_mm = height_mm + 2.0 * margin_mm
        origin       = self.origin_var.get()

        params = {
            'idx':           idx,
            'algorithm':     algorithm,
            'feed_rate':     feed_rate,
            'threshold_mm':  threshold_mm,
            'eff_width_mm':  eff_width_mm,
            'eff_height_mm': eff_height_mm,
            'max_depth':     max_depth,
            'origin':        origin,
            'extra_offset_mm': float(step['extra_offset'].get()),
            # Used only to key the erosion cache (see _compute_step_work) --
            # the actual depth map is precomputed on the main thread and
            # passed in separately, since it needs Tkinter reads.
            'source_visibility': tuple(step.get('source_visibility', ())),
        }

        if algorithm.startswith("V-Carving"):
            _profile    = step['profile_var'].get()
            _vbit_angle = float(step['vbit_angle'].get()) if _profile != "Ball" else 90.0
            tool        = Tool(_profile, float(step['diameter'].get()), _vbit_angle)
            params.update({
                'tool':                     tool,
                'min_feature_width_mm':     float(step['min_fw'].get()),
                'max_feature_width_mm':     float(step['max_fw'].get()),
                'vcarve_stepdown_mm':       float(step['vcarve_stepdown'].get()),
                'vcarve_final_stepdown_mm': float(step['vcarve_final_stepdown'].get()),
                'use_2opt':                 step['tsp_2opt_var'].get(),
                'skeleton_mode':            ("voronoi"    if algorithm == "V-Carving (Sub-pixel)"
                                             else "inscribed" if algorithm == "V-Carving (Feature Width)"
                                             else "medial_axis"),
                'gating_mode':              step['gating_mode_var'].get(),
            })
        elif algorithm == "Contour":
            _profile    = step['profile_var'].get()
            _vbit_angle = float(step['vbit_angle'].get()) if _profile == "V-Bit" else 0.0
            tool = Tool(_profile, float(step['diameter'].get()), _vbit_angle)
            params.update({
                'tool':                tool,
                'contour_z_increment_mm': float(step['contour_z_increment'].get()),
                'contour_adaptive':    step['contour_adaptive_var'].get(),
                'contour_lateral_tol': float(step['contour_lateral_tol'].get()),
                'use_2opt':            step['tsp_2opt_var'].get(),
                'contour_max_step_down': float(step['stepdown'].get()),
                'cut_direction':       step['cut_dir_var'].get(),
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
        else:  # Raster
            _profile    = step['profile_var'].get()
            _vbit_angle = float(step['vbit_angle'].get()) if _profile == "V-Bit" else 0.0
            tool     = Tool(_profile, float(step['diameter'].get()), _vbit_angle)
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
            })

        return params

    def _compute_step_work(self, step, params, progress_fn, prev_tp, depth_map, n_grey_levels):
        """Synchronous algorithm computation — call from a background thread.

        depth_map/n_grey_levels are precomputed on the main thread (see
        _get_depth_map_for_step) rather than read here, both because this
        runs off the main thread and because each step must be computed
        against its OWN source_visibility, not whichever step tab happens
        to be selected.

        Returns a toolpath result. Raises ProgressDialog.Cancelled or any
        algorithm exception on failure.
        """
        algorithm    = params['algorithm']

        tool = params['tool']
        erosion_cache_key = (tool.profile, tool.diameter_mm, tool.vbit_angle_deg,
                             params['max_depth'], params['eff_width_mm'], params['eff_height_mm'],
                             params['extra_offset_mm'], params['source_visibility'])

        if algorithm.startswith("V-Carving"):
            gating_mode = params['gating_mode']
            if gating_mode == "Residual from prior step":
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
                progress             = progress_fn,
                erosion_cache        = self._erosion_cache,
                erosion_cache_key    = erosion_cache_key,
            )
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
                use_2opt           = params['use_2opt'],
                cut_direction      = params['cut_direction'],
                max_step_down_mm   = params['contour_max_step_down'],
                extra_offset_mm    = params['extra_offset_mm'],
                erosion_cache      = self._erosion_cache,
                erosion_cache_key  = erosion_cache_key,
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
                    extra_offset_mm    = params['cutout_extra_offset_mm'],
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
                extra_offset_mm    = params['cutout_extra_offset_mm'],
                cut_direction      = params['cut_direction'],
                use_2opt           = params['use_2opt'],
                z_safe             = self._settings['z_safe'],
                feed_rapid         = self._settings['feed_rapid'],
                tab_settings       = params['tab_settings'],
                progress           = progress_fn,
            )
        else:  # Raster
            gating_mode = params['gating_mode']
            if gating_mode == "Residual from prior step":
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
                progress          = progress_fn,
                erosion_cache     = self._erosion_cache,
                erosion_cache_key = erosion_cache_key,
            )

    def _stale_stl_sources(self):
        """STL sources whose rasterized detail no longer matches what the
        canvas's current Precision and the source's own Scale call for --
        e.g. Precision was tightened, or Scale was increased, since the
        source was last (re)rasterized from its mesh."""
        stale = []
        for layer in self.layers:
            if layer.source_kind != 'stl' or not layer.stl_params:
                continue
            mesh_w_mm = layer.stl_params.get('mesh_w_mm')
            mesh_h_mm = layer.stl_params.get('mesh_h_mm')
            if mesh_w_mm is None or mesh_h_mm is None:
                continue   # legacy project file predates this bookkeeping
            target_w, target_h = self._stl_raster_px(
                mesh_w_mm, mesh_h_mm, scale_pct=layer.scale_pct)
            cur_w = layer.stl_params.get('width_px', target_w)
            cur_h = layer.stl_params.get('height_px', target_h)
            if (abs(target_w - cur_w) > 0.05 * max(target_w, 1)
                    or abs(target_h - cur_h) > 0.05 * max(target_h, 1)):
                stale.append((layer, target_w, target_h))
        return stale

    def _confirm_stale_stl_sources(self):
        """Offer to re-rasterize stale STL sources (see _stale_stl_sources)
        from their source mesh before a step is computed. Always returns
        True (computing proceeds either way) -- declining just means the
        step is computed against the existing, less-detailed rasters, same
        as it would be if this check didn't exist."""
        stale = self._stale_stl_sources()
        if not stale:
            return True
        names = ", ".join(layer.name for layer, _, _ in stale)
        if not messagebox.askyesno(
                "Mesh detail out of date",
                f"{len(stale)} mesh source(s) ({names}) were rasterized at a "
                "different detail level than the canvas currently calls for.\n\n"
                "Re-rasterize from source for full detail before computing?"):
            return True
        for layer, target_w, target_h in stale:
            try:
                triangles = stl_mesh.load_stl(layer.path)
                rot_x = layer.stl_params.get('rot_x', 0)
                rot_y = layer.stl_params.get('rot_y', 0)
                rot_z = layer.stl_params.get('rot_z', 0)
                rotated = stl_mesh.apply_rotation(triangles, rot_x, rot_y, rot_z)
                raw     = stl_mesh.generate_heightmap(rotated, target_w, target_h)
                image, alpha = stl_mesh.normalize_and_orient(raw)
            except Exception as exc:
                messagebox.showerror("Re-rasterize Error",
                                     f"Could not re-rasterize {layer.name}:\n{exc}")
                continue
            layer.raw, layer.alpha = image, alpha
            layer.stl_params['width_px']  = target_w
            layer.stl_params['height_px'] = target_h
        self._recompute_composite()
        self._erosion_cache.clear()
        self._render()
        return True

    def _compute_step(self, step):
        if self.current_image is None:
            messagebox.showinfo("No image", "Please open an image first.")
            return

        if not self._confirm_stale_stl_sources():
            return

        idx = self.steps.index(step)
        try:
            params = self._gather_step_params(step)
        except ValueError:
            messagebox.showerror("Invalid input",
                                 f"Please check the settings for Step {idx + 1}.")
            return
        depth_map, n_grey_levels = self._get_depth_map_for_step(step)

        # ── Prior-step checks (all before the dialog opens) ─────────────────
        prev_tp   = self.toolpaths[idx - 1] if idx > 0 else None
        algorithm = params['algorithm']

        if algorithm == "Contour":
            if idx > 0 and prev_tp is None:
                if not messagebox.askyesno(
                        "Previous step not computed",
                        f"Step {idx} has not been computed yet.\n\n"
                        "Without its result this step will treat the entire "
                        "depth map as unfinished.\n\nContinue anyway?"):
                    return
        elif algorithm == "Cutout":
            pass  # Cutout cuts based on its own depth_map threshold regardless
                  # of any prior step -- a missing prev_tp only affects this
                  # step's own result/accessible/alpha bookkeeping, not which
                  # pixels get cut, so there is nothing to warn about here.
        else:
            gating_mode = params['gating_mode']
            if gating_mode != "Full depth map" and idx > 0 and prev_tp is None:
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

        feed_rate = params['feed_rate']
        dlg = ProgressDialog(self.root,
                             title=f"Computing Step {idx + 1} Tool Path")

        def _worker():
            try:
                result = self._compute_step_work(
                    step, params, dlg.report, prev_tp, depth_map, n_grey_levels)
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
                result.metadata['feed_rate'] = feed_rate
                stats = result.get_stats(
                    feed_rate   = feed_rate,
                    feed_rapid  = self._settings['feed_rapid'],
                    z_safe      = self._settings['z_safe'],
                    accel_mm_s2 = self._settings['accel_mm_s2'],
                    ramp        = self._ramp_settings(step),
                    backoff     = self._backoff_settings(step),
                    arc         = self._arc_settings(step),
                )
                dlg.close("Completed.")
                step['info_var'].set(
                    f"Step {idx + 1}: {stats['motion_count']} motion lines\n"
                    f"Est. time: {_fmt_time(stats['est_minutes'])}"
                    f" @ {feed_rate:.0f} mm/min")
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

        # Validate every step upfront so the dialog never opens on bad input,
        # and precompute each step's own depth map here on the main thread
        # (each step has its own source_visibility, so these can differ).
        all_params = []
        all_depth_maps = []
        for step in self.steps:
            try:
                all_params.append(self._gather_step_params(step))
            except ValueError:
                idx = self.steps.index(step)
                messagebox.showerror("Invalid input",
                                     f"Please check the settings for Step {idx + 1}.")
                return
            all_depth_maps.append(self._get_depth_map_for_step(step))

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
                try:
                    dlg.report(i * 100.0 / n, f"Starting Step {i + 1} of {n}…")
                    result = self._compute_step_work(
                        step, params, step_progress(i), prev_tp, depth_map, n_grey_levels)
                except ProgressDialog.Cancelled:
                    self.root.after(0, lambda: dlg.close("Cancelled."))
                    return
                except Exception as exc:
                    self.root.after(0, lambda e=exc: dlg.close(f"Error: {e}"))
                    self.root.after(0, lambda e=exc: messagebox.showerror(
                        "Computation error", str(e)))
                    return

                # Store immediately so the next step's prev_tp is up to date.
                self.toolpaths[idx]          = result
                result.metadata['feed_rate'] = params['feed_rate']

                feed_rate = params['feed_rate']
                def _update_ui(idx=idx, s=step, r=result, fr=feed_rate):
                    stats = r.get_stats(
                        feed_rate   = fr,
                        feed_rapid  = self._settings['feed_rapid'],
                        z_safe      = self._settings['z_safe'],
                        accel_mm_s2 = self._settings['accel_mm_s2'],
                        ramp        = self._ramp_settings(s),
                        backoff     = self._backoff_settings(s),
                        arc         = self._arc_settings(s),
                    )
                    s['info_var'].set(
                        f"Step {idx + 1}: {stats['motion_count']} motion lines\n"
                        f"Est. time: {_fmt_time(stats['est_minutes'])}"
                        f" @ {fr:.0f} mm/min")
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
            info.append({
                'label':       f"Step {i + 1} — {profile} ∅{diam_mm:.4g} mm",
                'computed':    tp is not None,
                'profile':     profile,
                'diameter_mm': diam_mm,
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
        # supported in a single G-code file.
        tools = {(steps_info[i]['profile'], steps_info[i]['diameter_mm'])
                 for i in indices}
        if len(tools) > 1:
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
            for idx in indices:
                tp   = self.toolpaths[idx]
                step = self.steps[idx]
                feed_rate = tp.metadata.get('feed_rate', 800.0)
                bodies.append(tp.to_gcode(feed_rate, feed_rapid, z_safe, accel_mm_s2,
                                           ramp=self._ramp_settings(step),
                                           backoff=self._backoff_settings(step),
                                           arc=self._arc_settings(step)))

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
        self._erosion_cache.clear()
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
            and not (m == DISPLAY_STEP_UNACCESSIBLE and not has_accessible)
            and not (m in (DISPLAY_STEP_RES, DISPLAY_STEP_RESULT, DISPLAY_STEP_3D) and not step_computed)
            and not (m == DISPLAY_DIFF_RES_UNCUT and not (has_accessible and has_prev_result))
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
    def _pad_image(self, img):
        """Return *img* padded with the stock margin, or *img* unchanged when margin is 0."""
        try:
            margin_mm = float(self.margin_mm.get())
        except ValueError:
            return img
        if margin_mm <= 0.0:
            return img
        try:
            H, W = img.shape
            width_mm  = float(self.width_mm.get())
            height_mm = float(self.height_mm.get())
            pad_x = max(1, round(margin_mm * W / width_mm))
            pad_y = max(1, round(margin_mm * H / height_mm))
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
                img   = np.clip(depth_map - tp.result, 0.0, 1.0).astype(np.float32)
                cmap  = "hot"
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
                try:
                    threshold_mm = float(self.steps[step_idx]['threshold'].get())
                    max_depth    = float(self.max_depth.get())
                except (ValueError, IndexError):
                    threshold_mm, max_depth = 0.0, 5.0
                thr = threshold_mm / max_depth if max_depth else 0.0

                gating_mode = "Full depth map"
                if (0 <= step_idx < len(self.steps)
                        and self.steps[step_idx].get('gating_mode_var') is not None):
                    gating_mode = self.steps[step_idx]['gating_mode_var'].get()

                prior_idx = step_idx - 1
                if gating_mode == "Inaccessible by prior tool":
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

                min_fw_mm = 0.0
                max_fw_mm = 0.0
                try:
                    min_fw_mm = float(self.steps[step_idx]['min_fw'].get())
                    max_fw_mm = float(self.steps[step_idx]['max_fw'].get())
                except (ValueError, IndexError, KeyError):
                    pass
                try:
                    H, W = ref.shape
                    px_per_mm_x = W / float(self.width_mm.get())
                    px_per_mm_y = H / float(self.height_mm.get())
                    px_per_mm   = (px_per_mm_x + px_per_mm_y) / 2.0
                except (ValueError, ZeroDivisionError):
                    px_per_mm = 1.0
                _, mask = compute_vcarve_mask(ref, thr, min_fw_mm, px_per_mm, max_fw_mm)

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

        if mode not in (DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
                        DISPLAY_STEP_UNACCESSIBLE, DISPLAY_PROC_MASK,
                        DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT):
            img   = self._pad_image(self.current_image)
            cmap  = "gray"
            title = "Original"

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
        self.ax.imshow(img, interpolation="none", aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, alpha=img_alpha)
        self._overlay_toolpath_2d(img, step_idx)
        self.ax.axis("off")
        if title:
            self.ax.set_title(title, color="white", fontsize=10, pad=6)
        self.fig.tight_layout()

        if saved_xlim is not None and saved_ylim is not None:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)

        self.mpl_canvas.draw()

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
        try:
            margin_mm = float(self.margin_mm.get())
            width_mm  = float(self.width_mm.get())  + 2.0 * margin_mm
            height_mm = float(self.height_mm.get()) + 2.0 * margin_mm
        except ValueError:
            return
        if width_mm == 0 or height_mm == 0:
            return

        H, W = img.shape[:2]
        sx = W / width_mm
        sy = H / height_mm

        origin = self.origin_var.get()
        _ox = {"Bottom-Left": 0,          "Bottom-Right": width_mm,
               "Top-Left":    0,          "Top-Right":    width_mm,
               "Center":      width_mm  / 2}.get(origin, 0)
        _oy = {"Bottom-Left": 0,          "Bottom-Right": 0,
               "Top-Left":    height_mm, "Top-Right":    height_mm,
               "Center":      height_mm / 2}.get(origin, 0)

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
            margin_mm = float(self.margin_mm.get())
            width_mm  = float(self.width_mm.get())  + 2.0 * margin_mm
            height_mm = float(self.height_mm.get()) + 2.0 * margin_mm
            max_depth = float(self.max_depth.get())
        except ValueError:
            self.coord_var.set("")
            return

        px_per_mm_x = W / width_mm
        px_per_mm_y = H / height_mm

        origin = self.origin_var.get()
        _ox = {"Bottom-Left": 0,         "Bottom-Right": width_mm,
               "Top-Left":    0,         "Top-Right":    width_mm,
               "Center":      width_mm / 2}.get(origin, 0)
        _oy = {"Bottom-Left": 0,         "Bottom-Right": 0,
               "Top-Left":    height_mm, "Top-Right":    height_mm,
               "Center":      height_mm / 2}.get(origin, 0)

        x_mm = col / px_per_mm_x - _ox
        y_mm = (H - 1 - row) / px_per_mm_y - _oy

        mode = self.display_mode_var.get()
        if mode == DISPLAY_PROC_MASK:
            val = float(img[row, col])
            z_str = f"Z: {'1' if val > 0.5 else '0'}"
        elif mode in (DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT):
            val = float(img[row, col])
            z_str = f"ΔZ: {val * max_depth:+.2f} mm"
        else:
            if mode == DISPLAY_ORIGINAL:
                depth_map, _ = self._get_depth_map()
                val = float(depth_map[row, col])
            else:
                val = float(img[row, col])
            sign = +1 if mode in (DISPLAY_STEP_RES, DISPLAY_STEP_UNACCESSIBLE) else -1
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
