import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading

from constants import (
    TOOL_PROFILES, RASTER_DIRECTIONS, ALGORITHMS, DEFAULTS,
    DEFAULT_PREAMBLE, DEFAULT_POSTAMBLE,
    DISPLAY_ORIGINAL, DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
    DISPLAY_STEP_3D, DISPLAY_STEP_UNACCESSIBLE, DISPLAY_PROC_MASK,
    DISPLAY_MODES, DISPLAY_3D_MODES, DISPLAY_STEP_MODES,
)
from widgets import (ProgressDialog, LabeledEntry, SectionFrame,
                     SettingsDialog, ExportToolpathDialog, _fmt_time)
from algorithms import (compute_raster_toolpath, compute_vcarve_toolpath,
                        compute_contour_toolpath, compute_vcarve_mask)
from tool import Tool


class GCodeGeneratorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("CNC Relief Carving — Parameter Setup")
        self.root.geometry("1200x780")
        self.root.minsize(900, 560)

        self._settings = {
            'z_safe':     float(DEFAULTS['z_safe']),
            'feed_rapid': float(DEFAULTS['feed_rapid']),
            'preamble':   DEFAULTS['preamble'],
            'postamble':  DEFAULTS['postamble'],
        }

        self.current_image   = None   # 2-D float32 [0,1] grayscale array
        self.n_grey_levels   = 256
        self.image_path      = None
        self.steps     = []   # list of step-widget dicts, built by _build_sidebar
        self.toolpaths = []   # list of ToolPath|None, parallel to self.steps

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

        # Currently rendered 2D array (used for hover coordinate readout)
        self._displayed_array = None

        self._build_menu()
        self._build_layout()

    # ------------------------------------------------------------------ #
    #  Menu
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        self.display_mode_var   = tk.StringVar(value=DISPLAY_ORIGINAL)
        self.display_step_var   = tk.IntVar(value=1)
        self.show_cut_moves_var = tk.BooleanVar(value=True)
        self.show_rapids_var    = tk.BooleanVar(value=True)

        menubar   = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image…",  accelerator="Ctrl+O",
                              command=self.open_image)
        file_menu.add_command(label="Open Project…", accelerator="Ctrl+Shift+O",
                              command=self._load_project)
        file_menu.add_command(label="Save Project…", accelerator="Ctrl+S",
                              command=self._save_project)
        file_menu.add_command(label="Export Toolpath…",
                              command=self._export_toolpaths)
        file_menu.add_separator()
        file_menu.add_command(label="Settings…",     command=self._open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit",          accelerator="Ctrl+Q",
                              command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        self._view_menu = tk.Menu(menubar, tearoff=0)

        _display_hotkeys = [
            (DISPLAY_ORIGINAL,          "1"),
            (DISPLAY_STEP_3D,           "2"),
            (DISPLAY_STEP_RESULT,       "3"),
            (DISPLAY_STEP_RES,          "4"),
            (DISPLAY_STEP_UNACCESSIBLE, "5"),
            (DISPLAY_PROC_MASK,         "6"),
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
        self._view_menu.add_separator()
        self._view_menu.add_command(label="Reset Zoom  (double-click)",
                                    command=self._reset_zoom_and_render)

        menubar.add_cascade(label="View", menu=self._view_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self._save_project())
        self.root.bind("<Control-O>", lambda e: self._load_project())
        self.root.bind("<Control-q>", lambda e: self.root.quit())

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
        pane.add(right, minsize=420)
        self._build_right_panel(right)

        # ── Sidebar ────────────────────────────────────────────────────
        sidebar_outer = ttk.Frame(pane, width=275)
        sidebar_outer.pack_propagate(False)
        pane.add(sidebar_outer, minsize=240)
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

        # ── 1. Physical Image Size ─────────────────────────────────────
        sf = SectionFrame(p, "Physical Image Size")
        sf.pack(fill=tk.X, pady=(0, 8))
        self.width_cm    = LabeledEntry(sf, "Width",         "cm", DEFAULTS["width_cm"])
        self.height_cm   = LabeledEntry(sf, "Height",        "cm", DEFAULTS["height_cm"])
        self.width_cm.pack(fill=tk.X, pady=2)
        self.height_cm.pack(fill=tk.X, pady=2)

        self._aspect_lock_var = tk.BooleanVar(value=True)
        self._image_aspect    = None   # pixel W/H of the loaded image
        ttk.Checkbutton(self.height_cm, text="Lock",
                        variable=self._aspect_lock_var,
                        command=self._on_aspect_lock_toggle
                        ).pack(side=tk.RIGHT, padx=(0, 4))

        self.max_depth = LabeledEntry(sf, "Max cut depth", "mm", DEFAULTS["max_depth"])
        self.max_depth.pack(fill=tk.X, pady=2)
        self.margin_cm = LabeledEntry(sf, "Stock margin", "cm", DEFAULTS["margin_cm"])
        self.margin_cm.pack(fill=tk.X, pady=2)
        self.margin_depth_pct = LabeledEntry(sf, "Margin depth", "%", DEFAULTS["margin_depth_pct"])
        self.margin_depth_pct.pack(fill=tk.X, pady=2)
        self.width_cm.var.trace_add("write", self._on_width_changed)
        self.margin_cm.var.trace_add("write", lambda *_: self._render())
        self.margin_depth_pct.var.trace_add("write", lambda *_: self._render())
        self._on_aspect_lock_toggle()   # apply initial locked state

        polarity_row = ttk.Frame(sf)
        polarity_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(polarity_row, text="z=0 colour:").pack(side=tk.LEFT)
        self.surface = tk.StringVar(value=DEFAULTS["surface"])
        ttk.Combobox(polarity_row, textvariable=self.surface,
                     values=["white", "black"],
                     state="readonly", width=22).pack(side=tk.RIGHT)
        self.surface.trace_add("write", lambda *_: self._render())

        origin_row = ttk.Frame(sf)
        origin_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(origin_row, text="G-code origin:").pack(side=tk.LEFT)
        self.origin_var = tk.StringVar(value=DEFAULTS["origin"])
        ttk.Combobox(origin_row, textvariable=self.origin_var,
                     values=["Bottom-Left", "Bottom-Right",
                             "Top-Left", "Top-Right", "Center"],
                     state="readonly", width=22).pack(side=tk.RIGHT)

        # ── 2. Tool path steps (dynamic notebook) ─────────────────────
        self._steps_nb = ttk.Notebook(p)
        self._steps_nb.pack(fill=tk.X, pady=(0, 4))
        self._steps_nb.bind("<<NotebookTabChanged>>", self._on_step_tab_changed)

        # ── 3. G-code Info ────────────────────────────────────────────
        # Must be created before _add_step() so that info labels can attach.
        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self._info_frame = ttk.Frame(p)
        self._info_frame.pack(fill=tk.X)

        self._add_step()   # start with one step

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

        step = {'tab': tab}
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
        ttk.Button(bf, text="Export…",
                   command=lambda s=step: self._export_from_step(s)
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
        for key in ('diameter', 'stepover', 'stepdown', 'vbit_angle',
                    'vcarve_stepdown', 'vcarve_final_stepdown',
                    'contour_stepdown', 'feed_rate', 'threshold',
                    'min_fw', 'max_fw'):
            dst[key].set(src[key].get())
            if dst[key].auto_var is not None and src[key].auto_var is not None:
                dst[key].auto_var.set(src[key].auto_var.get())
        dst['raster_dir_var'].set(src['raster_dir_var'].get())
        dst['contour_adaptive_var'].set(src['contour_adaptive_var'].get())
        dst['gating_mode_var'].set(src['gating_mode_var'].get())

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

        stepover = LabeledEntry(raster_frame, "Step-over", "%", DEFAULTS["step_stepover"])
        stepover.pack(fill=tk.X, pady=2)
        step['stepover'] = stepover

        raster_offset = LabeledEntry(raster_frame, "Line offset", "%", DEFAULTS["step_raster_offset"])
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

        # ── Contour-only op controls (hidden unless Contour algorithm) ──
        contour_frame = ttk.Frame(operation_lf)
        step['contour_frame'] = contour_frame
        # (not packed initially — shown on demand by _refresh_layout)

        contour_stepdown = LabeledEntry(contour_frame, "Step-down", "mm",
                                        DEFAULTS["step_contour_stepdown"])
        contour_stepdown.pack(fill=tk.X, pady=2)
        step['contour_stepdown'] = contour_stepdown

        contour_adaptive_var = tk.BooleanVar(
            value=bool(int(DEFAULTS["step_contour_adaptive"])))
        ttk.Checkbutton(contour_frame, text="Adaptive levels (CDF)",
                        variable=contour_adaptive_var).pack(anchor=tk.W, pady=2)
        step['contour_adaptive_var'] = contour_adaptive_var

        tsp_frame = ttk.Frame(operation_lf)
        tsp_frame.pack(fill=tk.X)   # shown initially (Raster Scan is the default)
        step['tsp_frame'] = tsp_frame

        tsp_2opt_var = tk.BooleanVar(
            value=bool(int(DEFAULTS["step_tsp_2opt"])))
        ttk.Checkbutton(tsp_frame, text="Optimize path order (2-opt)",
                        variable=tsp_2opt_var).pack(anchor=tk.W, pady=2)
        step['tsp_2opt_var'] = tsp_2opt_var

        gating_frame = ttk.Frame(operation_lf)
        gating_frame.pack(fill=tk.X)
        step['gating_frame'] = gating_frame

        region_row = ttk.Frame(gating_frame)
        region_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(region_row, text="Process region:").pack(side=tk.LEFT)
        gating_var = tk.StringVar(value=DEFAULTS["step_gating_mode"])
        step['gating_mode_var'] = gating_var
        gating_cb = ttk.Combobox(region_row, textvariable=gating_var,
                                 values=["Full depth map",
                                         "Residual from prior step",
                                         "Inaccessible by prior tool"],
                                 state="readonly", width=22)
        gating_cb.pack(side=tk.RIGHT)
        step['gating_cb'] = gating_cb

        threshold = LabeledEntry(gating_frame, "Z-threshold", "mm", DEFAULTS["step_threshold"])
        threshold.pack(fill=tk.X, pady=2)
        step['threshold'] = threshold

        min_fw = LabeledEntry(gating_frame, "Min feature width", "mm", DEFAULTS["step_min_fw"])
        step['min_fw'] = min_fw
        max_fw = LabeledEntry(gating_frame, "Max feature width", "mm", DEFAULTS["step_max_fw"],
                              auto_label="Auto")
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

        diam = LabeledEntry(tool_lf, "Diameter", "mm", DEFAULTS["step_diameter"])
        diam.pack(fill=tk.X, pady=2)
        step['diameter'] = diam

        # ── Raster step-down controls (shown for Raster Scan only) ───────
        stepdown = LabeledEntry(tool_lf, "Max step-down", "mm", DEFAULTS["step_stepdown"])
        stepdown.pack(fill=tk.X, pady=2)   # initially shown (default = Raster Scan)
        step['stepdown'] = stepdown

        raster_final_stepdown = LabeledEntry(tool_lf, "Final step-down", "mm",
                                             DEFAULTS["step_raster_final_stepdown"])
        raster_final_stepdown.pack(fill=tk.X, pady=2)
        step['raster_final_stepdown'] = raster_final_stepdown

        # ── V-bit angle (shown whenever V-Bit profile is active) ──────
        vbit_frame = ttk.Frame(tool_lf)
        step['vbit_frame'] = vbit_frame
        # (not packed initially — shown on demand)

        vbit_angle = LabeledEntry(vbit_frame, "V-bit angle", "°", DEFAULTS["step_vbit_angle"])
        vbit_angle.pack(fill=tk.X, pady=2)
        step['vbit_angle'] = vbit_angle

        # ── V-carving-only controls (hidden when Raster Scan) ─────────
        vcarve_frame = ttk.Frame(tool_lf)
        step['vcarve_frame'] = vcarve_frame
        # (not packed initially — shown on demand by _refresh_layout)

        vcarve_stepdown = LabeledEntry(vcarve_frame, "Max step-down", "mm", DEFAULTS["step_vcarve_stepdown"])
        vcarve_stepdown.pack(fill=tk.X, pady=2)
        step['vcarve_stepdown'] = vcarve_stepdown

        vcarve_final_stepdown = LabeledEntry(vcarve_frame, "Final step-down", "mm", DEFAULTS["step_vcarve_final_stepdown"])
        vcarve_final_stepdown.pack(fill=tk.X, pady=2)
        step['vcarve_final_stepdown'] = vcarve_final_stepdown

        feed = LabeledEntry(tool_lf, "Feed rate", "mm/min", DEFAULTS["step_feed_rate"])
        feed.pack(fill=tk.X, pady=2)
        step['feed_rate'] = feed

        # ── Wire up algorithm / profile toggle ────────────────────────
        _VCARVE_PROFILES  = ["V-Bit", "Ball"]
        _RASTER_PROFILES  = ["Ball", "End Mill", "V-Bit"]
        _prev_alg = [algorithm_var.get()]

        def _refresh_layout(*_):
            alg     = algorithm_var.get()
            profile = step['profile_var'].get()
            alg_changed = (alg != _prev_alg[0])
            _prev_alg[0] = alg

            if alg.startswith("V-Carving"):
                if alg_changed:
                    step['gating_mode_var'].set("Inaccessible by prior tool")
                step['raster_frame'].pack_forget()
                step['contour_frame'].pack_forget()
                step['stepdown'].pack_forget()
                step['raster_final_stepdown'].pack_forget()
                step['vbit_frame'].pack(fill=tk.X, after=step['diameter'])
                step['vcarve_frame'].pack(fill=tk.X, after=step['vbit_frame'])
                step['min_fw'].pack(fill=tk.X, pady=2)
                step['max_fw'].pack(fill=tk.X, pady=2)
                step['gating_frame'].pack(fill=tk.X)
                step['tsp_frame'].pack(fill=tk.X)
                step['profile_cb'].config(state="readonly", values=_VCARVE_PROFILES)
                if profile not in _VCARVE_PROFILES:
                    step['profile_var'].set("V-Bit")
                step['vbit_angle'].set_state(
                    "disabled" if profile == "Ball" else "normal")
            elif alg == "Contour":
                step['raster_frame'].pack_forget()
                step['vcarve_frame'].pack_forget()
                step['stepdown'].pack_forget()
                step['raster_final_stepdown'].pack_forget()
                step['min_fw'].pack_forget()
                step['max_fw'].pack_forget()
                step['gating_frame'].pack_forget()
                step['tsp_frame'].pack_forget()
                step['profile_cb'].config(state="readonly", values=_RASTER_PROFILES)
                if profile not in _RASTER_PROFILES:
                    step['profile_var'].set("Ball")
                if profile == "V-Bit":
                    step['vbit_frame'].pack(fill=tk.X, after=step['diameter'])
                else:
                    step['vbit_frame'].pack_forget()
                step['contour_frame'].pack(fill=tk.X, after=step['alg_row'])
                step['vbit_angle'].set_state("normal")
            else:  # Raster Scan
                if alg_changed:
                    step['gating_mode_var'].set("Residual from prior step")
                step['vcarve_frame'].pack_forget()
                step['contour_frame'].pack_forget()
                step['min_fw'].pack_forget()
                step['max_fw'].pack_forget()
                step['gating_frame'].pack(fill=tk.X)
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
                step['raster_final_stepdown'].pack(fill=tk.X, pady=2,
                                                   after=step['stepdown'])
                step['raster_frame'].pack(fill=tk.X, after=step['alg_row'])
                step['vbit_angle'].set_state("normal")
            if hasattr(self, '_view_menu'):
                self._on_display_mode_changed()

        profile_var.trace_add("write", _refresh_layout)
        algorithm_var.trace_add("write", _refresh_layout)

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
            max_fw.var.set(f"{2.0 * r:.3f}")

        max_fw.auto_var.trace_add("write", _update_auto_max_fw)
        step['profile_var'].trace_add("write", _update_auto_max_fw)
        step['vbit_angle'].var.trace_add("write", _update_auto_max_fw)
        step['diameter'].var.trace_add("write", _update_auto_max_fw)
        self.max_depth.var.trace_add("write", _update_auto_max_fw)

        threshold.var.trace_add("write", _on_mask_param_changed)
        min_fw.var.trace_add("write", _on_mask_param_changed)
        max_fw.var.trace_add("write", _on_mask_param_changed)
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
        step['feed_rate'].set(DEFAULTS["step_feed_rate"])
        step['threshold'].set(DEFAULTS["step_threshold"])
        step['raster_dir_var'].set(DEFAULTS["step_raster_dir"])
        step['vbit_angle'].set(DEFAULTS["step_vbit_angle"])
        step['min_fw'].set(DEFAULTS["step_min_fw"])
        step['max_fw'].auto_var.set(False)
        step['max_fw'].set(DEFAULTS["step_max_fw"])
        step['vcarve_stepdown'].set(DEFAULTS["step_vcarve_stepdown"])
        step['vcarve_final_stepdown'].set(DEFAULTS["step_vcarve_final_stepdown"])
        step['contour_stepdown'].set(DEFAULTS["step_contour_stepdown"])
        step['contour_adaptive_var'].set(bool(int(DEFAULTS["step_contour_adaptive"])))
        step['tsp_2opt_var'].set(bool(int(DEFAULTS["step_tsp_2opt"])))
        step['gating_mode_var'].set(DEFAULTS["step_gating_mode"])

    # ------------------------------------------------------------------ #
    #  Right panel: image canvas
    # ------------------------------------------------------------------ #
    def _build_right_panel(self, parent):
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self._build_image_canvas(canvas_frame)

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
            text="Open a PNG image via File → Open Image…",
            font=("Helvetica", 13),
            foreground="#888888",
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # ------------------------------------------------------------------ #
    #  File / image actions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_grayscale(raw: np.ndarray) -> np.ndarray:
        """Convert any loaded image array to a 2-D float32 [0, 1] grayscale array.

        Alpha-channel handling: fully transparent pixels (alpha == 0) become white
        (1.0), so a PNG that encodes shapes with black ink on a fully-transparent
        background is treated as black-on-white rather than all-black.
        """
        img = raw.astype(np.float32)

        # Normalise to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        # Composite alpha onto a white background before discarding the channel
        if img.ndim == 3 and img.shape[2] == 4:
            rgb   = img[..., :3]
            alpha = img[..., 3:4]          # keep dims for broadcasting
            img   = rgb * alpha + (1.0 - alpha)   # blend onto white

        # Convert RGB to grayscale (ITU-R BT.601 luminance weights)
        if img.ndim == 3:
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

        return np.clip(img, 0.0, 1.0)

    def _on_aspect_lock_toggle(self):
        if self._aspect_lock_var.get():
            if self.current_image is not None:
                h, w = self.current_image.shape
                self._image_aspect = w / h
            self.height_cm.set_state("disabled")
            self._recalc_height()
        else:
            self._image_aspect = None
            self.height_cm.set_state("normal")

    def _recalc_height(self):
        if not self._aspect_lock_var.get() or self._image_aspect is None:
            return
        try:
            w = float(self.width_cm.get())
            self.height_cm.set(f"{w / self._image_aspect:.4f}")
        except (ValueError, ZeroDivisionError):
            pass

    def _on_width_changed(self, *_):
        self._recalc_height()

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select a PNG image",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            raw = mpimg.imread(path)
            self.current_image = self._to_grayscale(raw)
            self.image_path    = path
        except Exception as exc:
            messagebox.showerror("Error", f"Could not load image:\n{exc}")
            return

        # Update stored aspect ratio from the new image's pixel dimensions
        h, w = self.current_image.shape
        if self._aspect_lock_var.get():
            self._image_aspect = w / h
            self._recalc_height()

        # Reset any previously computed tool paths
        self.toolpaths = [None] * len(self.steps)
        for i, s in enumerate(self.steps):
            s['info_var'].set(f"Step {i + 1}: not computed")
        self.display_mode_var.set(DISPLAY_ORIGINAL)
        self._reset_zoom()

        self.placeholder.place_forget()
        levels = int(np.round((self.current_image.max() - self.current_image.min()) * 255)) + 1
        self.n_grey_levels = levels
        self.status_var.set(f"Loaded: {path}  |  {w} x {h} px  |  {levels} gray levels")
        self._render()

    def _open_settings(self):
        SettingsDialog(self.root, self._settings)

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
            'raster_dir':           step['raster_dir_var'].get(),
            'vbit_angle':           step['vbit_angle'].get(),
            'vcarve_stepdown':      step['vcarve_stepdown'].get(),
            'vcarve_final_stepdown': step['vcarve_final_stepdown'].get(),
            'contour_stepdown':     step['contour_stepdown'].get(),
            'contour_adaptive':     step['contour_adaptive_var'].get(),
            'tsp_2opt':             step['tsp_2opt_var'].get(),
            'feed_rate':            step['feed_rate'].get(),
            'threshold':            step['threshold'].get(),
            'min_fw':               step['min_fw'].get(),
            'max_fw':               step['max_fw'].get(),
            'max_fw_auto':          bool(step['max_fw'].auto_var.get()),
            'gating_mode':          step['gating_mode_var'].get(),
        }

    def _apply_step_dict(self, step, d):
        step['algorithm_var'].set(d.get('algorithm', DEFAULTS['step_algorithm']))
        step['profile_var'].set(d.get('profile', DEFAULTS['step_profile']))
        step['diameter'].set(d.get('diameter', DEFAULTS['step_diameter']))
        step['stepover'].set(d.get('stepover', DEFAULTS['step_stepover']))
        step['raster_offset'].set(d.get('raster_offset', DEFAULTS['step_raster_offset']))
        step['stepdown'].set(d.get('stepdown', DEFAULTS['step_stepdown']))
        step['raster_dir_var'].set(d.get('raster_dir', DEFAULTS['step_raster_dir']))
        step['vbit_angle'].set(d.get('vbit_angle', DEFAULTS['step_vbit_angle']))
        step['vcarve_stepdown'].set(d.get('vcarve_stepdown', DEFAULTS['step_vcarve_stepdown']))
        step['vcarve_final_stepdown'].set(d.get('vcarve_final_stepdown', DEFAULTS['step_vcarve_final_stepdown']))
        step['contour_stepdown'].set(d.get('contour_stepdown', DEFAULTS['step_contour_stepdown']))
        step['contour_adaptive_var'].set(bool(d.get('contour_adaptive', False)))
        step['tsp_2opt_var'].set(bool(d.get('tsp_2opt', False)))
        step['feed_rate'].set(d.get('feed_rate', DEFAULTS['step_feed_rate']))
        step['threshold'].set(d.get('threshold', DEFAULTS['step_threshold']))
        step['min_fw'].set(d.get('min_fw', DEFAULTS['step_min_fw']))
        step['max_fw'].auto_var.set(bool(d.get('max_fw_auto', False)))
        step['max_fw'].set(d.get('max_fw', DEFAULTS['step_max_fw']))
        step['gating_mode_var'].set(d.get('gating_mode', DEFAULTS['step_gating_mode']))

    def _save_project(self):
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".bm2",
            filetypes=[("BM2GCODE project", "*.bm2"), ("All files", "*.*")],
        )
        if not path:
            return
        project = {
            'version': 1,
            'image_path': self.image_path,
            'global': {
                'width_cm':     self.width_cm.get(),
                'height_cm':    self.height_cm.get(),
                'max_depth':    self.max_depth.get(),
                'surface':      self.surface.get(),
                'origin':       self.origin_var.get(),
                'aspect_locked': self._aspect_lock_var.get(),
            },
            'machine': {
                'z_safe':     self._settings['z_safe'],
                'feed_rapid': self._settings['feed_rapid'],
                'preamble':   self._settings['preamble'],
                'postamble':  self._settings['postamble'],
            },
            'steps': [self._step_to_dict(s) for s in self.steps],
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(project, f, indent=2)
            self.status_var.set(f"Project saved: {path}")
        except Exception as exc:
            messagebox.showerror("Save Error", f"Could not save project:\n{exc}")

    def _load_project(self):
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("BM2GCODE project", "*.bm2"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                project = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read project file:\n{exc}")
            return

        if project.get('version', 1) != 1:
            messagebox.showerror("Load Error", "Unsupported project file version.")
            return

        # ── Global image-size parameters ──────────────────────────────
        g = project.get('global', {})
        self._aspect_lock_var.set(bool(g.get('aspect_locked', True)))
        self.width_cm.set(g.get('width_cm', DEFAULTS['width_cm']))
        self.height_cm.set(g.get('height_cm', DEFAULTS['height_cm']))
        self.max_depth.set(g.get('max_depth', DEFAULTS['max_depth']))
        self.surface.set(g.get('surface', DEFAULTS['surface']))
        self.origin_var.set(g.get('origin', DEFAULTS['origin']))

        # ── Machine settings ──────────────────────────────────────────
        m = project.get('machine', {})
        self._settings['z_safe']     = m.get('z_safe',     self._settings['z_safe'])
        self._settings['feed_rapid'] = m.get('feed_rapid', self._settings['feed_rapid'])
        self._settings['preamble']   = m.get('preamble',   self._settings['preamble'])
        self._settings['postamble']  = m.get('postamble',  self._settings['postamble'])

        # ── Steps ─────────────────────────────────────────────────────
        step_dicts = project.get('steps', [])
        if not step_dicts:
            messagebox.showwarning("Load Warning", "Project contains no steps.")
            return

        # Trim or extend the steps list to match the saved count.
        while len(self.steps) > len(step_dicts):
            self._remove_step(self.steps[-1])
        while len(self.steps) < len(step_dicts):
            self._add_step()

        for step, d in zip(self.steps, step_dicts):
            self._apply_step_dict(step, d)

        # ── Image ─────────────────────────────────────────────────────
        saved_image_path = project.get('image_path')
        if saved_image_path:
            if os.path.isfile(saved_image_path):
                try:
                    raw = mpimg.imread(saved_image_path)
                    self.current_image = self._to_grayscale(raw)
                    self.image_path    = saved_image_path
                    h, w = self.current_image.shape
                    if self._aspect_lock_var.get():
                        self._image_aspect = w / h
                        self._recalc_height()
                    self.toolpaths = [None] * len(self.steps)
                    for i, s in enumerate(self.steps):
                        s['info_var'].set(f"Step {i + 1}: not computed")
                    self.display_mode_var.set(DISPLAY_ORIGINAL)
                    self._reset_zoom()
                    self.placeholder.place_forget()
                    levels = int(np.round(
                        (self.current_image.max() - self.current_image.min()) * 255)) + 1
                    self.n_grey_levels = levels
                    self._render()
                    self.status_var.set(
                        f"Project loaded: {path}  |  Image: {saved_image_path}")
                    return
                except Exception as exc:
                    messagebox.showwarning(
                        "Image Load Warning",
                        f"Project loaded but the image could not be opened:\n{exc}")
            else:
                messagebox.showwarning(
                    "Image Not Found",
                    f"Project loaded, but the saved image was not found:\n{saved_image_path}\n\n"
                    "Use File → Open Image… to load an image manually.")

        self.status_var.set(f"Project loaded: {path}")

    # ------------------------------------------------------------------ #
    #  Compute / save tool paths
    # ------------------------------------------------------------------ #
    def _compute_step(self, step):
        if self.current_image is None:
            messagebox.showinfo("No image", "Please open an image first.")
            return
        idx       = self.steps.index(step)
        algorithm = step['algorithm_var'].get()
        try:
            feed_rate    = float(step['feed_rate'].get())
            threshold_mm = float(step['threshold'].get())
            width_cm      = float(self.width_cm.get())
            height_cm     = float(self.height_cm.get())
            margin_cm     = float(self.margin_cm.get())
            eff_width_cm  = width_cm  + 2.0 * margin_cm
            eff_height_cm = height_cm + 2.0 * margin_cm
            max_depth    = float(self.max_depth.get())
            if algorithm.startswith("V-Carving"):
                _profile   = step['profile_var'].get()
                _vbit_angle = float(step['vbit_angle'].get()) if _profile != "Ball" else 90.0
                tool       = Tool(_profile, float(step['diameter'].get()), _vbit_angle)
                min_feature_width_mm     = float(step['min_fw'].get())
                max_feature_width_mm     = float(step['max_fw'].get())
                vcarve_stepdown_mm       = float(step['vcarve_stepdown'].get())
                vcarve_final_stepdown_mm = float(step['vcarve_final_stepdown'].get())
                use_2opt                 = step['tsp_2opt_var'].get()
            elif algorithm == "Contour":
                _profile    = step['profile_var'].get()
                _vbit_angle = (float(step['vbit_angle'].get())
                               if _profile == "V-Bit" else 0.0)
                tool             = Tool(_profile, float(step['diameter'].get()), _vbit_angle)
                contour_stepdown  = float(step['contour_stepdown'].get())
                contour_adaptive  = step['contour_adaptive_var'].get()
            else:
                _profile   = step['profile_var'].get()
                _vbit_angle = (float(step['vbit_angle'].get())
                               if _profile == "V-Bit" else 0.0)
                tool             = Tool(_profile, float(step['diameter'].get()), _vbit_angle)
                stepover         = float(step['stepover'].get())
                raster_offset_pct = float(step['raster_offset'].get())
                stepdown                  = float(step['stepdown'].get())
                raster_final_stepdown_mm  = float(step['raster_final_stepdown'].get())
                use_2opt                  = step['tsp_2opt_var'].get()
        except ValueError:
            messagebox.showerror("Invalid input",
                                 f"Please check the settings for Step {idx + 1}.")
            return

        # ── Derive prior step and gating reference ──────────────────────────
        _prev_tp = self.toolpaths[idx - 1] if idx > 0 else None

        if algorithm != "Contour":
            gating_mode = step['gating_mode_var'].get()
            if gating_mode == "Residual from prior step":
                prior_reference = _prev_tp.result if _prev_tp is not None else None
            elif gating_mode == "Inaccessible by prior tool":
                prior_reference = _prev_tp.accessible if _prev_tp is not None else None
            else:
                prior_reference = None
            if gating_mode != "Full depth map" and idx > 0 and _prev_tp is None:
                surf_name = ("accessible surface"
                             if gating_mode == "Inaccessible by prior tool"
                             else "result surface")
                if not messagebox.askyesno(
                        "Previous step not computed",
                        f"Step {idx} has not been computed yet.\n\n"
                        f"Without its {surf_name} this step will process the "
                        f"full depth map.\n\nContinue anyway?"):
                    return

        dlg = ProgressDialog(self.root,
                             title=f"Computing Step {idx + 1} Tool Path")

        if algorithm.startswith("V-Carving"):
            def _worker():
                try:
                    depth_map, n_grey_levels = self._get_depth_map()
                    result = compute_vcarve_toolpath(
                        depth_map       = depth_map,
                        n_grey_levels   = n_grey_levels,
                        tool            = tool,
                        prior_reference = prior_reference,
                        previous_result = _prev_tp.result if _prev_tp is not None else None,
                        max_depth      = max_depth,
                        width_cm       = eff_width_cm,
                        height_cm      = eff_height_cm,
                        origin         = self.origin_var.get(),
                        feed_rate            = feed_rate,
                        threshold_um         = threshold_mm * 1000.0,
                        min_feature_width_mm = min_feature_width_mm,
                        max_feature_width_mm = max_feature_width_mm,
                        stepdown_mm          = vcarve_stepdown_mm,
                        final_stepdown_mm    = vcarve_final_stepdown_mm,
                        skeleton_mode        = ("voronoi"    if algorithm == "V-Carving (Voronoi)"
                                                else "inscribed" if algorithm == "V-Carving (Inscribed)"
                                                else "medial_axis"),
                        use_2opt             = use_2opt,
                        progress             = dlg.report,
                    )
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
                        feed_rate=feed_rate,
                        feed_rapid=self._settings['feed_rapid'],
                        z_safe=self._settings['z_safe'],
                    )
                    dlg.close("Completed.")
                    step['info_var'].set(
                        f"Step {idx + 1}: {stats['motion_count']} motion lines\n"
                        f"Est. time: {_fmt_time(stats['est_minutes'])}"
                        f" @ {feed_rate:.0f} mm/min")
                    self.display_mode_var.set(DISPLAY_STEP_3D)
                    self.display_step_var.set(idx + 1)
                    self._on_display_mode_changed()

                self.root.after(0, _done)

            threading.Thread(target=_worker, daemon=True).start()

        elif algorithm == "Contour":
            # ── Contour (waterline) ─────────────────────────────────────
            if idx > 0 and _prev_tp is None:
                if not messagebox.askyesno(
                        "Previous step not computed",
                        f"Step {idx} has not been computed yet.\n\n"
                        "Without its result this step will treat the entire "
                        "depth map as unfinished.\n\nContinue anyway?"):
                    return

            def _worker():
                try:
                    depth_map, n_grey_levels = self._get_depth_map()
                    result = compute_contour_toolpath(
                        depth_map        = depth_map,
                        n_grey_levels    = n_grey_levels,
                        tool             = tool,
                        stepdown_mm      = contour_stepdown,
                        max_depth        = max_depth,
                        width_cm         = eff_width_cm,
                        height_cm        = eff_height_cm,
                        origin           = self.origin_var.get(),
                        feed_rate        = feed_rate,
                        prior_accessible = (_prev_tp.accessible
                                            if _prev_tp is not None else None),
                        previous_result  = _prev_tp.result if _prev_tp is not None else None,
                        progress         = dlg.report,
                        adaptive_levels  = contour_adaptive,
                    )
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
                        feed_rate=feed_rate,
                        feed_rapid=self._settings['feed_rapid'],
                        z_safe=self._settings['z_safe'],
                    )
                    dlg.close("Completed.")
                    step['info_var'].set(
                        f"Step {idx + 1}: {stats['motion_count']} motion lines\n"
                        f"Est. time: {_fmt_time(stats['est_minutes'])}"
                        f" @ {feed_rate:.0f} mm/min")
                    self.display_mode_var.set(DISPLAY_STEP_3D)
                    self.display_step_var.set(idx + 1)
                    self._on_display_mode_changed()

                self.root.after(0, _done)

            threading.Thread(target=_worker, daemon=True).start()

        else:
            # ── Raster scan ─────────────────────────────────────────────
            work_surface_tol = threshold_mm / max_depth if max_depth else 0.0

            step_mm = tool.diameter_mm * stepover / 100.0

            def _worker():
                try:
                    depth_map, n_grey_levels = self._get_depth_map()
                    result = compute_raster_toolpath(
                        depth_map         = depth_map,
                        n_grey_levels     = n_grey_levels,
                        tool              = tool,
                        step_mm           = step_mm,
                        stepdown_mm       = stepdown,
                        final_stepdown_mm = raster_final_stepdown_mm,
                        max_depth         = max_depth,
                        width_cm          = eff_width_cm,
                        height_cm         = eff_height_cm,
                        raster_dir        = step['raster_dir_var'].get(),
                        origin            = self.origin_var.get(),
                        work_surface_tol  = work_surface_tol,
                        previous_result   = _prev_tp.result if _prev_tp is not None else None,
                        prior_reference   = prior_reference,
                        raster_offset_pct = raster_offset_pct,
                        feed_rate         = feed_rate,
                        use_2opt          = use_2opt,
                        progress          = dlg.report,
                    )
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
                        feed_rate=feed_rate,
                        feed_rapid=self._settings['feed_rapid'],
                        z_safe=self._settings['z_safe'],
                    )
                    dlg.close("Completed.")
                    step['info_var'].set(
                        f"Step {idx + 1}: {stats['motion_count']} motion lines\n"
                        f"Est. time: {_fmt_time(stats['est_minutes'])}"
                        f" @ {feed_rate:.0f} mm/min")
                    self.display_mode_var.set(DISPLAY_STEP_3D)
                    self.display_step_var.set(idx + 1)
                    self._on_display_mode_changed()

                self.root.after(0, _done)

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

    def _export_from_step(self, step):
        idx = self.steps.index(step)
        if self.toolpaths[idx] is None:
            messagebox.showinfo("Not computed",
                                f"Compute Step {idx + 1} first.")
            return
        self._export_toolpaths(precheck_indices=[idx])

    def _export_toolpaths(self, precheck_indices=None):
        steps_info = self._build_steps_info()
        dlg = ExportToolpathDialog(self.root, steps_info,
                                   precheck_indices=precheck_indices)
        indices = dlg.selected_indices
        if not indices:
            return

        z_safe     = self._settings['z_safe']
        feed_rapid = self._settings['feed_rapid']
        preamble   = (self._settings['preamble']
                      or DEFAULT_PREAMBLE.format(z_safe=z_safe))
        postamble  = (self._settings['postamble']
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
                tp = self.toolpaths[idx]
                feed_rate = tp.metadata.get('feed_rate', 800.0)
                bodies.append(tp.to_gcode(feed_rate, feed_rapid, z_safe))

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
        """Sync display_step_var to whichever step tab is now selected."""
        tab_id = self._steps_nb.select()
        if not tab_id or not hasattr(self, 'display_mode_var'):
            return
        selected = self._steps_nb.index(tab_id)
        self.display_step_var.set(selected + 1)
        self._on_display_mode_changed()

    def _on_display_mode_changed(self):
        mode = self.display_mode_var.get()
        idx  = self.display_step_var.get() - 1

        # Update combobox to show only modes relevant to this step's algorithm/state
        has_step       = 0 <= idx < len(self.steps)
        has_accessible = self._accessible_for_step(idx) is not None
        step_computed  = (0 <= idx < len(self.toolpaths) and self.toolpaths[idx] is not None)
        available_modes = [
            m for m in DISPLAY_MODES
            if not (m == DISPLAY_PROC_MASK  and not has_step)
            and not (m == DISPLAY_STEP_UNACCESSIBLE and not has_accessible)
            and not (m in (DISPLAY_STEP_RES, DISPLAY_STEP_RESULT, DISPLAY_STEP_3D) and not step_computed)
        ]
        for m in DISPLAY_MODES:
            state = "normal" if m in available_modes else "disabled"
            self._view_menu.entryconfig(self._display_menu_indices[m], state=state)
        if mode not in available_modes:
            self.display_mode_var.set(DISPLAY_ORIGINAL)
            mode = DISPLAY_ORIGINAL
        # Reset 3D view state when entering a 3D mode fresh
        if mode in DISPLAY_3D_MODES:
            self._3d_elev = 30.0
            self._3d_azim = -60.0
            self._3d_zoom = 1.0
            self._3d_pan_offset = np.array([0.0, 0.0, 0.0])
            # Ensure at least the path is visible — nothing to see without it
            if not self.show_cut_moves_var.get() and not self.show_rapids_var.get():
                self.show_cut_moves_var.set(True)
                self.show_rapids_var.set(True)
        self._render()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _pad_image(self, img):
        """Return *img* padded with the stock margin, or *img* unchanged when margin is 0."""
        try:
            margin_cm = float(self.margin_cm.get())
        except ValueError:
            return img
        if margin_cm <= 0.0:
            return img
        try:
            H, W = img.shape
            width_cm  = float(self.width_cm.get())
            height_cm = float(self.height_cm.get())
            pad_x = max(1, round(margin_cm * W / width_cm))
            pad_y = max(1, round(margin_cm * H / height_cm))
            depth_pct = float(self.margin_depth_pct.get()) / 100.0
            if self.surface.get() == "white":
                pad_val = np.float32(1.0 - depth_pct)  # white = z=0; deeper cut → darker
            else:
                pad_val = np.float32(depth_pct)         # black = z=0; deeper cut → brighter
        except (ValueError, ZeroDivisionError):
            return img
        return np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)),
                      mode="constant", constant_values=pad_val)

    def _get_depth_map(self):
        """Return ``(depth_map, n_grey_levels)``.

        depth_map   : 2-D float32 [0, 1]  — 1.0 = maximum cut depth (intaglio).
        n_grey_levels: number of distinct depth steps the source image can encode.
                      current_image is stored as float32 so we use 65536 (16-bit
                      equivalent) as a conservative estimate.
        """
        img = self._pad_image(np.clip(self.current_image, 0.0, 1.0))
        n_grey_levels = self.n_grey_levels
        if self.surface.get() == "white":
            return 1.0 - img, n_grey_levels   # white = z=0, black = deepest cut
        else:
            return img, n_grey_levels          # black = z=0, white = deepest cut

    def _is_3d_mode(self):
        return self.display_mode_var.get() in DISPLAY_3D_MODES

    def _accessible_for_step(self, idx):
        """Return the achievable depth surface (morphological opening) for step *idx*, or None.

        For a raster/contour step: returns that step's own opening array.
        For a V-carve step (accessible=None): falls back to the prior step's opening.
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

    # ------------------------------------------------------------------ #
    #  3D rendering
    # ------------------------------------------------------------------ #
    def _render_3d(self, toolpath, title):
        """Render the depth-map surface and G-code tool path in 3D."""
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

        z_safe = self._settings['z_safe']
        rapids = []
        cuts   = []
        prev_end = None
        for pts in toolpath.segments:
            if not pts:
                continue
            if prev_end is not None and (
                    abs(prev_end[0] - pts[0][0]) > 1e-4
                    or abs(prev_end[1] - pts[0][1]) > 1e-4):
                px, py, pz = prev_end
                nx, ny, nz = pts[0]
                rapids.append([
                    (px, py, pz),        # current position
                    (px, py, z_safe),    # retract
                    (nx, ny, z_safe),    # traverse at safe height
                    (nx, ny, nz),        # plunge
                ])
            cuts.append(pts)
            prev_end = pts[-1]

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
            width_cm  = float(self.width_cm.get())
            height_cm = float(self.height_cm.get())
            max_depth = float(self.max_depth.get())
        except ValueError:
            width_cm = height_cm = max_depth = 1.0

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
            add_segments(rapids, "#ff0000",   1.0, 1.0, 'Rapids')
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
            margin_cm = float(self.margin_cm.get())
        except ValueError:
            margin_cm = 0.0
        W_mm = (width_cm  + 2.0 * margin_cm) * 10
        H_mm = (height_cm + 2.0 * margin_cm) * 10
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

        # Legend
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
            return

        mode = self.display_mode_var.get()
        step_idx = self.display_step_var.get() - 1

        # ── 3D mode ───────────────────────────────────────────────────
        if mode == DISPLAY_STEP_3D:
            tp = (self.toolpaths[step_idx]
                  if 0 <= step_idx < len(self.toolpaths) else None)
            if tp:
                self._render_3d(tp, f"Step {step_idx + 1} Tool Path")
                return

        # ── 2D modes — ensure we have a flat 2D axes ──────────────────
        if getattr(self.ax, 'name', '') == '3d':
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.fig.patch.set_facecolor("#2b2b2b")
            self.ax.set_facecolor("#2b2b2b")

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
                    px_per_mm_x = W / (float(self.width_cm.get())  * 10.0)
                    px_per_mm_y = H / (float(self.height_cm.get()) * 10.0)
                    px_per_mm   = (px_per_mm_x + px_per_mm_y) / 2.0
                except (ValueError, ZeroDivisionError):
                    px_per_mm = 1.0
                _, mask = compute_vcarve_mask(ref, thr, min_fw_mm, px_per_mm, max_fw_mm)

                img  = mask.astype(np.float32)
                cmap = "Greens"
                title = f"Processing Mask (Step {step_idx + 1})"

        if mode not in (DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
                        DISPLAY_STEP_UNACCESSIBLE, DISPLAY_PROC_MASK):
            img   = self._pad_image(self.current_image)
            cmap  = "gray"
            title = "Original"

        saved_xlim = self._view_xlim
        saved_ylim = self._view_ylim

        self._displayed_array = img
        self.ax.cla()
        self.ax.set_visible(True)
        self.ax.imshow(img, interpolation="none", aspect="equal",
                       cmap=cmap, vmin=0, vmax=1)
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
        """Overlay cut moves and rapids on the current 2D axes."""
        show_cuts   = self.show_cut_moves_var.get()
        show_rapids = self.show_rapids_var.get()
        if not show_cuts and not show_rapids:
            return
        tp = (self.toolpaths[step_idx]
              if 0 <= step_idx < len(self.toolpaths) else None)
        if tp is None or not tp.segments:
            return
        try:
            margin_cm = float(self.margin_cm.get())
            width_mm  = (float(self.width_cm.get())  + 2.0 * margin_cm) * 10.0
            height_mm = (float(self.height_cm.get()) + 2.0 * margin_cm) * 10.0
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

        for pts in tp.segments:
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
                self._pan_origin = (event.xdata, event.ydata,
                                    self.ax.get_xlim(), self.ax.get_ylim())
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
                    w_mm = float(self.width_cm.get())  * 10
                    h_mm = float(self.height_cm.get()) * 10
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
            ox, oy, xlim0, ylim0 = self._pan_origin
            dx = ox - event.xdata
            dy = oy - event.ydata
            new_xlim = (xlim0[0] + dx, xlim0[1] + dx)
            new_ylim = (ylim0[0] + dy, ylim0[1] + dy)
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
            margin_cm = float(self.margin_cm.get())
            width_mm  = (float(self.width_cm.get())  + 2.0 * margin_cm) * 10.0
            height_mm = (float(self.height_cm.get()) + 2.0 * margin_cm) * 10.0
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
