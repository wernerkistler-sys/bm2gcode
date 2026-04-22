import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading

from constants import (
    TOOL_PROFILES, RASTER_DIRECTIONS, ALGORITHMS, DEFAULTS,
    DISPLAY_ORIGINAL, DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
    DISPLAY_STEP_3D, DISPLAY_FINE_DETAIL, DISPLAY_VCARVE_MASK,
    DISPLAY_MODES, DISPLAY_3D_MODES, DISPLAY_STEP_MODES,
)
from widgets import ProgressDialog, LabeledEntry, SectionFrame, _fmt_time
from algorithms import compute_raster_toolpath, compute_vcarve_toolpath


class GCodeGeneratorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("CNC Relief Carving — Parameter Setup")
        self.root.geometry("1200x780")
        self.root.minsize(900, 560)

        self.current_image   = None   # 2-D float32 [0,1] grayscale array
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

        self._build_menu()
        self._build_layout()

    # ------------------------------------------------------------------ #
    #  Menu
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        menubar   = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image…",  accelerator="Ctrl+O",
                              command=self.open_image)
        file_menu.add_command(label="Close Image",  command=self.close_image)
        file_menu.add_separator()
        file_menu.add_command(label="Reset Defaults", command=self.reset_defaults)
        file_menu.add_separator()
        file_menu.add_command(label="Exit",          accelerator="Ctrl+Q",
                              command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Reset Zoom  (double-click)",
                              command=self._reset_zoom_and_render)
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-q>", lambda e: self.root.quit())

    # ------------------------------------------------------------------ #
    #  Top-level layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        # Status bar (pack first so it anchors to bottom reliably)
        self.status_var = tk.StringVar(value="No image loaded.")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2)
                  ).pack(side=tk.BOTTOM, fill=tk.X)

        # Horizontal pane: sidebar | right column
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5,
                              sashrelief=tk.RAISED, bg="#bbbbbb")
        pane.pack(fill=tk.BOTH, expand=True)

        # ── Right column: image + display-mode bar ────────────────────
        # Built first so display_mode_var exists before the sidebar's
        # _add_step() call fires <<NotebookTabChanged>>.
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
        ttk.Checkbutton(sf, text="Keep aspect ratio",
                        variable=self._aspect_lock_var,
                        command=self._on_aspect_lock_toggle
                        ).pack(anchor=tk.W, pady=(0, 2))

        self.max_depth = LabeledEntry(sf, "Max cut depth", "mm", DEFAULTS["max_depth"])
        self.max_depth.pack(fill=tk.X, pady=2)
        self.width_cm.var.trace_add("write", self._on_width_changed)
        self._on_aspect_lock_toggle()   # apply initial locked state

        polarity_row = ttk.Frame(sf)
        polarity_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(polarity_row, text="z=0 colour:").pack(side=tk.LEFT)
        self.surface = tk.StringVar(value=DEFAULTS["surface"])
        ttk.Combobox(polarity_row, textvariable=self.surface,
                     values=["white", "black"],
                     state="readonly", width=6).pack(side=tk.RIGHT)

        origin_row = ttk.Frame(sf)
        origin_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(origin_row, text="G-code origin:").pack(side=tk.LEFT)
        self.origin_var = tk.StringVar(value=DEFAULTS["origin"])
        ttk.Combobox(origin_row, textvariable=self.origin_var,
                     values=["Bottom-Left", "Bottom-Right",
                             "Top-Left", "Top-Right", "Center"],
                     state="readonly", width=14).pack(side=tk.RIGHT)

        # ── 2. Tool path steps (dynamic notebook) ─────────────────────
        self._steps_nb = ttk.Notebook(p)
        self._steps_nb.pack(fill=tk.X, pady=(0, 4))
        self._steps_nb.bind("<<NotebookTabChanged>>", self._on_step_tab_changed)

        ttk.Button(p, text="Add Step",
                   command=self._add_step).pack(fill=tk.X, pady=(0, 8))

        # ── 3. G-code Info ────────────────────────────────────────────
        # Must be created before _add_step() so that info labels can attach.
        ttk.Separator(p, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self._info_frame = ttk.Frame(p)
        self._info_frame.pack(fill=tk.X)

        self._add_step()   # start with one step

    def _add_step(self):
        """Append a new step tab to the notebook."""
        idx = len(self.steps)
        tab = ttk.Frame(self._steps_nb, padding=(8, 6))
        self._steps_nb.add(tab, text=f"Step {idx + 1}")

        step = {'tab': tab}
        self.steps.append(step)
        self.toolpaths.append(None)

        self._build_step_widgets(tab, step)

        # Compute / Save buttons
        bf = ttk.Frame(tab)
        bf.pack(fill=tk.X, pady=(8, 2))
        ttk.Button(bf, text="Compute",
                   command=lambda s=step: self._compute_step(s)
                   ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(bf, text="Save…",
                   command=lambda s=step: self._save_step(s)
                   ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # Remove button (disabled when only one step exists)
        step['remove_btn'] = ttk.Button(
            tab, text="Remove this step",
            command=lambda s=step: self._remove_step(s))
        step['remove_btn'].pack(fill=tk.X, pady=(2, 0))

        # Info label (shown in the G-code Info area below the notebook)
        step['info_var'] = tk.StringVar(value=f"Step {idx + 1}: not computed")
        lbl = ttk.Label(self._info_frame, textvariable=step['info_var'],
                        wraplength=230, justify=tk.LEFT)
        lbl.pack(anchor=tk.W, pady=(0, 4))
        step['info_label'] = lbl

        self._update_remove_buttons()
        self.display_mode_var.set(DISPLAY_ORIGINAL)
        self._steps_nb.select(tab)

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

        # ── Algorithm selector ─────────────────────────────────────────
        ttk.Label(frame, text="Algorithm:").pack(anchor=tk.W)
        algorithm_var = tk.StringVar(value=DEFAULTS["step_algorithm"])
        step['algorithm_var'] = algorithm_var
        ttk.Combobox(frame, textvariable=algorithm_var,
                     values=ALGORITHMS, state="readonly", width=18
                     ).pack(anchor=tk.W, pady=(2, 6))

        # ── Profile ────────────────────────────────────────────────────
        ttk.Label(frame, text="Profile:").pack(anchor=tk.W)
        profile_var = tk.StringVar(value=DEFAULTS["step_profile"])
        step['profile_var'] = profile_var
        profile_cb = ttk.Combobox(frame, textvariable=profile_var,
                                  values=TOOL_PROFILES, state="readonly", width=18)
        profile_cb.pack(anchor=tk.W, pady=(2, 6))
        step['profile_cb'] = profile_cb

        diam = LabeledEntry(frame, "Diameter", "mm", DEFAULTS["step_diameter"])
        diam.pack(fill=tk.X, pady=2)
        step['diameter'] = diam

        # ── Raster-only controls (hidden when V-Carving) ───────────────
        raster_frame = ttk.Frame(frame)
        raster_frame.pack(fill=tk.X)
        step['raster_frame'] = raster_frame

        stepover = LabeledEntry(raster_frame, "Step-over",     "%",  DEFAULTS["step_stepover"])
        stepover.pack(fill=tk.X, pady=2)
        step['stepover'] = stepover

        stepdown = LabeledEntry(raster_frame, "Max step-down", "mm", DEFAULTS["step_stepdown"])
        stepdown.pack(fill=tk.X, pady=2)
        step['stepdown'] = stepdown

        ttk.Label(raster_frame, text="Scan direction:").pack(anchor=tk.W, pady=(6, 2))
        dir_var = tk.StringVar(value=DEFAULTS["step_raster_dir"])
        step['raster_dir_var'] = dir_var
        dir_row = ttk.Frame(raster_frame)
        dir_row.pack(anchor=tk.W)
        for d in RASTER_DIRECTIONS:
            ttk.Radiobutton(dir_row, text=d, variable=dir_var,
                            value=d).pack(side=tk.LEFT, padx=(0, 6))

        # ── V-carving-only controls (hidden when Raster Scan) ─────────
        vcarve_frame = ttk.Frame(frame)
        step['vcarve_frame'] = vcarve_frame
        # (not packed initially — shown on demand by _on_algorithm_changed)

        vbit_angle = LabeledEntry(vcarve_frame, "V-bit angle",      "°",  DEFAULTS["step_vbit_angle"])
        vbit_angle.pack(fill=tk.X, pady=2)
        step['vbit_angle'] = vbit_angle

        max_vcarve = LabeledEntry(vcarve_frame, "Max V-carve depth", "mm", DEFAULTS["step_max_vcarve"])
        max_vcarve.pack(fill=tk.X, pady=2)
        step['max_vcarve'] = max_vcarve

        vcarve_stepdown = LabeledEntry(vcarve_frame, "Max step-down", "mm", DEFAULTS["step_vcarve_stepdown"])
        vcarve_stepdown.pack(fill=tk.X, pady=2)
        step['vcarve_stepdown'] = vcarve_stepdown

        vcarve_final_stepdown = LabeledEntry(vcarve_frame, "Final step-down", "mm", DEFAULTS["step_vcarve_final_stepdown"])
        vcarve_final_stepdown.pack(fill=tk.X, pady=2)
        step['vcarve_final_stepdown'] = vcarve_final_stepdown

        # ── Shared controls ────────────────────────────────────────────
        feed = LabeledEntry(frame, "Feed rate", "mm/min", DEFAULTS["step_feed_rate"])
        feed.pack(fill=tk.X, pady=2)
        step['feed_rate'] = feed

        threshold = LabeledEntry(frame, "Threshold", "mm", DEFAULTS["step_threshold"])
        threshold.pack(fill=tk.X, pady=2)
        step['threshold'] = threshold

        min_fw = LabeledEntry(frame, "Min feature width", "mm", DEFAULTS["step_min_fw"])
        step['min_fw'] = min_fw
        # packed/unpacked by _on_algorithm_changed below

        # ── Wire up algorithm toggle ───────────────────────────────────
        _VCARVE_PROFILES  = ["V-Bit", "Ball"]
        _RASTER_PROFILES  = ["Ball", "End Mill"]

        def _update_vbit_angle_state():
            alg     = algorithm_var.get()
            profile = step['profile_var'].get()
            if alg.startswith("V-Carving") and profile == "Ball":
                step['vbit_angle'].set_state("disabled")
            else:
                step['vbit_angle'].set_state("normal")

        def _on_profile_changed(*_):
            _update_vbit_angle_state()

        profile_var.trace_add("write", _on_profile_changed)

        def _on_algorithm_changed(*_):
            alg = algorithm_var.get()
            if alg.startswith("V-Carving"):
                step['raster_frame'].pack_forget()
                step['vcarve_frame'].pack(fill=tk.X, after=step['diameter'])
                step['min_fw'].pack(fill=tk.X, pady=2, after=step['threshold'])
                step['profile_cb'].config(state="readonly", values=_VCARVE_PROFILES)
                if step['profile_var'].get() not in _VCARVE_PROFILES:
                    step['profile_var'].set("V-Bit")
            else:
                step['vcarve_frame'].pack_forget()
                step['min_fw'].pack_forget()
                step['raster_frame'].pack(fill=tk.X, after=step['diameter'])
                step['profile_cb'].config(state="readonly", values=_RASTER_PROFILES)
                if step['profile_var'].get() not in _RASTER_PROFILES:
                    step['profile_var'].set("Ball")
            _update_vbit_angle_state()
            if hasattr(self, '_display_combo'):
                self._on_display_mode_changed()

        algorithm_var.trace_add("write", _on_algorithm_changed)

    def _reset_step_widgets(self, step):
        """Reset a step's widgets to DEFAULTS."""
        step['algorithm_var'].set(DEFAULTS["step_algorithm"])
        step['profile_var'].set(DEFAULTS["step_profile"])
        step['profile_cb'].config(state="readonly")
        step['diameter'].set(DEFAULTS["step_diameter"])
        step['stepover'].set(DEFAULTS["step_stepover"])
        step['stepdown'].set(DEFAULTS["step_stepdown"])
        step['feed_rate'].set(DEFAULTS["step_feed_rate"])
        step['threshold'].set(DEFAULTS["step_threshold"])
        step['raster_dir_var'].set(DEFAULTS["step_raster_dir"])
        step['vbit_angle'].set(DEFAULTS["step_vbit_angle"])
        step['max_vcarve'].set(DEFAULTS["step_max_vcarve"])
        step['min_fw'].set(DEFAULTS["step_min_fw"])
        step['vcarve_stepdown'].set(DEFAULTS["step_vcarve_stepdown"])
        step['vcarve_final_stepdown'].set(DEFAULTS["step_vcarve_final_stepdown"])

    # ------------------------------------------------------------------ #
    #  Right panel: image canvas + display-mode selector
    # ------------------------------------------------------------------ #
    def _build_right_panel(self, parent):
        # Image canvas (expands to fill available space)
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self._build_image_canvas(canvas_frame)

        # Display-mode selector bar
        mode_bar = ttk.Frame(parent, padding=(8, 4))
        mode_bar.pack(fill=tk.X, side=tk.BOTTOM)

        ttk.Label(mode_bar, text="Display:",
                  font=("Helvetica", 9, "bold")).grid(row=0, column=0,
                                                       padx=(0, 8),
                                                       sticky=tk.W)
        self.display_mode_var = tk.StringVar(value=DISPLAY_ORIGINAL)
        self._vcarve_mask_rb = None  # not used with combobox; kept for compatibility
        self._display_combo = ttk.Combobox(
            mode_bar,
            textvariable=self.display_mode_var,
            values=DISPLAY_MODES,
            state="readonly",
            width=16,
        )
        self._display_combo.grid(row=0, column=1, padx=(0, 8), sticky=tk.W)
        self._display_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_display_mode_changed())

        self.display_step_var = tk.IntVar(value=1)

        ttk.Button(mode_bar, text="Reset View",
                   command=self._reset_zoom_and_render
                   ).grid(row=0, column=2, padx=(12, 0))

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
        self.status_var.set(f"Loaded: {path}  |  {w} x {h} px  |  {levels} gray levels")
        self._render()

    def close_image(self):
        self.current_image = None
        self.image_path    = None
        self.toolpaths     = [None] * len(self.steps)
        for i, s in enumerate(self.steps):
            s['info_var'].set(f"Step {i + 1}: not computed")
        self.display_mode_var.set(DISPLAY_ORIGINAL)
        self._reset_zoom()
        self.ax.cla()
        self.ax.set_visible(False)
        self.mpl_canvas.draw()
        self.placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.status_var.set("No image loaded.")

    def reset_defaults(self):
        self.width_cm.set(DEFAULTS["width_cm"])
        if not self._aspect_lock_var.get():
            self.height_cm.set(DEFAULTS["height_cm"])
        self.max_depth.set(DEFAULTS["max_depth"])
        self.origin_var.set(DEFAULTS["origin"])
        self.surface.set(DEFAULTS["surface"])
        # Remove all steps except the first, then reset its fields
        while len(self.steps) > 1:
            self._remove_step(self.steps[-1])
        self._reset_step_widgets(self.steps[0])

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
            width_cm     = float(self.width_cm.get())
            max_depth    = float(self.max_depth.get())
            if algorithm.startswith("V-Carving"):
                vcarve_profile   = step['profile_var'].get()
                vbit_angle_deg   = float(step['vbit_angle'].get()) if vcarve_profile != "Ball" else 90.0
                max_vcarve_mm    = float(step['max_vcarve'].get())
                min_feature_width_mm = float(step['min_fw'].get())
                diam_mm          = float(step['diameter'].get())
                vcarve_stepdown_mm       = float(step['vcarve_stepdown'].get())
                vcarve_final_stepdown_mm = float(step['vcarve_final_stepdown'].get())
            else:
                diam_mm  = float(step['diameter'].get())
                stepover = float(step['stepover'].get())
                stepdown = float(step['stepdown'].get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 f"Please check the settings for Step {idx + 1}.")
            return

        dlg = ProgressDialog(self.root,
                             title=f"Computing Step {idx + 1} Tool Path")

        if algorithm.startswith("V-Carving"):
            # Retrieve accessible surface and residual from previous raster step
            accessible_prior  = None
            prior_residual_tp = None
            if idx == 0:
                pass   # no prior step — V-carve the full depth map
            elif self.toolpaths[idx - 1] is None:
                if not messagebox.askyesno(
                        "Previous step not computed",
                        f"Step {idx} has not been computed yet.\n\n"
                        "Without its accessible surface this V-carving step "
                        "will treat the entire depth map as fine detail.\n\n"
                        "Continue anyway?"):
                    return
            else:
                accessible_prior  = self.toolpaths[idx - 1].accessible
                prior_residual_tp = self.toolpaths[idx - 1].residual

            def _worker():
                try:
                    depth_map, n_grey_levels = self._get_depth_map()
                    result = compute_vcarve_toolpath(
                        depth_map      = depth_map,
                        n_grey_levels  = n_grey_levels,
                        accessible     = accessible_prior,
                        prior_residual = prior_residual_tp,
                        vbit_angle_deg = vbit_angle_deg,
                        max_vcarve_mm  = max_vcarve_mm,
                        tool_diameter_mm = diam_mm,
                        max_depth      = max_depth,
                        width_cm       = width_cm,
                        height_cm      = float(self.height_cm.get()),
                        origin         = self.origin_var.get(),
                        feed_rate            = feed_rate,
                        threshold_um         = threshold_mm * 1000.0,
                        min_feature_width_mm = min_feature_width_mm,
                        stepdown_mm          = vcarve_stepdown_mm,
                        final_stepdown_mm    = vcarve_final_stepdown_mm,
                        skeleton_mode        = ("voronoi"
                                                if algorithm == "V-Carving (Voronoi)"
                                                else "medial_axis"),
                        profile              = vcarve_profile,
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
                    stats = result.get_stats(feed_rate=feed_rate)
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
            # Prior residual: ones (flat stock) for first step, previous otherwise
            if idx == 0:
                prior_residual = None   # treats entire depth map as unfinished
            elif self.toolpaths[idx - 1] is None:
                if not messagebox.askyesno(
                        "Previous step not computed",
                        f"Step {idx} has not been computed yet.\n\n"
                        "Without its residual this step will treat the entire "
                        "depth map as unfinished.\n\nContinue anyway?"):
                    return
                prior_residual = None
            else:
                prior_residual = self.toolpaths[idx - 1].residual.copy()

            prior_resid_tol = threshold_mm / max_depth if max_depth else 0.0

            step_mm = diam_mm * stepover / 100.0
            profile = step['profile_var'].get()

            def _worker():
                try:
                    depth_map, n_grey_levels = self._get_depth_map()
                    result = compute_raster_toolpath(
                        depth_map       = depth_map,
                        n_grey_levels   = n_grey_levels,
                        radius_mm       = diam_mm / 2.0,
                        profile         = profile,
                        step_mm         = step_mm,
                        stepdown        = stepdown,
                        max_depth       = max_depth,
                        width_cm        = width_cm,
                        height_cm       = float(self.height_cm.get()),
                        raster_dir      = step['raster_dir_var'].get(),
                        origin          = self.origin_var.get(),
                        prior_residual  = prior_residual,
                        prior_resid_tol = prior_resid_tol,
                        feed_rate       = feed_rate,
                        progress        = dlg.report,
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
                    stats = result.get_stats(feed_rate=feed_rate)
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

    def _save_step(self, step):
        idx = self.steps.index(step)
        if self.toolpaths[idx] is None:
            messagebox.showinfo("Not computed",
                                f"Compute Step {idx + 1} first.")
            return
        self._save_toolpath(self.toolpaths[idx], f"step{idx + 1}_toolpath")

    def _save_toolpath(self, toolpath, default_stem):
        path = filedialog.asksaveasfilename(
            title="Save tool path",
            defaultextension=".nc",
            initialfile=default_stem,
            filetypes=[("G-code",       "*.nc *.gcode *.tap"),
                       ("NumPy array",  "*.npy"),
                       ("CSV file",     "*.csv"),
                       ("PNG image",    "*.png")]
        )
        if not path:
            return
        try:
            ext = path.rsplit(".", 1)[-1].lower()
            if ext in ("nc", "gcode", "tap"):
                # Feed rate was stored in metadata when the toolpath was computed
                feed_rate = toolpath.metadata.get('feed_rate', 800.0)
                gcode = toolpath.to_gcode(feed_rate=feed_rate, feed_rapid=3000.0)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(gcode)
            elif ext == "csv":
                np.savetxt(path, toolpath.residual, delimiter=",")
            elif ext == "png":
                import matplotlib
                matplotlib.image.imsave(path, toolpath.residual, cmap="gray")
            else:
                np.save(path, toolpath.residual)
            self.status_var.set(f"Saved: {path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

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
        is_vcarve_step = (
            0 <= idx < len(self.steps)
            and self.steps[idx].get('algorithm_var') is not None
            and self.steps[idx]['algorithm_var'].get().startswith("V-Carving")
        )
        has_accessible = self._accessible_for_step(idx) is not None
        step_computed  = (0 <= idx < len(self.toolpaths) and self.toolpaths[idx] is not None)
        available_modes = [
            m for m in DISPLAY_MODES
            if not (m == DISPLAY_VCARVE_MASK and not is_vcarve_step)
            and not (m == DISPLAY_FINE_DETAIL and not has_accessible)
            and not (m in (DISPLAY_STEP_RES, DISPLAY_STEP_RESULT, DISPLAY_STEP_3D) and not step_computed)
        ]
        self._display_combo['values'] = available_modes
        if mode not in available_modes:
            self.display_mode_var.set(DISPLAY_ORIGINAL)
            mode = DISPLAY_ORIGINAL
        # Reset 3D view state when entering a 3D mode fresh
        if mode in DISPLAY_3D_MODES:
            self._3d_elev = 30.0
            self._3d_azim = -60.0
            self._3d_zoom = 1.0
            self._3d_pan_offset = np.array([0.0, 0.0, 0.0])
        self._render()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _get_depth_map(self):
        """Return ``(depth_map, n_grey_levels)``.

        depth_map   : 2-D float32 [0, 1]  — 1.0 = maximum cut depth (intaglio).
        n_grey_levels: number of distinct depth steps the source image can encode.
                      current_image is stored as float32 so we use 65536 (16-bit
                      equivalent) as a conservative estimate.
        """
        img = np.clip(self.current_image, 0.0, 1.0)
        n_grey_levels = 65536
        if self.surface.get() == "white":
            return 1.0 - img, n_grey_levels   # white = z=0, black = deepest cut
        else:
            return img, n_grey_levels          # black = z=0, white = deepest cut

    def _is_3d_mode(self):
        return self.display_mode_var.get() in DISPLAY_3D_MODES

    def _accessible_for_step(self, idx):
        """Return the accessible surface relevant to step *idx*, or None.

        For a raster step: returns that step's own accessible array.
        For a V-carve step (accessible=None): returns the prior step's accessible.
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
        import matplotlib.colors as mcolors

        z_safe = 5.0  # must match to_gcode default
        rapids = []
        cuts   = []
        links  = []
        prev_end = None
        for typ, pts in toolpath.segments:
            if not pts:
                continue
            if prev_end is not None and prev_end != pts[0]:
                px, py, pz = prev_end
                nx, ny, nz = pts[0]
                rapids.append([
                    (px, py, pz),        # current position
                    (px, py, z_safe),    # retract
                    (nx, ny, z_safe),    # traverse at safe height
                    (nx, ny, nz),        # plunge
                ])
            if typ == 'cut':
                cuts.append(pts)
            elif typ == 'link':
                links.append(pts)
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

        depth_map, _ = self._get_depth_map()
        H, W = depth_map.shape
        try:
            width_cm  = float(self.width_cm.get())
            height_cm = float(self.height_cm.get())
            max_depth = float(self.max_depth.get())
        except ValueError:
            width_cm = height_cm = max_depth = 1.0

        # Subsample depth map for surface mesh (max 80×80 for performance)
        step = max(1, max(H, W) // 80)
        zs   = depth_map[::step, ::step] * max_depth   # mm
        xs   = np.linspace(0, width_cm  * 10, zs.shape[1])  # mm
        ys   = np.linspace(0, height_cm * 10, zs.shape[0])  # mm
        Xm, Ym = np.meshgrid(xs, ys)

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

        add_segments(rapids, "#ff0000",   1.0, 1.0, 'Rapids')
        add_segments(cuts,   "#61b970ff", 1.0, 1.0, 'Cut moves')
        add_segments(links,  "#0026ff",   1.0, 1.0, 'Terrain links')

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

        W_mm = width_cm  * 10
        H_mm = height_cm * 10
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
        half = max(width_cm * 10, height_cm * 10, max_depth) / 2 * self._3d_zoom
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
            if tp is not None:
                img   = tp.residual
                cmap  = "hot"
                title = f"Step {step_idx + 1} Residual"
            else:
                mode  = DISPLAY_ORIGINAL   # fall through to original

        elif mode == DISPLAY_STEP_RESULT:
            tp = (self.toolpaths[step_idx]
                  if 0 <= step_idx < len(self.toolpaths) else None)
            if tp is not None and tp.residual is not None:
                depth_map, _ = self._get_depth_map()
                img   = np.clip(depth_map - tp.residual, 0.0, 1.0).astype(np.float32)
                # Invert so that deeper cuts appear darker when z=0 is white
                cmap  = "gray_r" if self.surface.get() == "white" else "gray"
                title = f"Step {step_idx + 1} Result"
            else:
                mode  = DISPLAY_ORIGINAL   # fall through to original

        elif mode in (DISPLAY_FINE_DETAIL, DISPLAY_VCARVE_MASK):
            accessible = self._accessible_for_step(step_idx)
            depth_map, _ = self._get_depth_map()
            if accessible is None:
                fine_detail = depth_map.copy()
            else:
                fine_detail = np.clip(depth_map - accessible, 0.0, 1.0)
            if mode == DISPLAY_FINE_DETAIL:
                img   = fine_detail
                cmap  = "hot"
                title = f"Fine Detail (Step {step_idx + 1})"
            else:
                try:
                    threshold_mm = float(self.steps[step_idx]['threshold'].get())
                    max_depth    = float(self.max_depth.get())
                except (ValueError, IndexError):
                    threshold_mm, max_depth = 0.0, 5.0
                thr  = threshold_mm / max_depth if max_depth else 0.0
                mask = fine_detail > thr
                # Apply min-feature-width erosion if set (same logic as compute_vcarve_toolpath)
                try:
                    min_fw_mm = float(self.steps[step_idx]['min_fw'].get())
                except (ValueError, IndexError, KeyError):
                    min_fw_mm = 0.0
                if min_fw_mm > 0.0:
                    try:
                        H, W = fine_detail.shape
                        px_per_mm_x = W / (float(self.width_cm.get())  * 10.0)
                        px_per_mm_y = H / (float(self.height_cm.get()) * 10.0)
                        px_per_mm   = (px_per_mm_x + px_per_mm_y) / 2.0
                    except (ValueError, ZeroDivisionError):
                        px_per_mm = 1.0
                    min_half_px = (min_fw_mm / 2.0) * px_per_mm
                    from scipy.ndimage import distance_transform_edt
                    mask = distance_transform_edt(mask) > min_half_px
                img  = mask.astype(np.float32)
                cmap = "Greens"
                title = f"V-carve Mask (Step {step_idx + 1})"

        if mode not in (DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
                        DISPLAY_FINE_DETAIL, DISPLAY_VCARVE_MASK):
            img   = self.current_image
            cmap  = "gray"
            title = ""

        saved_xlim = self._view_xlim
        saved_ylim = self._view_ylim

        self.ax.cla()
        self.ax.set_visible(True)
        self.ax.imshow(img, interpolation="bilinear", aspect="equal",
                       cmap=cmap, vmin=0, vmax=1)
        self.ax.axis("off")
        if title:
            self.ax.set_title(title, color="white", fontsize=10, pad=6)
        self.fig.tight_layout()

        if saved_xlim is not None and saved_ylim is not None:
            self.ax.set_xlim(saved_xlim)
            self.ax.set_ylim(saved_ylim)

        self.mpl_canvas.draw()

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
