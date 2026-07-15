import tkinter as tk
from tkinter import ttk, messagebox
import queue
import re

from constants import DEFAULT_PREAMBLE, DEFAULT_POSTAMBLE, BLEND_MODES
import stl_mesh


class ProgressDialog(tk.Toplevel):
    """Modal progress dialog with a scrollable message log, a progress bar,
    a Cancel button (active during computation) and a Close button (enabled
    when computation finishes or is cancelled).

    Thread-safe: call ``report(pct, message)`` from any thread.
    The dialog polls an internal queue via ``after()`` so all Tk updates
    happen on the main thread.

    Cancellation
    ------------
    ``report()`` raises ``ProgressDialog.Cancelled`` when the user has clicked
    Cancel.  The worker should let this exception propagate; the caller catches
    it and calls ``close()``.

    Usage
    -----
    dlg = ProgressDialog(parent, title="Computing…")
    try:
        dlg.report(25, "Sampling depth map")   # raises Cancelled if user clicked
        dlg.report(75, "Minimising travel")
    except ProgressDialog.Cancelled:
        dlg.close("Cancelled.")
    else:
        dlg.close("Completed.")
    """

    class Cancelled(Exception):
        pass

    _POLL_MS = 40   # how often (ms) the dialog polls the queue

    def __init__(self, parent, title="Computing…"):
        super().__init__(parent)
        self.title(title)
        self.resizable(True, True)
        self.minsize(460, 320)
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self.grab_set()

        self._q          = queue.Queue()
        self._cancelled  = False   # set by Cancel button, read by worker via report()

        # ── Layout ────────────────────────────────────────────────────
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(outer, text=title,
                  font=("Helvetica", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        # Scrollable message log
        txt_frame = ttk.Frame(outer)
        txt_frame.grid(row=1, column=0, columnspan=2, sticky="nsew",
                       pady=(0, 8))
        txt_frame.columnconfigure(0, weight=1)
        txt_frame.rowconfigure(0, weight=1)
        self._log_text = tk.Text(txt_frame, height=10, wrap=tk.WORD,
                                 state=tk.DISABLED, relief=tk.SUNKEN,
                                 bg="#f7f7f7", font=("Courier", 9))
        sb = ttk.Scrollbar(txt_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        # Progress bar + percentage
        self._bar = ttk.Progressbar(outer, orient=tk.HORIZONTAL,
                                    mode="determinate", maximum=100)
        self._bar.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        self._pct_var = tk.StringVar(value="0 %")
        ttk.Label(outer, textvariable=self._pct_var,
                  font=("Helvetica", 10, "bold"), width=6,
                  anchor=tk.E).grid(row=2, column=1, sticky=tk.E, padx=(6, 0))

        # Buttons
        btn_row = ttk.Frame(outer)
        btn_row.grid(row=3, column=0, columnspan=2, sticky=tk.E, pady=(8, 0))
        self._close_btn = ttk.Button(btn_row, text="Close",
                                     state=tk.DISABLED,
                                     command=self._on_close)
        self._close_btn.pack(side=tk.RIGHT, padx=(6, 0))
        self._cancel_btn = ttk.Button(btn_row, text="Cancel",
                                      command=self._on_cancel)
        self._cancel_btn.pack(side=tk.RIGHT)

        # Centre on parent
        self.update_idletasks()
        pw = (parent.winfo_rootx() + parent.winfo_width()  // 2
              - self.winfo_width()  // 2)
        ph = (parent.winfo_rooty() + parent.winfo_height() // 2
              - self.winfo_height() // 2)
        self.geometry(f"+{pw}+{ph}")

        self._poll()

    # ------------------------------------------------------------------ #
    def report(self, pct: float, message: str = ""):
        """Thread-safe progress update.  Raises ``Cancelled`` if the user
        has clicked Cancel.  ``pct`` is 0–100."""
        if self._cancelled:
            raise ProgressDialog.Cancelled()
        pct = max(0.0, min(100.0, float(pct)))
        self._q.put((pct, message))

    def close(self, final_message: str = "Completed."):
        """Signal that computation has finished.  Appends ``final_message``
        to the log, disables Cancel, and enables Close.  Does NOT destroy
        the dialog — the user does that by clicking Close."""
        self._q.put(None)           # sentinel: computation done
        self._q.put((100.0, final_message))

    # ------------------------------------------------------------------ #
    def _on_cancel(self):
        self._cancelled = True
        self._cancel_btn.config(state=tk.DISABLED)
        self._append("Cancelling…")

    def _on_close(self):
        self.grab_release()
        self.destroy()

    def _on_window_close(self):
        """X button: cancel if running, close if done."""
        if str(self._close_btn["state"]) == tk.NORMAL:
            self._on_close()
        else:
            self._on_cancel()

    def _append(self, text: str):
        """Append a line to the log widget (main thread only)."""
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, text + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _poll(self):
        """Drain the queue; reschedule until done sentinel is received."""
        done = False
        try:
            while True:
                item = self._q.get_nowait()
                if item is None:           # completion sentinel
                    done = True
                    continue
                pct, message = item
                self._bar["value"] = pct
                self._pct_var.set(f"{pct:.0f} %")
                if message:
                    self._append(message)
                if done:
                    self._cancel_btn.config(state=tk.DISABLED)
                    self._close_btn.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        if not done:
            try:
                self.after(self._POLL_MS, self._poll)
            except tk.TclError:
                pass


class LabeledEntry(ttk.Frame):
    """Single-row widget: label on the left, unit tag + entry on the right.

    The entry is a ttk.Spinbox, so the value can be nudged with the up/down
    arrows or the mouse wheel (while hovering over it) in addition to typing.
    ``minval``/``maxval`` bound the value and ``increment`` sets the step size
    for both the arrows and the wheel.

    If auto_label is given, a checkbutton with that text is inserted between
    the unit tag and the label.  When checked, the entry is disabled;
    unchecking re-enables it.  The BooleanVar is exposed as ``auto_var``.
    Layout: [label ← spacer → [☐ auto_label] [unit] [entry]]
    """

    def __init__(self, parent, label, unit, default, width=8, auto_label=None,
                 minval=0.0, maxval=100000.0, increment=1.0, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=label).pack(side=tk.LEFT)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Spinbox(self, textvariable=self.var, width=width,
                                 from_=minval, to=maxval, increment=increment)
        self.entry.pack(side=tk.RIGHT)
        ttk.Label(self, text=unit, foreground="#666666").pack(side=tk.RIGHT, padx=(0, 3))
        self.auto_var = None
        if auto_label is not None:
            self.auto_var = tk.BooleanVar(value=False)
            self.auto_var.trace_add("write", self._on_auto_toggle)
            ttk.Checkbutton(self, text=auto_label,
                            variable=self.auto_var).pack(side=tk.RIGHT, padx=(0, 4))

        # The ttk.Spinbox class bindings already handle the mouse wheel, but
        # they don't stop the event, so it would also bubble up to whatever
        # scrollable container the spinbox lives in (e.g. a scrolling
        # sidebar), scrolling the view underneath the pointer at the same
        # time as nudging the value. Re-dispatch as the same <<Increment>>/
        # <<Decrement>> virtual events the arrows use, then swallow it.
        self.entry.bind("<MouseWheel>", self._on_mousewheel)
        self.entry.bind("<Button-4>", self._on_mousewheel)
        self.entry.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if "disabled" in self.entry.state():
            return None
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            self.entry.event_generate("<<Decrement>>")
        else:
            self.entry.event_generate("<<Increment>>")
        return "break"

    def _on_auto_toggle(self, *_):
        if self.auto_var and self.auto_var.get():
            self.entry.config(state="disabled")
        else:
            self.entry.config(state="normal")

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)

    def set_state(self, state):
        self.entry.config(state=state)


class SectionFrame(ttk.LabelFrame):
    """Labelled group frame with consistent internal padding.

    Pass ``collapsible=True`` to get a clickable disclosure-triangle title
    instead of a plain label; callers must then pack their content into
    ``self.body`` rather than ``self`` directly. Default (``collapsible=False``)
    behaves exactly as before -- ``self.body`` is just an alias for ``self``.
    """

    def __init__(self, parent, title, collapsible=False, start_collapsed=False, **kw):
        if not collapsible:
            super().__init__(parent, text=title, padding=(8, 6), **kw)
            self.body = self
            return

        self._title = title
        self._collapsed = start_collapsed

        # A LabelFrame's title is normally drawn straight from -text, which
        # can't be made clickable. -labelwidget lets us supply our own Label
        # (must be created with `parent`, not `self`, as its master) so the
        # disclosure caption sits properly embedded in the border notch,
        # like every other SectionFrame's title, instead of floating as a
        # separate line of ordinary content inside the frame.
        self._toggle_lbl = ttk.Label(parent, text=self._caption(),
                                     cursor="hand2", font=("Helvetica", 9, "bold"))
        super().__init__(parent, labelwidget=self._toggle_lbl, padding=(8, 6), **kw)
        self._toggle_lbl.bind("<Button-1>", lambda e: self._toggle())

        self.body = ttk.Frame(self)
        if start_collapsed:
            self._collapse()
        else:
            self.body.pack(fill=tk.X)

    def _caption(self):
        # Plain ASCII, not a Unicode triangle glyph (">"/"v") -- some
        # environments' default Tk font substitutes render geometric-shape
        # code points as invisible/blank, which silently hides the
        # disclosure indicator entirely rather than just looking different.
        return f"{'>' if self._collapsed else 'v'} {self._title}"

    def _collapse(self):
        # A plain pack_forget() here is not enough: once a pack master's
        # last remaining child is forgotten, Tk's geometry manager does not
        # reliably recompute the master's (now much smaller) requested
        # size on its own -- the LabelFrame is left occupying exactly the
        # screen space it had before, so nothing below it ever slides up
        # and collapsing buys back no room at all. Forcing the height
        # explicitly (pack_propagate(False) + config(height=...)) sidesteps
        # that and shrinks the frame down to just its title row.
        self.body.pack_forget()
        self.update_idletasks()
        self.pack_propagate(False)
        self.config(height=self._toggle_lbl.winfo_reqheight())

    def _expand(self):
        self.pack_propagate(True)   # let content dictate height again
        self.body.pack(fill=tk.X)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._toggle_lbl.config(text=self._caption())
        if self._collapsed:
            self._collapse()
        else:
            self._expand()


def _fmt_time(minutes):
    """Format a duration in minutes as a human-readable string."""
    if minutes < 1.0:
        return f"{minutes * 60:.0f} sec"
    if minutes < 60.0:
        return f"{minutes:.1f} min"
    h = int(minutes // 60)
    m = minutes % 60
    return f"{h}h {m:.0f}min"


class SettingsDialog(tk.Toplevel):
    """Modal dialog for editing global G-code settings (z_safe, feed_rapid,
    preamble, postamble).  Modifies *settings* dict in-place on OK."""

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(True, True)
        self.minsize(480, 620)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self._settings = settings
        self.confirmed = False

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Machine parameters ────────────────────────────────────────
        mf = SectionFrame(outer, "Machine Parameters")
        mf.pack(fill=tk.X, pady=(0, 8))

        self._z_safe = LabeledEntry(mf, "Safe height (z_safe)", "mm",
                                    str(settings.get('z_safe', 5.0)),
                                    minval=0.0, maxval=200.0, increment=1.0)
        self._z_safe.pack(fill=tk.X, pady=2)

        self._feed_rapid = LabeledEntry(mf, "Rapid feed (time estimate only)", "mm/min",
                                        str(settings.get('feed_rapid', 3000.0)),
                                        minval=1.0, maxval=30000.0, increment=100.0)
        self._feed_rapid.pack(fill=tk.X, pady=2)

        self._accel = LabeledEntry(mf, "Max acceleration (time estimate only)", "mm/s²",
                                   str(settings.get('accel_mm_s2', 300.0)),
                                   minval=1.0, maxval=10000.0, increment=10.0)
        self._accel.pack(fill=tk.X, pady=2)

        # ── G-code preamble ───────────────────────────────────────────
        pre_sf = SectionFrame(outer, "G-code Preamble")
        pre_sf.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        pre_hdr = ttk.Frame(pre_sf)
        pre_hdr.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(pre_hdr, text="Inserted before tool path moves:").pack(side=tk.LEFT)
        ttk.Button(pre_hdr, text="Reset to default",
                   command=self._reset_preamble).pack(side=tk.RIGHT)

        self._pre_text = self._make_text_area(pre_sf)
        pre_init = settings.get('preamble') or self._default_preamble()
        self._pre_text.insert('1.0', pre_init)

        # ── G-code postamble ──────────────────────────────────────────
        post_sf = SectionFrame(outer, "G-code Postamble")
        post_sf.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        post_hdr = ttk.Frame(post_sf)
        post_hdr.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(post_hdr, text="Appended after tool path moves:").pack(side=tk.LEFT)
        ttk.Button(post_hdr, text="Reset to default",
                   command=self._reset_postamble).pack(side=tk.RIGHT)

        self._post_text = self._make_text_area(post_sf, height=4)
        post_init = settings.get('postamble') or self._default_postamble()
        self._post_text.insert('1.0', post_init)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_row, text="OK",     command=self._confirm).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)

        # Centre on parent
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")

        self.wait_window()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_text_area(parent, height=6):
        f = ttk.Frame(parent)
        f.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(f, height=height, wrap=tk.NONE,
                      font=("Courier", 9), relief=tk.SUNKEN)
        sb_y = ttk.Scrollbar(f, orient=tk.VERTICAL,   command=txt.yview)
        sb_x = ttk.Scrollbar(f, orient=tk.HORIZONTAL, command=txt.xview)
        txt.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side=tk.RIGHT,  fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return txt

    def _default_preamble(self):
        try:
            z = float(self._z_safe.get())
        except ValueError:
            z = 5.0
        return DEFAULT_PREAMBLE.format(z_safe=z)

    @staticmethod
    def _default_postamble():
        return DEFAULT_POSTAMBLE

    def _reset_preamble(self):
        self._pre_text.delete('1.0', tk.END)
        self._pre_text.insert('1.0', self._default_preamble())

    def _reset_postamble(self):
        self._post_text.delete('1.0', tk.END)
        self._post_text.insert('1.0', self._default_postamble())

    def _confirm(self):
        try:
            z_safe      = float(self._z_safe.get())
            feed_rapid  = float(self._feed_rapid.get())
            accel_mm_s2 = float(self._accel.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Safe height, rapid feed and max acceleration "
                                 "must be numbers.",
                                 parent=self)
            return

        preamble  = self._pre_text.get('1.0', tk.END).rstrip('\n')
        postamble = self._post_text.get('1.0', tk.END).rstrip('\n')

        # Store '' when text matches the built-in default so to_gcode uses its
        # own default (which stays in sync with z_safe at export time).
        if preamble  == self._default_preamble():
            preamble  = ''
        if postamble == self._default_postamble():
            postamble = ''

        self._settings['z_safe']      = z_safe
        self._settings['feed_rapid']  = feed_rapid
        self._settings['accel_mm_s2'] = accel_mm_s2
        self._settings['preamble']    = preamble
        self._settings['postamble']   = postamble
        self.confirmed = True
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.grab_release()
        self.destroy()


class DressUpDialog(tk.Toplevel):
    """Modal dialog for editing one step's G-code tool path dress-ups
    (ramp-in, back-off, arc-rounded rapid corners). Modifies *settings*
    dict in-place on OK.

    *z_safe* is only used to validate the arc radius against the machine's
    safe height (arc rounding sweeps laterally below z_safe).
    """

    def __init__(self, parent, settings, z_safe, title="Tool Path Dress-Up"):
        super().__init__(parent)
        self.title(title)
        self.resizable(True, True)
        self.minsize(420, 360)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self._settings = settings
        self._z_safe   = z_safe
        self.confirmed = False

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        du_sf = SectionFrame(outer, "Tool Path Dress-Up")
        du_sf.pack(fill=tk.X, pady=(0, 8))

        self._ramp_enabled_var = tk.BooleanVar(value=bool(settings.get('ramp_in_enabled', False)))
        ttk.Checkbutton(du_sf, text="Ramp-in (replace vertical plunges with a zigzag ramp)",
                        variable=self._ramp_enabled_var,
                        command=self._on_ramp_toggle).pack(anchor=tk.W, pady=2)

        self._ramp_angle = LabeledEntry(du_sf, "Ramp angle", "deg",
                                        str(settings.get('ramp_angle_deg', 30.0)),
                                        minval=5.0, maxval=85.0, increment=5)
        self._ramp_angle.pack(fill=tk.X, pady=2)

        self._ramp_length = LabeledEntry(du_sf, "Max ramp length", "mm",
                                         str(settings.get('ramp_max_length_mm', 15.0)),
                                         minval=0.5, maxval=200.0, increment=1.0)
        self._ramp_length.pack(fill=tk.X, pady=2)

        ttk.Separator(du_sf).pack(fill=tk.X, pady=4)

        self._backoff_enabled_var = tk.BooleanVar(value=bool(settings.get('backoff_enabled', False)))
        ttk.Checkbutton(du_sf, text="Back off before retract",
                        variable=self._backoff_enabled_var,
                        command=self._on_backoff_toggle).pack(anchor=tk.W, pady=2)

        self._backoff_dist = LabeledEntry(du_sf, "Back-off distance", "mm",
                                          str(settings.get('backoff_distance_mm', 0.5)),
                                          minval=0.0, maxval=20.0, increment=0.1)
        self._backoff_dist.pack(fill=tk.X, pady=2)

        ttk.Separator(du_sf).pack(fill=tk.X, pady=4)

        self._arc_enabled_var = tk.BooleanVar(value=bool(settings.get('arc_rapids_enabled', False)))
        ttk.Checkbutton(du_sf, text="Arc-round rapid corners",
                        variable=self._arc_enabled_var,
                        command=self._on_arc_toggle).pack(anchor=tk.W, pady=2)

        self._arc_radius = LabeledEntry(du_sf, "Arc radius", "mm",
                                        str(settings.get('arc_radius_mm', 3.0)),
                                        minval=0.0, maxval=50.0, increment=1.0)
        self._arc_radius.pack(fill=tk.X, pady=2)

        self._on_ramp_toggle()
        self._on_backoff_toggle()
        self._on_arc_toggle()

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_row, text="OK",     command=self._confirm).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)

        # Centre on parent
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")

        self.wait_window()

    def _on_ramp_toggle(self):
        state = tk.NORMAL if self._ramp_enabled_var.get() else tk.DISABLED
        self._ramp_angle.set_state(state)
        self._ramp_length.set_state(state)

    def _on_backoff_toggle(self):
        self._backoff_dist.set_state(tk.NORMAL if self._backoff_enabled_var.get() else tk.DISABLED)

    def _on_arc_toggle(self):
        self._arc_radius.set_state(tk.NORMAL if self._arc_enabled_var.get() else tk.DISABLED)

    def _confirm(self):
        try:
            ramp_angle   = float(self._ramp_angle.get())
            ramp_length  = float(self._ramp_length.get())
            backoff_dist = float(self._backoff_dist.get())
            arc_radius   = float(self._arc_radius.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "The dress-up parameters must be numbers.",
                                 parent=self)
            return

        if not (0.0 < ramp_angle < 90.0):
            messagebox.showerror("Invalid input",
                                 "Ramp angle must be between 0 and 90 degrees.",
                                 parent=self)
            return
        if ramp_length <= 0.0:
            messagebox.showerror("Invalid input",
                                 "Max ramp length must be positive.",
                                 parent=self)
            return
        if backoff_dist < 0.0:
            messagebox.showerror("Invalid input",
                                 "Back-off distance cannot be negative.",
                                 parent=self)
            return
        if arc_radius < 0.0:
            messagebox.showerror("Invalid input",
                                 "Arc radius cannot be negative.",
                                 parent=self)
            return
        if arc_radius > self._z_safe:
            messagebox.showerror("Invalid input",
                                 "Arc radius cannot exceed the safe height (z_safe) -- "
                                 "the arc's lateral sweep is only safe above the stock "
                                 "surface, so it can never be larger than the available "
                                 "clearance below z_safe.",
                                 parent=self)
            return

        self._settings['ramp_in_enabled']     = self._ramp_enabled_var.get()
        self._settings['ramp_angle_deg']      = ramp_angle
        self._settings['ramp_max_length_mm']  = ramp_length
        self._settings['backoff_enabled']     = self._backoff_enabled_var.get()
        self._settings['backoff_distance_mm'] = backoff_dist
        self._settings['arc_rapids_enabled']  = self._arc_enabled_var.get()
        self._settings['arc_radius_mm']       = arc_radius
        self.confirmed = True
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.grab_release()
        self.destroy()


class LayerPropertiesDialog(tk.Toplevel):
    """Non-modal popup for editing one layer's placement/blend settings.

    Unlike DressUpDialog, this dialog is NOT modal (no grab_set()/wait_window())
    and has no OK/Cancel batching -- every field is live: each edit is written
    straight onto *layer* and `on_change()` is called immediately so the
    caller can recompute the composite and re-render, the same trace_add-on-
    every-keystroke idiom already used for the Physical Image Size fields.
    Just a single Close button; there is nothing to confirm or discard.
    """

    def __init__(self, parent, layer, on_change, title=None, is_base=False):
        super().__init__(parent)
        self.layer = layer
        self._on_change = on_change
        self._is_base = is_base
        self.title(title or f"Layer Properties — {layer.name}")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        sf = SectionFrame(outer, "Placement")
        sf.pack(fill=tk.X, pady=(0, 8))

        self._ox = LabeledEntry(sf, "Offset X", "mm", str(layer.offset_x_mm),
                                minval=-100000.0, maxval=100000.0, increment=1.0)
        self._ox.pack(fill=tk.X, pady=2)
        self._oy = LabeledEntry(sf, "Offset Y", "mm", str(layer.offset_y_mm),
                                minval=-100000.0, maxval=100000.0, increment=1.0)
        self._oy.pack(fill=tk.X, pady=2)
        self._rot = LabeledEntry(sf, "Rotation", "deg", str(layer.rotation_deg),
                                 minval=-360.0, maxval=360.0, increment=1.0)
        self._rot.pack(fill=tk.X, pady=2)
        self._scale = LabeledEntry(sf, "Scale", "%", str(layer.scale_pct),
                                   minval=1.0, maxval=1000.0, increment=1.0)
        self._scale.pack(fill=tk.X, pady=2)

        self._blend_sf = SectionFrame(outer, "Blend")
        self._blend_sf.pack(fill=tk.X, pady=(0, 8))

        self._blend_row = ttk.Frame(self._blend_sf)
        self._blend_row.pack(fill=tk.X, pady=2)
        ttk.Label(self._blend_row, text="Blend mode:").pack(side=tk.LEFT)
        self._blend_var = tk.StringVar(value=layer.blend_mode)
        self._blend_widget = None
        self._build_blend_widget()

        self._opacity = LabeledEntry(self._blend_sf, "Opacity", "%", str(layer.opacity_pct),
                                     minval=0.0, maxval=100.0, increment=5.0)
        self._opacity.pack(fill=tk.X, pady=2)

        self._depth_scale = LabeledEntry(self._blend_sf, "Depth scale", "%",
                                         str(layer.depth_scale_pct),
                                         minval=0.0, maxval=1000.0, increment=5.0)
        self._depth_scale.pack(fill=tk.X, pady=2)

        self._invert_var = tk.BooleanVar(value=layer.invert)
        ttk.Checkbutton(self._blend_sf, text="Invert (flip this layer's polarity)",
                        variable=self._invert_var).pack(anchor=tk.W, pady=2)

        self._visible_var = tk.BooleanVar(value=layer.visible)
        ttk.Checkbutton(self._blend_sf, text="Visible",
                        variable=self._visible_var).pack(anchor=tk.W, pady=2)

        for var in (self._ox.var, self._oy.var, self._rot.var, self._scale.var,
                    self._opacity.var, self._depth_scale.var, self._blend_var):
            var.trace_add("write", self._apply)
        self._invert_var.trace_add("write", self._apply)
        self._visible_var.trace_add("write", self._apply)

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_row, text="Close", command=self.destroy).pack(side=tk.RIGHT)

        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")

    def _build_blend_widget(self):
        """(Re)build the blend-mode row's right-hand widget: a Combobox for
        a normal layer, or a static label for the base layer, whose blend
        mode is always forced to Replace at composite time regardless of
        what's stored (see layers.composite) -- so it isn't user-editable."""
        if self._blend_widget is not None:
            self._blend_widget.destroy()
        if self._is_base:
            self._blend_widget = ttk.Label(self._blend_row, text="Replace (base layer)",
                                           foreground="#666666")
        else:
            self._blend_widget = ttk.Combobox(self._blend_row, textvariable=self._blend_var,
                                              values=BLEND_MODES, state="readonly", width=14)
        self._blend_widget.pack(side=tk.RIGHT)

    def retarget(self, layer, title=None, is_base=False):
        """Point this already-open dialog at a different layer, so selecting
        a new row in the Layers list re-uses the one popup instead of
        stacking a second window."""
        self.layer = layer
        self.title(title or f"Layer Properties — {layer.name}")
        self._ox.set(str(layer.offset_x_mm))
        self._oy.set(str(layer.offset_y_mm))
        self._rot.set(str(layer.rotation_deg))
        self._scale.set(str(layer.scale_pct))
        self._opacity.set(str(layer.opacity_pct))
        self._depth_scale.set(str(layer.depth_scale_pct))
        self._blend_var.set(layer.blend_mode)
        if is_base != self._is_base:
            self._is_base = is_base
            self._build_blend_widget()
        self._invert_var.set(layer.invert)
        self._visible_var.set(layer.visible)

    def _apply(self, *_):
        try:
            self.layer.offset_x_mm    = float(self._ox.get())
            self.layer.offset_y_mm    = float(self._oy.get())
            self.layer.rotation_deg   = float(self._rot.get())
            self.layer.scale_pct      = float(self._scale.get())
            self.layer.opacity_pct    = float(self._opacity.get())
            self.layer.depth_scale_pct = float(self._depth_scale.get())
        except ValueError:
            return   # mid-edit invalid text (e.g. "-" while typing "-5") -- skip silently
        self.layer.blend_mode = self._blend_var.get()   # no-op when is_base (no Combobox to edit it)
        self.layer.invert     = self._invert_var.get()
        self.layer.visible    = self._visible_var.get()
        self._on_change()


class CanvasSetupDialog(tk.Toplevel):
    """Modal dialog shown when (re)establishing the base/canvas-defining
    layer (File -> Open Image, or "+ Image..." when no layer exists yet).
    Lets the user pick the canvas's pixel resolution independent of the
    loaded image's own native size (defaulting to that native size, so the
    common case is unchanged) and a background depth for any canvas area no
    layer ends up covering. Mirrors ImportSTLDialog's "Output Resolution"
    section, since STL import already offers the equivalent width/height
    choice for its own case.

    After ``wait_window()`` returns, read:
    - ``self.confirmed``                     -- bool
    - ``self.width_px``, ``self.height_px``  -- int
    - ``self.background_depth_pct``          -- float
    """

    def __init__(self, parent, native_width_px, native_height_px,
                 default_background_depth_pct=0.0):
        super().__init__(parent)
        self.title("Canvas Setup")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self.confirmed = False
        self._native_w = native_width_px
        self._native_h = native_height_px
        self._updating = False

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        res_sf = SectionFrame(outer, "Canvas Resolution")
        res_sf.pack(fill=tk.X, pady=(0, 8))

        self.width_var  = tk.StringVar(value=str(native_width_px))
        self.height_var = tk.StringVar(value=str(native_height_px))
        self.keep_aspect_var = tk.BooleanVar(value=True)

        row_w = ttk.Frame(res_sf)
        row_w.pack(fill=tk.X, pady=2)
        ttk.Label(row_w, text="Width", width=8, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(row_w, from_=2, to=20000, textvariable=self.width_var,
                    width=8).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(row_w, text="px", foreground="#666666").pack(side=tk.LEFT)

        row_h = ttk.Frame(res_sf)
        row_h.pack(fill=tk.X, pady=2)
        ttk.Label(row_h, text="Height", width=8, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(row_h, from_=2, to=20000, textvariable=self.height_var,
                    width=8).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(row_h, text="px", foreground="#666666").pack(side=tk.LEFT)

        ttk.Checkbutton(res_sf, text="Keep aspect ratio",
                        variable=self.keep_aspect_var).pack(anchor=tk.W, pady=2)

        self.width_var.trace_add("write", self._on_width_changed)
        self.height_var.trace_add("write", self._on_height_changed)

        bg_sf = SectionFrame(outer, "Background")
        bg_sf.pack(fill=tk.X, pady=(0, 8))
        self._bg_depth = LabeledEntry(bg_sf, "Background depth", "%",
                                      str(default_background_depth_pct),
                                      minval=0.0, maxval=100.0, increment=5.0)
        self._bg_depth.pack(fill=tk.X, pady=2)

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="OK", command=self._confirm).pack(
            side=tk.RIGHT, padx=(0, 6))

        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")

        self.wait_window()

    def _on_width_changed(self, *_):
        if self._updating or not self.keep_aspect_var.get():
            return
        try:
            w = int(self.width_var.get())
        except (tk.TclError, ValueError):
            return
        self._updating = True
        try:
            self.height_var.set(str(max(2, round(w * self._native_h / self._native_w))))
        finally:
            self._updating = False

    def _on_height_changed(self, *_):
        if self._updating or not self.keep_aspect_var.get():
            return
        try:
            h = int(self.height_var.get())
        except (tk.TclError, ValueError):
            return
        self._updating = True
        try:
            self.width_var.set(str(max(2, round(h * self._native_w / self._native_h))))
        finally:
            self._updating = False

    def _confirm(self):
        try:
            width_px  = int(self.width_var.get())
            height_px = int(self.height_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid Input",
                                 "Width and height must be integers.", parent=self)
            return
        if width_px < 2 or height_px < 2:
            messagebox.showerror("Invalid Input",
                                 "Width and height must each be at least 2.",
                                 parent=self)
            return
        try:
            background_depth_pct = float(self._bg_depth.get())
        except ValueError:
            background_depth_pct = 0.0
        self.width_px, self.height_px = width_px, height_px
        self.background_depth_pct = background_depth_pct
        self.confirmed = True
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.confirmed = False
        self.grab_release()
        self.destroy()


class ExportToolpathDialog(tk.Toplevel):
    """Modal dialog for selecting which computed steps to include in a single
    exported G-code file.

    After ``wait_window`` returns, read:
    - ``self.selected_indices`` — list of int step indices, or ``None`` if cancelled
    - ``self.suggested_stem``   — suggested filename stem (no extension)
    """

    def __init__(self, parent, steps_info, precheck_indices=None):
        """
        steps_info      — list of dicts, one per step:
                          'label' (str), 'computed' (bool),
                          'profile' (str), 'diameter_mm' (float)
        precheck_indices — list of indices to pre-check, or None to pre-check
                           all computed steps
        """
        super().__init__(parent)
        self.title("Export Toolpath")
        self.resizable(True, True)
        self.minsize(360, 300)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self.selected_indices = None
        self.suggested_stem   = ""
        self._steps_info      = steps_info

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Steps checkboxes ─────────────────────────────────────────────
        sf = SectionFrame(outer, "Steps")
        sf.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self._check_vars = []
        for i, info in enumerate(steps_info):
            var = tk.BooleanVar()
            if info['computed']:
                checked = (precheck_indices is None) or (i in precheck_indices)
                var.set(checked)
                state = tk.NORMAL
            else:
                var.set(False)
                state = tk.DISABLED
            var.trace_add("write", self._update_filename)
            self._check_vars.append(var)
            ttk.Checkbutton(sf, text=info['label'], variable=var,
                            state=state).pack(anchor=tk.W, pady=1)

        # ── Suggested filename ────────────────────────────────────────────
        self._filename_var = tk.StringVar()
        ttk.Label(outer, textvariable=self._filename_var,
                  foreground="#555555").pack(anchor=tk.W, pady=(0, 8))

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="OK",     command=self._confirm).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)

        # Centre on parent
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")

        self._update_filename()
        self.wait_window()

    # ------------------------------------------------------------------ #
    def _update_filename(self, *_):
        seen = []
        seen_set = set()
        for var, info in zip(self._check_vars, self._steps_info):
            if var.get() and info['computed']:
                key = (info['profile'], info['diameter_mm'])
                if key not in seen_set:
                    seen_set.add(key)
                    seen.append(key)

        if not seen:
            self._filename_var.set("Suggested filename: (none selected)")
            self.suggested_stem = ""
            return

        tokens = []
        for profile, diam in seen:
            diam_str = f"{diam:.4g}"   # drops trailing zeros: 6.0→6, 3.175→3.175
            token = f"{profile}_{diam_str}mm".replace(" ", "_")
            token = re.sub(r"[^\w.\-]", "_", token)
            tokens.append(token)

        stem = "_".join(tokens)
        self.suggested_stem = stem
        self._filename_var.set(f"Suggested filename: {stem}.nc")

    def _confirm(self):
        indices = [i for i, v in enumerate(self._check_vars) if v.get()]
        if not indices:
            messagebox.showwarning("No steps selected",
                                   "Select at least one step to export.",
                                   parent=self)
            return
        self.selected_indices = indices
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.selected_indices = None
        self.grab_release()
        self.destroy()


class ImportSTLDialog(tk.Toplevel):
    """Modal dialog for configuring an STL import: rasterization resolution,
    orientation (a quarter-turn rotation around each of the X/Y/Z axes,
    applied to the mesh before rasterization), and whether to auto-populate
    bm2gcode's physical-size fields from the mesh's bounding box.

    *triangles*/*xy_bounds*/*z_bounds* describe the mesh at rotation
    (0, 0, 0). Rotation-dependent values (aspect ratio, physical-size
    preview) are recomputed live via ``stl_mesh.rotate_bounds`` as the
    rotation dropdowns change, so they never drift from what actually gets
    applied on Import.

    After ``wait_window()`` returns, read:
    - ``self.confirmed``                        -- bool
    - ``self.width_px``, ``self.height_px``      -- int
    - ``self.rot_x``, ``self.rot_y``, ``self.rot_z`` -- int, one of 0/90/180/270
    - ``self.populate_physical``                 -- bool
    """

    _ROT_VALUES = ["0", "90", "180", "270"]

    def __init__(self, parent, triangles, xy_bounds, z_bounds, is_base=True):
        super().__init__(parent)
        self.title("Import STL")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self.confirmed = False
        self._triangles  = triangles
        self._xy_bounds  = xy_bounds
        self._z_bounds   = z_bounds
        self._updating   = False
        self._is_base    = is_base
        self.background_depth_pct = 0.0

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Orientation ──────────────────────────────────────────────
        rot_sf = SectionFrame(outer, "Orientation")
        rot_sf.pack(fill=tk.X, pady=(0, 8))

        self.rot_x_var = tk.StringVar(value="0")
        self.rot_y_var = tk.StringVar(value="0")
        self.rot_z_var = tk.StringVar(value="0")
        for label, var in (("Rotate around X axis", self.rot_x_var),
                           ("Rotate around Y axis", self.rot_y_var),
                           ("Rotate around Z axis", self.rot_z_var)):
            row = ttk.Frame(rot_sf)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=20, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Combobox(row, textvariable=var, values=self._ROT_VALUES,
                        state="readonly", width=6).pack(side=tk.RIGHT)

        self.rot_x_var.trace_add("write", self._on_rotation_changed)
        self.rot_y_var.trace_add("write", self._on_rotation_changed)
        self.rot_z_var.trace_add("write", self._on_rotation_changed)

        # ── Output Resolution ───────────────────────────────────────
        res_sf = SectionFrame(outer, "Output Resolution")
        res_sf.pack(fill=tk.X, pady=(0, 8))

        default_w, default_h = self._default_dims()
        self.width_var  = tk.StringVar(value=str(default_w))
        self.height_var = tk.StringVar(value=str(default_h))
        self.keep_aspect_var = tk.BooleanVar(value=True)

        row_w = ttk.Frame(res_sf)
        row_w.pack(fill=tk.X, pady=2)
        ttk.Label(row_w, text="Width", width=8, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(row_w, from_=2, to=9999, textvariable=self.width_var,
                    width=8).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(row_w, text="px", foreground="#666666").pack(side=tk.LEFT)

        row_h = ttk.Frame(res_sf)
        row_h.pack(fill=tk.X, pady=2)
        ttk.Label(row_h, text="Height", width=8, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(row_h, from_=2, to=9999, textvariable=self.height_var,
                    width=8).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(row_h, text="px", foreground="#666666").pack(side=tk.LEFT)

        ttk.Checkbutton(res_sf, text="Keep aspect ratio",
                        variable=self.keep_aspect_var).pack(anchor=tk.W, pady=2)

        self.width_var.trace_add("write", self._on_width_changed)
        self.height_var.trace_add("write", self._on_height_changed)

        # ── Physical Size (only relevant when establishing the base/canvas-
        #    defining layer -- appending an STL layer onto an existing
        #    canvas never touches Physical Image Size, since resizing it to
        #    fit a newly-appended decorative layer would silently move
        #    every other layer's placement relative to the stock) ────────
        self.populate_var = tk.BooleanVar(value=self._is_base)
        self._preview_var = tk.StringVar()
        if self._is_base:
            phys_sf = SectionFrame(outer, "Physical Size")
            phys_sf.pack(fill=tk.X, pady=(0, 8))

            ttk.Checkbutton(phys_sf,
                            text="Set width/height/max depth from mesh bounding box",
                            variable=self.populate_var).pack(anchor=tk.W)

            ttk.Label(phys_sf, textvariable=self._preview_var,
                      foreground="#555555").pack(anchor=tk.W, pady=(4, 0))

        # ── Background depth (only relevant when establishing the base/
        #    canvas-defining layer -- appending an STL layer onto an
        #    existing canvas never touches the background) ─────────────
        self._bg_depth = None
        if self._is_base:
            bg_sf = SectionFrame(outer, "Background")
            bg_sf.pack(fill=tk.X, pady=(0, 8))
            self._bg_depth = LabeledEntry(bg_sf, "Background depth", "%", "0.0",
                                          minval=0.0, maxval=100.0, increment=5.0)
            self._bg_depth.pack(fill=tk.X, pady=2)

        # ── Buttons ──────────────────────────────────────────────────
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Import", command=self._on_import).pack(
            side=tk.RIGHT, padx=(0, 6))

        self._update_preview()

        # Centre on parent
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")

        self.wait_window()

    # ------------------------------------------------------------------ #
    def _current_rotation(self):
        try:
            return (int(self.rot_x_var.get()), int(self.rot_y_var.get()),
                    int(self.rot_z_var.get()))
        except (tk.TclError, ValueError):
            return (0, 0, 0)

    def _rotated_bounds(self):
        rot_x, rot_y, rot_z = self._current_rotation()
        return stl_mesh.rotate_bounds(self._xy_bounds, self._z_bounds,
                                      rot_x, rot_y, rot_z)

    def _default_dims(self):
        (x_min, x_max, y_min, y_max), _ = self._rotated_bounds()
        x_range = x_max - x_min
        y_range = y_max - y_min
        if y_range < 1e-12:
            return 1024, 1024
        ar = x_range / y_range
        if ar >= 1.0:
            return 1024, max(2, round(1024 / ar))
        return max(2, round(1024 * ar)), 1024

    def _update_preview(self):
        (x_min, x_max, y_min, y_max), (z_min, z_max) = self._rotated_bounds()
        width_mm, height_mm, max_depth_mm = stl_mesh.physical_size_from_bounds(
            x_max - x_min, y_max - y_min, z_max - z_min)
        self._preview_var.set(
            f"→ {width_mm:.2f} mm × {height_mm:.2f} mm × {max_depth_mm:.2f} mm")

    def _on_rotation_changed(self, *_):
        self._update_preview()
        if self.keep_aspect_var.get():
            try:
                w = int(self.width_var.get())
            except (tk.TclError, ValueError):
                return
            (x_min, x_max, y_min, y_max), _ = self._rotated_bounds()
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range < 1e-12:
                return
            self._updating = True
            try:
                self.height_var.set(str(max(2, round(w * y_range / x_range))))
            finally:
                self._updating = False

    def _on_width_changed(self, *_):
        if self._updating or not self.keep_aspect_var.get():
            return
        try:
            w = int(self.width_var.get())
        except (tk.TclError, ValueError):
            return
        (x_min, x_max, y_min, y_max), _ = self._rotated_bounds()
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < 1e-12:
            return
        self._updating = True
        try:
            self.height_var.set(str(max(2, round(w * y_range / x_range))))
        finally:
            self._updating = False

    def _on_height_changed(self, *_):
        if self._updating or not self.keep_aspect_var.get():
            return
        try:
            h = int(self.height_var.get())
        except (tk.TclError, ValueError):
            return
        (x_min, x_max, y_min, y_max), _ = self._rotated_bounds()
        x_range = x_max - x_min
        y_range = y_max - y_min
        if y_range < 1e-12:
            return
        self._updating = True
        try:
            self.width_var.set(str(max(2, round(h * x_range / y_range))))
        finally:
            self._updating = False

    def _on_import(self):
        try:
            width_px  = int(self.width_var.get())
            height_px = int(self.height_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid Input",
                                 "Width and height must be integers.", parent=self)
            return
        if width_px < 2 or height_px < 2:
            messagebox.showerror("Invalid Input",
                                 "Width and height must each be at least 2.",
                                 parent=self)
            return
        self.width_px, self.height_px = width_px, height_px
        self.rot_x, self.rot_y, self.rot_z = self._current_rotation()
        self.populate_physical = self.populate_var.get()
        if self._bg_depth is not None:
            try:
                self.background_depth_pct = float(self._bg_depth.get())
            except ValueError:
                self.background_depth_pct = 0.0
        self.confirmed = True
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.confirmed = False
        self.grab_release()
        self.destroy()
