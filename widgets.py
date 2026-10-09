import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import copy
import math
import queue
import re

from constants import DEFAULT_PREAMBLE, DEFAULT_POSTAMBLE, TOOL_PROFILES
from tool_library import (ToolLibrary, ToolDef, Material, CuttingData,
                          new_id, format_value)
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
        self._minval = minval
        self._maxval = maxval
        self._label_widget = ttk.Label(self, text=label)
        self._label_widget.pack(side=tk.LEFT)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Spinbox(self, textvariable=self.var, width=width,
                                 from_=minval, to=maxval, increment=increment)
        self.entry.pack(side=tk.RIGHT)
        ttk.Label(self, text=unit, foreground="#666666").pack(side=tk.RIGHT, padx=(0, 3))
        self.auto_var = None
        self._auto_checkbutton = None
        if auto_label is not None:
            self.auto_var = tk.BooleanVar(value=False)
            self.auto_var.trace_add("write", self._on_auto_toggle)
            self._auto_checkbutton = ttk.Checkbutton(self, text=auto_label,
                                                       variable=self.auto_var)
            self._auto_checkbutton.pack(side=tk.RIGHT, padx=(0, 4))

        # The ttk.Spinbox class bindings already handle the mouse wheel, but
        # they don't stop the event, so it would also bubble up to whatever
        # scrollable container the spinbox lives in (e.g. a scrolling
        # sidebar), scrolling the view underneath the pointer at the same
        # time as nudging the value. Re-dispatch as the same <<Increment>>/
        # <<Decrement>> virtual events the arrows use, then swallow it.
        self.entry.bind("<MouseWheel>", self._on_mousewheel)
        self.entry.bind("<Button-4>", self._on_mousewheel)
        self.entry.bind("<Button-5>", self._on_mousewheel)

        # ``self.var`` mirrors every keystroke (it's the Spinbox's own
        # textvariable -- Tk offers no way to decouple what's displayed from
        # it), so callers that want a *reactive* callback (trace_add) rather
        # than a one-off read (get()) should watch ``committed_var`` instead:
        # it only updates once editing is "done" -- Enter, tab/click-away, or
        # an arrow/wheel nudge (which always lands on a complete, valid
        # value, unlike a value built up digit by digit) -- so a callback
        # doesn't get invoked with a truncated in-progress number, e.g. "1"
        # and "10" on the way to typing "100".
        self.committed_var = tk.StringVar(value=self.var.get())
        self.entry.bind("<Return>",   self._commit)
        self.entry.bind("<KP_Enter>", self._commit)
        self.entry.bind("<FocusOut>", self._commit)
        # <<Increment>>/<<Decrement>> fire *before* ttk.Spinbox's own C-level
        # button handler writes its new (unclamped -- from_/to only bound
        # the arrows' own arithmetic, not what's rejected) value into
        # self.var, not after -- so committing synchronously here validates
        # the value the click is about to replace, then gets silently
        # overwritten a moment later by Tk's own out-of-range write. Defer
        # to after_idle so _commit runs once Tk's own update has landed.
        self.entry.bind("<<Increment>>", self._commit_idle, add="+")
        self.entry.bind("<<Decrement>>", self._commit_idle, add="+")

    def _clamped(self, value):
        """Parse *value* as a float and clamp it into [minval, maxval].
        Returns the clamped value as a string, or None if *value* isn't a
        valid number (caller should then leave the field untouched)."""
        try:
            f = float(value)
        except ValueError:
            return None
        f = min(max(f, self._minval), self._maxval)
        return str(int(f)) if f == int(f) else repr(f)

    def _commit(self, *_):
        # The Spinbox's from_/to only clamp the arrow buttons and mouse
        # wheel -- typed text (then Enter/Tab/click-away) bypasses them
        # entirely, so e.g. a Diameter or Step-over field can commit "0"
        # even though downstream toolpath generation divides by it. Re-clamp
        # here so a committed value is always in-range, whether it arrived
        # by typing or by nudging.
        clamped = self._clamped(self.var.get())
        if clamped is None:
            clamped = self.committed_var.get() or str(self._minval)
        self.var.set(clamped)
        self.committed_var.set(clamped)

    def _commit_idle(self, *_):
        self.after_idle(self._commit)

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
        # committed_var, not var: var mirrors the raw Spinbox text, which
        # can transiently hold an out-of-range value -- e.g. between an
        # arrow-key click and the deferred _commit_idle correction (see
        # <<Decrement>> binding above) -- that callers must never see.
        return self.committed_var.get()

    def set(self, value):
        # A programmatic set is already a complete, deliberate value (not a
        # keystroke on the way to one), so it should land immediately --
        # update both vars in lockstep rather than waiting for a commit
        # event that will never come (nothing is focused/being typed into).
        # Still re-clamped like a typed commit: a value loaded from an old
        # or hand-edited project file (e.g. a stray min_px_per_mm: 0) is no
        # more trustworthy than something typed into the box.
        clamped = self._clamped(value)
        if clamped is None:
            clamped = str(value)
        self.var.set(clamped)
        self.committed_var.set(clamped)

    def set_state(self, state):
        self.entry.config(state=state)

    def set_label(self, text):
        self._label_widget.config(text=text)

    def set_auto_visible(self, visible):
        """Show or hide the auto checkbutton (a no-op if this entry has
        none). Re-packing after a forget lands it back in its original
        position -- entry and unit stay packed throughout, so it's always
        the newest side=RIGHT slave, i.e. immediately left of the unit
        label, as at construction."""
        if self._auto_checkbutton is None:
            return
        if visible:
            self._auto_checkbutton.pack(side=tk.RIGHT, padx=(0, 4))
        else:
            self._auto_checkbutton.pack_forget()


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
    acceleration, spindle spin-up dwell, preamble, postamble).  Modifies *settings* dict in-place on OK."""

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

        self._dwell = LabeledEntry(mf, "Spindle spin-up dwell", "s",
                                   str(settings.get('spindle_dwell_s', 3.0)),
                                   minval=0.0, maxval=600.0, increment=1.0)
        self._dwell.pack(fill=tk.X, pady=2)

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
            dwell_s     = float(self._dwell.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Safe height, rapid feed, max acceleration and "
                                 "spindle spin-up dwell must be numbers.",
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
        self._settings['spindle_dwell_s'] = max(0.0, dwell_s)
        self._settings['preamble']    = preamble
        self._settings['postamble']   = postamble
        self.confirmed = True
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.grab_release()
        self.destroy()


class DressUpDialog(tk.Toplevel):
    """Modal dialog for editing one step's tool path dress-ups (ramp-in,
    back-off, arc-rounded rapid corners, and tabs). Modifies *settings*
    dict in-place on OK.

    *z_safe* is only used to validate the arc radius against the machine's
    safe height (arc rounding sweeps laterally below z_safe).

    *algorithm* is the step's current algorithm -- tabs only make sense on a
    Cutout step's closed, full-depth ring, so the Tabs section is greyed out
    (but its settings are still shown/persisted) whenever this isn't
    "Cutout".
    """

    def __init__(self, parent, settings, z_safe, title="Tool Path Dress-Up",
                 algorithm=None):
        super().__init__(parent)
        self.title(title)
        self.resizable(True, True)
        self.minsize(420, 360)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self._settings = settings
        self._z_safe   = z_safe
        self._algorithm = algorithm
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

        self._arc_feed = LabeledEntry(du_sf, "Arc feed", "mm/min",
                                      str(settings.get('arc_feed_mm_min', 1500.0)),
                                      minval=1.0, maxval=30000.0, increment=100.0)
        self._arc_feed.pack(fill=tk.X, pady=2)

        ttk.Separator(du_sf).pack(fill=tk.X, pady=4)

        self._tab_enabled_var = tk.BooleanVar(value=bool(settings.get('tab_enabled', False)))
        self._tab_checkbutton = ttk.Checkbutton(
            du_sf, text="Leave tabs (uncut bridges)",
            variable=self._tab_enabled_var,
            command=self._on_tab_toggle)
        self._tab_checkbutton.pack(anchor=tk.W, pady=2)

        self._tab_count = LabeledEntry(du_sf, "Tab count", "",
                                       str(settings.get('tab_count', 4)),
                                       minval=1, maxval=50, increment=1)
        self._tab_count.pack(fill=tk.X, pady=2)

        self._tab_width = LabeledEntry(du_sf, "Tab width", "mm",
                                       str(settings.get('tab_width_mm', 5.0)),
                                       minval=0.5, maxval=200.0, increment=0.5)
        self._tab_width.pack(fill=tk.X, pady=2)

        self._tab_height = LabeledEntry(
            du_sf, "Tab height", "mm",
            str(settings.get('tab_height_mm', 1.0)),
            minval=0.1, maxval=50.0, increment=0.1)
        self._tab_height.pack(fill=tk.X, pady=2)

        self._tab_phase = LabeledEntry(du_sf, "Phase", "%",
                                       str(settings.get('tab_phase_pct', 0.0)),
                                       minval=0.0, maxval=100.0, increment=1.0)
        self._tab_phase.pack(fill=tk.X, pady=2)

        self._tab_angle = LabeledEntry(du_sf, "Tab angle", "deg",
                                       str(settings.get('tab_angle_deg', 60.0)),
                                       minval=5.0, maxval=90.0, increment=5.0)
        self._tab_angle.pack(fill=tk.X, pady=2)

        self._on_ramp_toggle()
        self._on_backoff_toggle()
        self._on_arc_toggle()
        self._on_tab_toggle()
        if self._algorithm is not None and self._algorithm != "Cutout":
            self._tab_checkbutton.config(state=tk.DISABLED)

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
        state = tk.NORMAL if self._arc_enabled_var.get() else tk.DISABLED
        self._arc_radius.set_state(state)
        self._arc_feed.set_state(state)

    def _on_tab_toggle(self):
        applicable = self._algorithm is None or self._algorithm == "Cutout"
        state = tk.NORMAL if (applicable and self._tab_enabled_var.get()) else tk.DISABLED
        self._tab_count.set_state(state)
        self._tab_width.set_state(state)
        self._tab_height.set_state(state)
        self._tab_phase.set_state(state)
        self._tab_angle.set_state(state)

    def _confirm(self):
        try:
            ramp_angle    = float(self._ramp_angle.get())
            ramp_length   = float(self._ramp_length.get())
            backoff_dist  = float(self._backoff_dist.get())
            arc_radius    = float(self._arc_radius.get())
            arc_feed      = float(self._arc_feed.get())
            tab_count     = int(float(self._tab_count.get()))
            tab_width     = float(self._tab_width.get())
            tab_height    = float(self._tab_height.get())
            tab_phase     = float(self._tab_phase.get())
            tab_angle     = float(self._tab_angle.get())
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
        if arc_feed <= 0.0:
            messagebox.showerror("Invalid input",
                                 "Arc feed must be positive.",
                                 parent=self)
            return
        if tab_count < 1:
            messagebox.showerror("Invalid input",
                                 "Tab count must be at least 1.",
                                 parent=self)
            return
        if tab_width <= 0.0:
            messagebox.showerror("Invalid input",
                                 "Tab width must be positive.",
                                 parent=self)
            return
        if tab_height <= 0.0:
            messagebox.showerror("Invalid input",
                                 "Tab height must be positive.",
                                 parent=self)
            return
        if not (0.0 <= tab_phase < 100.0):
            messagebox.showerror("Invalid input",
                                 "Tab phase must be between 0 and 100 percent.",
                                 parent=self)
            return
        if not (0.0 < tab_angle <= 90.0):
            messagebox.showerror("Invalid input",
                                 "Tab angle must be between 0 and 90 degrees.",
                                 parent=self)
            return

        self._settings['ramp_in_enabled']     = self._ramp_enabled_var.get()
        self._settings['ramp_angle_deg']      = ramp_angle
        self._settings['ramp_max_length_mm']  = ramp_length
        self._settings['backoff_enabled']     = self._backoff_enabled_var.get()
        self._settings['backoff_distance_mm'] = backoff_dist
        self._settings['arc_rapids_enabled']  = self._arc_enabled_var.get()
        self._settings['arc_radius_mm']       = arc_radius
        self._settings['arc_feed_mm_min']     = arc_feed
        self._settings['tab_enabled']    = self._tab_enabled_var.get()
        self._settings['tab_count']      = tab_count
        self._settings['tab_width_mm']   = tab_width
        self._settings['tab_height_mm']  = tab_height
        self._settings['tab_phase_pct']  = tab_phase
        self._settings['tab_angle_deg']  = tab_angle
        self.confirmed = True
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.grab_release()
        self.destroy()


class ToolLibraryDialog(tk.Toplevel):
    """Modal editor for the tool library.

    Edits a deep copy of *library*. After ``wait_window`` returns, read
    ``self.confirmed`` and, if True, ``self.library`` (the edited copy).
    Saving the library file is up to the caller.
    """

    _PREVIEW_W = 120
    _PREVIEW_H = 220

    def __init__(self, parent, library, select_tool_id=None):
        super().__init__(parent)
        self.title("Tool Library")
        self.resizable(True, True)
        self.minsize(720, 560)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self.library   = copy.deepcopy(library)
        self.confirmed = False
        self._tool     = None    # ToolDef shown in the geometry form
        self._loading  = False   # True while forms are filled programmatically

        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(outer)
        nb.pack(fill=tk.BOTH, expand=True)
        tools_tab = ttk.Frame(nb, padding=8)
        mats_tab  = ttk.Frame(nb, padding=8)
        nb.add(tools_tab, text="Tools")
        nb.add(mats_tab,  text="Materials")
        self._build_tools_tab(tools_tab)
        self._build_materials_tab(mats_tab)

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="Import…", command=self._import).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Export…", command=self._export).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btn_row, text="OK",     command=self._confirm).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)

        self._refresh_tool_tree(select_tool_id)
        self._refresh_material_list()

        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")

        self.wait_window()

    # ------------------------------------------------------------------ #
    #  Tools tab
    # ------------------------------------------------------------------ #
    def _build_tools_tab(self, tab):
        top = ttk.Frame(tab)
        top.pack(fill=tk.BOTH, expand=True)

        # ── Tool list ─────────────────────────────────────────────────
        left = ttk.Frame(top)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._tool_tree = ttk.Treeview(left, columns=("num", "name", "profile", "diam"),
                                       show="headings", selectmode="browse", height=8)
        for col, text, width, anchor in (("num", "#", 36, tk.E),
                                         ("name", "Name", 170, tk.W),
                                         ("profile", "Profile", 80, tk.W),
                                         ("diam", "Ø mm", 60, tk.E)):
            self._tool_tree.heading(col, text=text)
            self._tool_tree.column(col, width=width, anchor=anchor,
                                   stretch=(col == "name"))
        self._tool_tree.pack(fill=tk.BOTH, expand=True)
        self._tool_tree.bind("<<TreeviewSelect>>", self._on_tool_selected)

        list_btns = ttk.Frame(left)
        list_btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(list_btns, text="New",       command=self._new_tool).pack(side=tk.LEFT)
        ttk.Button(list_btns, text="Duplicate", command=self._duplicate_tool).pack(side=tk.LEFT, padx=4)
        ttk.Button(list_btns, text="Delete",    command=self._delete_tool).pack(side=tk.LEFT)

        # ── Geometry form ─────────────────────────────────────────────
        geo = SectionFrame(top, "Geometry")
        geo.pack(side=tk.LEFT, fill=tk.BOTH, padx=(8, 0))
        form = ttk.Frame(geo)
        form.pack(side=tk.LEFT, fill=tk.Y)

        self._name_var = tk.StringVar()
        row = ttk.Frame(form)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Name").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._name_var, width=22).pack(side=tk.RIGHT)

        self._number_var = tk.StringVar()
        row = ttk.Frame(form)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Tool number").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._number_var, width=8).pack(side=tk.RIGHT)

        self._profile_var = tk.StringVar()
        row = ttk.Frame(form)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Profile").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self._profile_var, values=TOOL_PROFILES,
                     state="readonly", width=12).pack(side=tk.RIGHT)

        self._diam = LabeledEntry(form, "Diameter", "mm", "6",
                                  minval=0.01, maxval=200.0, increment=0.5)
        self._diam.pack(fill=tk.X, pady=2)
        self._angle = LabeledEntry(form, "V-bit angle", "°", "60",
                                   minval=1.0, maxval=179.0, increment=5.0)
        self._angle.pack(fill=tk.X, pady=2)
        self._flute = LabeledEntry(form, "Flute length", "mm", "0",
                                   minval=0.0, maxval=500.0, increment=1.0)
        self._flute.pack(fill=tk.X, pady=2)
        self._shank = LabeledEntry(form, "Shank Ø (0 = Ø)", "mm", "0",
                                   minval=0.0, maxval=200.0, increment=0.5)
        self._shank.pack(fill=tk.X, pady=2)
        self._length = LabeledEntry(form, "Overall length", "mm", "0",
                                    minval=0.0, maxval=1000.0, increment=1.0)
        self._length.pack(fill=tk.X, pady=2)

        self._notes_var = tk.StringVar()
        row = ttk.Frame(form)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Notes").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._notes_var, width=22).pack(side=tk.RIGHT)

        self._preview = tk.Canvas(geo, width=self._PREVIEW_W, height=self._PREVIEW_H,
                                  background="#2b2b2b", highlightthickness=0)
        self._preview.pack(side=tk.LEFT, padx=(8, 0), anchor=tk.N)

        for var in (self._name_var, self._number_var, self._profile_var, self._notes_var):
            var.trace_add("write", self._on_geometry_changed)
        for le in (self._diam, self._angle, self._flute, self._shank, self._length):
            le.committed_var.trace_add("write", self._on_geometry_changed)

        # ── Cutting data ──────────────────────────────────────────────
        cut = SectionFrame(tab, "Cutting data")
        cut.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self._cut_tree = ttk.Treeview(cut, columns=("mat", "feed", "plunge", "rpm", "sd"),
                                      show="headings", selectmode="browse", height=4)
        for col, text, width in (("mat", "Material", 160), ("feed", "Feed mm/min", 90),
                                 ("plunge", "Plunge mm/min", 100), ("rpm", "RPM", 70),
                                 ("sd", "Step-down mm", 90)):
            self._cut_tree.heading(col, text=text)
            self._cut_tree.column(col, width=width,
                                  anchor=tk.W if col == "mat" else tk.E,
                                  stretch=(col == "mat"))
        self._cut_tree.pack(fill=tk.BOTH, expand=True)
        self._cut_tree.bind("<<TreeviewSelect>>", self._on_cutting_selected)

        cform = ttk.Frame(cut)
        cform.pack(fill=tk.X, pady=(6, 0))
        self._cut_mat_var = tk.StringVar()
        ttk.Label(cform, text="Material").grid(row=0, column=0, sticky=tk.W)
        self._cut_mat_cb = ttk.Combobox(cform, textvariable=self._cut_mat_var,
                                        state="readonly", width=18)
        self._cut_mat_cb.grid(row=0, column=1, sticky=tk.W, padx=(4, 12))
        self._cut_mat_var.trace_add("write", self._on_cutting_material_chosen)
        self._cut_feed   = LabeledEntry(cform, "Feed", "mm/min", "800",
                                        minval=1.0, maxval=30000.0, increment=50.0)
        self._cut_plunge = LabeledEntry(cform, "Plunge (0 = feed)", "mm/min", "0",
                                        minval=0.0, maxval=30000.0, increment=50.0)
        self._cut_rpm    = LabeledEntry(cform, "Spindle (0 = none)", "RPM", "0",
                                        minval=0.0, maxval=100000.0, increment=1000.0)
        self._cut_sd     = LabeledEntry(cform, "Step-down", "mm", "1",
                                        minval=0.0, maxval=50.0, increment=0.1)
        self._cut_feed.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=2)
        self._cut_plunge.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)
        self._cut_rpm.grid(row=1, column=2, sticky=tk.EW, padx=(12, 0), pady=2)
        self._cut_sd.grid(row=2, column=2, sticky=tk.EW, padx=(12, 0), pady=2)
        cbtns = ttk.Frame(cform)
        cbtns.grid(row=0, column=2, sticky=tk.E)
        ttk.Button(cbtns, text="Set", command=self._set_cutting).pack(side=tk.LEFT)
        ttk.Button(cbtns, text="Remove", command=self._remove_cutting).pack(side=tk.LEFT, padx=(4, 0))
        cform.columnconfigure(2, weight=1)

    def _refresh_tool_tree(self, select_id=None):
        tree = self._tool_tree
        tree.delete(*tree.get_children())
        for t in self.library.tools:
            tree.insert("", tk.END, iid=t.id, values=self._tool_row(t))
        if select_id is None and self._tool is not None:
            select_id = self._tool.id
        if select_id is None or not tree.exists(select_id):
            children = tree.get_children()
            select_id = children[0] if children else None
        if select_id is not None:
            tree.selection_set(select_id)
            tree.see(select_id)
        # <<TreeviewSelect>> is only delivered on the next event-loop pass;
        # show the tool now so edits made before then land on the right one.
        self._show_tool(self.library.find_tool(select_id) if select_id else None)

    @staticmethod
    def _tool_row(t):
        return ("" if t.number is None else t.number, t.name, t.profile,
                format_value(t.diameter_mm))

    def _on_tool_selected(self, _event=None):
        sel = self._tool_tree.selection()
        self._show_tool(self.library.find_tool(sel[0]) if sel else None)

    def _show_tool(self, tool):
        self._tool = tool
        self._loading = True
        try:
            if tool is None:
                for var in (self._name_var, self._number_var, self._notes_var):
                    var.set("")
            else:
                self._name_var.set(tool.name)
                self._number_var.set("" if tool.number is None else str(tool.number))
                self._profile_var.set(tool.profile)
                self._diam.set(format_value(tool.diameter_mm))
                self._angle.set(format_value(tool.vbit_angle_deg))
                self._flute.set(format_value(tool.flute_length_mm))
                self._shank.set(format_value(tool.shank_diameter_mm))
                self._length.set(format_value(tool.overall_length_mm))
                self._notes_var.set(tool.notes)
        finally:
            self._loading = False
        self._angle.set_state("normal" if tool is not None and tool.profile == "V-Bit"
                              else "disabled")
        self._refresh_cutting_tree()
        self._draw_preview()

    def _on_geometry_changed(self, *_):
        if self._loading or self._tool is None:
            return
        t = self._tool
        t.name    = self._name_var.get()
        t.profile = self._profile_var.get()
        t.notes   = self._notes_var.get()
        number = self._number_var.get().strip()
        if not number:
            t.number = None
        else:
            try:
                t.number = int(number)
            except ValueError:
                pass
        for le, attr in ((self._diam, 'diameter_mm'), (self._angle, 'vbit_angle_deg'),
                         (self._flute, 'flute_length_mm'),
                         (self._shank, 'shank_diameter_mm'),
                         (self._length, 'overall_length_mm')):
            try:
                setattr(t, attr, float(le.get()))
            except ValueError:
                pass
        self._angle.set_state("normal" if t.profile == "V-Bit" else "disabled")
        self._tool_tree.item(t.id, values=self._tool_row(t))
        self._draw_preview()

    def _draw_preview(self):
        """Side view of the whole tool -- cutting part and shank -- scaled to
        fit the canvas. The cutting part is the flute length measured from
        the tip, never shorter than the tip itself (cone or ball); without a
        flute length a default of the tip plus one radius, and at least one
        diameter, is drawn. Without
        an overall length (or one shorter than the cutting part) only a stub
        of the shank is drawn, open at the top."""
        c = self._preview
        c.delete("all")
        t = self._tool
        if t is None or t.diameter_mm <= 0:
            return
        W, H, pad, label_h = self._PREVIEW_W, self._PREVIEW_H, 12, 14
        R  = t.diameter_mm / 2.0
        Rs = (t.shank_diameter_mm if t.shank_diameter_mm > 0 else t.diameter_mm) / 2.0
        if t.profile == "V-Bit":
            half = math.radians(max(1.0, min(179.0, t.vbit_angle_deg)) / 2.0)
            tip_h = R / math.tan(half)
        elif t.profile == "Ball":
            tip_h = R
        else:
            tip_h = 0.0
        if t.flute_length_mm > 0:
            body_h = max(t.flute_length_mm, tip_h)
        else:
            body_h = max(tip_h + R, 2.0 * R)
        shank_closed = t.overall_length_mm > body_h
        shank_h = (t.overall_length_mm - body_h if shank_closed
                   else max(2.0 * Rs, 0.4 * body_h))
        scale = min((W - 2 * pad) / (2.0 * max(R, Rs)),
                    (H - 2 * pad - label_h) / (body_h + shank_h))
        cx, bottom = W / 2.0, H - pad
        r, th, bh = R * scale, tip_h * scale, body_h * scale
        top = bottom - bh
        if t.profile == "V-Bit":
            pts = [cx - r, top, cx + r, top, cx + r, bottom - th, cx, bottom, cx - r, bottom - th]
        elif t.profile == "Ball":
            n = 16
            arc = [(cx + r * math.cos(math.pi * i / n), bottom - r + r * math.sin(math.pi * i / n))
                   for i in range(n + 1)]
            pts = [cx - r, top, cx + r, top]
            for x, y in arc:
                pts += [x, y]
        else:
            pts = [cx - r, top, cx + r, top, cx + r, bottom, cx - r, bottom]

        line, shank_fill = "#d0d8e0", "#a3abb5"
        rs, s_top = Rs * scale, top - shank_h * scale
        if shank_closed:
            c.create_polygon([cx - rs, s_top, cx + rs, s_top, cx + rs, top, cx - rs, top],
                             fill=shank_fill, outline=line)
        else:
            c.create_rectangle(cx - rs, s_top, cx + rs, top, fill=shank_fill, width=0)
            c.create_line(cx - rs, s_top, cx - rs, top, fill=line)
            c.create_line(cx + rs, s_top, cx + rs, top, fill=line)
            c.create_line(cx - rs, s_top, cx + rs, s_top, fill=line, dash=(2, 2))
        c.create_polygon(pts, fill="#7a8ca3", outline=line)

        label = f"Ø {format_value(t.diameter_mm)} mm"
        if t.overall_length_mm > 0:
            label = f"Ø {format_value(t.diameter_mm)} · L {format_value(t.overall_length_mm)} mm"
        c.create_text(W / 2.0, pad / 2.0 + 2, text=label,
                      fill=line, font=("TkDefaultFont", 8))

    def _new_tool(self):
        t = ToolDef(id=new_id(), name="New tool", profile="End Mill", diameter_mm=6.0)
        self.library.tools.append(t)
        self._refresh_tool_tree(t.id)

    def _duplicate_tool(self):
        if self._tool is None:
            return
        t = copy.deepcopy(self._tool)
        t.id = new_id()
        t.name = f"{t.name} (copy)"
        self.library.tools.append(t)
        self._refresh_tool_tree(t.id)

    def _delete_tool(self):
        if self._tool is None:
            return
        if not messagebox.askyesno(
                "Delete tool",
                f"Delete \"{self._tool.name}\" from the library?\n\n"
                "Steps linked to it keep their values and show \"not in library\".",
                parent=self):
            return
        self.library.tools.remove(self._tool)
        self._tool = None
        self._refresh_tool_tree()

    # ── Cutting data ──────────────────────────────────────────────────
    def _material_name(self, material_id):
        m = self.library.find_material(material_id)
        return m.name if m is not None else f"(unknown: {material_id})"

    def _refresh_cutting_tree(self, select_material_id=None):
        tree = self._cut_tree
        tree.delete(*tree.get_children())
        self._cut_mat_cb.config(values=[m.name for m in self.library.materials])
        if self._tool is None:
            return
        for mid, cd in self._tool.cutting.items():
            tree.insert("", tk.END, iid=mid, values=(
                self._material_name(mid), format_value(cd.feed_rate),
                format_value(cd.plunge_rate), format_value(cd.spindle_rpm),
                format_value(cd.stepdown_mm)))
        if select_material_id is not None and tree.exists(select_material_id):
            tree.selection_set(select_material_id)

    def _on_cutting_selected(self, _event=None):
        sel = self._cut_tree.selection()
        if not sel or self._tool is None:
            return
        cd = self._tool.cutting.get(sel[0])
        if cd is None:
            return
        m = self.library.find_material(sel[0])
        self._loading = True
        try:
            self._cut_mat_var.set(m.name if m is not None else "")
        finally:
            self._loading = False
        self._fill_cutting_form(cd)

    def _fill_cutting_form(self, cd):
        self._cut_feed.set(format_value(cd.feed_rate))
        self._cut_plunge.set(format_value(cd.plunge_rate))
        self._cut_rpm.set(format_value(cd.spindle_rpm))
        self._cut_sd.set(format_value(cd.stepdown_mm))

    def _on_cutting_material_chosen(self, *_):
        """Choosing a material with existing data loads it into the form."""
        if self._loading or self._tool is None:
            return
        mid = self._chosen_material_id()
        if mid is not None and mid in self._tool.cutting:
            self._cut_tree.selection_set(mid)

    def _chosen_material_id(self):
        name = self._cut_mat_var.get()
        for m in self.library.materials:
            if m.name == name:
                return m.id
        return None

    def _set_cutting(self):
        if self._tool is None:
            return
        mid = self._chosen_material_id()
        if mid is None:
            messagebox.showwarning("No material", "Choose a material first.", parent=self)
            return
        try:
            cd = CuttingData(feed_rate=float(self._cut_feed.get()),
                             stepdown_mm=float(self._cut_sd.get()),
                             plunge_rate=float(self._cut_plunge.get()),
                             spindle_rpm=float(self._cut_rpm.get()))
        except ValueError:
            messagebox.showerror("Invalid input", "Cutting data must be numbers.", parent=self)
            return
        self._tool.cutting[mid] = cd
        self._refresh_cutting_tree(mid)

    def _remove_cutting(self):
        sel = self._cut_tree.selection()
        if not sel or self._tool is None:
            return
        self._tool.cutting.pop(sel[0], None)
        self._refresh_cutting_tree()

    # ------------------------------------------------------------------ #
    #  Materials tab
    # ------------------------------------------------------------------ #
    def _build_materials_tab(self, tab):
        self._mat_list = tk.Listbox(tab, height=10, exportselection=False)
        self._mat_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        btns = ttk.Frame(tab)
        btns.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        ttk.Button(btns, text="Add…",    command=self._add_material).pack(fill=tk.X)
        ttk.Button(btns, text="Rename…", command=self._rename_material).pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Delete",  command=self._delete_material).pack(fill=tk.X)

    def _refresh_material_list(self):
        self._mat_list.delete(0, tk.END)
        for m in self.library.materials:
            self._mat_list.insert(tk.END, m.name)
        self._refresh_cutting_tree()

    def _selected_material(self):
        sel = self._mat_list.curselection()
        return self.library.materials[sel[0]] if sel else None

    def _ask_material_name(self, title, initial=""):
        name = simpledialog.askstring(title, "Material name:", initialvalue=initial,
                                      parent=self)
        if name is None:
            return None
        name = name.strip()
        if not name:
            return None
        if any(m.name == name for m in self.library.materials):
            messagebox.showerror("Duplicate name",
                                 f"A material named \"{name}\" already exists.", parent=self)
            return None
        return name

    def _add_material(self):
        name = self._ask_material_name("Add material")
        if name is not None:
            self.library.materials.append(Material(new_id(), name))
            self._refresh_material_list()

    def _rename_material(self):
        m = self._selected_material()
        if m is None:
            return
        name = self._ask_material_name("Rename material", m.name)
        if name is not None:
            m.name = name
            self._refresh_material_list()

    def _delete_material(self):
        m = self._selected_material()
        if m is None:
            return
        users = self.library.tools_using_material(m.id)
        if users:
            if not messagebox.askyesno(
                    "Delete material",
                    f"{len(users)} tool(s) have cutting data for \"{m.name}\".\n"
                    "Delete the material and that cutting data?", parent=self):
                return
        for t in users:
            t.cutting.pop(m.id, None)
        self.library.materials.remove(m)
        self._refresh_material_list()

    # ------------------------------------------------------------------ #
    #  Import / export / close
    # ------------------------------------------------------------------ #
    def _import(self):
        path = filedialog.askopenfilename(
            parent=self, title="Import Tool Library",
            filetypes=[("Tool library", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            other = ToolLibrary.load(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Import error", f"Could not read tool library:\n{exc}",
                                 parent=self)
            return

        def on_conflict(existing, incoming):
            answer = messagebox.askyesnocancel(
                "Tool already in library",
                f"\"{incoming.name}\" is already in the library as \"{existing.name}\".\n\n"
                "Yes: replace it\nNo: keep both\nCancel: skip this tool", parent=self)
            if answer is None:
                return "skip"
            return "replace" if answer else "keep_both"

        n = self.library.merge(other, on_conflict)
        self._tool = None
        self._refresh_tool_tree()
        self._refresh_material_list()
        messagebox.showinfo("Import", f"{n} tool(s) imported.", parent=self)

    def _export(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Export Tool Library", defaultextension=".json",
            filetypes=[("Tool library", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.library.save(path)
        except OSError as exc:
            messagebox.showerror("Export error", f"Could not write tool library:\n{exc}",
                                 parent=self)

    def _confirm(self):
        for t in self.library.tools:
            if not t.name.strip():
                messagebox.showerror("Missing name", "Every tool needs a name.", parent=self)
                self._tool_tree.selection_set(t.id)
                return
        self.confirmed = True
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.grab_release()
        self.destroy()


class ApplyMaterialDialog(tk.Toplevel):
    """Modal dialog asking which linked steps should receive the cutting
    data for a newly chosen material.

    *rows* is a list of ``(label, enabled)`` pairs, one per step; disabled
    rows (e.g. the tool has no data for the material) are shown unchecked.
    After ``wait_window`` returns, ``self.selected_indices`` is the list of
    checked row indices, or None if cancelled.
    """

    def __init__(self, parent, material_name, rows):
        super().__init__(parent)
        self.title("Apply Cutting Data")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()
        self.selected_indices = None

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text=f"Apply cutting data for {material_name} "
                              f"to these steps that use library tools?",
                  wraplength=380).pack(anchor=tk.W, pady=(0, 8))
        self._vars = []
        for label, enabled in rows:
            var = tk.BooleanVar(value=enabled)
            self._vars.append(var)
            ttk.Checkbutton(outer, text=label, variable=var,
                            state=tk.NORMAL if enabled else tk.DISABLED).pack(anchor=tk.W, pady=1)

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btn_row, text="Apply", command=self._confirm).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Don't apply", command=self._cancel).pack(side=tk.RIGHT)

        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")
        self.wait_window()

    def _confirm(self):
        self.selected_indices = [i for i, v in enumerate(self._vars) if v.get()]
        self.grab_release()
        self.destroy()

    def _cancel(self):
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
    """Modal dialog for configuring an STL import: orientation (a quarter-
    turn rotation around each of the X/Y/Z axes, applied to the mesh before
    rasterization) and whether to auto-populate bm2gcode's physical-size
    fields from the mesh's bounding box. Rasterization resolution is not
    chosen here -- it's derived from the canvas's Precision setting and the
    source's own placement once the source exists (see app._stl_raster_dims).

    *triangles*/*xy_bounds*/*z_bounds* describe the mesh at rotation
    (0, 0, 0). Rotation-dependent values (aspect ratio, physical-size
    preview) are recomputed live via ``stl_mesh.rotate_bounds`` as the
    rotation dropdowns change, so they never drift from what actually gets
    applied on Import.

    After ``wait_window()`` returns, read:
    - ``self.confirmed``                             -- bool
    - ``self.rot_x``, ``self.rot_y``, ``self.rot_z``  -- int, one of 0/90/180/270
    - ``self.populate_physical``                      -- bool
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
        self._is_base    = is_base

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

        # ── Physical Size (only relevant when establishing the base/canvas-
        #    defining source -- appending an STL source onto an existing
        #    canvas never touches Physical Image Size, since resizing it to
        #    fit a newly-appended decorative source would silently move
        #    every other source's placement relative to the stock) ───────
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

    def _update_preview(self):
        (x_min, x_max, y_min, y_max), (z_min, z_max) = self._rotated_bounds()
        width_mm, height_mm, max_depth_mm = stl_mesh.physical_size_from_bounds(
            x_max - x_min, y_max - y_min, z_max - z_min)
        self._preview_var.set(
            f"→ {width_mm:.2f} mm × {height_mm:.2f} mm × {max_depth_mm:.2f} mm")

    def _on_rotation_changed(self, *_):
        self._update_preview()

    def _on_import(self):
        self.rot_x, self.rot_y, self.rot_z = self._current_rotation()
        self.populate_physical = self.populate_var.get()
        self.confirmed = True
        self.grab_release()
        self.destroy()

    def _cancel(self):
        self.confirmed = False
        self.grab_release()
        self.destroy()
