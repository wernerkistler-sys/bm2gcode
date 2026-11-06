import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from tkinter import font as tkfont
import copy
import math
import queue
import re

import numpy as np

from constants import DEFAULT_PREAMBLE, DEFAULT_POSTAMBLE, TOOL_PROFILES
from tool import UNDERCUT_FIELDS
from tool_library import (ToolLibrary, ToolDef, Material, CuttingData,
                          new_id, format_value, recommend_cutting,
                          FZ_DIAMETERS_MM, MATERIAL_PRESETS, PLUNGE_FRACTION)
import machine_defaults
import stl_mesh
import tool_profile


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
        if label:   # an empty label would still take up a few pixels
            self._label_widget.pack(side=tk.LEFT)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Spinbox(self, textvariable=self.var, width=width,
                                 from_=minval, to=maxval, increment=increment)
        self.entry.pack(side=tk.RIGHT)
        if unit:
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


class LabeledRangeEntry(ttk.Frame):
    """Single-row widget for a (low, high) pair sharing one unit:
    [label ← spacer → unit [low] – [high]], the unit placed before the
    values as in LabeledEntry.

    Each bound is a label-less LabeledEntry (``lo_entry``/``hi_entry``) of
    the usual entry width, so both clamp to [minval, maxval], nudge with
    arrows/wheel and commit like any LabeledEntry, and [high] lines up
    with the entries of the rows around it. The dash between them fills
    the rest of a ``span``-character Combobox's width, so [low] starts
    where such a combobox does. low < high is not enforced here; callers
    validate the pair when they use it.
    """

    def __init__(self, parent, label, unit, default_lo, default_hi, width=8,
                 minval=0.0, maxval=100.0, increment=1.0, span=22, **kw):
        super().__init__(parent, **kw)
        self._label_widget = ttk.Label(self, text=label)
        self._label_widget.pack(side=tk.LEFT)
        self.hi_entry = LabeledEntry(self, "", "", default_hi, width=width,
                                     minval=minval, maxval=maxval, increment=increment)
        self.hi_entry.pack(side=tk.RIGHT)
        dash = ttk.Label(self, text="–", anchor=tk.CENTER)
        # Measured, not assumed: character widths depend on font and DPI.
        probe = ttk.Combobox(self, width=span)
        gap = probe.winfo_reqwidth() - 2 * self.hi_entry.entry.winfo_reqwidth()
        probe.destroy()
        # The glyph, not the label's requested width: the label's own 1 px
        # border may be clipped by the fixed-width box without harm.
        glyph = tkfont.Font(font=dash.cget("font") or "TkDefaultFont").measure("–")
        if gap >= glyph:
            dash_box = ttk.Frame(self, width=gap, height=dash.winfo_reqheight())
            dash_box.pack_propagate(False)
            dash_box.pack(side=tk.RIGHT, fill=tk.Y)
            dash.destroy()
            dash = ttk.Label(dash_box, text="–", anchor=tk.CENTER)
            dash.pack(fill=tk.BOTH, expand=True)
        else:   # too narrow to line up: keep the dash readable instead
            dash.pack(side=tk.RIGHT, padx=3)
        self.lo_entry = LabeledEntry(self, "", unit, default_lo, width=width,
                                     minval=minval, maxval=maxval, increment=increment)
        self.lo_entry.pack(side=tk.RIGHT)

    def get(self):
        """Committed (low, high) strings."""
        return self.lo_entry.get(), self.hi_entry.get()

    def set(self, lo, hi):
        self.lo_entry.set(lo)
        self.hi_entry.set(hi)

    def set_state(self, state):
        self.lo_entry.set_state(state)
        self.hi_entry.set_state(state)

    def set_label(self, text):
        self._label_widget.config(text=text)


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
    acceleration, spindle spin-up dwell, spindle speed range, max feed rate,
    preamble, postamble).  Modifies *settings* dict in-place on OK.

    *defaults* holds the user-wide default machine settings: Save as Default
    writes the dialog's values to *defaults_path* and into *defaults*, Use
    Defaults fills the dialog with them."""

    def __init__(self, parent, settings, defaults, defaults_path):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(True, True)
        self.minsize(480, 620)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self._settings = settings
        self._defaults = defaults
        self._defaults_path = defaults_path
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

        self._max_rpm = LabeledEntry(mf, "Max spindle speed (0 = no limit)", "RPM",
                                     str(settings.get('spindle_max_rpm', 24000.0)),
                                     minval=0.0, maxval=100000.0, increment=1000.0)
        self._max_rpm.pack(fill=tk.X, pady=2)

        self._min_rpm = LabeledEntry(mf, "Min spindle speed (0 = no limit)", "RPM",
                                     str(settings.get('spindle_min_rpm', 6000.0)),
                                     minval=0.0, maxval=100000.0, increment=1000.0)
        self._min_rpm.pack(fill=tk.X, pady=2)

        self._feed_max = LabeledEntry(mf, "Max feed rate (0 = no limit)", "mm/min",
                                      str(settings.get('feed_max', 5000.0)),
                                      minval=0.0, maxval=100000.0, increment=100.0)
        self._feed_max.pack(fill=tk.X, pady=2)

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
        ttk.Button(btn_row, text="Save as Default",
                   command=self._save_as_default).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Use Defaults",
                   command=self._use_defaults).pack(side=tk.LEFT, padx=(6, 0))

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

    def _read_fields(self):
        """The dialog's values as a settings dict, or None (after an error
        message) when a numeric field isn't a number."""
        try:
            z_safe      = float(self._z_safe.get())
            feed_rapid  = float(self._feed_rapid.get())
            accel_mm_s2 = float(self._accel.get())
            dwell_s     = float(self._dwell.get())
            max_rpm     = float(self._max_rpm.get())
            min_rpm     = float(self._min_rpm.get())
            feed_max    = float(self._feed_max.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Safe height, rapid feed, max acceleration, "
                                 "spindle spin-up dwell, spindle speeds and max "
                                 "feed rate must be numbers.",
                                 parent=self)
            return None

        preamble  = self._pre_text.get('1.0', tk.END).rstrip('\n')
        postamble = self._post_text.get('1.0', tk.END).rstrip('\n')

        # Store '' when text matches the built-in default so to_gcode uses its
        # own default (which stays in sync with z_safe at export time).
        if preamble  == self._default_preamble():
            preamble  = ''
        if postamble == self._default_postamble():
            postamble = ''

        return {
            'z_safe':          z_safe,
            'feed_rapid':      feed_rapid,
            'accel_mm_s2':     accel_mm_s2,
            'spindle_dwell_s': max(0.0, dwell_s),
            'spindle_max_rpm': max(0.0, max_rpm),
            'spindle_min_rpm': max(0.0, min_rpm),
            'feed_max':        max(0.0, feed_max),
            'preamble':        preamble,
            'postamble':       postamble,
        }

    def _save_as_default(self):
        values = self._read_fields()
        if values is None:
            return
        try:
            machine_defaults.save(self._defaults_path, values)
        except OSError as exc:
            messagebox.showerror("Save as Default",
                                 f"The default settings could not be saved:\n{exc}",
                                 parent=self)
            return
        self._defaults.update(values)
        messagebox.showinfo("Save as Default",
                            "These settings are now the defaults for new "
                            "projects.", parent=self)

    def _use_defaults(self):
        d = self._defaults
        self._z_safe.set(str(d['z_safe']))
        self._feed_rapid.set(str(d['feed_rapid']))
        self._accel.set(str(d['accel_mm_s2']))
        self._dwell.set(str(d['spindle_dwell_s']))
        self._max_rpm.set(str(d['spindle_max_rpm']))
        self._min_rpm.set(str(d['spindle_min_rpm']))
        self._feed_max.set(str(d['feed_max']))
        self._pre_text.delete('1.0', tk.END)
        self._pre_text.insert('1.0', d['preamble'] or self._default_preamble())
        self._post_text.delete('1.0', tk.END)
        self._post_text.insert('1.0', d['postamble'] or self._default_postamble())

    def _confirm(self):
        values = self._read_fields()
        if values is None:
            return
        self._settings.update(values)
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
        self._ramp_checkbutton = ttk.Checkbutton(
            du_sf, text="Ramp-in (replace vertical plunges with a zigzag ramp)",
            variable=self._ramp_enabled_var, command=self._on_ramp_toggle)
        self._ramp_checkbutton.pack(anchor=tk.W, pady=2)

        self._ramp_angle = LabeledEntry(du_sf, "Ramp angle", "deg",
                                        str(settings.get('ramp_angle_deg', 30.0)),
                                        minval=5.0, maxval=85.0, increment=5)
        self._ramp_angle.pack(fill=tk.X, pady=2)

        self._ramp_length = LabeledEntry(du_sf, "Max ramp length", "mm",
                                         str(settings.get('ramp_max_length_mm', 15.0)),
                                         minval=0.5, maxval=200.0, increment=1.0)
        self._ramp_length.pack(fill=tk.X, pady=2)

        self._ramp_max_depth = LabeledEntry(du_sf, "Max ramp depth (0 = no limit)", "mm",
                                            str(settings.get('ramp_max_depth_mm', 0.0)),
                                            minval=0.0, maxval=100.0, increment=0.5)
        self._ramp_max_depth.pack(fill=tk.X, pady=2)

        self._ramp_start_z = LabeledEntry(du_sf, "Ramp start Z", "mm",
                                          str(settings.get('ramp_start_z_mm', 0.0)),
                                          minval=-100.0, maxval=100.0, increment=0.5)
        self._ramp_start_z.pack(fill=tk.X, pady=2)

        ttk.Separator(du_sf).pack(fill=tk.X, pady=4)

        self._backoff_enabled_var = tk.BooleanVar(value=bool(settings.get('backoff_enabled', False)))
        self._backoff_checkbutton = ttk.Checkbutton(
            du_sf, text="Back off before retract",
            variable=self._backoff_enabled_var, command=self._on_backoff_toggle)
        self._backoff_checkbutton.pack(anchor=tk.W, pady=2)

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
        if self._sideways_only():
            self._ramp_checkbutton.config(state=tk.DISABLED)
            self._backoff_checkbutton.config(state=tk.DISABLED)
            ttk.Label(du_sf, text="Ramp-in and back-off are not used by Undercut: "
                                  "both move sideways at cutting depth, which "
                                  "under an overhang could hit the part.",
                      foreground="#666666", wraplength=380, justify=tk.LEFT
                      ).pack(fill=tk.X, pady=(4, 0))

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

    def _sideways_only(self):
        return self._algorithm == "Undercut"

    def _on_ramp_toggle(self):
        on = self._ramp_enabled_var.get() and not self._sideways_only()
        state = tk.NORMAL if on else tk.DISABLED
        self._ramp_angle.set_state(state)
        self._ramp_length.set_state(state)
        self._ramp_start_z.set_state(state)
        self._ramp_max_depth.set_state(state)

    def _on_backoff_toggle(self):
        on = self._backoff_enabled_var.get() and not self._sideways_only()
        self._backoff_dist.set_state(tk.NORMAL if on else tk.DISABLED)

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
            ramp_start_z  = float(self._ramp_start_z.get())
            ramp_max_depth = float(self._ramp_max_depth.get())
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
        if ramp_start_z >= self._z_safe:
            messagebox.showerror("Invalid input",
                                 "Ramp start Z must be below the safe height (z_safe).",
                                 parent=self)
            return
        if ramp_max_depth < 0.0:
            messagebox.showerror("Invalid input",
                                 "Max ramp depth cannot be negative.",
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
        self._settings['ramp_start_z_mm']     = ramp_start_z
        self._settings['ramp_max_depth_mm']   = ramp_max_depth
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


class ProfileEditorDialog(tk.Toplevel):
    """Modal editor for a Custom tool profile: the right half of the
    cutter's side silhouette as a chain of cubic Bezier segments (see
    tool_profile for the node format).

    The canvas shows the drawn curve and its handles on the right, the
    mirrored curve greyed on the left, and the filled silhouette of the
    radial height curve the toolpath engine actually uses (the lower
    envelope, dashed orange) behind them. After ``wait_window`` returns,
    ``result`` holds the edited node list, or None when cancelled.
    """

    _CANVAS_W = 460
    _CANVAS_H = 420
    _MARGIN   = 28
    _HIT_PX   = 7
    _GRID_STEPS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

    def __init__(self, parent, nodes=None, title="Edit Profile"):
        super().__init__(parent)
        self.title(title)
        self.resizable(True, True)
        self.minsize(720, 480)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()

        self.result   = None
        self._nodes   = (copy.deepcopy(nodes) if nodes
                         else tool_profile.point_roundover(12.7, 6.35))
        self._undo    = []      # earlier node lists, most recent last
        self._sel     = None    # index of the selected node
        self._drag    = None    # (node index, "p" | "in" | "out") being dragged
        self._pan     = None    # (x, y) of the last pan event
        self._loading = False   # True while node entries are filled programmatically
        self._scale   = 1.0     # canvas px per mm
        self._ox = self._oy = 0.0   # canvas position of r = 0, z = 0

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._canvas = tk.Canvas(left, width=self._CANVAS_W, height=self._CANVAS_H,
                                 background="#2b2b2b", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._info_var = tk.StringVar()
        ttk.Label(left, textvariable=self._info_var, justify=tk.LEFT,
                  wraplength=self._CANVAS_W).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(left, foreground="#666666", justify=tk.LEFT, wraplength=self._CANVAS_W,
                  text="Drag points and handles (Shift+drag a handle for a sharp corner). "
                       "Double-click the curve to add a point, Delete removes the "
                       "selected one. Wheel zooms, right-drag pans, Ctrl+Z undoes."
                  ).pack(anchor=tk.W, pady=(2, 0))

        right = ttk.Frame(outer, padding=(10, 0, 0, 0))
        right.pack(side=tk.LEFT, fill=tk.Y)

        preset = SectionFrame(right, "Start from preset")
        preset.pack(fill=tk.X)
        self._preset_var = tk.StringVar(value=tool_profile.PRESETS[0])
        ttk.Combobox(preset, textvariable=self._preset_var, values=tool_profile.PRESETS,
                     state="readonly", width=20).pack(fill=tk.X, pady=(0, 4))
        self._p_diam   = LabeledEntry(preset, "Diameter", "mm", "12.7",
                                      minval=0.1, maxval=200.0, increment=0.5)
        self._p_height = LabeledEntry(preset, "Height", "mm", "6.35",
                                      minval=0.0, maxval=200.0, increment=0.5)
        self._p_angle  = LabeledEntry(preset, "Angle", "°", "0",
                                      minval=0.0, maxval=179.0, increment=5.0)
        self._p_radius = LabeledEntry(preset, "Corner radius", "mm", "1",
                                      minval=0.0, maxval=100.0, increment=0.5)
        for le in (self._p_diam, self._p_height, self._p_angle, self._p_radius):
            le.pack(fill=tk.X, pady=2)
        ttk.Button(preset, text="Apply preset", command=self._apply_preset
                   ).pack(anchor=tk.E, pady=(4, 0))
        self._preset_var.trace_add("write", self._on_preset_chosen)
        self._on_preset_chosen()

        node = SectionFrame(right, "Selected point")
        node.pack(fill=tk.X, pady=(8, 0))
        self._node_r = LabeledEntry(node, "r (from axis)", "mm", "0",
                                    minval=0.0, maxval=200.0, increment=0.1)
        self._node_z = LabeledEntry(node, "z (above tip)", "mm", "0",
                                    minval=-200.0, maxval=500.0, increment=0.1)
        self._node_r.pack(fill=tk.X, pady=2)
        self._node_z.pack(fill=tk.X, pady=2)
        self._node_r.committed_var.trace_add("write", self._on_node_entry)
        self._node_z.committed_var.trace_add("write", self._on_node_entry)
        self._smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(node, text="Smooth (mirror handles)",
                        variable=self._smooth_var).pack(anchor=tk.W, pady=2)
        row = ttk.Frame(node)
        row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(row, text="Add point", command=self._add_point).pack(side=tk.LEFT)
        ttk.Button(row, text="Delete point", command=self._delete_point
                   ).pack(side=tk.LEFT, padx=(4, 0))

        view = SectionFrame(right, "View")
        view.pack(fill=tk.X, pady=(8, 0))
        self._snap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view, text="Snap to grid", variable=self._snap_var
                        ).pack(anchor=tk.W)
        row = ttk.Frame(view)
        row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(row, text="Fit", command=self._fit).pack(side=tk.LEFT)
        ttk.Button(row, text="Undo", command=self._undo_last).pack(side=tk.LEFT, padx=(4, 0))

        btn_row = ttk.Frame(right)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="OK", command=self._confirm).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)

        c = self._canvas
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_motion)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Double-Button-1>", self._on_double_click)
        for btn in ("2", "3"):
            c.bind(f"<ButtonPress-{btn}>", self._on_pan_start)
            c.bind(f"<B{btn}-Motion>", self._on_pan)
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Button-4>", self._on_wheel)
        c.bind("<Button-5>", self._on_wheel)
        c.bind("<Delete>", lambda _e: self._delete_point())
        c.bind("<BackSpace>", lambda _e: self._delete_point())
        c.bind("<Configure>", lambda _e: self._fit())
        self.bind("<Control-z>", lambda _e: self._undo_last())

        self._select(None)
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{max(0, pw)}+{max(0, ph)}")
        self._fit()
        self.wait_window()

    # ── Coordinates ───────────────────────────────────────────────────
    def _w2c(self, r, z):
        return self._ox + r * self._scale, self._oy - z * self._scale

    def _c2w(self, x, y):
        return (x - self._ox) / self._scale, (self._oy - y) / self._scale

    def _canvas_size(self):
        c = self._canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w <= 1 or h <= 1:
            w, h = int(c['width']), int(c['height'])
        return w, h

    def _grid_step(self):
        for step in self._GRID_STEPS:
            if step * self._scale >= 24:
                return step
        return self._GRID_STEPS[-1]

    def _fit(self):
        pts = [tool_profile.sample_profile(self._nodes)]
        for n in self._nodes:
            pts.append([n['p'], n['in'], n['out']])
        pts = np.vstack(pts)
        r_max = max(float(np.abs(pts[:, 0]).max()), 0.5)
        z_min = min(float(pts[:, 1].min()), 0.0)
        z_max = max(float(pts[:, 1].max()), z_min + 0.5)
        W, H = self._canvas_size()
        m = self._MARGIN
        self._scale = max(1e-3, min((W - 2 * m) / (2.0 * r_max),
                                    (H - 2 * m) / (z_max - z_min)))
        extra = (H - 2 * m) - (z_max - z_min) * self._scale
        self._ox = W / 2.0
        self._oy = H - m - extra / 2.0 + z_min * self._scale
        self._redraw()

    # ── Drawing ───────────────────────────────────────────────────────
    def _redraw(self):
        c = self._canvas
        c.delete("all")
        W, H = self._canvas_size()

        step = self._grid_step()
        r0, z1 = self._c2w(0, 0)
        r1, z0 = self._c2w(W, H)
        k = math.floor(r0 / step)
        while k * step <= r1:
            x, _ = self._w2c(k * step, 0)
            c.create_line(x, 0, x, H, fill="#353535")
            k += 1
        k = math.floor(z0 / step)
        while k * step <= z1:
            _, y = self._w2c(0, k * step)
            c.create_line(0, y, W, y, fill="#353535")
            k += 1
        x_axis, y_tip = self._w2c(0, 0)
        c.create_line(x_axis, 0, x_axis, H, fill="#707070", dash=(4, 3))
        c.create_line(0, y_tip, W, y_tip, fill="#707070")
        c.create_text(W - 4, H - 4, anchor=tk.SE, fill="#808080",
                      text=f"grid {format_value(step)} mm", font=("TkDefaultFont", 8))

        pts = tool_profile.sample_profile(self._nodes)
        rh = tool_profile.radial_height(self._nodes)
        if rh is not None:
            # radial_height measures from the curve's lowest point (where
            # the toolpath engine puts Z = 0); draw it back in curve space.
            r, h = rh
            h = h + float(pts[:, 1].min())
            top = max(float(h.max()), h[0] + 0.25 * float(r[-1]))
            poly = [self._w2c(-r[-1], top)]
            poly += [self._w2c(-ri, hi) for ri, hi in zip(r[::-1], h[::-1])]
            poly += [self._w2c(ri, hi) for ri, hi in zip(r, h)]
            poly.append(self._w2c(r[-1], top))
            c.create_polygon(*[v for p in poly for v in p], fill="#41506a", outline="")
            env = [self._w2c(ri, hi) for ri, hi in zip(r, h)]
            c.create_line(*[v for p in env for v in p], fill="#ff9933", width=2, dash=(5, 3))

        right = [self._w2c(rr, zz) for rr, zz in pts]
        left  = [self._w2c(-rr, zz) for rr, zz in pts]
        c.create_line(*[v for p in left for v in p], fill="#808a96", width=1)
        c.create_line(*[v for p in right for v in p], fill="#ffffff", width=2)

        last = len(self._nodes) - 1
        for i, n in enumerate(self._nodes):
            px, py = self._w2c(*n['p'])
            sel = (i == self._sel)
            for key in ('in', 'out'):
                if (key == 'in' and i == 0) or (key == 'out' and i == last):
                    continue
                hx, hy = self._w2c(*n[key])
                c.create_line(px, py, hx, hy, fill="#8899aa")
                c.create_oval(hx - 4, hy - 4, hx + 4, hy + 4,
                              fill="#ffcc66" if sel else "#9a7a40", outline="")
            c.create_rectangle(px - 4, py - 4, px + 4, py + 4,
                               fill="#ff6644" if sel else "#ffffff", outline="#202020")
        self._update_info(rh)

    def _update_info(self, rh):
        if rh is None:
            self._info_var.set("The profile has no width.")
            return
        r, h = rh
        text = (f"Ø {format_value(2.0 * r[-1])} mm · "
                f"edge height {format_value(float(h.max()))} mm · "
                f"{len(self._nodes)} points")
        issues = tool_profile.validate(self._nodes)
        if issues:
            text += "\n⚠ " + "\n⚠ ".join(issues)
        self._info_var.set(text)

    # ── Editing ───────────────────────────────────────────────────────
    def _push_undo(self):
        self._undo.append(copy.deepcopy(self._nodes))
        del self._undo[:-100]

    def _undo_last(self):
        if not self._undo:
            return
        self._nodes = self._undo.pop()
        if self._sel is not None and self._sel >= len(self._nodes):
            self._sel = None
        self._select(self._sel)
        self._redraw()

    def _select(self, idx):
        self._sel = idx
        self._loading = True
        try:
            if idx is None:
                self._node_r.set("0")
                self._node_z.set("0")
            else:
                r, z = self._nodes[idx]['p']
                self._node_r.set(format_value(r))
                self._node_z.set(format_value(z))
        finally:
            self._loading = False
        self._node_r.set_state("disabled" if idx in (None, 0) else "normal")
        self._node_z.set_state("disabled" if idx is None else "normal")

    def _snap(self, r, z):
        if self._snap_var.get():
            step = self._grid_step()
            r, z = round(r / step) * step, round(z / step) * step
        return r, z

    def _move_anchor(self, i, r, z):
        n = self._nodes[i]
        r = 0.0 if i == 0 else max(0.0, r)
        dr, dz = r - n['p'][0], z - n['p'][1]
        for key in ('p', 'in', 'out'):
            n[key] = [max(0.0, n[key][0] + dr), n[key][1] + dz]

    def _move_handle(self, i, key, r, z, mirror):
        n = self._nodes[i]
        n[key] = [max(0.0, r), z]
        other = 'out' if key == 'in' else 'in'
        if not mirror or (other == 'in' and i == 0) \
                or (other == 'out' and i == len(self._nodes) - 1):
            return
        px, pz = n['p']
        vx, vz = px - n[key][0], pz - n[key][1]
        length = math.hypot(vx, vz)
        other_len = math.hypot(n[other][0] - px, n[other][1] - pz)
        if length < 1e-9 or other_len < 1e-9:
            return
        n[other] = [max(0.0, px + vx / length * other_len), pz + vz / length * other_len]

    def _hit(self, x, y):
        """(node index, key) of the handle or anchor under (x, y), or None.
        Handles win over anchors since they sit on top of them when
        retracted."""
        best, best_d = None, None
        last = len(self._nodes) - 1
        for i, n in enumerate(self._nodes):
            for key in ('in', 'out', 'p'):
                if (key == 'in' and i == 0) or (key == 'out' and i == last):
                    continue
                hx, hy = self._w2c(*n[key])
                d = math.hypot(hx - x, hy - y)
                if d > self._HIT_PX:
                    continue
                d += 1.0 if key == 'p' else 0.0
                if best_d is None or d < best_d:
                    best, best_d = (i, key), d
        return best

    def _on_press(self, event):
        self._canvas.focus_set()
        hit = self._hit(event.x, event.y)
        if hit is None:
            self._drag = None
            self._select(None)
        else:
            self._push_undo()
            self._drag = hit
            self._select(hit[0])
        self._redraw()

    def _on_motion(self, event):
        if self._drag is None:
            return
        i, key = self._drag
        r, z = self._snap(*self._c2w(event.x, event.y))
        if key == 'p':
            self._move_anchor(i, r, z)
            self._select(i)
        else:
            shift = bool(event.state & 0x0001)
            self._move_handle(i, key, r, z, self._smooth_var.get() and not shift)
        self._redraw()

    def _on_release(self, _event):
        self._drag = None

    def _split(self, i, t):
        """Split segment i (node i → node i+1) at parameter t, keeping its
        shape, and select the new node."""
        a, b = self._nodes[i], self._nodes[i + 1]
        p0, p1 = np.asarray(a['p']), np.asarray(a['out'])
        p2, p3 = np.asarray(b['in']), np.asarray(b['p'])
        q0, q1, q2 = p0 + (p1 - p0) * t, p1 + (p2 - p1) * t, p2 + (p3 - p2) * t
        r0, r1 = q0 + (q1 - q0) * t, q1 + (q2 - q1) * t
        mid = r0 + (r1 - r0) * t
        a['out'] = q0.tolist()
        b['in'] = q2.tolist()
        self._nodes.insert(i + 1, tool_profile.make_node(mid, r0, r1))
        self._select(i + 1)

    def _on_double_click(self, event):
        t = np.linspace(0.0, 1.0, 65)
        best = None
        for i in range(len(self._nodes) - 1):
            seg = tool_profile.sample_profile(self._nodes[i:i + 2], per_segment=64)
            xy = np.array([self._w2c(r, z) for r, z in seg])
            d = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
            j = int(np.argmin(d))
            if d[j] <= 1.5 * self._HIT_PX and (best is None or d[j] < best[0]):
                best = (d[j], i, t[j])
        if best is None or best[2] in (0.0, 1.0):
            return
        self._push_undo()
        self._split(best[1], best[2])
        self._redraw()

    def _add_point(self):
        i = self._sel if self._sel is not None else len(self._nodes) - 2
        i = min(i, len(self._nodes) - 2)
        self._push_undo()
        self._split(i, 0.5)
        self._redraw()

    def _delete_point(self):
        i = self._sel
        if i is None or i == 0 or len(self._nodes) <= 2:
            return
        self._push_undo()
        del self._nodes[i]
        self._select(None)
        self._redraw()

    def _on_node_entry(self, *_):
        if self._loading or self._sel is None:
            return
        try:
            r, z = float(self._node_r.get()), float(self._node_z.get())
        except ValueError:
            return
        if [r, z] == self._nodes[self._sel]['p']:
            return
        self._push_undo()
        self._move_anchor(self._sel, r, z)
        self._redraw()

    # ── View ──────────────────────────────────────────────────────────
    def _on_wheel(self, event):
        up = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
        f = 1.2 if up else 1.0 / 1.2
        wr, wz = self._c2w(event.x, event.y)
        self._scale *= f
        self._ox = event.x - wr * self._scale
        self._oy = event.y + wz * self._scale
        self._redraw()
        return "break"

    def _on_pan_start(self, event):
        self._pan = (event.x, event.y)

    def _on_pan(self, event):
        if self._pan is None:
            return
        self._ox += event.x - self._pan[0]
        self._oy += event.y - self._pan[1]
        self._pan = (event.x, event.y)
        self._redraw()

    # ── Presets ───────────────────────────────────────────────────────
    _PRESET_FIELDS = {
        "Point round-over": ("diam", "height", "angle"),
        "Ball":             ("diam",),
        "V-Bit":            ("diam", "angle"),
        "Flat":             ("diam",),
        "Bull nose":        ("diam", "radius"),
    }

    def _on_preset_chosen(self, *_):
        preset = self._preset_var.get()
        fields = self._PRESET_FIELDS.get(preset, ())
        for name, le in (("diam", self._p_diam), ("height", self._p_height),
                         ("angle", self._p_angle), ("radius", self._p_radius)):
            le.set_state("normal" if name in fields else "disabled")
        self._p_angle.set_label("Tip angle" if preset == "Point round-over" else "Angle")
        if preset == "V-Bit" and float(self._p_angle.get()) < 1.0:
            self._p_angle.set("60")

    def _apply_preset(self):
        try:
            d = float(self._p_diam.get())
            height = float(self._p_height.get())
            angle = float(self._p_angle.get())
            radius = float(self._p_radius.get())
        except ValueError:
            return
        preset = self._preset_var.get()
        if preset == "Point round-over":
            nodes = tool_profile.point_roundover(d, height, angle)
        elif preset == "Ball":
            nodes = tool_profile.ball(d)
        elif preset == "V-Bit":
            nodes = tool_profile.vbit(d, max(1.0, angle))
        elif preset == "Flat":
            nodes = tool_profile.flat(d)
        else:
            nodes = tool_profile.bull_nose(d, radius)
        self._push_undo()
        self._nodes = nodes
        self._select(None)
        self._fit()

    # ── Close ─────────────────────────────────────────────────────────
    def _confirm(self):
        if tool_profile.radial_height(self._nodes) is None:
            messagebox.showerror("Profile", "The profile has no width.", parent=self)
            return
        self.result = tool_profile.normalize_nodes(self._nodes)
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

    def __init__(self, parent, library, select_tool_id=None, spindle_max_rpm=0.0,
                 spindle_min_rpm=0.0, feed_max=0.0):
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
        self._machine_limits = (spindle_max_rpm, spindle_min_rpm, feed_max)
        self._material = None    # Material shown in the reference-data form

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
        left = SectionFrame(top, "Tools")
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
        self._edit_profile_btn = ttk.Button(form, text="Edit profile…",
                                            command=self._edit_profile)
        self._edit_profile_btn.pack(anchor=tk.E, pady=2)
        self._flutes = LabeledEntry(form, "Flutes", "", "2",
                                    minval=1.0, maxval=12.0, increment=1.0)
        self._flutes.pack(fill=tk.X, pady=2)
        self._flute = LabeledEntry(form, "Flute length", "mm", "0",
                                   minval=0.0, maxval=500.0, increment=1.0)
        self._flute.pack(fill=tk.X, pady=2)
        self._length = LabeledEntry(form, "Overall length", "mm", "0",
                                    minval=0.0, maxval=1000.0, increment=1.0)
        self._length.pack(fill=tk.X, pady=2)
        self._shank = LabeledEntry(form, "Shank Ø (0 = Ø)", "mm", "0",
                                   minval=0.0, maxval=200.0, increment=0.5)
        self._shank.pack(fill=tk.X, pady=2)
        self._max_rpm = LabeledEntry(form, "Max speed (0 = no limit)", "RPM", "0",
                                     minval=0.0, maxval=100000.0, increment=1000.0)
        self._max_rpm.pack(fill=tk.X, pady=2)
        # Undercut-profile geometry; each field is editable only for the
        # profiles that use it (tool.UNDERCUT_FIELDS). Their shank is the
        # Shank field above.
        self._undercut_entries = {}
        for attr, label, unit, default, lo_, hi_, inc in (
                ('neck_diameter_mm',   "Neck diameter",  "mm", "3",  0.01, 200.0, 0.5),
                ('neck_length_mm',     "Neck length",    "mm", "15", 0.0, 500.0, 1.0),
                ('head_height_mm',     "Head height",    "mm", "2",  0.01, 100.0, 0.5),
                ('dovetail_angle_deg', "Dovetail angle", "°",  "60", 1.0, 89.0, 5.0)):
            le = LabeledEntry(form, label, unit, default,
                              minval=lo_, maxval=hi_, increment=inc)
            le.pack(fill=tk.X, pady=2)
            self._undercut_entries[attr] = le

        self._notes_var = tk.StringVar()
        row = ttk.Frame(form)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Notes").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._notes_var, width=22).pack(side=tk.RIGHT)

        self._preview = tk.Canvas(geo, width=self._PREVIEW_W, height=self._PREVIEW_H,
                                  background="#2b2b2b", highlightthickness=0)
        self._preview.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=(8, 0))
        self._preview.bind("<Configure>", lambda _e: self._draw_preview())

        for var in (self._name_var, self._number_var, self._profile_var, self._notes_var):
            var.trace_add("write", self._on_geometry_changed)
        for le in (self._diam, self._flutes, self._max_rpm, self._angle, self._flute,
                   self._shank, self._length, *self._undercut_entries.values()):
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
        self._cut_stats = ttk.Label(cform, foreground="#666666")
        self._cut_stats.grid(row=1, column=3, sticky=tk.W, padx=(8, 0))
        for le in (self._cut_feed, self._cut_rpm, self._diam, self._flutes):
            le.committed_var.trace_add("write", self._update_cut_stats)
        self._update_cut_stats()
        cbtns = ttk.Frame(cform)
        cbtns.grid(row=0, column=2, columnspan=2, sticky=tk.E)
        ttk.Button(cbtns, text="Set", command=self._set_cutting).pack(side=tk.LEFT)
        ttk.Button(cbtns, text="Remove", command=self._remove_cutting).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(cbtns, text="Recommend…",
                   command=self._recommend_cutting).pack(side=tk.LEFT, padx=(4, 0))
        cform.columnconfigure(3, weight=1)

    def _update_cut_stats(self, *_):
        """Show cutting speed vc = π·D·n / 1000 (m/min) and chip load
        fz = feed / (n · flutes) (mm/tooth) for the feed and spindle speed in
        the cutting-data form and the diameter and flutes in the geometry
        form."""
        try:
            feed   = float(self._cut_feed.get())
            rpm    = float(self._cut_rpm.get())
            diam   = float(self._diam.get())
            flutes = max(1, int(float(self._flutes.get())))
        except ValueError:
            rpm = 0.0
        if rpm <= 0:
            self._cut_stats.config(text="vc –  ·  fz –")
            return
        vc = math.pi * diam * rpm / 1000.0
        fz = feed / (rpm * flutes)
        self._cut_stats.config(text=f"vc {vc:.0f} m/min  ·  fz {fz:.3f} mm")

    def _refresh_tool_tree(self, select_id=None):
        tree = self._tool_tree
        tree.delete(*tree.get_children())
        for t in self.library.sorted_tools():
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
                self._flutes.set(str(tool.flutes))
                self._max_rpm.set(format_value(tool.max_rpm))
                self._angle.set(format_value(tool.vbit_angle_deg))
                self._flute.set(format_value(tool.flute_length_mm))
                self._shank.set(format_value(tool.shank_diameter_mm))
                self._length.set(format_value(tool.overall_length_mm))
                for attr, le in self._undercut_entries.items():
                    le.set(format_value(getattr(tool, attr)))
                self._notes_var.set(tool.notes)
        finally:
            self._loading = False
        self._update_geometry_states()
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
        try:
            t.flutes = max(1, int(float(self._flutes.get())))
        except ValueError:
            pass
        try:
            t.max_rpm = max(0.0, float(self._max_rpm.get()))
        except ValueError:
            pass
        for le, attr in ((self._diam, 'diameter_mm'), (self._angle, 'vbit_angle_deg'),
                         (self._flute, 'flute_length_mm'),
                         (self._shank, 'shank_diameter_mm'),
                         (self._length, 'overall_length_mm'),
                         *((le, attr) for attr, le in self._undercut_entries.items())):
            try:
                setattr(t, attr, float(le.get()))
            except ValueError:
                pass
        if t.profile == "Custom":
            if not t.profile_points:
                t.profile_points = tool_profile.point_roundover(
                    t.diameter_mm, t.diameter_mm / 2.0)
            self._sync_custom_diameter()
        self._update_geometry_states()
        self._tool_tree.item(t.id, values=self._tool_row(t))
        self._tool_tree.move(t.id, "", self.library.sorted_tools().index(t))
        self._tool_tree.see(t.id)
        self._draw_preview()

    def _update_geometry_states(self):
        """Enable the fields that apply to the shown tool's profile: the V
        angle for V-bits, the neck/head fields for undercut profiles; for
        Custom profiles the curve sets the diameter and is edited in the
        profile editor."""
        profile = self._tool.profile if self._tool is not None else None
        self._angle.set_state("normal" if profile == "V-Bit" else "disabled")
        self._diam.set_state("disabled" if profile == "Custom" else "normal")
        self._edit_profile_btn.config(state="normal" if profile == "Custom" else "disabled")
        used = {attr for attr, _ in UNDERCUT_FIELDS.get(profile, ())}
        for attr, le in self._undercut_entries.items():
            le.set_state("normal" if attr in used else "disabled")
        # An undercut cutter's shank only matters when wider than its neck.
        self._shank.set_label("Shank Ø (0 = neck)" if profile in UNDERCUT_FIELDS
                              else "Shank Ø (0 = Ø)")

    def _sync_custom_diameter(self):
        """Set the shown Custom tool's diameter from its curve."""
        t = self._tool
        rh = tool_profile.radial_height(t.profile_points)
        if rh is None:
            return
        t.diameter_mm = round(2.0 * float(rh[0][-1]), 6)
        self._loading = True
        try:
            self._diam.set(format_value(t.diameter_mm))
        finally:
            self._loading = False

    def _edit_profile(self):
        t = self._tool
        if t is None or t.profile != "Custom":
            return
        dlg = ProfileEditorDialog(self, t.profile_points, title=f"Profile — {t.name}")
        self.grab_set()
        if dlg.result is None:
            return
        t.profile_points = dlg.result
        self._sync_custom_diameter()
        self._tool_tree.item(t.id, values=self._tool_row(t))
        self._draw_preview()

    def _undercut_outline(self, t):
        """Side-view outline (x, height) in mm of an undercut cutter's head,
        neck and shank, tip at height 0, right half then mirrored."""
        R = t.diameter_mm / 2.0
        rn = min(max(t.neck_diameter_mm / 2.0, 0.0), R)
        if t.profile == "Lollipop":
            n = 24
            head_top = R + math.sqrt(max(0.0, R * R - rn * rn))
            start = -math.pi / 2
            stop = math.asin(min(1.0, (head_top - R) / R)) if R > 0 else 0.0
            right = [(R * math.cos(start + (stop - start) * i / n),
                      R + R * math.sin(start + (stop - start) * i / n)) for i in range(n + 1)]
        elif t.profile == "T-Slot":
            head_top = max(t.head_height_mm, 0.0)
            right = [(0.0, 0.0), (R, 0.0), (R, head_top), (rn, head_top)]
        else:   # Dovetail
            head_top = max(t.head_height_mm, 0.0)
            tan_a = math.tan(math.radians(max(1.0, min(89.0, t.dovetail_angle_deg))))
            right = [(0.0, 0.0), (R, 0.0),
                     (max(rn, R - head_top / tan_a), head_top), (rn, head_top)]
        neck_top = max(t.neck_length_mm, head_top + R)
        rs = t.shank_diameter_mm / 2.0
        right += [(rn, neck_top)]
        if rs > rn:
            right += [(rs, neck_top), (rs, neck_top + 2.0 * R)]
        else:
            right += [(rn, neck_top + 2.0 * R)]
        return right, max(rs, R)

    def _draw_preview(self):
        """Side view of the whole tool -- cutting part and shank -- scaled to
        fit the canvas. The cutting part is the flute length measured from
        the tip, never shorter than the tip itself (cone, ball or custom
        edge); without a
        flute length a default of the tip plus one radius, and at least one
        diameter, is drawn. Without
        an overall length (or one shorter than the cutting part) only a stub
        of the shank is drawn, open at the top."""
        c = self._preview
        c.delete("all")
        t = self._tool
        if t is None or t.diameter_mm <= 0:
            return
        W, H = c.winfo_width(), c.winfo_height()
        if W <= 1 or H <= 1:    # not mapped yet
            W, H = self._PREVIEW_W, self._PREVIEW_H
        pad, label_h = 12, 14
        if t.profile in UNDERCUT_FIELDS:
            right, half_w = self._undercut_outline(t)
            body_h = max(y for _, y in right)
            scale = min((W - 2 * pad) / (2.0 * half_w), (H - 2 * pad - label_h) / body_h)
            cx, bottom = W / 2.0, H - pad
            outline = right + [(-x, y) for x, y in reversed(right)]
            pts = []
            for x, y in outline:
                pts += [cx + x * scale, bottom - y * scale]
            c.create_polygon(pts, fill="#7a8ca3", outline="#d0d8e0")
            c.create_text(W / 2.0, pad / 2.0 + 2, text=f"Ø {format_value(t.diameter_mm)} mm",
                          fill="#d0d8e0", font=("TkDefaultFont", 8))
            return
        R  = t.diameter_mm / 2.0
        Rs = (t.shank_diameter_mm if t.shank_diameter_mm > 0 else t.diameter_mm) / 2.0
        if t.profile == "V-Bit":
            half = math.radians(max(1.0, min(179.0, t.vbit_angle_deg)) / 2.0)
            tip_h = R / math.tan(half)
        elif t.profile == "Ball":
            tip_h = R
        elif t.profile == "Custom" and t.profile_points:
            curve = tool_profile.profile_curve(t.profile_points)
            tip_h = float(max(h for _, h in curve)) if curve else 0.0
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
        elif t.profile == "Custom" and t.profile_points and curve:
            rr, hh = tool_profile.curve_arrays(curve)
            pts = [cx - r, top]
            for ri, hi in zip(rr[::-1], hh[::-1]):
                pts += [cx - ri * scale, bottom - hi * scale]
            for ri, hi in zip(rr, hh):
                pts += [cx + ri * scale, bottom - hi * scale]
            pts += [cx + r, top]
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

    def _recommend_cutting(self):
        """Open the recommendation for the shown tool and chosen material;
        accepted values fill the cutting form (Set stores them)."""
        if self._tool is None:
            return
        m = self.library.find_material(self._chosen_material_id())
        if m is None:
            messagebox.showwarning("No material", "Choose a material first.", parent=self)
            return
        if not m.has_reference():
            messagebox.showinfo(
                "No reference data",
                f"\"{m.name}\" has no reference cutting data.\n\n"
                "Enter it, or load a preset, on the Materials tab.", parent=self)
            return
        dlg = CuttingRecommendationDialog(self, self._tool, m, *self._machine_limits)
        self.grab_set()
        if dlg.result is not None:
            self._fill_cutting_form(dlg.result)

    # ------------------------------------------------------------------ #
    #  Materials tab
    # ------------------------------------------------------------------ #
    def _build_materials_tab(self, tab):
        self._mat_list = tk.Listbox(tab, height=10, exportselection=False)
        self._mat_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._mat_list.bind("<<ListboxSelect>>", self._on_material_selected)
        btns = ttk.Frame(tab)
        btns.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        ttk.Button(btns, text="Add…",    command=self._add_material).pack(fill=tk.X)
        ttk.Button(btns, text="Rename…", command=self._rename_material).pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Delete",  command=self._delete_material).pack(fill=tk.X)

        # ── Reference data for cutting recommendations ────────────────
        ref = SectionFrame(tab, "Reference data")
        ref.pack(side=tk.LEFT, fill=tk.BOTH, padx=(8, 0))
        row = ttk.Frame(ref)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Load preset").pack(side=tk.LEFT)
        self._preset_var = tk.StringVar()
        self._preset_cb = ttk.Combobox(row, textvariable=self._preset_var,
                                       values=list(MATERIAL_PRESETS), state="readonly",
                                       width=24)
        self._preset_cb.pack(side=tk.RIGHT)
        self._preset_cb.bind("<<ComboboxSelected>>", self._on_preset_chosen)
        self._mat_vc = LabeledEntry(ref, "Cutting speed vc", "m/min", "0",
                                    minval=0.0, maxval=5000.0, increment=25.0)
        self._mat_vc.pack(fill=tk.X, pady=2)
        self._mat_sd = LabeledEntry(ref, "Step-down per pass", "× Ø", "0",
                                    minval=0.0, maxval=10.0, increment=0.05)
        self._mat_sd.pack(fill=tk.X, pady=2)
        ttk.Label(ref, text="Feed per tooth fz (mm) by cutter diameter:").pack(
            anchor=tk.W, pady=(6, 2))
        grid = ttk.Frame(ref)
        grid.pack(fill=tk.X)
        self._mat_fz_vars = []
        self._mat_fz_entries = []
        for i, d in enumerate(FZ_DIAMETERS_MM):
            r, c = divmod(i, 3)
            ttk.Label(grid, text=f"Ø {format_value(d)}").grid(
                row=r, column=2 * c, sticky=tk.E, padx=(8 if c else 0, 2), pady=1)
            var = tk.StringVar()
            entry = ttk.Entry(grid, textvariable=var, width=7, justify=tk.RIGHT)
            entry.grid(row=r, column=2 * c + 1, sticky=tk.W, pady=1)
            var.trace_add("write", self._on_material_data_changed)
            self._mat_fz_vars.append(var)
            self._mat_fz_entries.append(entry)
        ttk.Label(ref, text="Values come from the Stepcraft milling-parameter sheet;\n"
                            "between diameters fz is interpolated.",
                  foreground="#666666").pack(anchor=tk.W, pady=(6, 0))
        for le in (self._mat_vc, self._mat_sd):
            le.committed_var.trace_add("write", self._on_material_data_changed)
        self._show_material(None)

    def _refresh_material_list(self):
        self._mat_list.delete(0, tk.END)
        for m in self.library.materials:
            self._mat_list.insert(tk.END, m.name)
        if self._material in self.library.materials:
            i = self.library.materials.index(self._material)
            self._mat_list.selection_set(i)
            self._mat_list.see(i)
        else:
            self._show_material(None)
        self._refresh_cutting_tree()

    def _on_material_selected(self, _event=None):
        self._show_material(self._selected_material())

    def _show_material(self, material):
        """Fill the reference-data form from *material* (None clears and
        disables it)."""
        self._material = material
        self._loading = True
        try:
            self._preset_var.set("")
            if material is None:
                self._mat_vc.set("0")
                self._mat_sd.set("0")
                for var in self._mat_fz_vars:
                    var.set("")
            else:
                self._mat_vc.set(format_value(material.cutting_speed_m_min))
                self._mat_sd.set(format_value(material.stepdown_factor))
                fz = material.feed_per_tooth_mm or [None] * len(FZ_DIAMETERS_MM)
                for var, v in zip(self._mat_fz_vars, fz):
                    var.set("" if v is None else format_value(v))
        finally:
            self._loading = False
        state = "disabled" if material is None else "normal"
        self._mat_vc.set_state(state)
        self._mat_sd.set_state(state)
        for entry in self._mat_fz_entries:
            entry.config(state=state)
        self._preset_cb.config(state="disabled" if material is None else "readonly")

    def _on_preset_chosen(self, _event=None):
        name = self._preset_var.get()
        if self._material is None or name not in MATERIAL_PRESETS:
            return
        self._material.apply_preset(name)
        self._show_material(self._material)
        self._preset_var.set(name)

    def _on_material_data_changed(self, *_):
        """Write the form into the shown material. Fields that aren't
        numbers leave the stored value unchanged; an empty fz field clears
        that diameter's value, which makes the row incomplete."""
        if self._loading or self._material is None:
            return
        m = self._material
        for le, attr in ((self._mat_vc, 'cutting_speed_m_min'),
                         (self._mat_sd, 'stepdown_factor')):
            try:
                setattr(m, attr, float(le.get()))
            except ValueError:
                pass
        fz = list(m.feed_per_tooth_mm or [0.0] * len(FZ_DIAMETERS_MM))
        for i, var in enumerate(self._mat_fz_vars):
            text = var.get().strip().replace(',', '.')
            if not text:
                fz[i] = 0.0
                continue
            try:
                fz[i] = max(0.0, float(text))
            except ValueError:
                pass
        m.feed_per_tooth_mm = fz if any(fz) else None

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
        for t in self.library.sorted_tools():
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


class CuttingRecommendationDialog(tk.Toplevel):
    """Modal dialog showing recommended cutting data for *tool* in
    *material* and how it is derived (see ``tool_library.recommend_cutting``).

    The adjustment, machine limits and custom feed per tooth set here
    apply to this calculation only.
    After ``wait_window`` returns, ``self.result`` is the accepted
    ``CuttingData``, or None if cancelled.
    """

    _ADJUSTMENTS = {"Standard": 0.0, "Roughing (vc −25 %)": -25.0,
                    "Finishing (vc +25 %)": 25.0}

    def __init__(self, parent, tool, material, spindle_max_rpm, spindle_min_rpm=0.0,
                 feed_max=0.0):
        super().__init__(parent)
        self.title("Recommended Cutting Data")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grab_set()
        self.result = None
        self._tool = tool
        self._material = material
        self._rec = None

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text=f"{tool.name} in {material.name}",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 8))

        opts = SectionFrame(outer, "Options")
        opts.pack(fill=tk.X)
        row = ttk.Frame(opts)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Adjustment").pack(side=tk.LEFT)
        self._adjust_var = tk.StringVar(value="Standard")
        ttk.Combobox(row, textvariable=self._adjust_var, values=list(self._ADJUSTMENTS),
                     state="readonly", width=20).pack(side=tk.RIGHT)
        self._max_rpm = LabeledEntry(opts, "Max spindle speed (0 = no limit)", "RPM",
                                     format_value(spindle_max_rpm),
                                     minval=0.0, maxval=100000.0, increment=1000.0)
        self._max_rpm.pack(fill=tk.X, pady=2)
        self._min_rpm = LabeledEntry(opts, "Min spindle speed (0 = no limit)", "RPM",
                                     format_value(spindle_min_rpm),
                                     minval=0.0, maxval=100000.0, increment=1000.0)
        self._min_rpm.pack(fill=tk.X, pady=2)
        self._feed_max = LabeledEntry(opts, "Max feed rate (0 = no limit)", "mm/min",
                                      format_value(feed_max),
                                      minval=0.0, maxval=100000.0, increment=100.0)
        self._feed_max.pack(fill=tk.X, pady=2)
        self._fz_override = LabeledEntry(opts, "Feed per tooth (0 = material table)", "mm",
                                         "0", minval=0.0, maxval=5.0, increment=0.005)
        self._fz_override.pack(fill=tk.X, pady=2)

        calc = SectionFrame(outer, "Calculation")
        calc.pack(fill=tk.X, pady=(8, 0))
        self._calc_labels = {}
        rows = (("d",       "Cutter diameter d"),
                ("z",       "Flutes z"),
                ("vc",      "Cutting speed vc"),
                ("fz",      "Feed per tooth fz"),
                ("n_calc",  "n = vc·1000 / (π·d)"),
                ("rpm",     "Spindle speed"),
                ("feed",    "Feed vf = n·z·fz"),
                ("vc_eff",  "Effective vc"),
                ("fz_eff",  "Effective fz"),
                ("plunge",  f"Plunge = vf / {round(1 / PLUNGE_FRACTION)}"),
                ("sd",      "Step-down"))
        grid = ttk.Frame(calc)
        grid.pack(fill=tk.X)
        for r, (key, text) in enumerate(rows):
            ttk.Label(grid, text=text).grid(row=r, column=0, sticky=tk.W, pady=1)
            lbl = ttk.Label(grid, text="", anchor=tk.E)
            lbl.grid(row=r, column=1, sticky=tk.E, padx=(16, 0), pady=1)
            self._calc_labels[key] = lbl
        grid.columnconfigure(1, weight=1)
        self._notes = ttk.Label(calc, text="", wraplength=360, foreground="#a06000",
                                justify=tk.LEFT)
        self._notes.pack(anchor=tk.W, pady=(6, 0))
        self._warnings = ttk.Label(calc, text="", wraplength=360, foreground="#c02020",
                                   justify=tk.LEFT)
        self._warnings.pack(anchor=tk.W, pady=(4, 0))

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(12, 0))
        self._use_btn = ttk.Button(btn_row, text="Use values", command=self._confirm)
        self._use_btn.pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)

        self._adjust_var.trace_add("write", self._recompute)
        for le in (self._max_rpm, self._min_rpm, self._feed_max, self._fz_override):
            le.committed_var.trace_add("write", self._recompute)
        self._recompute()

        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2 - self.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{pw}+{ph}")
        self.wait_window()

    def _recompute(self, *_):
        limits = []
        for le in (self._max_rpm, self._min_rpm, self._feed_max, self._fz_override):
            try:
                limits.append(float(le.get()))
            except ValueError:
                limits.append(0.0)
        max_rpm, min_rpm, feed_max, fz_override = limits
        adjust = self._ADJUSTMENTS.get(self._adjust_var.get(), 0.0)
        try:
            rec = recommend_cutting(self._tool, self._material, max_rpm, adjust,
                                    max_feed=feed_max, min_rpm=min_rpm,
                                    fz_override=fz_override)
        except ValueError as exc:
            self._rec = None
            for lbl in self._calc_labels.values():
                lbl.config(text="—")
            self._notes.config(text=str(exc))
            self._warnings.config(text="")
            self._use_btn.config(state="disabled")
            return
        self._rec = rec
        L = self._calc_labels
        L['d'].config(text=f"{format_value(rec.diameter_mm)} mm")
        L['z'].config(text=str(rec.flutes))
        L['vc'].config(text=f"{rec.vc_m_min:.0f} m/min")
        L['fz'].config(text=f"{rec.fz_mm:.4f} mm")
        L['n_calc'].config(text=f"{rec.rpm_calc:.0f} RPM")
        L['rpm'].config(text=f"{rec.rpm:.0f} RPM")
        L['feed'].config(text=f"{rec.feed_rate:.0f} mm/min")
        L['vc_eff'].config(text=f"{rec.vc_eff_m_min:.0f} m/min")
        L['fz_eff'].config(text=f"{rec.fz_eff_mm:.4f} mm")
        L['plunge'].config(text=f"{rec.plunge_rate:.0f} mm/min")
        L['sd'].config(text=f"{format_value(self._material.stepdown_factor)} × d → "
                            f"{format_value(rec.stepdown_mm)} mm")
        self._notes.config(text="\n".join(rec.notes))
        self._warnings.config(text="\n".join(rec.warnings))
        self._use_btn.config(state="normal")

    def _confirm(self):
        if self._rec is None:
            return
        self.result = self._rec.to_cutting()
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
