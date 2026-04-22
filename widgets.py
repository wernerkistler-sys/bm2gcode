import tkinter as tk
from tkinter import ttk
import queue


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
    """Single-row widget: label on the left, unit tag + entry on the right."""

    def __init__(self, parent, label, unit, default, width=8, **kw):
        super().__init__(parent, **kw)
        ttk.Label(self, text=label).pack(side=tk.LEFT)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(side=tk.RIGHT)
        ttk.Label(self, text=unit, foreground="#666666").pack(side=tk.RIGHT, padx=(0, 3))

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)

    def set_state(self, state):
        self.entry.config(state=state)


class SectionFrame(ttk.LabelFrame):
    """Labelled group frame with consistent internal padding."""

    def __init__(self, parent, title, **kw):
        super().__init__(parent, text=title, padding=(8, 6), **kw)


def _fmt_time(minutes):
    """Format a duration in minutes as a human-readable string."""
    if minutes < 1.0:
        return f"{minutes * 60:.0f} sec"
    if minutes < 60.0:
        return f"{minutes:.1f} min"
    h = int(minutes // 60)
    m = minutes % 60
    return f"{h}h {m:.0f}min"
