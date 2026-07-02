import locale
import tkinter as tk
from app import GCodeGeneratorApp

if __name__ == "__main__":
    root = tk.Tk()
    app  = GCodeGeneratorApp(root)
    # matplotlib's first figure/canvas setup switches the process' numeric
    # locale to the OS locale (e.g. de_DE, which uses ',' as the decimal
    # separator). Tk's spinbox widgets format values through the C library
    # under that locale, so a comma would leak into fields that float()
    # elsewhere in the app can't parse. Force '.' as the decimal point.
    locale.setlocale(locale.LC_NUMERIC, "C")
    root.mainloop()
