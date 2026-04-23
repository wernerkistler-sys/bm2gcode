import tkinter as tk
from app import GCodeGeneratorApp

if __name__ == "__main__":
    root = tk.Tk()
    app  = GCodeGeneratorApp(root)
    root.mainloop()
