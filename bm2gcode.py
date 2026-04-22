"""
CNC Relief Carving — Image Viewer & Parameter Setup

Sidebar sections (scrollable):
  • Physical Image Size  (width cm, height cm, max cut depth mm)
  • Roughing Tool        (profile, diameter, raster step, raster direction)
  • Refining Tool        (profile, diameter, raster step, raster direction)
  • Actions             (Open, Reset Defaults,
                         Compute Roughing Path, Save Roughing Path,
                         Compute Refining Path, Save Refining Path)
  • Image Info

Below the image:
  • Radio buttons to switch display between Original / Post-Roughing Residual /
    Post-Refining Residual
"""

import tkinter as tk
from app import GCodeGeneratorApp

if __name__ == "__main__":
    root = tk.Tk()
    app  = GCodeGeneratorApp(root)
    root.mainloop()
