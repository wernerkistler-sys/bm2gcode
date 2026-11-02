"""Time + peak traced memory of Contour / V-carve / Pocket / Cutout at one size.
usage: bench_algos.py W_mm H_mm px_per_mm algo [mem]"""
import os, sys, time, tracemalloc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tool import Tool
import algorithms as A

W_mm, H_mm, ppm, algo = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
mem = len(sys.argv) > 5
H, W = int(H_mm * ppm), int(W_mm * ppm)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
depth = (0.5 + 0.25 * np.sin(xx / (7 * ppm)) * np.cos(yy / (11 * ppm))
         + 0.2 * np.sin((xx + yy) / (2.3 * ppm))).astype(np.float32)
# a raised plateau region in the middle so pocket/cutout/vcarve have features
r = np.hypot(xx - W / 2, yy - H / 2)
depth = np.where(r < min(H, W) * 0.35, depth * 0.3, 1.0).astype(np.float32)
depth = np.clip(depth, 0, 1)
del yy, xx, r
print(f"{algo}: canvas {W}x{H} = {W*H/1e6:.1f} Mpx")
if mem:
    tracemalloc.start()
t0 = last = time.perf_counter()
def progress(pct, msg):
    global last
    now = time.perf_counter()
    m = ""
    if mem:
        cur, peak = tracemalloc.get_traced_memory()
        m = f"  cur {cur/1e6:6.0f} MB peak {peak/1e6:6.0f} MB"
        tracemalloc.reset_peak()
    print(f"{now-t0:7.1f}s (+{now-last:6.1f}s){m}  {msg[:70]}", flush=True)
    last = now

common = dict(width_mm=W_mm, height_mm=H_mm, progress=progress)
if algo == "contour":
    tp = A.compute_contour_toolpath(depth, 256, Tool(profile="Ball", diameter_mm=3.0),
                                    z_increment_mm=0.5, max_depth=5.0, **common)
elif algo == "vcarve":
    tp = A.compute_vcarve_toolpath(depth, 256, Tool(profile="V-Bit", diameter_mm=6.0, vbit_angle_deg=60),
                                   max_depth=5.0, **common)
elif algo == "pocket":
    tp = A.compute_pocket_toolpath(depth, 256, Tool(profile="End Mill", diameter_mm=6.0),
                                   step_mm=2.4, stepdown_mm=2.0, max_depth=5.0,
                                   raster_dir="Onion Skin (Outside-In)", target_z_mm=5.0, **common)
elif algo == "cutout":
    tp = A.compute_cutout_toolpath(depth, 256, Tool(profile="End Mill", diameter_mm=6.0),
                                   cutout_depth_mm=10.0, max_depth=5.0, stepdown_mm=2.0,
                                   threshold_mm=1.0, **common)
if mem:
    print(f"overall peak {tracemalloc.get_tracemalloc_memory()/1e6:.0f}")
print(f"total {time.perf_counter()-t0:.1f}s, {len(tp.segments)} segs, "
      f"{sum(len(s) for s in tp.segments)} pts")
