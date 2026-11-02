"""Stage-by-stage time and memory of compute_raster_toolpath on a synthetic relief.
usage: bench_raster.py W_mm H_mm px_per_mm diameter profile stepdown [mem]"""
import os, sys, time, tracemalloc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tool import Tool
from algorithms import compute_raster_toolpath

W_mm, H_mm, ppm, dia = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
prof, stepdown = sys.argv[5], float(sys.argv[6])
mem = len(sys.argv) > 7
H, W = int(H_mm * ppm), int(W_mm * ppm)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
depth = (0.5 + 0.25 * np.sin(xx / (7 * ppm)) * np.cos(yy / (11 * ppm))
         + 0.2 * np.sin((xx + yy) / (2.3 * ppm))).astype(np.float32)
depth = np.clip(depth, 0, 1)
del yy, xx
print(f"canvas {W}x{H} = {W*H/1e6:.1f} Mpx, one float32 map = {W*H*4/1e6:.0f} MB")

if mem:
    tracemalloc.start()
t0 = last = time.perf_counter()
def progress(pct, msg):
    global last
    now = time.perf_counter()
    m = ""
    if mem:
        cur, peak = tracemalloc.get_traced_memory()
        m = f"  cur {cur/1e6:7.0f} MB  peak {peak/1e6:7.0f} MB"
        tracemalloc.reset_peak()
    print(f"{now-t0:7.1f}s (+{now-last:6.1f}s){m}  {pct:5.1f}% {msg[:80]}", flush=True)
    last = now

tp = compute_raster_toolpath(depth, 256, Tool(profile=prof, diameter_mm=dia),
                             step_mm=dia * 0.2, stepdown_mm=stepdown, max_depth=5.0,
                             width_mm=W_mm, height_mm=H_mm, raster_dir="Horizontal",
                             cut_direction="Either", progress=progress)
n = sum(len(s) for s in tp.segments)
print(f"total {time.perf_counter()-t0:.1f}s, {len(tp.segments)} segs, {n} pts")
