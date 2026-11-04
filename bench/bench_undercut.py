"""Stage-by-stage time and memory of compute_undercut_toolpath on the umbilic
torus in the repo root, with a 6 mm Lollipop (3 mm neck).
usage: bench_undercut.py px_per_mm [mem]"""
import os, sys, time, tracemalloc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from stl_mesh import (load_stl, generate_dexels, generate_heightmap, mesh_xy_bounds,
                      mesh_z_bounds, normalize_and_orient)
from layers import Layer, composite, composite_gaps, to_depth
from tool import Tool
from undercut import compute_undercut_toolpath

ppm = float(sys.argv[1])
mem = len(sys.argv) > 2
tris = load_stl(os.path.join(ROOT, "umbilic-torus-3-1.stl"))
x0, x1, y0, y1 = mesh_xy_bounds(tris)
z0, z1 = mesh_z_bounds(tris)
W_mm, H_mm, max_depth = x1 - x0, y1 - y0, z1 - z0
W, H = int(round(W_mm * ppm)), int(round(H_mm * ppm))
raw, alpha = normalize_and_orient(generate_heightmap(tris, W, H), (z0, z1))
layer = Layer(name="torus", raw=raw, source_kind="stl", alpha=alpha)
layer.dexels = generate_dexels(tris, W, H)
depth = to_depth(composite([layer], (H, W), ppm, ppm, "white"), "white", False)
gaps, _ = composite_gaps([layer], (H, W), ppm, ppm, "white", depth)
print(f"canvas {W}x{H} = {W*H/1e6:.2f} Mpx, {len(gaps.ceil)} gaps, "
      f"{100 * np.mean(gaps.counts > 0):.0f}% of pixels with gaps")

if mem:
    tracemalloc.start()
t0 = last = time.perf_counter()
max_peak = 0
def progress(pct, msg):
    global last, max_peak
    now = time.perf_counter()
    m = ""
    if mem:
        cur, peak = tracemalloc.get_traced_memory()
        max_peak = max(max_peak, peak)
        m = f"  cur {cur/1e6:6.0f} MB  peak {peak/1e6:6.0f} MB"
        tracemalloc.reset_peak()
    print(f"{now-t0:7.1f}s (+{now-last:5.1f}s){m}  {pct:5.1f}% {msg[:70]}", flush=True)
    last = now

tp = compute_undercut_toolpath(depth, gaps, Tool("Lollipop", 6.0, neck_diameter_mm=3.0),
                               z_increment_mm=2.0, stepover_pct=20, max_depth=max_depth,
                               width_mm=W_mm, height_mm=H_mm, progress=progress)
print(f"total {time.perf_counter()-t0:.1f}s, {len(tp.segments)} chains, "
      f"{sum(len(s) for s in tp.segments)} pts"
      + (f", highest stage peak {max_peak/1e6:.0f} MB = {max_peak/(W*H):.0f} B/px" if mem else ""))
