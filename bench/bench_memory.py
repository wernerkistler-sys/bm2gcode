"""Peak and held memory per canvas pixel (B/px) of one computation, at 20 px/mm.
usage: bench_memory.py case [N] [one-worker] [snapshot]

case        composite | raster | contour | vcarve | cutout | pocket | pocket_zz |
            levelclear | levelclear_zz | view | view_shaded | all
N           canvas side in px (default 2000, i.e. a 100x100 mm stock)
one-worker  Pocket / Level Clearing regions one at a time (_RegionRunner with
            one worker), for the step's own need rather than the pool's budget
snapshot    also list the allocations that make up the peak, from a
            tracemalloc snapshot at each new high (slows the run several
            times over; its RSS figure is then meaningless)

"all" runs every case in its own process. Fixed costs (imports, numba's
compilation, module state) count per pixel too, so canvases much smaller than
the default overstate every figure. Peak traced is tracemalloc's peak
(numpy allocations); peak RSS is ru_maxrss above the process's baseline
(Linux and macOS; it also counts freed memory the allocator keeps); held is
what the finished computation keeps; toolpath arrays are the ToolPath's maps.
The view cases draw the canvas with matplotlib's Agg backend the way app.py
does, plain and with LightSource shading."""
import os, subprocess, sys, threading, time, tracemalloc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CASES = ["composite", "raster", "contour", "vcarve", "cutout", "pocket", "pocket_zz",
         "levelclear", "levelclear_zz", "view", "view_shaded"]

case = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2000
flags = set(sys.argv[2:])
if case == "all":
    extra = [f for f in ("one-worker",) if f in flags]
    for c in CASES:
        subprocess.run([sys.executable, os.path.abspath(__file__), c, str(N)] + extra)
    sys.exit(0)

import resource
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import algorithms as A
import layers as L
from tool import Tool

ppm = 20.0
mm = N / ppm
rng = np.random.default_rng(1)
depth = zoom(gaussian_filter(rng.random((200, 200)), 6), N / 200, order=1)
depth = gaussian_filter(depth, 2)
depth = ((depth - depth.min()) / (depth.max() - depth.min())).astype(np.float32)
prev = (depth * 0.5).astype(np.float32)
nop = lambda pct, msg: None
ball = Tool(profile="Ball", diameter_mm=6.0)
em = Tool(profile="End Mill", diameter_mm=6.0)
vbit = Tool(profile="V-Bit", diameter_mm=12.0, vbit_angle_deg=60.0)

if "one-worker" in flags:
    _init = A._RegionRunner.__init__
    A._RegionRunner.__init__ = lambda self, max_workers=None, memory=None: _init(self, 1, memory)

if case.startswith("view"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax = fig.add_subplot()


def run():
    if case == "composite":
        a = L.Layer(name="a", raw=depth.copy(), source_kind="image")
        b = L.Layer(name="b", raw=depth[::-1].copy(), source_kind="image",
                    blend_mode="Merge high", rotation_deg=10.0)
        return L.composite([a, b], (N, N), ppm, ppm, "white")
    if case == "raster":
        return A.compute_raster_toolpath(depth, 256, ball, 1.0, 3.0, 10.0, mm, mm, "Horizontal",
                                         previous_result=prev, progress=nop)
    if case == "contour":
        return A.compute_contour_toolpath(depth, 256, ball, 1.0, 10.0, mm, mm,
                                          previous_result=prev, progress=nop)
    if case == "vcarve":
        return A.compute_vcarve_toolpath(depth, 256, vbit, 10.0, mm, mm, threshold_mm=3.0,
                                         progress=nop)
    if case == "cutout":
        return A.compute_cutout_toolpath(depth, 256, em, 12.0, 10.0, mm, mm, threshold_mm=2.0,
                                         stepdown_mm=3.0, progress=nop)
    if case in ("pocket", "pocket_zz"):
        pattern = "Zig-Zag (0°)" if case == "pocket_zz" else "Onion Skin (Outside-In)"
        return A.compute_pocket_toolpath(depth, 256, em, 2.4, 3.0, 10.0, mm, mm, pattern,
                                         target_z_mm=5.0, previous_result=prev, progress=nop)
    if case in ("levelclear", "levelclear_zz"):
        pattern = "Zig-Zag (0°)" if case == "levelclear_zz" else "Onion Skin (Outside-In)"
        return A.compute_level_clearing_toolpath(depth, 256, em, 2.4, 1.0, 10.0, mm, mm, pattern,
                                                 previous_result=prev, progress=nop)
    if case == "view":
        ax.imshow(depth, interpolation="none", cmap="gray", vmin=0, vmax=1)
        fig.canvas.draw()
        return None
    if case == "view_shaded":
        rgb = LightSource(azdeg=315, altdeg=45).shade(
            depth, cmap=plt.get_cmap("gray"), vert_exag=1.0, blend_mode="soft")
        ax.imshow(rgb, interpolation="none")
        fig.canvas.draw()
        return rgb
    raise SystemExit(f"unknown case {case!r}; see usage")


px = N * N
base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_unit = 1 if sys.platform == "darwin" else 1024     # ru_maxrss: bytes on macOS, KiB elsewhere
tracemalloc.start(12 if "snapshot" in flags else 1)
best = [0, None]
stop = [False]


def poll():
    while not stop[0]:
        cur, _ = tracemalloc.get_traced_memory()
        if cur > best[0] * 1.03:
            best[0], best[1] = cur, tracemalloc.take_snapshot()
        time.sleep(0.005)


if "snapshot" in flags:
    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
t0 = time.perf_counter()
out = run()
elapsed = time.perf_counter() - t0
stop[0] = True
cur, peak = tracemalloc.get_traced_memory()
rss = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base_rss) * rss_unit
kept = sum(getattr(out, f).nbytes for f in ("result", "accessible", "alpha", "undercut_reach")
           if isinstance(getattr(out, f, None), np.ndarray))
label = case + (" (one worker)" if "one-worker" in flags else "")
print(f"{label:28s} {N}x{N}  peak traced {peak / px:6.1f} B/px  peak RSS {rss / px:6.1f} B/px  "
      f"held {cur / px:5.1f} B/px  toolpath arrays {kept / px:4.1f} B/px  {elapsed:5.1f} s",
      flush=True)
if "snapshot" in flags:
    poller.join()
    snap = best[1].filter_traces([tracemalloc.Filter(False, tracemalloc.__file__)])
    print(f"  at the peak ({best[0] / px:.1f} B/px), largest allocations by line:")
    for st in snap.statistics("lineno")[:12]:
        frame = st.traceback[0]
        path = frame.filename
        path = (os.path.relpath(path, ROOT) if path.startswith(ROOT)
                else "/".join(path.split(os.sep)[-2:]))
        print(f"    {st.size / px:6.1f} B/px  {path}:{frame.lineno}")
