"""Plain assert-based tests for grey_morphology.py: results must match
scipy.ndimage's grey_erosion/grey_dilation/grey_opening exactly, while
memory stays bounded for large kernels.

Run with: python tests/test_grey_morphology.py
"""
import os
import resource
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grey_morphology as gm
from algorithms import _build_kernel
from tool import Tool

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


rng = np.random.default_rng(0)

print("Memory stays bounded for a large kernel")
# Runs first so the peak RSS isn't masked by the scipy comparisons below.
# scipy itself needs ~6.4 GB for this footprint.
r = 90
ky, kx = np.mgrid[-r:r + 1, -r:r + 1]
rr = np.hypot(kx, ky)
fp_big = rr <= r
ball_big = -np.where(fp_big, r - np.sqrt(np.maximum(0.0, r * r - rr ** 2)), 1e9) / 50.0
img_big = rng.random((400, 400))
before = peak_rss_mb()
gm.grey_erosion(img_big, fp_big, ball_big)
gm.grey_dilation(img_big, fp_big, ball_big)
gm.grey_erosion(img_big, fp_big)
check(f"r={r} px: peak RSS grows < 500 MB (grew {peak_rss_mb() - before:.0f} MB)",
      peak_rss_mb() - before < 500.0)

print("Matches scipy for tool kernels")
px_per_mm = 10.0
max_depth = 5.0
tools = [Tool("End Mill", 0.1), Tool("End Mill", 1.0), Tool("End Mill", 4.0),
         Tool("Ball", 1.0), Tool("Ball", 4.0),
         Tool("V-Bit", 4.0, 60.0), Tool("V-Bit", 4.0, 90.0)]
images = {"64x64": rng.random((64, 64)),
          "77x51": rng.random((77, 51)),
          "small 9x30": rng.random((9, 30)),
          "small 30x7": rng.random((30, 7))}
for tool in tools:
    kernel, footprint = _build_kernel(tool, px_per_mm, max_depth)
    structure = -kernel.astype(np.float64)
    label = f"{tool.profile} {tool.diameter_mm} mm ({kernel.shape[0]} px)"
    for img_name, img in images.items():
        check(f"erosion  {label} {img_name}",
              np.array_equal(gm.grey_erosion(img, footprint, structure),
                             ndi.grey_erosion(img, footprint=footprint, structure=structure)))
        check(f"dilation {label} {img_name}",
              np.array_equal(gm.grey_dilation(img, footprint, structure),
                             ndi.grey_dilation(img, footprint=footprint, structure=structure)))

print("Matches scipy for an asymmetric structure and footprint")
fp_asym = rng.random((11, 15)) > 0.4
fp_asym[5, 7] = True
s_asym = rng.random((11, 15))
for img_name, img in images.items():
    check(f"erosion  asymmetric {img_name}",
          np.array_equal(gm.grey_erosion(img, fp_asym, s_asym),
                         ndi.grey_erosion(img, footprint=fp_asym, structure=s_asym)))
    check(f"dilation asymmetric {img_name}",
          np.array_equal(gm.grey_dilation(img, fp_asym, s_asym),
                         ndi.grey_dilation(img, footprint=fp_asym, structure=s_asym)))

print("Matches scipy for flat footprints (structure=None)")
from skimage.morphology import disk
for rad in (1, 5, 20):
    fp = disk(rad).astype(bool)
    for img_name, img in images.items():
        check(f"erosion  flat disk r={rad} {img_name}",
              np.array_equal(gm.grey_erosion(img, fp),
                             ndi.grey_erosion(img, footprint=fp)))
        check(f"dilation flat disk r={rad} {img_name}",
              np.array_equal(gm.grey_dilation(img, fp),
                             ndi.grey_dilation(img, footprint=fp)))
        check(f"opening  flat disk r={rad} {img_name}",
              np.array_equal(gm.grey_opening(img, fp),
                             ndi.grey_opening(img, footprint=fp)))

print("Matches scipy when split into row tiles")
# Tall enough for several tiles whatever the core count; the last tile is
# shorter than the others.
_, fp_ball = _build_kernel(Tool("Ball", 2.0), px_per_mm, max_depth)
ball = -_build_kernel(Tool("Ball", 2.0), px_per_mm, max_depth)[0].astype(np.float64)
for H in (gm._MIN_TILE_ROWS * 3 + 17, 1000):
    tall = rng.random((H, 90))
    n_tiles = -(-H // max(gm._MIN_TILE_KERNELS * fp_ball.shape[0], gm._MIN_TILE_ROWS,
                          -(-H // gm._POOL._max_workers)))
    check(f"{H} rows split into {n_tiles} tiles", n_tiles > 1)
    check(f"erosion  ball {H} rows",
          np.array_equal(gm.grey_erosion(tall, fp_ball, ball),
                         ndi.grey_erosion(tall, footprint=fp_ball, structure=ball)))
    check(f"dilation ball {H} rows",
          np.array_equal(gm.grey_dilation(tall, fp_ball, ball),
                         ndi.grey_dilation(tall, footprint=fp_ball, structure=ball)))
    check(f"erosion  flat disk {H} rows",
          np.array_equal(gm.grey_erosion(tall, disk(12).astype(bool)),
                         ndi.grey_erosion(tall, footprint=disk(12).astype(bool))))
tall32 = rng.random((1000, 90)).astype(np.float32)
check("tiled float32 in, float32 out, matches scipy",
      gm.grey_erosion(tall32, fp_ball, ball).dtype == np.float32
      and np.array_equal(gm.grey_erosion(tall32, fp_ball, ball),
                         ndi.grey_erosion(tall32, footprint=fp_ball, structure=ball)))

print("Keeps the input dtype")
img32 = rng.random((40, 40)).astype(np.float32)
fp = disk(4).astype(bool)
check("float32 in, float32 out", gm.grey_erosion(img32, fp).dtype == np.float32)
check("float32 matches scipy",
      np.array_equal(gm.grey_erosion(img32, fp), ndi.grey_erosion(img32, footprint=fp)))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
