"""Plain assert-based tests for grey_morphology.py: results must match
scipy.ndimage's grey_erosion/grey_dilation/grey_opening/binary_erosion/
binary_dilation exactly, while memory stays bounded for large kernels.

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
gm.binary_erosion(img_big > 0.5, fp_big)
gm.binary_dilation(img_big > 0.5, fp_big)
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
# shorter than the others. Tiles are the scipy path's, so numba is off.
gm.USE_NUMBA = False
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
gm.USE_NUMBA = True

print("Keeps the input dtype")
img32 = rng.random((40, 40)).astype(np.float32)
fp = disk(4).astype(bool)
check("float32 in, float32 out", gm.grey_erosion(img32, fp).dtype == np.float32)
check("float32 matches scipy",
      np.array_equal(gm.grey_erosion(img32, fp), ndi.grey_erosion(img32, footprint=fp)))

print("Compiled kernel and scipy fallback")
gm._get_numba_kernel()
have_numba = gm._numba_kernel is not False
print(f"  (numba {'installed' if have_numba else 'not installed: fallback only'})")
_, fp_l = _build_kernel(Tool("Ball", 3.0), px_per_mm, max_depth)
s_l = -_build_kernel(Tool("Ball", 3.0), px_per_mm, max_depth)[0].astype(np.float64)
sentinel = rng.random((150, 120))
sentinel[10:30, 40:80] = np.inf
sentinel[60:65] = -2.0e6
sentinel[:, 100:] = 1.0e6
band_fps = [fp_l & (np.hypot(*np.mgrid[-15:16, -15:16]) < 8), fp_l & (np.hypot(*np.mgrid[-15:16, -15:16]) >= 8)]
band_ins = [rng.random((150, 120)), sentinel]
for use in ([True, False] if have_numba else [False]):
    gm.USE_NUMBA = use
    tag = "numba" if use else "fallback"
    check(f"{tag}: ball erosion with inf and sentinels matches scipy",
          np.array_equal(gm.grey_erosion(sentinel, fp_l, s_l),
                         ndi.grey_erosion(sentinel, footprint=fp_l, structure=s_l)))
    check(f"{tag}: ball dilation with inf and sentinels matches scipy",
          np.array_equal(gm.grey_dilation(sentinel, fp_l, s_l),
                         ndi.grey_dilation(sentinel, footprint=fp_l, structure=s_l)))
    check(f"{tag}: asymmetric structure matches scipy",
          np.array_equal(gm.grey_erosion(images["77x51"], fp_asym, s_asym),
                         ndi.grey_erosion(images["77x51"], footprint=fp_asym, structure=s_asym)))
    e32 = gm.grey_erosion(img32, fp_l[8:23, 8:23], s_l[8:23, 8:23])
    check(f"{tag}: float32 stays float32 and matches scipy",
          e32.dtype == np.float32 and np.array_equal(
              e32, ndi.grey_erosion(img32, footprint=fp_l[8:23, 8:23],
                                    structure=s_l[8:23, 8:23])))
    tall32 = rng.random((700, 90)).astype(np.float32)        # several row tiles
    for op in (gm.grey_erosion, gm.grey_dilation):
        r32, r64 = op(tall32, fp_l, s_l), op(tall32.astype(np.float64), fp_l, s_l)
        check(f"{tag}: float32 {op.__name__} is the float64 result rounded to float32",
              r32.dtype == np.float32 and np.array_equal(r32, r64.astype(np.float32)))
    check(f"{tag}: banded erosion is the min of the parts' erosions",
          np.array_equal(gm.banded_grey_erosion(band_ins, band_fps, s_l),
                         np.minimum(*[ndi.grey_erosion(a, footprint=fp, structure=s_l)
                                      for a, fp in zip(band_ins, band_fps)])))
    check(f"{tag}: banded dilation is the max of the parts' dilations",
          np.array_equal(gm.banded_grey_dilation(band_ins, band_fps, s_l),
                         np.maximum(*[ndi.grey_dilation(a, footprint=fp, structure=s_l)
                                      for a, fp in zip(band_ins, band_fps)])))
gm.USE_NUMBA = True

print("Binary operations match scipy")
masks = {name: img > 0.5 for name, img in images.items()}
masks["touches every edge"] = ndi.gaussian_filter(rng.random((120, 90)), 4) > 0.48
masks["all True"] = np.ones((50, 60), bool)
masks["all False"] = np.zeros((50, 60), bool)
masks["one pixel"] = np.zeros((50, 60), bool)
masks["one pixel"][25, 30] = True
masks["full-width band"] = np.zeros((90, 60), bool)
masks["full-width band"][:30] = True
fp_empty_center = rng.random((11, 15)) > 0.4
fp_empty_center[5, 7] = False
footprints = {"disk r=1": disk(1), "disk r=5": disk(5), "disk r=20": disk(20),
              "disk r=40 (wider than the small images)": disk(40),
              "End Mill": _build_kernel(Tool("End Mill", 4.0), px_per_mm, max_depth)[1],
              "V-Bit": _build_kernel(Tool("V-Bit", 4.0, 60.0), px_per_mm, max_depth)[1],
              "asymmetric": fp_asym, "asymmetric, center False": fp_empty_center,
              "rows only": np.ones((1, 9), bool), "columns only": np.ones((9, 1), bool)}
for fp_name, fp in footprints.items():
    for mask_name, mask in masks.items():
        e = gm.binary_erosion(mask, fp)
        d = gm.binary_dilation(mask, fp)
        check(f"binary erosion  {fp_name} {mask_name}",
              e.dtype == bool and np.array_equal(e, ndi.binary_erosion(mask, structure=fp)))
        check(f"binary dilation {fp_name} {mask_name}",
              d.dtype == bool and np.array_equal(d, ndi.binary_dilation(mask, structure=fp)))

print("Binary operations match scipy when split into row tiles")
tall_mask = ndi.gaussian_filter(rng.random((1000, 90)), 4) > 0.5
check("binary erosion  disk r=12 1000 rows",
      np.array_equal(gm.binary_erosion(tall_mask, disk(12)),
                     ndi.binary_erosion(tall_mask, structure=disk(12))))
check("binary dilation disk r=12 1000 rows",
      np.array_equal(gm.binary_dilation(tall_mask, disk(12)),
                     ndi.binary_dilation(tall_mask, structure=disk(12))))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
