"""Plain assert-based tests for the onion-skin pocket-clearing helpers
(algorithms.py's _onion_skin_chains / _ring_chain_to_points /
_order_fixed_orientation_pass).

Run with: python tests/test_algorithms.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from algorithms import (_onion_skin_chains, _ring_chain_to_points,
                        _order_fixed_orientation_pass, compute_pocket_toolpath)
from tool import Tool

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


def disk_mask(shape, center, radius):
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    return (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2


def dist_from(points, center):
    return np.hypot(points[:, 0] - center[0], points[:, 1] - center[1])


# --------------------------------------------------------------------------
# a region smaller than one step_px still gets its own single perimeter ring,
# rather than nothing -- and the function must terminate (an unclamped
# step_px would erode nothing and loop forever).
# --------------------------------------------------------------------------
print("tiny region")

tiny = np.zeros((10, 10), dtype=bool)
tiny[4:6, 4:6] = True   # 2x2 block, well under one 6px step-over
chains = _onion_skin_chains(tiny, step_px=6, min_area_px=100)
check("a tiny region still returns exactly one chain", len(chains) == 1)
check("the tiny region's chain has at least 3 points", len(chains[0]) >= 3)


# --------------------------------------------------------------------------
# a region whose tight bounding box is only 1 pixel tall (or wide) is a
# regression case: compute_pocket_toolpath crops each region to its exact
# find_objects bounding box, which touches the mask on every side by
# definition, so a sliver region's cropped array can be Nx1 or 1xN --
# find_contours raises "Input array must be at least 2x2" on that without
# the 1px padding _onion_skin_chains applies internally.
# --------------------------------------------------------------------------
print("1-pixel-tall sliver region (regression)")

sliver = np.ones((1, 8), dtype=bool)
chains = _onion_skin_chains(sliver, step_px=2, min_area_px=4)
check("a 1-row-tall region does not crash find_contours", len(chains) == 1)
check("the sliver's chain has at least 3 points", len(chains[0]) >= 3)


# --------------------------------------------------------------------------
# a filled disk stays one island and erodes through multiple rings
# --------------------------------------------------------------------------
print("filled disk")

R = 40
disk = disk_mask((100, 100), (50, 50), R)
chains = _onion_skin_chains(disk, step_px=4, min_area_px=20)
check("a convex filled disk never splits into more than one chain", len(chains) == 1)
d = dist_from(chains[0], (50, 50))
check("the chain's outermost points sit near the disk's true radius",
      abs(d.max() - R) < 3.0)
check("the chain reaches well inward of the outer radius (multiple rings, not just one)",
      d.min() < R - 3 * 4)


# --------------------------------------------------------------------------
# an annulus (pocket around an island / a hole) is one connected component --
# find_contours already returns both its outer and inner ring in one call
# --------------------------------------------------------------------------
print("annulus")

outer_r, inner_r = 40, 30
annulus = disk_mask((100, 100), (50, 50), outer_r) & ~disk_mask((100, 100), (50, 50), inner_r)
# step_px bigger than the annulus's own thickness (10 px) collapses it to
# nothing after the very first ring extraction, so the whole returned chain
# is exactly that first iteration's outer+inner ring pair.
chains = _onion_skin_chains(annulus, step_px=15, min_area_px=20)
check("an annulus stays one connected chain (not a split)", len(chains) == 1)
d = dist_from(chains[0], (50, 50))
check("the chain includes points near the annulus's outer boundary",
      abs(d.max() - outer_r) < 3.0)
check("the chain includes points near the annulus's inner (hole) boundary",
      abs(d.min() - inner_r) < 3.0)


# --------------------------------------------------------------------------
# a dumbbell-shaped region (two lobes joined by a narrow neck): the ring(s)
# traced before the neck erodes away form one chain spanning both lobes;
# once erosion disconnects the neck, each lobe continues as its own
# independent chain rather than bridging back across the former neck.
# --------------------------------------------------------------------------
print("dumbbell split")

shape = (60, 160)
c1, c2 = (30, 40), (30, 120)
lobe_r = 25
neck = np.zeros(shape, dtype=bool)
neck[27:33, 40:120] = True   # a 6px-tall bridge, narrower than either lobe
dumbbell = disk_mask(shape, c1, lobe_r) | disk_mask(shape, c2, lobe_r) | neck
chains = _onion_skin_chains(dumbbell, step_px=4, min_area_px=20)

spans_both = [(dist_from(c, c1).min() < lobe_r) and (dist_from(c, c2).min() < lobe_r)
             for c in chains]
check("exactly one chain spans both lobes (the joined outline before the neck erodes away)",
      sum(spans_both) == 1)
check("every other chain stays local to a single lobe (post-split, independent per lobe)",
      len(chains) - sum(spans_both) == 2)


# --------------------------------------------------------------------------
# regression test: when a split produces two sibling branches of very
# different remaining depth, the shallow sibling's chain must not be
# stranded until after the deep sibling's entire remaining descent
# finishes. A depth-first "fully recurse this branch, then start its
# sibling" implementation would bury the small lobe's chain at the very
# end, well behind rings that are actually shallower (closer to the
# surface) than much of the big lobe's own later rings. The big lobe is
# placed first in scan order (top-left) so scipy.ndimage.label assigns it
# component 1 -- a depth-first bug would fully finish it before even
# starting component 2 (the small lobe), regardless of which is actually
# shallower.
# --------------------------------------------------------------------------
print("sibling branches of unequal depth stay interleaved (regression)")

shape2 = (100, 220)
big_c, small_c = (50, 50), (50, 190)
big_r, small_r = 45, 12
neck2 = np.zeros(shape2, dtype=bool)
neck2[47:53, 50:190] = True
uneven = disk_mask(shape2, big_c, big_r) | disk_mask(shape2, small_c, small_r) | neck2
chains2 = _onion_skin_chains(uneven, step_px=4, min_area_px=20)

is_small_lobe = [dist_from(c, small_c).min() < small_r / 2 for c in chains2]
is_big_lobe = [dist_from(c, big_c).min() < big_r / 2 and not is_small_lobe[i]
              for i, c in enumerate(chains2)]
check("both the small and big lobe end up with their own post-split chain",
      any(is_small_lobe) and any(is_big_lobe))
small_idx = is_small_lobe.index(True)
last_big_idx = max(i for i, v in enumerate(is_big_lobe) if v)
check("the small lobe's chain is not stranded after the big lobe's entire "
      "remaining descent (depth-first recursion bug)",
      small_idx < last_big_idx)


# --------------------------------------------------------------------------
# regression test: a stray fragment thrown off by erosion at a concave cusp
# (connected to the main body only by a thin bridge that vanishes on the
# very first erosion) must be dropped rather than spawned as its own tiny
# standalone chain/toolpath segment. The fragment (a 4-pixel remnant) sits
# well under min_area_px, unlike a genuine second lobe.
# --------------------------------------------------------------------------
print("stray erosion fragment at a cusp is dropped (regression)")

shape3 = (100, 100)
main_c = (50, 50)
main_r = 35
wart_c = (3, 50)
wart = disk_mask(shape3, wart_c, 4)
bridge = np.zeros(shape3, dtype=bool)
bridge[6:16, 49:51] = True   # thin bridge, narrower than the step_px=3 erosion disk
cusp_mask = disk_mask(shape3, main_c, main_r) | wart | bridge

step_px3 = 3
min_area_px3 = max(4, int(round(np.pi * (0.5 * step_px3) ** 2)))
chains3 = _onion_skin_chains(cusp_mask, step_px=step_px3, min_area_px=min_area_px3)
check("no chain is a tiny standalone ring stranded at the stray fragment's location",
      not any(len(c) < min_area_px3 * 2 and dist_from(c, wart_c).max() < 6 for c in chains3))
check("the main lobe still gets its own full chain covering its own interior",
      any(dist_from(c, main_c).min() < 5 for c in chains3))


# --------------------------------------------------------------------------
# _ring_chain_to_points: pixel -> mm conversion matches the same
# row/col -> y/x convention used throughout algorithms.py (row 0 = top =
# max Y, col 0 = left = min X), at a fixed constant floor depth
# --------------------------------------------------------------------------
print("_ring_chain_to_points")

chain_px = np.array([[0.0, 0.0], [10.0, 20.0]])
px_per_mm = 2.0
H = 21
pts = _ring_chain_to_points(chain_px, z_mm=3.0, px_per_mm=px_per_mm, H=H, _ox=0.0, _oy=0.0)
check("row 0 maps to the top of the canvas (max Y)",
      np.isclose(pts[0][1], (H - 1) / px_per_mm))
check("col 0 maps to the left edge of the canvas (X = 0)",
      np.isclose(pts[0][0], 0.0))
check("z is the negated flat floor depth for every point",
      all(np.isclose(p[2], -3.0) for p in pts))
check("row/col offsets scale by px_per_mm correctly for the second point",
      np.isclose(pts[1][0], 20.0 / px_per_mm) and
      np.isclose(pts[1][1], (H - 1 - 10.0) / px_per_mm))


# --------------------------------------------------------------------------
# _order_fixed_orientation_pass: regression test for a real bug -- ordering
# multiple onion-skin regions under an unlocked cut direction must never
# reverse a whole region's chain, since that silently swaps its
# Outside-In/Inside-Out winding. Each segment here stands in for one
# region's whole ring chain (as compute_pocket_toolpath emits it): its own
# entry/exit points must survive untouched, no matter which one the
# nearest-neighbour search would rather enter from.
# --------------------------------------------------------------------------
print("_order_fixed_orientation_pass")

# seg_a's *exit* point sits exactly at the origin -- a reversal-capable
# ordering pass (like _order_free_raster_pass) would jump at the chance to
# enter it there at zero cost, reversing it. seg_b's entry is merely close
# (not free) but requires no reversal. A fixed-orientation pass must still
# prefer seg_b, since it never considers entering a segment from its exit.
seg_a = [(10.0, 0.0), (0.0, 0.0)]
seg_b = [(5.0, 0.0), (5.0, 1.0)]
pass_segs = [seg_a, seg_b]
ordered = _order_fixed_orientation_pass(pass_segs, origin=(0.0, 0.0))
check("no segment's point order is altered (identity, not just equal-valued)",
      ordered[0] is pass_segs[0] or ordered[0] is pass_segs[1])
check("the near-zero-cost *reversed* entry into seg_a is never taken -- "
      "seg_b (whose own entry point is merely close) is visited first",
      ordered[0] == seg_b)
check("both segments are present exactly once, each unreversed",
      len(ordered) == 2 and seg_a in ordered and seg_b in ordered)


# --------------------------------------------------------------------------
# regression test: Onion Skin (Inside-Out) under a locked cut direction, on
# a region that erosion forks into sibling chains (a big lobe and a small
# lobe). Outside-In's chain list already runs shallow-to-deep overall (the
# shared outer rings first, each lobe's own deeper rings after), so under a
# locked direction (ordered_segs = pass_segs, unchanged) it plays back
# correctly as-is. Inside-Out must reverse not just each chain's own point
# order but also which chain is visited first -- otherwise every individual
# chain runs backwards (deep-to-shallow) while the chains themselves still
# play back in their original shallow-to-deep list order, so cutting jumps
# between unrelated depths from one chain to the next instead of
# progressing deep-to-shallow overall.
# --------------------------------------------------------------------------
print("Onion Skin (Inside-Out) under a locked direction (regression)")

shape4 = (100, 220)
big_c4, small_c4 = (50, 50), (50, 190)
big_r4, small_r4 = 45, 12
neck4 = np.zeros(shape4, dtype=bool)
neck4[47:53, 50:190] = True
mask4 = disk_mask(shape4, big_c4, big_r4) | disk_mask(shape4, small_c4, small_r4) | neck4
depth4 = np.zeros(shape4, dtype=np.float32)
depth4[mask4] = 1.0

tool4 = Tool(profile="End Mill", diameter_mm=1.0, vbit_angle_deg=90.0)
px_per_mm4 = shape4[1] / 220.0


def _mm_center(rc):
    row, col = rc
    return (col / px_per_mm4, (shape4[0] - 1 - row) / px_per_mm4)


big_mm, small_mm = _mm_center(big_c4), _mm_center(small_c4)

tp = compute_pocket_toolpath(
    depth_map=depth4, n_grey_levels=256, tool=tool4,
    step_mm=0.5, stepdown_mm=0.0, max_depth=3.0,
    width_mm=220.0, height_mm=100.0, raster_dir="Onion Skin (Inside-Out)",
    target_z_mm=3.0, feed_rate=800.0, cut_direction="Climb",
    progress=lambda pct, msg: None,
)


def _dist_mm(pt, center_mm):
    return ((pt[0] - center_mm[0]) ** 2 + (pt[1] - center_mm[1]) ** 2) ** 0.5


first_seg, last_seg = tp.segments[0], tp.segments[-1]
first_start_depth = min(_dist_mm(first_seg[0], big_mm), _dist_mm(first_seg[0], small_mm))
last_end_depth = min(_dist_mm(last_seg[-1], big_mm), _dist_mm(last_seg[-1], small_mm))
check("cutting starts at a deep point (near a lobe's own centre), not the outer boundary",
      first_start_depth < 5.0)
check("cutting ends at a shallow point (near the outer boundary), not a lobe's centre",
      last_end_depth > big_r4 - 5.0)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
