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
                        _order_fixed_orientation_pass, compute_pocket_toolpath,
                        _diag_grid_line_ids, _diag_line_bounds, _diag_line_points,
                        _region_contour_rings, _make_contour_ring,
                        _nearest_ring_vertex, _ring_arc_between,
                        _order_locked_raster_pass, _order_zigzag_pass,
                        _order_free_region_pass, prune_air_segments,
                        compute_air_free_depth, _retract_breakeven_mm, _build_kernel,
                        simulate_toolpath_result, compute_raster_toolpath)
from tool import Tool
from toolpath import ToolPath

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
# well under min_area_px, unlike a genuine second lobe. Discarding it must
# also not, by itself, count as a fork: the main lobe -- the only
# substantial survivor -- must continue as ONE unbroken chain rather than
# being closed off and restarted the moment the split is detected, which
# would silently break an ordinary, non-forking pocket's ring sequence into
# two chains for no real reason (and let downstream ordering interleave
# them with an unrelated pocket in between, since ordering has no idea
# they were ever the same pocket).
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
check("discarding the stray fragment is not itself treated as a fork -- "
      "the main lobe stays exactly one continuous chain, not two",
      len(chains3) == 1)


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


# --------------------------------------------------------------------------
# regression test: even under a locked cut direction, visiting order between
# *separate* pockets should still minimise travel -- a locked direction only
# needs each chain's own point order left untouched (its winding is decided
# at generation time), not the pockets themselves visited in whatever
# arbitrary order scipy.ndimage.label happened to discover them in. The
# "far" pocket is placed so label discovers it FIRST (small row index, top
# of the image) even though it is geometrically far from the tool's actual
# (0,0) mm start under a Bottom-Left origin; the "near" pocket is discovered
# SECOND (large row index) despite sitting right next to the start.
# --------------------------------------------------------------------------
print("cross-pocket travel order survives a locked cut direction (regression)")

shape5 = (100, 100)
far_c5, near_c5 = (5, 90), (95, 5)
depth5 = np.zeros(shape5, dtype=np.float32)
depth5[disk_mask(shape5, far_c5, 10)] = 1.0
depth5[disk_mask(shape5, near_c5, 10)] = 1.0

tool5 = Tool(profile="End Mill", diameter_mm=1.0, vbit_angle_deg=90.0)
px_per_mm5 = shape5[1] / 100.0


def _mm5(rc):
    row, col = rc
    return (col / px_per_mm5, (shape5[0] - 1 - row) / px_per_mm5)


far_mm5, near_mm5 = _mm5(far_c5), _mm5(near_c5)

tp5 = compute_pocket_toolpath(
    depth_map=depth5, n_grey_levels=256, tool=tool5,
    step_mm=0.5, stepdown_mm=0.0, max_depth=3.0,
    width_mm=100.0, height_mm=100.0, raster_dir="Onion Skin (Outside-In)",
    target_z_mm=3.0, feed_rate=800.0, cut_direction="Climb", origin="Bottom-Left",
    progress=lambda pct, msg: None,
)
check("exactly two chains (one per separate pocket)", len(tp5.segments) == 2)
first_pt = tp5.segments[0][0]
d_near = ((first_pt[0] - near_mm5[0]) ** 2 + (first_pt[1] - near_mm5[1]) ** 2) ** 0.5
d_far = ((first_pt[0] - far_mm5[0]) ** 2 + (first_pt[1] - far_mm5[1]) ** 2) ** 0.5
check("the pocket nearest the tool's actual start is visited first, "
      "not whichever pocket label() happened to discover first",
      d_near < d_far)


# --------------------------------------------------------------------------
# regression test: the original reported bug, combined -- one pocket whose
# own erosion forks into sibling lobes, alongside a second, wholly separate
# pocket. The forked pocket's own chains (parent + each lobe) must all stay
# adjacent, in one contiguous run, never interleaved with the unrelated
# pocket's chain in between.
# --------------------------------------------------------------------------
print("a forked pocket's chains never interleave with an unrelated pocket (regression)")

shape6 = (100, 300)
big_c6, small_c6, third_c6 = (50, 50), (50, 190), (10, 280)
depth6 = np.zeros(shape6, dtype=np.float32)
depth6[disk_mask(shape6, big_c6, 45)] = 1.0
depth6[disk_mask(shape6, small_c6, 12)] = 1.0
depth6[47:53, 50:190] = 1.0   # neck joining big_c6 and small_c6 into one forking pocket
depth6[disk_mask(shape6, third_c6, 10)] = 1.0   # unrelated, separate pocket

tool6 = Tool(profile="End Mill", diameter_mm=1.0, vbit_angle_deg=90.0)
tp6 = compute_pocket_toolpath(
    depth_map=depth6, n_grey_levels=256, tool=tool6,
    step_mm=0.5, stepdown_mm=0.0, max_depth=3.0,
    width_mm=300.0, height_mm=100.0, raster_dir="Onion Skin (Outside-In)",
    target_z_mm=3.0, feed_rate=800.0, cut_direction="Climb",
    progress=lambda pct, msg: None,
)
px_per_mm6 = shape6[1] / 300.0


def _mm6(rc):
    row, col = rc
    return (col / px_per_mm6, (shape6[0] - 1 - row) / px_per_mm6)


third_mm6 = _mm6(third_c6)
is_third = [dist_from(np.array(seg), third_mm6).min() < 15 for seg in tp6.segments]
check("the unrelated pocket contributes exactly one chain",
      sum(is_third) == 1)
forked_indices = [i for i, v in enumerate(is_third) if not v]
check("the forked pocket's chains form one contiguous run "
      "(no unrelated-pocket chain sandwiched in between)",
      forked_indices == list(range(min(forked_indices), max(forked_indices) + 1)))


# --------------------------------------------------------------------------
# _diag_grid_line_ids / _diag_line_bounds
# --------------------------------------------------------------------------
print("_diag_grid_line_ids / _diag_line_bounds")

H = W = 5   # rows/cols 0..4, row+col in [0,8], row-col in [-4,4]

k_ids, adj_step = _diag_grid_line_ids(H, W, "D45", px_per_mm=1.0,
                                      step_mm=1.0, raster_offset_pct=0.0)
check("D45 k_ids span the full row+col range", k_ids[0] == 0 and k_ids[-1] == 8)

k_ids135, _ = _diag_grid_line_ids(H, W, "D135", px_per_mm=1.0,
                                  step_mm=1.0, raster_offset_pct=0.0)
check("D135 k_ids span the full row-col range", k_ids135[0] == -4 and k_ids135[-1] == 4)

check("D45 bounds on the main diagonal (k=4) span the whole box",
      _diag_line_bounds("D45", 4, 0, 4, 0, 4) == (0, 4))
check("D45 bounds at a corner (k=8) reduce to a single point",
      _diag_line_bounds("D45", 8, 0, 4, 0, 4) == (4, 4))
check("D45 bounds for a k outside the box's range return None",
      _diag_line_bounds("D45", 20, 0, 4, 0, 4) is None)
check("D135 bounds on the main diagonal (k=0) span the whole box",
      _diag_line_bounds("D135", 0, 0, 4, 0, 4) == (0, 4))


# --------------------------------------------------------------------------
# _diag_line_points: sampling stays on the diagonal, and mask gaps split
# a line into multiple segments the same way _line_points does.
# --------------------------------------------------------------------------
print("_diag_line_points")

depth = np.ones((5, 5), dtype=np.float32)
segs = _diag_line_points("D45", 4, 0, 4, False, depth, None, 0.5,
                         px_per_mm=1.0, H=5, _ox=0.0, _oy=0.0, max_depth=1.0)
check("D45 k=4, no mask: one segment covering all 5 points",
      len(segs) == 1 and len(segs[0]) == 5)
pts_rc = [(round(5 - 1 - y), round(x)) for x, y, z in segs[0]]
check("every D45 k=4 point satisfies row+col == 4",
      all(r + c == 4 for r, c in pts_rc))

gap_mask = np.ones((5, 5), dtype=np.float32)
gap_mask[2, 2] = 0.0   # the middle point of the k=4 diagonal (row=2,col=2)
segs_gap = _diag_line_points("D45", 4, 0, 4, False, depth, gap_mask, 0.5,
                             px_per_mm=1.0, H=5, _ox=0.0, _oy=0.0, max_depth=1.0)
check("a single masked-out point splits the diagonal into 2 segments",
      len(segs_gap) == 2)
check("the two split segments have 2 points each (5 total minus the gap)",
      sum(len(s) for s in segs_gap) == 4)


# --------------------------------------------------------------------------
# _order_locked_raster_pass: 4-way coord() key
# --------------------------------------------------------------------------
print("_order_locked_raster_pass 4-way grouping")

seg_a = [(0.0, 0.0, 0.0)]
seg_b = [(1.0, 1.0, 0.0)]
for direction in ("X", "Y", "D45", "D135"):
    ordered = _order_locked_raster_pass([seg_b, seg_a], direction, (0.0, 0.0))
    check(f"_order_locked_raster_pass runs for direction={direction!r}",
          len(ordered) == 2)


# --------------------------------------------------------------------------
# _region_contour_rings / _make_contour_ring: ring count and arc-length table
# --------------------------------------------------------------------------
print("_region_contour_rings / _make_contour_ring")

solid = np.ones((10, 10), dtype=bool)
rings = _region_contour_rings(solid)
check("a solid block has exactly one boundary ring", len(rings) == 1)

annulus_zz = np.ones((20, 20), dtype=bool)
annulus_zz[8:12, 8:12] = False
rings_annulus = _region_contour_rings(annulus_zz)
check("an annulus (region with a hole) has exactly two rings",
      len(rings_annulus) == 2)

ring = _make_contour_ring(rings[0])
check("_make_contour_ring drops find_contours' duplicate closing vertex",
      not np.allclose(ring.pts_px[0], ring.pts_px[-1]))
check("cumulative arc length is non-decreasing",
      np.all(np.diff(ring.cum) >= -1e-9))
check("ring perimeter L is positive and finite",
      ring.L > 0 and np.isfinite(ring.L))


# --------------------------------------------------------------------------
# _nearest_ring_vertex / _ring_arc_between: the "shorter arc never leaves
# the region" property, on a concave (C-shaped) region with a notch.
# --------------------------------------------------------------------------
print("_nearest_ring_vertex / _ring_arc_between")

c_shape = np.zeros((30, 30), dtype=bool)
c_shape[2:28, 2:28] = True
c_shape[10:20, 14:28] = False   # notch cut into the right side
ring_c = _make_contour_ring(_region_contour_rings(c_shape)[0])

# Two points that sit close together in a straight line (both near the
# notch's open right edge, on the top/bottom arm tips) but are only
# reachable by walking the long way around the notch along the boundary.
i_top = _nearest_ring_vertex(ring_c, row=2, col=27, tol_px=2.0)
i_bottom = _nearest_ring_vertex(ring_c, row=27, col=27, tol_px=2.0)
check("both notch-tip points snap onto the C-shape's ring", i_top is not None and i_bottom is not None)

check("_nearest_ring_vertex returns None past its tolerance",
      _nearest_ring_vertex(ring_c, row=15, col=20, tol_px=1.0) is None)   # inside the notch (empty space)

if i_top is not None and i_bottom is not None:
    arc = _ring_arc_between(ring_c, i_top, i_bottom)
    arc_len = float(np.sum(np.hypot(np.diff(arc[:, 0]), np.diff(arc[:, 1]))))
    straight_dist = float(np.hypot(arc[0, 0] - arc[-1, 0], arc[0, 1] - arc[-1, 1]))
    check("the contour arc detours well around the notch instead of cutting across it",
          arc_len > straight_dist * 2)
    check("_ring_arc_between picks the shorter of the two ring directions",
          arc_len <= ring_c.L / 2.0 + 1e-6)
    check("every arc point is one of the ring's own vertices (never leaves the region boundary)",
          all(any(np.allclose(p, v) for v in ring_c.pts_px) for p in arc))


# --------------------------------------------------------------------------
# compute_pocket_toolpath: Zig-Zag end-to-end smoke tests
# --------------------------------------------------------------------------
print("compute_pocket_toolpath Zig-Zag")

_noop_progress = lambda pct, msg: None
zz_tool = Tool(profile="End Mill", diameter_mm=1.0)

# A concave "C"-shaped pocket: a scan line crossing the notch splits into
# two segments, exercising contour-linking around a concave boundary.
depth_c = np.zeros((30, 30), dtype=np.float32)
depth_c[2:28, 2:28] = 1.0
depth_c[10:20, 14:28] = 0.0

tp_zz = compute_pocket_toolpath(
    depth_c, n_grey_levels=256, tool=zz_tool,
    step_mm=2.0, stepdown_mm=0.0, max_depth=5.0,
    width_mm=30.0, height_mm=30.0, raster_dir="Zig-Zag (0°)",
    cut_direction="Either", progress=_noop_progress,
)
check("Zig-Zag on a concave pocket produces a non-empty toolpath",
      len(tp_zz.segments) > 0)
check("Zig-Zag's contour linking fires at least once on a concave pocket",
      tp_zz.metadata.get('n_links_contour', 0) >= 1)
check("n_links_contour never exceeds n_links_inserted",
      tp_zz.metadata['n_links_contour'] <= tp_zz.metadata['n_links_inserted'])

# A pocket region with a hole (annulus): some gap endpoints resolve to the
# same ring (contour link), others don't (falls back to the geodesic link).
depth_annulus = np.zeros((30, 30), dtype=np.float32)
depth_annulus[4:26, 4:26] = 1.0
depth_annulus[12:18, 12:18] = 0.0

tp_hole = compute_pocket_toolpath(
    depth_annulus, n_grey_levels=256, tool=zz_tool,
    step_mm=2.0, stepdown_mm=0.0, max_depth=5.0,
    width_mm=30.0, height_mm=30.0, raster_dir="Zig-Zag (90°)",
    cut_direction="Either", progress=_noop_progress,
)
check("Zig-Zag on a region with a hole produces a non-empty toolpath",
      len(tp_hole.segments) > 0)
check("the annulus is a single connected region despite its hole",
      tp_hole.metadata['n_regions'] == 1)

# Two disjoint pocket regions close enough to link across: contour-linking
# must not fire (the two endpoints belong to different regions), so this
# must fall back to the geodesic link without raising.
depth_two = np.zeros((30, 30), dtype=np.float32)
depth_two[2:10, 2:10] = 1.0
depth_two[2:10, 13:21] = 1.0   # separated from the first block by 2 empty columns

tp_two = compute_pocket_toolpath(
    depth_two, n_grey_levels=256, tool=zz_tool,
    step_mm=2.0, stepdown_mm=0.0, max_depth=5.0,
    width_mm=30.0, height_mm=30.0, raster_dir="Zig-Zag (0°)",
    cut_direction="Either", progress=_noop_progress,
)
check("two disjoint regions are computed without error", len(tp_two.segments) > 0)
check("two disjoint regions are detected as such", tp_two.metadata['n_regions'] == 2)
sides = ["A" if seg[0][0] < 11.0 else "B" for seg in tp_two.segments]
switches = sum(1 for i in range(1, len(sides)) if sides[i] != sides[i - 1])
check("the two disjoint regions' segments are never interleaved "
      "(one region's segments finish before the other's begin)",
      switches <= 1)

for angle in ("Zig-Zag (0°)", "Zig-Zag (45°)", "Zig-Zag (90°)", "Zig-Zag (135°)"):
    tp_angle = compute_pocket_toolpath(
        depth_c, n_grey_levels=256, tool=zz_tool,
        step_mm=3.0, stepdown_mm=0.0, max_depth=5.0,
        width_mm=30.0, height_mm=30.0, raster_dir=angle,
        cut_direction="Climb", progress=_noop_progress,
    )
    check(f"{angle} under locked Climb direction produces a toolpath",
          len(tp_angle.segments) > 0)


# --------------------------------------------------------------------------
# _order_zigzag_pass: never interleaves two regions, and each region's own
# entry corner depends on the incoming tool position rather than always
# being the same fixed corner.
# --------------------------------------------------------------------------
print("_order_zigzag_pass")

# direction="Y": lines are vertical, grouped by constant X -- one region's
# two "lines" (single-point segments here, for simplicity) sit at x=0 and
# x=10; a second, disjoint region's lines sit at x=100 and x=110.
region1 = {1: [(0.0, 0.0, 0.0)], 2: [(10.0, 0.0, 0.0)]}
region2 = {1: [(100.0, 0.0, 0.0)], 2: [(110.0, 0.0, 0.0)]}
pass_segs_zz = [region1[1], region1[2], region2[1], region2[2]]
seg_region_ids_zz = [1, 1, 2, 2]

ordered_zz = _order_zigzag_pass(pass_segs_zz, seg_region_ids_zz, "Y", cur=(2.0, 0.0),
                                lock_direction=True)
region_seq = [1 if seg[0][0] < 50.0 else 2 for seg in ordered_zz]
check("the nearer region (region 1) is visited first",
      region_seq[0] == 1)
check("region 1's two segments are never separated by region 2's",
      region_seq == [1, 1, 2, 2])
check("region 1 enters at x=0 (nearer to cur=(2,0) than x=10)",
      ordered_zz[0][0][0] == 0.0)
check("region 2 enters at x=100 (nearer to region 1's exit at x=10 than x=110)",
      ordered_zz[2][0][0] == 100.0)

# Same single region's two lines, entered from opposite starting points
# depending purely on which is nearer the incoming cur -- not always the
# same corner.
one_region = [(0.0, 0.0, 0.0)], [(10.0, 0.0, 0.0)]
near_left = _order_zigzag_pass(list(one_region), [1, 1], "Y", cur=(2.0, 0.0),
                               lock_direction=True)
near_right = _order_zigzag_pass(list(one_region), [1, 1], "Y", cur=(8.0, 0.0),
                                lock_direction=True)
check("a region enters at its leftmost line when cur is nearer that end",
      near_left[0][0][0] == 0.0)
check("the SAME region enters at its rightmost line when cur is nearer that "
      "end instead -- the entry corner is not fixed",
      near_right[0][0][0] == 10.0)


# --------------------------------------------------------------------------
# _order_free_region_pass: under an unlocked (Either) cut direction, each
# line must be free to flip its own entry end for a short, alternating
# zig-zag -- regression test for a real bug where reusing the
# locked-direction sweep for Either forced every line back to the same
# starting side every time, turning every line-to-line gap into a full
# return trip across the region instead of a short hop to the near end.
# --------------------------------------------------------------------------
print("_order_free_region_pass")

# direction="X": three horizontal lines (as generation always stores them,
# left-to-right) stacked at y=0, 2, 4, each spanning x=0..10.
line_y0 = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
line_y2 = [(0.0, 2.0, 0.0), (10.0, 2.0, 0.0)]
line_y4 = [(0.0, 4.0, 0.0), (10.0, 4.0, 0.0)]
free_ordered = _order_free_region_pass([line_y0, line_y2, line_y4], "X", cur=(2.0, 0.0))

check("all three lines are present in the free-ordered result",
      len(free_ordered) == 3)
gaps = [((free_ordered[i][-1][0] - free_ordered[i + 1][0][0]) ** 2 +
        (free_ordered[i][-1][1] - free_ordered[i + 1][0][1]) ** 2) ** 0.5
       for i in range(len(free_ordered) - 1)]
check("consecutive lines are entered from alternating ends, keeping every "
      "line-to-line gap short (a full return trip across the region would "
      "be 10mm; a same-side alternating entry is ~2mm)",
      all(g < 3.0 for g in gaps))


# --------------------------------------------------------------------------
# Air-cut pruning (prune_air_segments / compute_air_free_depth): runs where
# the tool's tip is no deeper than what previous_result already cleared
# under its footprint are removed; everything that touches material stays.
# --------------------------------------------------------------------------
print("prune_air_segments")

AIR_PX = 4.0        # px/mm
AIR_H = AIR_W = 80  # 20 x 20 mm
AIR_MD = 5.0        # max depth mm
AIR_TOL = 1.0 / 255


def _air_prune(segs, free, breakeven_fn=None, parent_ids=None):
    return prune_air_segments(segs, free, AIR_PX, AIR_H, 0.0, 0.0, AIR_MD, AIR_TOL,
                              breakeven_fn=breakeven_fn, parent_ids=parent_ids)


# Left half fully cleared (free depth 1.0), right half untouched stock.
free_half = np.zeros((AIR_H, AIR_W), dtype=np.float32)
free_half[:, :40] = 1.0
line = [(0.0, 10.0, -2.5), (19.75, 10.0, -2.5)]
out, _, air = _air_prune([line], free_half)
check("a line half over cleared stock keeps exactly one piece", len(out) == 1)
check("the kept piece starts at the cleared/uncleared boundary (within 1 px)",
      len(out) == 1 and abs(out[0][0][0] - 40 / AIR_PX) <= 1.0 / AIR_PX)
check("the kept piece ends at the line's own end", len(out) == 1 and out[0][-1] == line[-1])
check("the reported air length matches what was removed (within 1 px)",
      abs(air - 40 / AIR_PX) <= 1.0 / AIR_PX)

# A narrow cleared band crossing the line: short interior air run.
free_band = np.zeros((AIR_H, AIR_W), dtype=np.float32)
free_band[:, 38:42] = 1.0     # 1 mm wide
breakeven = lambda d: _retract_breakeven_mm(800.0, 3000.0, 5.0, d)   # ~4 mm at 2.5 mm deep
out, _, _ = _air_prune([line], free_band, breakeven_fn=breakeven)
check("an interior air run shorter than the retract breakeven is fed through, not split",
      len(out) == 1 and out[0][0] == line[0] and out[0][-1] == line[-1])
out, _, _ = _air_prune([line], free_band, breakeven_fn=None)
check("without a breakeven, the same interior air run splits the line in two", len(out) == 2)

# Nothing cleared: the segment comes back untouched; all cleared: dropped.
out, _, air = _air_prune([line], np.zeros((AIR_H, AIR_W), dtype=np.float32))
check("a segment that cuts material everywhere is returned unchanged",
      out == [line] and air == 0.0)
out, _, _ = _air_prune([line], np.ones((AIR_H, AIR_W), dtype=np.float32))
check("a segment that cuts only air is dropped", out == [])

# A closed square loop whose left side lies over cleared stock becomes one
# open piece (the air run is rotated to the loop's ends), not two.
square = [(4.0, 4.0, -2.5), (4.0, 16.0, -2.5), (16.0, 16.0, -2.5),
          (16.0, 4.0, -2.5), (4.0, 4.0, -2.5)]
free_left = np.zeros((AIR_H, AIR_W), dtype=np.float32)
free_left[:, :24] = 1.0      # x < 6 mm cleared: the whole x=4 side
out, _, _ = _air_prune([square], free_left, breakeven_fn=None)
check("a closed loop with one air side becomes a single open piece", len(out) == 1)
check("that piece no longer visits the cleared side",
      len(out) == 1 and all(p[0] > 4.0 + 1e-9 for p in out[0]))

# parent_ids follow their segment through splits and drops.
seg_a = [(0.0, 5.0, -2.5), (19.75, 5.0, -2.5)]
seg_b = [(1.0, 15.0, -2.5), (8.0, 15.0, -2.5)]      # ends short of the band: untouched
seg_c = [(0.0, 12.0, -2.5), (19.75, 12.0, -2.5)]
out, ids, _ = _air_prune([seg_a, seg_b, seg_c], free_band, breakeven_fn=None,
                         parent_ids=["a", "b", "c"])
check("parent ids are remapped one per output segment", len(ids) == len(out))
check("each split segment's pieces keep their source id, in order",
      ids == ["a", "a", "b", "c", "c"])

# Invariant: removing the air never leaves material that the unpruned path
# would have cut -- compared via the same swept-volume simulation used for
# every step's result map, allowing one pixel of rim jitter (the pieces are
# resampled from a different phase) and one grey level of depth.
from scipy.ndimage import maximum_filter
rng = np.random.default_rng(7)
yy, xx = np.mgrid[0:AIR_H, 0:AIR_W]
worst = 0.0
for trial, tool in enumerate([Tool(profile="Ball", diameter_mm=3.0),
                              Tool(profile="End Mill", diameter_mm=2.0),
                              Tool(profile="V-Bit", diameter_mm=6.0, vbit_angle_deg=60)] * 3):
    k, fp = _build_kernel(tool, AIR_PX, AIR_MD)
    cy, cx = rng.uniform(20, 60, 2)
    prev = np.clip(0.7 - np.hypot(yy - cy, xx - cx) / 30, 0, 1).astype(np.float32)
    free = compute_air_free_depth(prev, k, fp, (AIR_H, AIR_W))
    segs = []
    for _ in range(10):
        n = int(rng.integers(2, 7))
        pts = np.column_stack([rng.uniform(0, (AIR_W - 1) / AIR_PX, n),
                               rng.uniform(0, (AIR_H - 1) / AIR_PX, n),
                               -rng.uniform(0, 0.9 * AIR_MD, n)])
        segs.append([tuple(p) for p in pts])
    out, _, _ = _air_prune(segs, free, breakeven_fn=breakeven)
    sim = lambda s: simulate_toolpath_result(ToolPath(segments=s), (AIR_H, AIR_W),
                                             AIR_PX, AIR_PX, 0.0, 0.0, AIR_MD, k, fp)
    r_off = np.maximum(prev, sim(segs))
    r_on = np.maximum(prev, sim(out))
    worst = max(worst, float((r_off - maximum_filter(r_on, size=3)).max()))
check("pruned paths remove the same material as unpruned ones (random segments, "
      "Ball/End Mill/V-Bit)", worst <= AIR_TOL)


print("avoid_air in compute_raster_toolpath")

# A dome, with the left half already carved to its final depth by a
# "previous step". 10 px/mm keeps the RDP tolerance small enough (0.1 mm)
# that it can't mask a pruning error in the result comparison.
dome_yy, dome_xx = np.mgrid[0:120, 0:120]
dome = (np.clip(1 - np.hypot(dome_yy - 60, dome_xx - 60) / 50, 0, 1) * 0.9).astype(np.float32)
dome_prev = np.where(dome_xx < 60, dome, 0).astype(np.float32)


def _raster(avoid_air, raster_dir):
    return compute_raster_toolpath(
        dome, n_grey_levels=256, tool=Tool(profile="Ball", diameter_mm=0.6),
        step_mm=0.2, stepdown_mm=1.0, max_depth=5.0, width_mm=12.0, height_mm=12.0,
        raster_dir=raster_dir, previous_result=dome_prev, cut_direction="Climb",
        avoid_air=avoid_air, progress=_noop_progress)


def _path_len(tp):
    return sum(float(np.hypot(*np.diff(np.asarray(s)[:, :2], axis=0).T).sum())
               for s in tp.segments if len(s) > 1)


for rdir in ("Horizontal", "Diagonal (45°)", "Spiral (Outside-In)"):
    tp_off, tp_on = _raster(False, rdir), _raster(True, rdir)
    deficit = float(np.max(np.minimum(tp_off.result, tp_off.accessible)
                           - maximum_filter(tp_on.result, size=3)))
    check(f"{rdir}: avoid_air removes feed length", _path_len(tp_on) < 0.7 * _path_len(tp_off))
    check(f"{rdir}: avoid_air leaves no material the unpruned path removes",
          deficit <= AIR_TOL)
    check(f"{rdir}: air_pruned_mm is reported", tp_on.metadata['air_pruned_mm'] > 0.0)


print("avoid_air in compute_pocket_toolpath")

# Two disjoint pocket regions, the left one's left half already cleared to
# the floor: splitting must keep each piece's region id so regions are
# still never interleaved.
for rdir in ("Zig-Zag (0°)", "Onion Skin (Outside-In)", "Spiral (Outside-In)"):
    prev_two = np.zeros((30, 30), dtype=np.float32)
    prev_two[:, :6] = 1.0
    tp_two_air = compute_pocket_toolpath(
        depth_two, n_grey_levels=256, tool=zz_tool,
        step_mm=1.0, stepdown_mm=1.0, max_depth=5.0,
        width_mm=30.0, height_mm=30.0, raster_dir=rdir, target_z_mm=2.0,
        previous_result=prev_two, cut_direction="Climb", avoid_air=True,
        progress=_noop_progress,
    )
    check(f"{rdir}: pocket with avoid_air is computed without error",
          len(tp_two_air.segments) > 0)
    sides = ["A" if sg[0][0] < 11.0 else "B" for sg in tp_two_air.segments]
    if rdir.startswith("Zig-Zag"):
        passes_switches = sum(1 for i in range(1, len(sides)) if sides[i] != sides[i - 1])
        # stepdown 1 mm to 2 mm: two passes, each may switch region once,
        # plus once between passes.
        check(f"{rdir}: regions are still never interleaved within a pass",
              passes_switches <= 3)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
