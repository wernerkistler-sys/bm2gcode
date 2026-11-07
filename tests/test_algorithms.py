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
                        _region_contour_rings, _make_contour_ring, _biarc, _sample_biarc,
                        _nearest_ring_vertex, _ring_arc_between,
                        _order_locked_raster_pass, _order_zigzag_pass,
                        _order_free_region_pass, prune_air_segments,
                        compute_air_free_depth, _retract_breakeven_mm, _build_kernel,
                        simulate_toolpath_result, compute_raster_toolpath,
                        registration_flip_axis, registration_hole_centers,
                        registration_auto_spacing, bore_hole_segments,
                        compute_registration_holes_toolpath, _spiral_px,
                        _extract_z_contours, _order_contours_graph,
                        _fill_level_gaps, _prune_contours,
                        compute_contour_toolpath, _geodesic_link,
                        _plan_passes, make_multipass, band_levels, check_z_range)
from tool import Tool
from toolpath import ToolPath
import tool_profile

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
# ring-to-ring links: every move between two consecutive chain points --
# ring edges, S-shaped biarc links, and straight fallback links alike --
# must stay inside the region (never gouging material outside a concave
# pocket boundary), and a straight fallback link between consecutive
# erosion levels should be about one step-over long, not a long chord
# across a cusp. "Inside" is the same 0.5 iso-level of the
# bilinearly-interpolated mask that find_contours traces, with a small
# tolerance for marching squares' own straight-chord approximation of
# that level set.
# --------------------------------------------------------------------------
print("ring-to-ring links stay inside the region and stay short")

from scipy.ndimage import map_coordinates


def link_jumps(chain, min_len=1.5):
    """(p, q) pairs of consecutive chain points further apart than a
    single find_contours edge, i.e. straight links between two rings
    (a biarc link is sampled finely enough to never show up here)."""
    steps = np.hypot(*(np.diff(chain, axis=0).T))
    return [(chain[i], chain[i + 1]) for i in np.nonzero(steps > min_len)[0]]


def all_moves_inside(mask, chains, tol=0.1):
    """Every straight move between consecutive points of every chain,
    sampled at most 0.25 px apart, stays inside mask."""
    mask_f = np.pad(mask, 1).astype(np.float32)
    for c in chains:
        p, q = c[:-1], c[1:]
        n = max(2, int(np.ceil(np.hypot(*(q - p).T).max() / 0.25)) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts = p[:, None, :] + t[None, :, None] * (q - p)[:, None, :]
        vals = map_coordinates(mask_f, pts.reshape(-1, 2).T + 1.0, order=1,
                               mode="constant", cval=0.0)
        if np.any(vals < 0.5 - tol):
            return False
    return True


def max_link(chains):
    lens = [np.hypot(*(q - p)) for c in chains for p, q in link_jumps(c)]
    return max(lens) if lens else 0.0


rr, cc = np.mgrid[0:120, 0:120]
u_shape = np.zeros((120, 120), dtype=bool)
u_shape[10:110, 10:110] = True
u_shape[10:90, 40:80] = False                     # deep notch from the top
l_shape = np.zeros((120, 120), dtype=bool)
l_shape[10:110, 10:45] = True
l_shape[75:110, 10:110] = True
crescent = disk_mask((120, 120), (60, 60), 50) & ~disk_mask((120, 120), (40, 60), 42)
triangle = (rr >= 10) & (rr <= 110) & (np.abs(cc - 60) <= (rr - 10) * 0.25)
rect = np.zeros((80, 120), dtype=bool)
rect[10:70, 10:110] = True
stub = np.zeros((60, 60), dtype=bool)
stub[5:32, 10:45] = True
stub[32:39, 10:14] = True                         # narrow stub off one corner, gone after one erosion

for name, mask, step in (("corner stub", stub, 4), ("U shape", u_shape, 5), ("L shape", l_shape, 5),
                         ("crescent", crescent, 4), ("acute triangle", triangle, 3),
                         ("stray-fragment cusp", cusp_mask, step_px3),
                         ("rectangle", rect, 5), ("disk", disk, 4)):
    min_area = max(4, int(round(np.pi * (0.5 * step) ** 2)))
    for rev in (False, True):
        cs = _onion_skin_chains(mask, step_px=step, min_area_px=min_area,
                                reverse_rings=rev)
        tag = f"{name}{' (reversed rings)' if rev else ''}"
        check(f"{tag}: every move of every chain stays inside the region",
              all_moves_inside(mask, cs))
        check(f"{tag}: no straight link is longer than about one diagonal step-over "
              f"(max {max_link(cs):.1f} px, step {step} px)",
              max_link(cs) <= step * np.sqrt(2) + 2.0)

check("annulus: every move stays inside the region",
      all_moves_inside(annulus, _onion_skin_chains(annulus, step_px=15, min_area_px=20)))
annulus_thick = (disk_mask((120, 120), (60, 60), 55)
                 & ~disk_mask((120, 120), (60, 60), 15))
check("thick annulus: every move stays inside the region",
      all_moves_inside(annulus_thick,
                       _onion_skin_chains(annulus_thick, step_px=4, min_area_px=20)))


# --------------------------------------------------------------------------
# S-shaped ring transitions: wherever a ring is big enough to hold arcs of
# one step-over radius, the chain changes ring along a tangent-continuous
# biarc, so its heading (measured over a few px, to see past find_contours'
# pixel stair-steps) never turns sharply. A straight chord between rings
# turns by roughly 90 degrees twice.
# --------------------------------------------------------------------------
print("S-shaped ring transitions")


def dedup(chain):
    """chain with consecutive duplicate points dropped."""
    return np.vstack([chain[:1], chain[1:][np.hypot(*np.diff(chain, axis=0).T) > 1e-9]])


def smoothed_turns(chain, window=3.0):
    """Heading change in degrees at every point of a (deduplicated) chain,
    comparing the chord from `window` px of arc length behind with the
    chord to `window` px ahead, plus a mask of the points far enough from
    either end for both chords to exist."""
    cum = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(chain, axis=0).T))))
    i = np.arange(len(chain))
    behind = np.clip(np.searchsorted(cum, cum - window), 0, len(chain) - 1)
    ahead = np.clip(np.searchsorted(cum, cum + window), 0, len(chain) - 1)
    ok = (cum - window >= 0) & (cum + window <= cum[-1])
    v1, v2 = chain[i] - chain[behind], chain[ahead] - chain[i]
    ang = np.degrees(np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0],
                                       np.sum(v1 * v2, axis=1))))
    return ang, ok


def link_zone(chain, window=3.0):
    """Mask of (deduplicated) chain points within `window` px of arc
    length of a link between rings. find_contours puts every ring vertex
    of a binary mask on a pixel edge's midpoint (one coordinate an
    integer, the other a half integer); a biarc's sample points generally
    aren't, and a straight link shows up as a jump longer than one ring
    edge."""
    is_int = np.abs(chain - np.round(chain)) < 1e-6
    is_half = np.abs(chain - np.round(chain - 0.5) - 0.5) < 1e-6
    on_lattice = (is_int[:, 0] & is_half[:, 1]) | (is_half[:, 0] & is_int[:, 1])
    step = np.hypot(*np.diff(chain, axis=0).T)
    in_link = ~on_lattice
    jump = np.nonzero(step > 1.5)[0]
    in_link[jump] = True
    in_link[jump + 1] = True
    cum = np.concatenate(([0.0], np.cumsum(step)))
    link_s = cum[in_link]
    if len(link_s) == 0:
        return np.zeros(len(chain), dtype=bool)
    idx = np.searchsorted(link_s, cum)
    after = np.abs(link_s[np.clip(idx, 0, len(link_s) - 1)] - cum)
    before = np.abs(link_s[np.clip(idx - 1, 0, len(link_s) - 1)] - cum)
    return np.minimum(after, before) <= window


big_disk = disk_mask((200, 200), (100, 100), 80)
for step in (4, 8):
    for rev in (False, True):
        worst = 0.0
        n_links = 0
        for c in _onion_skin_chains(big_disk, step_px=step, min_area_px=20, reverse_rings=rev):
            c = dedup(c)
            ang, ok = smoothed_turns(c)
            zone = link_zone(c)
            outer = np.hypot(c[:, 0] - 100, c[:, 1] - 100) > 4 * step
            sel = ok & zone & outer
            n_links += int(np.sum(np.diff(zone.astype(int)) == 1))
            if np.any(sel):
                worst = max(worst, float(ang[sel].max()))
        check(f"disk, step {step}{' (reversed rings)' if rev else ''}: ring-to-ring links "
              f"were found at all ({n_links})", n_links >= 5)
        check(f"disk, step {step}{' (reversed rings)' if rev else ''}: no sharp turn at "
              f"any ring-to-ring link outside the innermost rings (max {worst:.0f} deg)",
              worst < 75.0)

print("_biarc")
r = 5.0
p0 = np.array([[0.0, 0.0]]);        t0 = np.array([[0.0, 1.0]])
p1 = np.array([[r, np.sqrt(3) * r]]); t1 = np.array([[0.0, 1.0]])
arc = _biarc(p0, t0, p1, t1, min_radius=r - 1e-6)
check("parallel tangents, offset r sideways and sqrt(3) r forward: a valid S",
      bool(arc['valid'][0]))
check("... made of two 60 degree arcs",
      np.allclose(np.degrees(2 * np.abs([arc['half1'][0], arc['half2'][0]])), 60.0))
check("... of opposite turning sense",
      np.sign(arc['half1'][0]) == -np.sign(arc['half2'][0]))
check("... with total length 2 * r * pi / 3",
      np.isclose(arc['length'][0], 2 * r * np.pi / 3))
pts = _sample_biarc(p0, t0, p1, arc)[0]
check("sampled biarc starts at p0 and ends at p1",
      np.allclose(pts[0], p0[0]) and np.allclose(pts[-1], p1[0], atol=1e-6))
check("sampled biarc leaves along t0 and arrives along t1",
      np.allclose((pts[1] - pts[0]) / np.hypot(*(pts[1] - pts[0])), t0[0], atol=0.05)
      and np.allclose((pts[-1] - pts[-2]) / np.hypot(*(pts[-1] - pts[-2])), t1[0], atol=0.05))
check("sampled biarc points are at most 0.5 px apart",
      np.hypot(*np.diff(pts, axis=0).T).max() <= 0.5 + 1e-9)
check("a radius below min_radius is invalid",
      not _biarc(p0, t0, p1, t1, min_radius=r + 0.5)['valid'][0])
check("a target behind the start is invalid",
      not _biarc(p0, t0, np.array([[r, -np.sqrt(3) * r]]), t1, min_radius=1.0)['valid'][0])


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

# Tool at a corner on the stock side of the starting end line: the adjacent
# line's entry is nearer, but the sweep must still begin on the end line and
# visit every line in sequence.
stack = [[(0.0, y, 0.0), (10.0, y, 0.0)] for y in (4.0, 3.0, 2.0, 1.0, 0.0)]
ordered = _order_locked_raster_pass(stack, "X", (0.0, 0.0))
check("_order_locked_raster_pass starts on the end line, not its neighbour",
      [seg[0][1] for seg in ordered] == [0.0, 1.0, 2.0, 3.0, 4.0]
      or [seg[0][1] for seg in ordered] == [4.0, 3.0, 2.0, 1.0, 0.0])
check("_order_locked_raster_pass keeps one travel direction per sweep",
      len({seg[0][0] for seg in ordered}) == 1)
ordered_zz = _order_zigzag_pass(stack, [0] * len(stack), "X", (0.0, 0.0), True)
check("_order_zigzag_pass (locked) visits lines in sequence",
      [seg[0][1] for seg in ordered_zz] in ([0.0, 1.0, 2.0, 3.0, 4.0],
                                             [4.0, 3.0, 2.0, 1.0, 0.0]))


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


for rdir in ("Horizontal", "Diagonal (45°)", "Coil (Outside-In)",
             "Spiral (Outside-In)"):
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
for rdir in ("Zig-Zag (0°)", "Onion Skin (Outside-In)", "Coil (Outside-In)",
             "Spiral (Outside-In)"):
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


print("Spiral (morphed from the coil)")

# A flat depth map, so the path is never gated: one continuous curve that
# covers the whole stock no worse than the rectangular coil does, without
# its 90-degree corners.
MS_PX, MS_STEP = 4.0, 2.5           # px/mm, mm


def _spiral_raster(raster_dir, w_mm, h_mm, cut_direction="Climb"):
    flat = np.full((int(h_mm * MS_PX), int(w_mm * MS_PX)), 0.5, dtype=np.float32)
    return compute_raster_toolpath(
        flat, n_grey_levels=256, tool=Tool(profile="End Mill", diameter_mm=1.0),
        step_mm=MS_STEP, stepdown_mm=5.0, max_depth=5.0, width_mm=w_mm, height_mm=h_mm,
        raster_dir=raster_dir, origin="Bottom-Left", cut_direction=cut_direction,
        progress=_noop_progress)


def _max_gap_px(tp, shape):
    """Largest distance (px) from any stock pixel to the path."""
    from scipy.ndimage import distance_transform_edt
    H, W = shape
    free = np.ones(shape, dtype=bool)
    for seg in tp.segments:
        a = np.asarray(seg)[:, :2]
        for p, q in zip(a[:-1], a[1:]):
            n = int(np.ceil(np.hypot(*(q - p)) * MS_PX * 2)) + 1
            xy = p + np.linspace(0.0, 1.0, n)[:, None] * (q - p)
            free[np.clip(np.rint(H - 1 - xy[:, 1] * MS_PX).astype(int), 0, H - 1),
                 np.clip(np.rint(xy[:, 0] * MS_PX).astype(int), 0, W - 1)] = False
    return float(distance_transform_edt(free).max())


def _curvature_radii(rc, k=3):
    """Circumradius through every sample and its k-th neighbours."""
    a, b, c = rc[:-2 * k], rc[k:-k], rc[2 * k:]
    ab, bc, ca = (np.hypot(*(q - p).T) for p, q in ((a, b), (b, c), (c, a)))
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    return ab * bc * ca / np.maximum(2.0 * cross, 1e-12), b


def _first_ring_area(pts, perimeter):
    """Signed area (CCW positive) of the path's first `perimeter` mm."""
    a = np.asarray(pts)[:, :2]
    n = int(np.searchsorted(np.cumsum(np.hypot(*np.diff(a, axis=0).T)), perimeter))
    a = a[:n + 2]
    return 0.5 * float(np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1]))


for w_mm, h_mm in ((40.0, 25.0), (30.0, 30.0)):
    tag = f"{w_mm:.0f}x{h_mm:.0f} mm"
    shape = (int(h_mm * MS_PX), int(w_mm * MS_PX))
    W_mm, H_mm = (shape[1] - 1) / MS_PX, (shape[0] - 1) / MS_PX
    tp_out = _spiral_raster("Spiral (Outside-In)", w_mm, h_mm)
    tp_in = _spiral_raster("Spiral (Inside-Out)", w_mm, h_mm)
    tp_rect = _spiral_raster("Coil (Outside-In)", w_mm, h_mm)
    check(f"{tag}: one continuous segment", len(tp_out.segments) == 1)
    pts = np.asarray(tp_out.segments[0])
    check(f"{tag}: path stays on the stock",
          pts[:, 0].min() >= -1e-9 and pts[:, 0].max() <= W_mm + 1e-9
          and pts[:, 1].min() >= -1e-9 and pts[:, 1].max() <= H_mm + 1e-9)
    check(f"{tag}: Outside-In starts at the top-left corner",
          np.allclose(pts[0, :2], (0.0, H_mm), atol=1e-6))
    check(f"{tag}: Inside-Out is the reversal of Outside-In",
          len(tp_in.segments) == 1
          and np.allclose(np.asarray(tp_in.segments[0])[::-1, :2], pts[:, :2], atol=1e-6))
    gap, gap_rect = _max_gap_px(tp_out, shape), _max_gap_px(tp_rect, shape)
    check(f"{tag}: no gap wider than the rectangular coil's corners "
          f"({gap / (MS_STEP * MS_PX):.3f} vs {gap_rect / (MS_STEP * MS_PX):.3f} step)",
          gap <= 0.586 * MS_STEP * MS_PX + 1.0)
    # Away from the stock's own sharp-cornered outer rings and the final
    # half-turn into the centre, the curve's radius never drops below
    # about one step-over.
    step_px = MS_STEP * MS_PX
    radii, mid = _curvature_radii(_spiral_px(0, shape[0] - 1, 0, shape[1] - 1,
                                                   step_px, "CW"))
    to_edge = np.minimum.reduce([mid[:, 0], shape[0] - 1 - mid[:, 0],
                                 mid[:, 1], shape[1] - 1 - mid[:, 1]])
    inner = (to_edge >= 2.0 * step_px) & (to_edge <= (min(shape) - 1) / 2 - 1.5 * step_px)
    check(f"{tag}: no tight corners inside the outer rings "
          f"(min radius {radii[inner].min() / step_px:.2f} step)",
          radii[inner].min() >= 0.9 * step_px)
    rect_radii, _ = _curvature_radii(np.concatenate(
        [np.asarray(sg)[:, :2] for sg in tp_rect.segments]) * MS_PX, k=1)
    check(f"{tag}: rectangular coil does have sharp corners",
          rect_radii.min() < 1.0)
    ratio = _path_len(tp_out) / _path_len(tp_rect)
    check(f"{tag}: path length within 1.35x the rectangular coil ({ratio:.2f}x)",
          ratio < 1.35)
    perim = 2.0 * (W_mm + H_mm)
    check(f"{tag}: Climb Outside-In winds clockwise",
          _first_ring_area(pts, perim) < 0.0)
    pts_conv = _spiral_raster("Spiral (Outside-In)", w_mm, h_mm,
                             "Conventional").segments[0]
    check(f"{tag}: Conventional Outside-In winds counter-clockwise",
          _first_ring_area(pts_conv, perim) > 0.0)

# Line offset: a 50% spiral runs midway between the turns of the 0% one,
# halving the uncovered gap of the two together; 100% is the 0% path.
for w_mm, h_mm in ((40.0, 25.0), (30.0, 30.0)):
    tag = f"{w_mm:.0f}x{h_mm:.0f} mm, 50% line offset"
    shape = (int(h_mm * MS_PX), int(w_mm * MS_PX))
    step_px = MS_STEP * MS_PX
    box = (0, shape[0] - 1, 0, shape[1] - 1)
    sp0 = _spiral_px(*box, step_px, "CW", 0.0)
    sp50 = _spiral_px(*box, step_px, "CW", 50.0)
    check(f"{tag}: 100% offset gives the 0% path",
          np.allclose(_spiral_px(*box, step_px, "CW", 100.0), sp0))
    check(f"{tag}: starts at the same corner as the 0% path",
          np.allclose(sp50[0], sp0[0]))
    tp0 = ToolPath(segments=[[(c / MS_PX, (shape[0] - 1 - r) / MS_PX, 0.0) for r, c in sp]
                             for sp in (sp0,)])
    tp_both = ToolPath(segments=[[(c / MS_PX, (shape[0] - 1 - r) / MS_PX, 0.0) for r, c in sp]
                                 for sp in (sp0, sp50)])
    gap1, gap2 = _max_gap_px(tp0, shape), _max_gap_px(tp_both, shape)
    check(f"{tag}: both paths together halve the gap "
          f"({gap1 / step_px:.3f} -> {gap2 / step_px:.3f} step)",
          gap2 <= 0.5 * gap1 + 1.0)
    from scipy.spatial import cKDTree
    centre = np.array([(shape[0] - 1) / 2, (shape[1] - 1) / 2])
    to_edge = np.minimum.reduce([sp50[:, 0], shape[0] - 1 - sp50[:, 0],
                                 sp50[:, 1], shape[1] - 1 - sp50[:, 1]])
    away = (to_edge >= step_px) & (np.abs(sp50 - centre).max(axis=1)
                                   >= (abs(shape[1] - shape[0]) / 2 + step_px))
    d, _ = cKDTree(sp0).query(sp50[away])
    check(f"{tag}: runs between the 0% turns, never on them "
          f"(closest {d.min() / step_px:.2f} step)", d.min() >= 0.25 * step_px)

tp_two_morph = compute_pocket_toolpath(
    depth_two, n_grey_levels=256, tool=zz_tool,
    step_mm=1.0, stepdown_mm=2.0, max_depth=5.0,
    width_mm=30.0, height_mm=30.0, raster_dir="Spiral (Inside-Out)",
    target_z_mm=2.0, cut_direction="Climb", progress=_noop_progress)
sides = ["A" if sg[0][0] < 11.0 else "B" for sg in tp_two_morph.segments]
check("pocket: Spiral clears both regions",
      len(tp_two_morph.segments) > 0 and set(sides) == {"A", "B"})
check("pocket: Spiral never interleaves regions",
      sum(1 for i in range(1, len(sides)) if sides[i] != sides[i - 1]) <= 1)


# --------------------------------------------------------------------------
# registration holes: every pattern maps onto itself under its own flip, and
# the flip keeps the G-code origin at the same physical point on the back
# canvas (same W x H for the axis flips, H x W for Diagonal) -- the reason a
# two-sided job needs no X/Y re-zeroing.
# --------------------------------------------------------------------------
print("custom profile in compute_raster_toolpath")

# A Custom profile drawn as a flat-bottomed cylinder must mill exactly what
# an End Mill of the same diameter does: same kernel, same toolpath.
_flat = tool_profile.profile_curve(tool_profile.flat(1.0))
_res = {}
for _name, _tool in (("End Mill", Tool("End Mill", 1.0)),
                     ("Custom", Tool("Custom", 1.0, 90.0, _flat))):
    _res[_name] = compute_raster_toolpath(
        dome, n_grey_levels=256, tool=_tool, step_mm=0.3, stepdown_mm=5.0,
        max_depth=5.0, width_mm=12.0, height_mm=12.0, raster_dir="Horizontal",
        progress=_noop_progress)
check("Custom flat profile gives the End Mill result map",
      np.allclose(_res["Custom"].result, _res["End Mill"].result, atol=1e-6))
check("Custom flat profile gives the End Mill segments",
      len(_res["Custom"].segments) == len(_res["End Mill"].segments))


print("registration holes")


def _reflect(p, axis):
    (ax, ay), (dx, dy) = axis
    px, py = p[0] - ax, p[1] - ay
    t = px * dx + py * dy
    return (ax + 2 * t * dx - px, ay + 2 * t * dy - py)


def _same_set(a, b, tol=1e-9):
    b = list(b)
    for p in a:
        hit = [i for i, q in enumerate(b) if abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol]
        if not hit:
            return False
        b.pop(hit[0])
    return not b


def _origin_point(origin, W, H):
    return {"Bottom-Left": (0, 0), "Bottom-Right": (W, 0), "Top-Left": (0, H),
            "Top-Right": (W, H), "Center": (W / 2, H / 2)}[origin]


for (W, H) in [(100.0, 100.0), (120.0, 80.0), (80.0, 120.0)]:
    for pattern in ("Vertical axis", "Horizontal axis", "Diagonal"):
        for origin in ("Bottom-Left", "Bottom-Right", "Top-Left", "Top-Right", "Center"):
            dirs = ("+45°", "−45°") if (pattern == "Diagonal" and origin == "Center") else ("+45°",)
            for dd in dirs:
                tag = f"{W:g}x{H:g} {pattern} {origin} {dd}"
                axis = registration_flip_axis(pattern, W, H, origin, dd)
                holes = registration_hole_centers(pattern, W, H, origin, dd, inset_mm=7.0)
                check(f"{tag}: holes map onto themselves under the flip",
                      _same_set(holes, [_reflect(p, axis) for p in holes]))
                check(f"{tag}: auto-placed holes lie inside the stock",
                      all(7.0 - 1e-9 <= x <= W - 7.0 + 1e-9
                          and 7.0 - 1e-9 <= y <= H - 7.0 + 1e-9 for x, y in holes))
                if pattern == "Diagonal":
                    # The flipped stock occupies the reflected footprint; its
                    # own origin point (same origin setting, H x W canvas
                    # placed where the reflection puts it) must be the
                    # machine zero the front side used.
                    corners = [_reflect(p, axis) for p in
                               [(0, 0), (W, 0), (0, H), (W, H)]]
                    x0 = min(c[0] for c in corners)
                    y0 = min(c[1] for c in corners)
                    ox, oy = _origin_point(origin, H, W)
                    back_origin = (x0 + ox, y0 + oy)
                else:
                    back_origin = _origin_point(origin, W, H)
                front_origin = _origin_point(origin, W, H)
                check(f"{tag}: flip keeps the G-code origin at the same physical point",
                      np.hypot(back_origin[0] - front_origin[0],
                               back_origin[1] - front_origin[1]) < 1e-9)

check("auto spacing uses the shorter stock side",
      registration_auto_spacing(120.0, 80.0, 10.0) == 60.0)
diag = registration_hole_centers("Diagonal", 120.0, 80.0, "Bottom-Left", inset_mm=10.0)
check("Diagonal from Bottom-Left: corner hole at (inset, inset), arms along x and y",
      _same_set(diag, [(10.0, 10.0), (70.0, 10.0), (10.0, 70.0)]))
diag = registration_hole_centers("Diagonal", 120.0, 80.0, "Top-Right", inset_mm=10.0,
                                 spacing_mm=30.0)
check("Diagonal from Top-Right with explicit spacing",
      _same_set(diag, [(110.0, 70.0), (80.0, 70.0), (110.0, 40.0)]))

# Helix: tool-centre radius, full depth, bounded Z drop per revolution.
segs = bore_hole_segments(5.0, 5.0, 1.5, 10.0, 1.0, climb=True)
pts = np.array(segs[0])
r = np.hypot(pts[:-1, 0] - 5.0, pts[:-1, 1] - 5.0)
check("helix: one continuous segment", len(segs) == 1)
check("helix: every point but the lead-out sits on the path radius",
      np.allclose(r, 1.5))
check("helix: starts at the stock surface", abs(pts[0, 2]) < 1e-12)
check("helix: reaches full depth", abs(pts[:, 2].min() + 10.0) < 1e-9)
check("helix: ends back at the hole centre at full depth",
      np.allclose(pts[-1], (5.0, 5.0, -10.0)))
ang = np.unwrap(np.arctan2(pts[:-1, 1] - 5.0, pts[:-1, 0] - 5.0))
per_rev = np.interp(ang[0] + 2 * np.pi, ang, pts[:-1, 2])
check("helix: Climb winds counter-clockwise", ang[-1] > ang[0])
check("helix: Z drop per revolution is at most the step-down",
      abs(per_rev - pts[0, 2]) <= 1.0 + 1e-9)
check("helix: ends with a flat full circle at the bottom",
      np.all(np.abs(pts[:-1, 2][ang >= ang[-1] - 2 * np.pi + 1e-9] + 10.0) < 1e-9))
conv = np.array(bore_hole_segments(5.0, 5.0, 1.5, 10.0, 1.0, climb=False)[0])
ang_c = np.unwrap(np.arctan2(conv[:-1, 1] - 5.0, conv[:-1, 0] - 5.0))
check("helix: Conventional winds clockwise", ang_c[-1] < ang_c[0])

pecks = bore_hole_segments(5.0, 5.0, 0.0, 10.0, 4.0)
check("plunge: pecked in step-down increments", len(pecks) == 3)
check("plunge: all pecks at the hole centre",
      all(abs(p[0] - 5.0) < 1e-12 and abs(p[1] - 5.0) < 1e-12 for sg in pecks for p in sg))
check("plunge: every peck but the last lifts back out of the hole",
      all(sg[-1][2] == 0.0 for sg in pecks[:-1]) and abs(pecks[-1][-1][2] + 10.0) < 1e-12)
check("plunge: step-down 0 is a single plunge",
      len(bore_hole_segments(0.0, 0.0, 0.0, 10.0, 0.0)) == 1)

reg_tool = Tool("End Mill", 6.0, 90.0)
try:
    compute_registration_holes_toolpath(reg_tool, 100.0, 100.0, (200, 200),
                                        hole_diameter_mm=5.0)
    check("hole smaller than the tool raises", False)
except ValueError:
    check("hole smaller than the tool raises", True)

tp_reg = compute_registration_holes_toolpath(
    reg_tool, 100.0, 100.0, (201, 201), pattern="Diagonal", origin="Bottom-Left",
    hole_diameter_mm=8.0, hole_depth_mm=15.0, stepdown_mm=2.0, inset_mm=10.0,
    max_depth=5.0)
check("toolpath: one segment per helical hole", len(tp_reg.segments) == 3)
check("toolpath: result reaches full depth at the hole centres",
      tp_reg.result[200 - 20, 20] >= 1.0 - 1e-3)
check("toolpath: holes deeper than max depth are marked cut through",
      tp_reg.alpha is not None and tp_reg.alpha[200 - 20, 20] == 0.0)
check("toolpath: material between holes is untouched",
      tp_reg.result[100, 100] == 0.0 and tp_reg.alpha[100, 100] == 1.0)
# Like every algorithm, G-code X/Y span pixel centres: (W - 1) / px_per_mm.
reg_ext = 200 / (201 / 100.0)
check("toolpath: G-code centres follow the origin",
      _same_set(tp_reg.metadata['hole_centers'],
                [(10.0, 10.0), (reg_ext - 10.0, 10.0), (10.0, reg_ext - 10.0)], tol=1e-6))
tp_tr = compute_registration_holes_toolpath(
    reg_tool, 100.0, 100.0, (201, 201), pattern="Vertical axis", origin="Top-Right",
    hole_diameter_mm=6.0, hole_depth_mm=4.0, stepdown_mm=2.0, inset_mm=10.0,
    max_depth=5.0)
check("toolpath: Top-Right origin shifts centres to negative X/Y",
      _same_set(tp_tr.metadata['hole_centers'],
                [(-reg_ext / 2, -(reg_ext - 10.0)), (-reg_ext / 2, -10.0)], tol=1e-6))
check("toolpath: hole the tool's own size is plunged",
      tp_tr.metadata['cut_method'] == 'plunge')
check("toolpath: a hole shallower than max depth is not marked cut through",
      tp_tr.alpha is None)


print("_order_contours_graph")
N_C = 121
_rr, _cc = np.mgrid[0:N_C, 0:N_C]


def _dist_px(center):
    return np.hypot(_rr - center[0], _cc - center[1])


def _order_graph(surface, levels, pieces=None, lock=False):
    contours = pieces if pieces is not None else _extract_z_contours(
        surface, levels, 256, cut_direction="Climb" if lock else "Either")
    segs, _ = _order_contours_graph(contours, levels, surface, 256, max_depth=10.0,
                                    px_per_mm=1.0, H=N_C, ox=0.0, oy=0.0,
                                    lock_direction=lock, current_pos=(0.0, 0.0))
    return segs


def _feature(seg, centers):
    """Index of the centre (row, col) nearest the segment's centroid."""
    p = np.array(seg)
    r, c = N_C - 1 - p[:, 1].mean(), p[:, 0].mean()
    return int(np.argmin([np.hypot(r - cr, c - cc) for cr, cc in centers]))


def _top_down(zs):
    return all(a >= b - 1e-9 for a, b in zip(zs, zs[1:]))


def _features_in_turn(segs, centers):
    """Top-down within each feature, and each feature cut in one go."""
    feats = [_feature(s, centers) for s in segs]
    runs = [f for i, f in enumerate(feats) if i == 0 or f != feats[i - 1]]
    per_feature = all(_top_down([s[0][2] for s, f in zip(segs, feats) if f == k])
                      for k in set(feats))
    return per_feature, len(runs) == len(set(feats)), len(set(feats))


LEVELS = [k / 10 for k in range(1, 11)]
CENTERS = [(60, 30), (60, 90)]
hills = np.clip(np.minimum(_dist_px(CENTERS[0]), _dist_px(CENTERS[1])) / 28.0,
                0, 1).astype(np.float32)
for name, surf in [("two hills", hills), ("two craters", 1.0 - hills)]:
    for lock in (False, True):
        segs = _order_graph(surf, LEVELS, lock=lock)
        td, in_turn, n_f = _features_in_turn(segs, CENTERS)
        tag = f"{name}{' (locked)' if lock else ''}"
        check(f"{tag}: both features cut", n_f == 2)
        check(f"{tag}: top-down within each feature", td)
        check(f"{tag}: one feature after the other", in_turn)

# Pruned/clipped pieces: every ring split into open runs, given shuffled.
contours = _extract_z_contours(hills, LEVELS, 256)
rng = np.random.default_rng(1)
pieces = {}
for z, cs in contours.items():
    runs = [r for c in cs for r in np.array_split(c, 3) if len(r) >= 3]
    pieces[z] = [runs[i] for i in rng.permutation(len(runs))]
segs = _order_graph(hills, LEVELS, pieces=pieces)
td, in_turn, n_f = _features_in_turn(segs, CENTERS)
check("open pieces: top-down within each feature", td and n_f == 2)
check("open pieces: one feature after the other", in_turn)

# Crater with a central peak: the peak's rings follow only the peak's own.
r0 = _dist_px((60, 60))
peak = np.where(r0 < 12, r0 / 12.0, np.clip((40.0 - r0) / 15.0, 0, 1)).astype(np.float32)
segs = _order_graph(peak, LEVELS)
feats = ["peak" if np.hypot(*(np.array(s)[:, :2] - (60, 60)).T).mean() < 18 else "crater"
         for s in segs]
check("crater with central peak: top-down within each feature",
      all(_top_down([s[0][2] for s, f in zip(segs, feats) if f == k])
          for k in ("peak", "crater")))
check("crater with central peak: one feature after the other",
      sum(1 for a, b in zip(feats, feats[1:]) if a != b) == 1)

# Two hills merging below a saddle: each hill top-down, the merged rings
# only after both hills' rings above the saddle. Below the saddle the merged
# material spans the canvas, splitting the air into the halves above and
# below it: two branches, each top-down.
merge = np.clip(np.minimum(_dist_px(CENTERS[0]), _dist_px(CENTERS[1])) / 50.0,
                0, 1).astype(np.float32)          # saddle at 30/50 = 0.6
segs = _order_graph(merge, LEVELS)
feats = []
for s in segs:
    p = np.array(s)
    merged = p[:, 0].min() < 55 and p[:, 0].max() > 65
    feats.append(("merged", p[:, 1].mean() > 60) if merged else _feature(s, CENTERS))
last_hill = max(i for i, f in enumerate(feats) if f in (0, 1))
first_merged = min(i for i, f in enumerate(feats) if f not in (0, 1))
check("hills merging: each hill top-down",
      all(_top_down([s[0][2] for s, f in zip(segs, feats) if f == k]) for k in (0, 1)))
check("hills merging: merged rings after both hills", first_merged > last_hill)
check("hills merging: merged rings top-down on each side",
      all(_top_down([s[0][2] for s, f in zip(segs, feats) if f == ("merged", side)])
          for side in (False, True)))

# Vertical-walled bosses: the rings of every level coincide on the wall.
bosses = np.where(np.minimum(_dist_px(CENTERS[0]), _dist_px(CENTERS[1])) < 15,
                  0.0, 1.0).astype(np.float32)
segs = _order_graph(bosses, LEVELS)
td, in_turn, n_f = _features_in_turn(segs, CENTERS)
check("vertical-walled bosses: top-down within each boss", td and n_f == 2)
check("vertical-walled bosses: one boss after the other", in_turn)


print("Contour Max step-down")
check("gap fill: evenly spaced levels between the first level and the surface",
      np.allclose(_fill_level_gaps([0.3], 0.1), [0.1, 0.2, 0.3]))
check("gap fill: a gap of exactly the limit stays unfilled",
      np.allclose(_fill_level_gaps([0.1, 0.2, 0.3], 0.1), [0.1, 0.2, 0.3]))
filled = _fill_level_gaps([0.05, 0.1, 0.45, 0.5], 0.1)
check("gap fill: existing levels stay, no gap exceeds the limit",
      all(z in filled for z in (0.05, 0.1, 0.45, 0.5))
      and np.max(np.diff([0.0] + filled)) <= 0.1 + 1e-9
      and filled == sorted(filled))
check("gap fill: the gap below the deepest point of the surface stays unfilled",
      np.allclose(_fill_level_gaps([0.1, 0.6, 1.0], 0.25, deepest=0.6),
                  [0.1, 0.35, 0.6, 1.0]))

# A vertical-walled pit: every level's ring lies on the wall, so each one is
# laterally redundant to the rings below it.
pit = (_dist_px((60, 60)) < 30).astype(np.float32)
levels = [i / 10 for i in range(1, 11)]


def _pruned_depths(step_down_mm):
    pruned = _prune_contours(_extract_z_contours(pit, levels, 256), levels, 2.0,
                             N_C, N_C, max_depth=10.0, max_step_down_mm=step_down_mm)
    return [z * 10.0 for z in levels if pruned[z]]


def _max_step(depths):
    return float(np.max(np.diff([0.0] + depths)))


check("pruning keeps every level whose removal would exceed Max step-down",
      np.allclose(_pruned_depths(1.0), range(1, 11)))
check("pruning stays within a larger Max step-down",
      _max_step(_pruned_depths(3.0)) <= 3.0 + 1e-6)
check("pruning still collapses levels within Max step-down",
      len(_pruned_depths(3.0)) < len(levels))


def _contour_depths(**kwargs):
    # Most pixels are at 0.9 of full depth, a small pit reaches full depth,
    # so the quantile levels all land near the bottom of a tall wall.
    surface = np.where(_dist_px((60, 60)) < 50, 0.9, 0.0).astype(np.float32)
    surface[_dist_px((60, 60)) < 5] = 1.0
    tp = compute_contour_toolpath(surface, 256, Tool(profile="Flat", diameter_mm=1.0),
                                  z_increment_mm=2.5, max_depth=10.0,
                                  width_mm=N_C, height_mm=N_C,
                                  progress=lambda *a: None, **kwargs)
    return sorted({round(-p[2], 3) for seg in tp.segments for p in seg[:1]})


unbounded = _contour_depths(adaptive_levels=True)
bounded = _contour_depths(adaptive_levels=True, max_step_down_mm=1.0)
check("adaptive Contour steps down further than Max step-down without the limit",
      _max_step(unbounded) > 1.0)
check("adaptive Contour stays within Max step-down", _max_step(bounded) <= 1.0 + 1e-3)
check("adaptive Contour keeps the quantile levels",
      all(z in bounded for z in unbounded))
check("adaptive Contour stays within Max step-down with lateral pruning",
      _max_step(_contour_depths(adaptive_levels=True, max_step_down_mm=1.0,
                                lateral_tol_mm=2.0)) <= 1.0 + 1e-3)
check("uniform Contour levels ignore Max step-down",
      _contour_depths(max_step_down_mm=1.0) == _contour_depths())


# ---------------------------------------------------------------------------
print("ToolPath.resample / iter_resampled")


def _resample_loop(segments, spacing):
    """Point-by-point reference: per segment its first point, then a sample
    every *spacing* of XY arc length, the spacing carried across edges."""
    out = []
    for seg in segments:
        if not seg:
            continue
        pts = np.asarray(seg, dtype=np.float64)
        out.append(pts[0])
        carry = 0.0
        for i in range(1, len(pts)):
            diff = pts[i] - pts[i - 1]
            seg_len = float(np.hypot(diff[0], diff[1]))
            if seg_len < 1e-10:
                continue
            pos = carry
            while pos < seg_len:
                out.append(pts[i - 1] + pos / seg_len * diff)
                pos += spacing
            carry = pos - seg_len
    return np.array(out).reshape(-1, 3)


_rs_segs = [
    [(0.0, 0.0, 0.0), (1.0, 0.0, -1.0), (1.0, 0.0, -2.0), (1.0, 0.7, -2.0),
     (1.0, 0.7, -2.0), (0.3, 0.7, -0.5)],              # plunge + duplicate vertex
    [(5.0, 5.0, -1.0)],                                  # single point
    [],
    [(2.0, 2.0, 0.0), (2.0, 2.0, -3.0)],                 # pure plunge
    [(0.0, 3.0, -0.25), (0.13, 3.0, -0.25), (0.13, 3.29, -0.5), (0.9, 3.4, -0.1)],
]
# 0.0731 mm divides none of the edge lengths, so no sample falls on a vertex.
_rs_ref = _resample_loop(_rs_segs, 0.0731)
_rs_vec = ToolPath(segments=_rs_segs).resample(0.0731)
check("matches the point-by-point sampling (plunges, carried spacing, single points)",
      _rs_vec.shape == _rs_ref.shape and np.allclose(_rs_vec, _rs_ref, atol=1e-12))
_rs_chunks = list(ToolPath(segments=_rs_segs).iter_resampled(0.0731, max_samples=3))
check("chunks hold at most max_samples rows and concatenate to resample()",
      max(len(c) for c in _rs_chunks) <= 3
      and np.array_equal(np.concatenate(_rs_chunks), _rs_vec))
_rs_plunge = ToolPath(segments=[[(0.0, 0.0, 0.0), (1.0, 0.0, -1.0), (1.0, 0.0, -2.0),
                                 (2.0, 0.0, -2.0)]]).resample(0.5)
check("a sample on a plunge's vertex takes the plunge's bottom",
      np.allclose(_rs_plunge, [(0, 0, 0), (0, 0, 0), (0.5, 0, -0.5), (1, 0, -2),
                               (1.5, 0, -2)]))
check("an empty path resamples to an empty (0, 3) array",
      ToolPath(segments=[]).resample(0.1).shape == (0, 3))


# ---------------------------------------------------------------------------
print("onion skin rings on a large disk stay concentric")
_os_R, _os_step = 300, 10
_os_yy, _os_xx = np.mgrid[0:2 * _os_R + 21, 0:2 * _os_R + 21]
_os_c = _os_R + 10
_os_chains = _onion_skin_chains(np.hypot(_os_yy - _os_c, _os_xx - _os_c) <= _os_R,
                                _os_step, min_area_px=20)
_os_pts = np.vstack(_os_chains)
_os_rad = np.hypot(_os_pts[:, 0] - _os_c, _os_pts[:, 1] - _os_c)
_os_off = np.abs(_os_rad - (_os_R - _os_step * np.round((_os_R - _os_rad) / _os_step)))
# Ring points sit within a pixel and a half of R - k * step; only the links
# between rings (a few percent of the points) lie farther from them.
check(f"rings lie on R - k * step-over ({100 * np.mean(_os_off <= 1.5):.1f}% of points "
      "within 1.5 px)", len(_os_chains) == 1 and np.mean(_os_off <= 1.5) >= 0.93)


# ---------------------------------------------------------------------------
print("_geodesic_link")


def _geodesic_link_heapq(r0, c0, r1, c1, ceil_norm, depth, H, W, ppx, ppy,
                         max_depth, final_norm_val=0.0):
    """Point-by-point reference: heapq Dijkstra over the same window, moves
    and 3-D arc-length cost."""
    import heapq
    margin = max(2, max(abs(r1 - r0), abs(c1 - c0)) // 4 + 1)
    rlo, rhi = max(0, min(r0, r1) - margin), min(H - 1, max(r0, r1) + margin)
    clo, chi = max(0, min(c0, c1) - margin), min(W - 1, max(c0, c1) + margin)
    dist, prev, heap = {(r0, c0): 0.0}, {}, [(0.0, r0, c0)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if (r, c) == (r1, c1):
            break
        if d > dist.get((r, c), float('inf')):
            continue
        z_cur = min(max(float(depth[r, c]) - final_norm_val, 0.0), ceil_norm)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if not (rlo <= nr <= rhi and clo <= nc <= chi):
                continue
            z_nb = min(max(float(depth[nr, nc]) - final_norm_val, 0.0), ceil_norm)
            dz = (z_nb - z_cur) * max_depth
            nd = d + ((dr / ppy) ** 2 + (dc / ppx) ** 2 + dz * dz) ** 0.5
            if nd < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)], prev[(nr, nc)] = nd, (r, c)
                heapq.heappush(heap, (nd, nr, nc))
    path, cur = [], (r1, c1)
    while cur != (r0, c0):
        path.append(cur)
        cur = prev[cur]
    return [(r0, c0)] + path[::-1]


def _link_len(path, depth, ceil_norm, ppm, max_depth, fn=0.0):
    p = np.array(path)
    z = np.clip(depth[p[:, 0], p[:, 1]].astype(np.float64) - fn, 0.0, ceil_norm) * max_depth
    d = np.diff(p, axis=0) / ppm
    return float(np.sum(np.sqrt((d ** 2).sum(axis=1) + np.diff(z) ** 2)))


_gl_ppm, _gl_md = 20.0, 5.0
_gl_yy, _gl_xx = np.mgrid[0:300, 0:300].astype(np.float32)
_gl_relief = np.clip(0.5 + 0.3 * np.sin(_gl_xx / 31.0) * np.cos(_gl_yy / 17.0)
                     + 0.15 * np.sin((_gl_xx + _gl_yy) / 9.0), 0, 1).astype(np.float32)
_gl_step = np.full((300, 300), 0.6, np.float32)
_gl_step[:, 150:] = 0.2
_gl_step[100:120, 60:240] = 0.9                       # a deeper groove across the step
_gl_rng = np.random.default_rng(3)
_gl_cases = []
for _gl_depth in (_gl_relief, _gl_step):
    for _ in range(60):
        r0, c0 = (int(v) for v in _gl_rng.integers(0, 300, 2))
        r1 = int(np.clip(r0 + _gl_rng.integers(-24, 25), 0, 299))
        c1 = int(np.clip(c0 + _gl_rng.integers(-24, 25), 0, 299))
        _gl_cases.append((_gl_depth, r0, c0, r1, c1, float(_gl_rng.choice([0.4, 0.7, 1.0])),
                          float(_gl_rng.choice([0.0, 0.1]))))
_gl_same = _gl_len_ok = 0
for depth, r0, c0, r1, c1, cn, fn in _gl_cases:
    new = _geodesic_link(r0, c0, r1, c1, cn, depth, 300, 300, _gl_ppm, _gl_ppm, _gl_md, fn)
    ref = _geodesic_link_heapq(r0, c0, r1, c1, cn, depth, 300, 300, _gl_ppm, _gl_ppm, _gl_md, fn)
    _gl_same += new == ref
    _gl_len_ok += abs(_link_len(new, depth, cn, _gl_ppm, _gl_md, fn)
                      - _link_len(ref, depth, cn, _gl_ppm, _gl_md, fn)) < 1e-9
check("every link is as short as the heapq Dijkstra's (relief, flat floor with steps, "
      "ceilings, final allowance, canvas edges)", _gl_len_ok == len(_gl_cases))
check(f"links follow the heapq Dijkstra's pixels ({_gl_same}/{len(_gl_cases)} identical)",
      _gl_same >= 0.9 * len(_gl_cases))
_gl_path = _geodesic_link(10, 10, 14, 30, 1.0, _gl_relief, 300, 300, _gl_ppm, _gl_ppm, _gl_md)
check("a link runs from its start to its end in 8-connected steps",
      _gl_path[0] == (10, 10) and _gl_path[-1] == (14, 30)
      and all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
              for a, b in zip(_gl_path, _gl_path[1:])))
check("a link from a pixel to itself is that pixel",
      _geodesic_link(5, 5, 5, 5, 1.0, _gl_relief, 300, 300, _gl_ppm, _gl_ppm, _gl_md)
      == [(5, 5)])


# --------------------------------------------------------------------------
# Z range: the depth band of Raster Scan and Contour steps
# --------------------------------------------------------------------------
print("Z range")

check("passes step down from top_mm",
      _plan_passes(10.0, 2.0, 0.0, 4.0) == [6.0, 8.0, 10.0])
check("passes from top_mm keep the dedicated final pass",
      _plan_passes(10.0, 2.0, 1.0, 4.0) == [6.0, 8.0, 9.0, 10.0])
check("top_mm = 0 plans the same passes as before",
      _plan_passes(5.0, 1.5, 0.5) == _plan_passes(5.0, 1.5, 0.5, 0.0))
_mp = make_multipass([[(0.0, 0.0, -9.0), (1.0, 0.0, -9.0)]], 10.0, 2.0, top_mm=4.0)
check("make_multipass from top_mm: first ceiling is top + step-down",
      [c for c, _, _ in _mp] == [6.0, 8.0, 10.0]
      and _mp[0][2][0][0][2] == -6.0 and _mp[-1][2][0][0][2] == -9.0)
_mpf = make_multipass([[(0.0, 0.0, -4.5), (1.0, 0.0, -9.0)]], 10.0, 2.0, 1.0, top_mm=4.0)
check("make_multipass from top_mm with a final reserve roughs nothing above top_mm",
      all(p[2] <= -4.0 + 1e-9 for _, fin, segs in _mpf if not fin for sg in segs for p in sg))
_bl = band_levels((0.3, 0.7), 1.0, 10.0)
check("band levels lie in (lo, hi] and end at hi",
      len(_bl) == 4 and all(0.3 < z <= 0.7 for z in _bl) and _bl[-1] == 0.7)
check("full band levels match the uniform levels",
      band_levels((0.0, 1.0), 0.3, 1.0) == [min(i * 0.3 / 1.0, 1.0) for i in range(1, 5)])
for _bad in ((0.5, 0.5), (0.6, 0.4), (-0.1, 1.0), (0.0, 1.1)):
    try:
        check_z_range(_bad)
        check(f"Z range {_bad} is rejected", False)
    except ValueError:
        check(f"Z range {_bad} is rejected", True)

_zr_kw = dict(n_grey_levels=256, tool=Tool(profile="Ball", diameter_mm=0.6),
              step_mm=0.3, stepdown_mm=1.0, max_depth=5.0, width_mm=12.0,
              height_mm=12.0, raster_dir="Horizontal", cut_direction="Climb",
              progress=_noop_progress)
_zr_full = compute_raster_toolpath(dome, **_zr_kw)
_zr_full2 = compute_raster_toolpath(dome, z_range=(0.0, 1.0), **_zr_kw)
check("Raster: full Z range gives the default path",
      _zr_full.segments == _zr_full2.segments
      and np.array_equal(_zr_full.result, _zr_full2.result))
_zr_band = compute_raster_toolpath(dome, z_range=(0.3, 0.7), **_zr_kw)
_zr_z = np.array([-p[2] for s in _zr_band.segments for p in s])
check("Raster: no point is cut below the band's floor",
      _zr_z.size > 0 and _zr_z.max() <= 0.7 * 5.0 + 1e-6)
check("Raster: the result stays above the band's floor",
      float(_zr_band.result.max()) <= 0.7 + 1e-6)
check("Raster: accessible is capped at the band's floor",
      float(_zr_band.accessible.max()) <= 0.7 + 1e-6)
_zr_cuts = [s for s in _zr_band.segments if len(s) > 1]
_zr_deep = np.array([-p[2] for s in _zr_cuts for p in s])
check("Raster: the passes cut only below the band's top (links aside)",
      np.mean(_zr_deep >= 0.3 * 5.0 - 0.05) > 0.95)
check("Raster: the band takes fewer passes",
      _zr_band.metadata['n_passes'] < _zr_full.metadata['n_passes']
      and _zr_band.metadata['z_range'] == (0.3, 0.7))


def _contour_levels(**kwargs):
    tp = compute_contour_toolpath(dome, 256, Tool(profile="Ball", diameter_mm=0.6),
                                  z_increment_mm=0.5, max_depth=5.0,
                                  width_mm=12.0, height_mm=12.0,
                                  progress=_noop_progress, **kwargs)
    return tp, sorted({round(-p[2] / 5.0, 4) for seg in tp.segments for p in seg})


_zc_full, _ = _contour_levels()
_zc_full2, _ = _contour_levels(z_range=(0.0, 1.0))
check("Contour: full Z range gives the default path",
      _zc_full.segments == _zc_full2.segments)
_zc_band, _zc_lv = _contour_levels(z_range=(0.3, 0.7))
check("Contour: every cut lies in the band",
      bool(_zc_lv) and min(_zc_lv) >= 0.3 - 0.01 and max(_zc_lv) <= 0.7 + 1e-4)
check("Contour: the band's floor is its deepest level",
      abs(max(_zc_lv) - 0.7) < 1e-3)
check("Contour: accessible is capped at the band's floor",
      float(_zc_band.accessible.max()) <= 0.7 + 1e-6)
_zca_band, _zca_lv = _contour_levels(z_range=(0.3, 0.7), adaptive_levels=True,
                                     max_step_down_mm=0.4)
check("Contour (adaptive, Max step-down): every cut lies in the band",
      bool(_zca_lv) and min(_zca_lv) >= 0.3 - 0.01 and max(_zca_lv) <= 0.7 + 1e-4)
check("_fill_level_gaps starts from start",
      _fill_level_gaps([0.9], 0.2, 1.0, start=0.5) == [0.7, 0.9])


print("Depth map change gating (prior step's depth map as prior_reference)")

# Two plateaus; the prior step's depth map has only the left one, so gating
# against it cuts only the right one, and gating against an identical depth
# map cuts nothing.
_dmc = np.zeros((60, 60), dtype=np.float32)
_dmc[10:50, 5:20] = 0.5
_dmc[10:50, 40:55] = 0.5
_dmc_prior = _dmc.copy()
_dmc_prior[:, 30:] = 0.0


def _dmc_raster(prior_reference):
    return compute_raster_toolpath(
        _dmc, n_grey_levels=256, tool=Tool(profile="End Mill", diameter_mm=2.0),
        step_mm=1.0, stepdown_mm=5.0, max_depth=5.0, width_mm=60.0, height_mm=60.0,
        raster_dir="Horizontal", work_surface_tol=0.01,
        prior_reference=prior_reference, progress=_noop_progress)


_dmc_tp = _dmc_raster(_dmc_prior)
_dmc_x = (np.concatenate([np.asarray(s)[:, 0] for s in _dmc_tp.segments])
          if _dmc_tp.segments else np.zeros(0))
check("changed plateau is cut", _dmc_x.size > 0)
check("unchanged plateau is not cut", _dmc_x.size > 0 and float(_dmc_x.min()) >= 38.0)
check("identical depth maps cut nothing", len(_dmc_raster(_dmc).segments) == 0)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
