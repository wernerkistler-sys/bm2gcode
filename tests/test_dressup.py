"""Plain assert-based tests for the G-code dress-up helpers (dressup.py) and
their integration into ToolPath.iter_moves()/to_gcode()/get_stats().

Run with: python tests/test_dressup.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dressup import (RampSettings, BackoffSettings, ArcSettings,
                      generate_ramp, compute_backoff_path, generate_corner_arc)
from toolpath import ToolPath

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# --------------------------------------------------------------------------
# generate_ramp
# --------------------------------------------------------------------------
print("generate_ramp")

# Disabled / no settings -> plain plunge fallback
pts = [(0.0, 0.0, -5.0), (10.0, 0.0, -5.0)]
check("disabled settings returns single point",
      generate_ramp(pts, 0.0, None) == [pts[0]])
check("enabled=False returns single point",
      generate_ramp(pts, 0.0, RampSettings(enabled=False)) == [pts[0]])

# Constant-Z straight segment
straight = [(x, 0.0, -5.0) for x in range(0, 21)]  # 0..20mm, z=-5 constant
settings = RampSettings(enabled=True, angle_deg=30.0, max_length_mm=15.0)
ramp = generate_ramp(straight, 0.0, settings)
check("constant-Z ramp has multiple waypoints", len(ramp) > 1)
check("constant-Z ramp ends exactly at points[0]", ramp[-1] == straight[0])
zs = [p[2] for p in ramp]
check("constant-Z ramp Z is monotonically non-increasing",
      all(zs[i] >= zs[i + 1] - 1e-9 for i in range(len(zs) - 1)))
check("constant-Z ramp XY stays within lookahead bounds",
      all(0.0 <= p[0] <= settings.max_length_mm + 1e-6 and abs(p[1]) < 1e-9 for p in ramp))
check("constant-Z ramp never descends below the floor Z",
      all(p[2] >= -5.0 - 1e-9 for p in ramp))

# Varying-Z (sloped) segment: z(x) = -2 - 0.4*x for x in [0, 10]
sloped = [(float(x), 0.0, -2.0 - 0.4 * x) for x in range(0, 11)]
ramp2 = generate_ramp(sloped, 0.0, RampSettings(enabled=True, angle_deg=30.0, max_length_mm=10.0))
check("sloped-floor ramp ends exactly at points[0]", ramp2[-1] == sloped[0])


def floor_at(x):
    x = max(0.0, min(10.0, x))
    return -2.0 - 0.4 * x


check("sloped-floor ramp never goes below the true floor Z",
      all(p[2] >= floor_at(p[0]) - 1e-6 for p in ramp2))

# Short-segment fallback: total length well under 0.5mm
tiny = [(0.0, 0.0, -1.0), (0.1, 0.0, -1.0)]
check("too-short segment falls back to plain plunge",
      generate_ramp(tiny, 0.0, RampSettings(enabled=True)) == [tiny[0]])

# Angle degradation: force a tiny max_legs cap with a deep plunge relative to
# the available lookahead length -- should still terminate exactly at
# points[0] without crashing or overshooting the lookahead length.
deep = [(x, 0.0, -50.0) for x in range(0, 21)]
capped = RampSettings(enabled=True, angle_deg=5.0, max_length_mm=15.0, max_legs=2)
ramp3 = generate_ramp(deep, 0.0, capped)
check("max_legs cap still terminates exactly at points[0]", ramp3[-1] == deep[0])
check("max_legs cap never exceeds lookahead length",
      all(p[0] <= capped.max_length_mm + 1e-6 for p in ramp3))

# --------------------------------------------------------------------------
# compute_backoff_path
# --------------------------------------------------------------------------
print("compute_backoff_path")

leg = [(0.0, 0.0, -5.0), (10.0, 0.0, -5.0)]
bo = compute_backoff_path(leg, BackoffSettings(enabled=True, distance_mm=2.0))
check("backoff direction opposes travel and has the right magnitude",
      len(bo) == 1 and abs(bo[-1][0] - 8.0) < 1e-9 and abs(bo[-1][1]) < 1e-9
      and abs(bo[-1][2] - (-3.0)) < 1e-9)

short_leg = [(0.0, 0.0, -5.0), (1.0, 0.0, -5.0)]
bo_clamped = compute_backoff_path(short_leg, BackoffSettings(enabled=True, distance_mm=5.0))
check("backoff clamps to the segment's start",
      bo_clamped and abs(bo_clamped[-1][0] - 0.0) < 1e-9)

vertical_leg = [(3.0, 4.0, -1.0), (3.0, 4.0, -5.0)]
check("vertical-only path returns []",
      compute_backoff_path(vertical_leg, BackoffSettings(enabled=True, distance_mm=1.0)) == [])

check("disabled settings returns []",
      compute_backoff_path(leg, BackoffSettings(enabled=False, distance_mm=2.0)) == [])
check("zero distance returns []",
      compute_backoff_path(leg, BackoffSettings(enabled=True, distance_mm=0.0)) == [])

# Near-vertical final raise: the last leg has almost no XY travel, so the
# back-off has to continue into the previous leg to cover its distance.
raise_end = [(0.0, 0.0, -5.0), (10.0, 0.0, -5.0), (10.01, 0.0, -1.0)]
bo_raise = compute_backoff_path(raise_end, BackoffSettings(enabled=True, distance_mm=2.0))
check("near-vertical last move extends into the previous leg",
      bo_raise and abs(bo_raise[-1][0] - 8.01) < 1e-9 and abs(bo_raise[-1][1]) < 1e-9)
check("near-vertical last move back-off never descends",
      all(bo_raise[i][2] <= bo_raise[i + 1][2] + 1e-12 for i in range(len(bo_raise) - 1))
      and bo_raise[0][2] >= -1.0 - 1e-12)

# Steep descent (5:1): a plain 45° rise would land at z=-4.5, below the
# surface cut at x=0.5 (z=-2.5).
steep = [(0.0, 0.0, 0.0), (1.0, 0.0, -5.0)]
bo_steep = compute_backoff_path(steep, BackoffSettings(enabled=True, distance_mm=0.5))
check("steep descent back-off stays on or above the cut surface",
      bo_steep and abs(bo_steep[-1][0] - 0.5) < 1e-9 and bo_steep[-1][2] >= -2.5 - 1e-9)


def path_z_at(path, x, y):
    """Highest Z of `path` at XY (x, y), or None if (x, y) isn't on it."""
    best = None
    for (x1, y1, z1), (x2, y2, z2) in zip(path, path[1:]):
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 < 1e-18:
            if math.hypot(x - x1, y - y1) < 1e-7:
                z = max(z1, z2)
            else:
                continue
        else:
            t = ((x - x1) * dx + (y - y1) * dy) / L2
            if t < -1e-9 or t > 1 + 1e-9:
                continue
            if math.hypot(x1 + t * dx - x, y1 + t * dy - y) > 1e-7:
                continue
            z = z1 + t * (z2 - z1)
        best = z if best is None else max(best, z)
    return best


import random
rng = random.Random(1234)
all_ok = True
for _ in range(200):
    n = rng.randint(2, 6)
    poly = [(rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-6, 0)) for _ in range(n)]
    if rng.random() < 0.3:
        poly.append((poly[-1][0], poly[-1][1], rng.uniform(-6, 0)))  # pure-Z last leg
    pts = compute_backoff_path(poly, BackoffSettings(enabled=True, distance_mm=rng.uniform(0.1, 4.0)))
    for p in pts:
        pz = path_z_at(poly, p[0], p[1])
        if pz is None or p[2] < pz - 1e-9:
            all_ok = False
check("random polylines: back-off stays on the path's XY and never below its Z", all_ok)

# --------------------------------------------------------------------------
# generate_corner_arc
# --------------------------------------------------------------------------
print("generate_corner_arc")

corner = (10.0, 5.0, 3.0)
dir_in = (0.0, 0.0, 1.0)
dir_out = (0.6, 0.8, 0.0)  # arbitrary unit horizontal direction
R = 2.0
arc_pts = generate_corner_arc(corner, dir_in, dir_out, R, 8)
p_in_expected = (corner[0], corner[1], corner[2] - R)
p_out_expected = (corner[0] + R * dir_out[0], corner[1] + R * dir_out[1], corner[2])
check("arc starts exactly at P_in", dist(arc_pts[0], p_in_expected) < 1e-9)
check("arc ends exactly at P_out", dist(arc_pts[-1], p_out_expected) < 1e-9)

center = (p_in_expected[0] + R * dir_out[0],
          p_in_expected[1] + R * dir_out[1],
          p_in_expected[2])
check("all interior waypoints lie on the circle of radius R",
      all(abs(dist(p, center) - R) < 1e-9 for p in arc_pts))

# Tangent continuity: the first arc segment's direction should be close to
# dir_in, and the last arc segment's direction close to dir_out.
d0 = arc_pts[1]
first_dir = tuple((d0[i] - arc_pts[0][i]) for i in range(3))
first_len = math.sqrt(sum(c * c for c in first_dir))
first_dir = tuple(c / first_len for c in first_dir)
cos_in = sum(first_dir[i] * dir_in[i] for i in range(3))
check("arc's first segment direction is close to dir_in (tangent)", cos_in > 0.9)

dN = arc_pts[-2]
last_dir = tuple((arc_pts[-1][i] - dN[i]) for i in range(3))
last_len = math.sqrt(sum(c * c for c in last_dir))
last_dir = tuple(c / last_len for c in last_dir)
cos_out = sum(last_dir[i] * dir_out[i] for i in range(3))
check("arc's last segment direction is close to dir_out (tangent)", cos_out > 0.9)

check("zero radius returns empty list", generate_corner_arc(corner, dir_in, dir_out, 0.0, 8) == [])

# --------------------------------------------------------------------------
# ToolPath.iter_moves / to_gcode / get_stats integration
# --------------------------------------------------------------------------
print("ToolPath integration")

tp = ToolPath(segments=[
    [(0.0, 0.0, -5.0), (20.0, 0.0, -5.0)],
    [(20.0, 0.0, -5.0), (20.0, 0.0, -10.0)],   # same-XY step-down (multipass)
    [(50.0, 30.0, -5.0), (70.0, 30.0, -5.0)],  # gapped: retract/traverse/plunge
], metadata={'profile': 'End Mill', 'radius_mm': 3.0})

# -- disabled: no ramp/backoff/arc kinds should ever be yielded
kinds_disabled = {k for k, _ in tp.iter_moves(z_safe=5.0)}
check("no dress-up kinds appear when settings are all None",
      not ({'ramp', 'backoff', 'arc'} & kinds_disabled))
check("plain retract/traverse/plunge/cut still present when disabled",
      {'rapid_retract', 'rapid_traverse', 'plunge', 'cut'} <= kinds_disabled)

gcode_disabled = tp.to_gcode(feed_rate=800, feed_rapid=1500, z_safe=5.0)
check("disabled G-code has no ramp/back-off/arc comments",
      '; ramp' not in gcode_disabled and '; back off' not in gcode_disabled
      and '; arc' not in gcode_disabled)

# -- all three enabled
ramp_s = RampSettings(enabled=True, angle_deg=30.0, max_length_mm=15.0)
backoff_s = BackoffSettings(enabled=True, distance_mm=0.5)
arc_s = ArcSettings(enabled=True, radius_mm=1.0)

kinds_enabled = {k for k, _ in tp.iter_moves(z_safe=5.0, ramp=ramp_s, backoff=backoff_s, arc=arc_s)}
check("ramp/backoff/arc kinds appear when enabled",
      {'ramp', 'backoff', 'arc'} <= kinds_enabled)

gcode_enabled = tp.to_gcode(feed_rate=800, feed_rapid=1500, z_safe=5.0,
                            ramp=ramp_s, backoff=backoff_s, arc=arc_s)
check("enabled G-code contains ramp/back-off/arc comments",
      '; ramp' in gcode_enabled and '; back off' in gcode_enabled and '; arc' in gcode_enabled)
backoff_lines = [ln for ln in gcode_enabled.splitlines() if '; back off' in ln]
check("back-off G-code lines carry Z",
      backoff_lines and all(' Z' in ln for ln in backoff_lines))

for kind, pts in tp.iter_moves(z_safe=5.0, ramp=ramp_s, backoff=backoff_s, arc=arc_s):
    # The very first rapid_traverse is a lone target point -- the prior
    # machine position is unknown, so there's no "from" waypoint to include.
    min_len = 1 if kind == 'rapid_traverse' else 2
    check(f"{kind} move has >= {min_len} waypoint(s)", len(pts) >= min_len)

stats_disabled = tp.get_stats(feed_rate=800, feed_rapid=1500, z_safe=5.0)
stats_enabled = tp.get_stats(feed_rate=800, feed_rapid=1500, z_safe=5.0,
                              ramp=ramp_s, backoff=backoff_s, arc=arc_s)
check("get_stats and to_gcode/iter_moves never diverge on presence of dress-ups",
      stats_enabled['motion_count'] > stats_disabled['motion_count'])
check("segments themselves are untouched by dress-ups",
      tp.segments == [
          [(0.0, 0.0, -5.0), (20.0, 0.0, -5.0)],
          [(20.0, 0.0, -5.0), (20.0, 0.0, -10.0)],
          [(50.0, 30.0, -5.0), (70.0, 30.0, -5.0)],
      ])

# --------------------------------------------------------------------------
# Ramp-in only replaces the below-z=0 (in-stock) portion of a plunge; the
# air above the stock surface is always a plain vertical plunge down to
# z=0 first.
# --------------------------------------------------------------------------
print("ramp/plunge split at z=0")


def moves_for_second_segment(tp2, **kw):
    """Return the ordered list of (kind, pts) between the first 'cut' and
    the second 'cut' -- i.e. everything synthesized for the transition into
    the second segment."""
    events = list(tp2.iter_moves(**kw))
    cut_indices = [i for i, (k, _) in enumerate(events) if k == 'cut']
    return events[cut_indices[0] + 1:cut_indices[1]]


deep_ramp = RampSettings(enabled=True, angle_deg=30.0, max_length_mm=15.0)

# Deep target (-8mm): the descent crosses the stock surface, so we expect a
# plain vertical plunge from z_safe down to z=0, THEN a ramp from 0 to -8.
tp_deep = ToolPath(segments=[
    [(0.0, 0.0, -1.0), (20.0, 0.0, -1.0)],
    [(50.0, 0.0, -8.0), (70.0, 0.0, -8.0)],
])
moves_deep = moves_for_second_segment(tp_deep, z_safe=5.0, ramp=deep_ramp)
kinds_deep = [k for k, _ in moves_deep]
check("deep target: plunge (air) precedes ramp (material)",
      kinds_deep.index('plunge') < kinds_deep.index('ramp'))
plunge_air = next(pts for k, pts in moves_deep if k == 'plunge')
check("air plunge runs from z_safe to exactly z=0",
      abs(plunge_air[0][2] - 5.0) < 1e-9 and abs(plunge_air[-1][2] - 0.0) < 1e-9)
ramp_material = next(pts for k, pts in moves_deep if k == 'ramp')
check("ramp starts at exactly z=0 and ends at the target depth",
      abs(ramp_material[0][2] - 0.0) < 1e-9 and abs(ramp_material[-1][2] - (-8.0)) < 1e-9)
check("no ramp waypoint ever rises above the stock surface (z=0)",
      all(p[2] <= 1e-9 for p in ramp_material))

# Surface-level target (z=0.0mm, e.g. the top of an embossed area): this
# app's depth maps are always clipped to [0, -max_cut_depth], so a target
# can be AT the stock surface but never above it. With nothing below the
# surface to ramp into, this must be a single plain plunge from z_safe
# straight down to z=0 -- no ramp move at all.
tp_surface = ToolPath(segments=[
    [(0.0, 0.0, -1.0), (20.0, 0.0, -1.0)],
    [(50.0, 0.0, 0.0), (70.0, 0.0, 0.0)],
])
moves_surface = moves_for_second_segment(tp_surface, z_safe=5.0, ramp=deep_ramp)
kinds_surface = [k for k, _ in moves_surface]
check("surface-level (z=0) target produces no ramp move at all",
      'ramp' not in kinds_surface)
check("surface-level target is a single plain plunge from z_safe to z=0",
      kinds_surface.count('plunge') == 1)

# Same-XY step-down that's already entirely below the surface (ordinary
# multipass roughing): should behave like the old whole-descent ramp, i.e.
# no extra air-plunge segment since there's no air to plunge through.
tp_stepdown = ToolPath(segments=[
    [(0.0, 0.0, -3.0), (20.0, 0.0, -3.0)],
    [(20.0, 0.0, -3.0), (20.0, 0.0, -6.0)],
])
moves_stepdown = [ev for ev in tp_stepdown.iter_moves(z_safe=5.0, ramp=deep_ramp)
                  if ev[0] in ('plunge', 'ramp')]
# Only the first segment's own initial plunge/ramp (from z_safe) plus the
# step-down's own plunge/ramp should appear -- the step-down itself must not
# contain an extra air-plunge leg (already in material).
stepdown_kinds = [k for k, _ in moves_stepdown]
check("already-in-material step-down has no extra air-plunge segment",
      stepdown_kinds.count('plunge') <= 1)  # only the first segment's own entry plunge, if any

# Arc corner B must now be eligible even though ramp-in is enabled (it
# always transitions into a vertical plunge-through-air leg first).
arc_and_ramp = ArcSettings(enabled=True, radius_mm=1.0)
moves_deep_arc = moves_for_second_segment(tp_deep, z_safe=5.0, ramp=deep_ramp, arc=arc_and_ramp)
kinds_deep_arc = [k for k, _ in moves_deep_arc]
check("corner B is arced even when ramp-in is active",
      kinds_deep_arc.count('arc') == 2)

# Raised ramp start Z (stock extending above z=0): the vertical air plunge
# stops at start_z_mm and the zig-zag covers everything below it.
raised_ramp = RampSettings(enabled=True, angle_deg=30.0, max_length_mm=15.0,
                           start_z_mm=2.0)
moves_raised = moves_for_second_segment(tp_deep, z_safe=5.0, ramp=raised_ramp)
plunge_raised = next(pts for k, pts in moves_raised if k == 'plunge')
ramp_raised = next(pts for k, pts in moves_raised if k == 'ramp')
check("raised start Z: air plunge stops at start_z_mm",
      abs(plunge_raised[0][2] - 5.0) < 1e-9 and abs(plunge_raised[-1][2] - 2.0) < 1e-9)
check("raised start Z: ramp runs from start_z_mm to the target depth",
      abs(ramp_raised[0][2] - 2.0) < 1e-9 and abs(ramp_raised[-1][2] - (-8.0)) < 1e-9)
check("raised start Z: no ramp waypoint rises above start_z_mm",
      all(p[2] <= 2.0 + 1e-9 for p in ramp_raised))

# A surface-level target is ramped into when the start Z is above it.
moves_surface_raised = moves_for_second_segment(tp_surface, z_safe=5.0, ramp=raised_ramp)
check("raised start Z: surface-level (z=0) target is ramped into",
      'ramp' in [k for k, _ in moves_surface_raised])


def implied_radius(arc_pts, z_safe):
    """Both corner arcs' Z ranges span [z_safe - r, z_safe], so the implied
    radius is z_safe minus the arc's minimum Z."""
    return z_safe - min(p[2] for p in arc_pts)


# Corner A (retract-side) and corner B (plunge-side) must get the SAME
# radius, and that radius must NOT change depending on whether ramp-in is
# enabled -- ramp-in only changes what happens *below* z=0, it must not
# affect how large the corner fillets above z=0 are allowed to be.
tp_symmetric = ToolPath(segments=[
    [(0.0, 0.0, -3.0), (20.0, 0.0, -3.0)],
    [(50.0, 0.0, -3.0), (70.0, 0.0, -3.0)],  # same depth as the previous segment's end
])
big_radius_arc = ArcSettings(enabled=True, radius_mm=10.0, n_segments=8)

moves_no_ramp = moves_for_second_segment(tp_symmetric, z_safe=5.0, arc=big_radius_arc)
arcs_no_ramp = [pts for k, pts in moves_no_ramp if k == 'arc']
r_a_no_ramp = implied_radius(arcs_no_ramp[0], 5.0)
r_b_no_ramp = implied_radius(arcs_no_ramp[1], 5.0)
check("corner A and corner B match when ramp-in is disabled",
      abs(r_a_no_ramp - r_b_no_ramp) < 1e-6)

moves_with_ramp = moves_for_second_segment(tp_symmetric, z_safe=5.0, ramp=deep_ramp, arc=big_radius_arc)
arcs_with_ramp = [pts for k, pts in moves_with_ramp if k == 'arc']
r_a_with_ramp = implied_radius(arcs_with_ramp[0], 5.0)
r_b_with_ramp = implied_radius(arcs_with_ramp[1], 5.0)
check("corner A and corner B still match when ramp-in is enabled",
      abs(r_a_with_ramp - r_b_with_ramp) < 1e-6)
check("enabling ramp-in does not change corner B's radius",
      abs(r_b_no_ramp - r_b_with_ramp) < 1e-6)
check("enabling ramp-in does not change corner A's radius either",
      abs(r_a_no_ramp - r_a_with_ramp) < 1e-6)

# Same check, but with the previous and next segments at DIFFERENT depths,
# so the retract climb and plunge descent are genuinely different lengths.
# Both corners must still share the same (smaller) radius.
tp_asymmetric_depth = ToolPath(segments=[
    [(0.0, 0.0, -2.0), (20.0, 0.0, -2.0)],
    [(50.0, 0.0, -8.0), (70.0, 0.0, -8.0)],
])
moves_asym = moves_for_second_segment(tp_asymmetric_depth, z_safe=5.0, arc=big_radius_arc)
arcs_asym = [pts for k, pts in moves_asym if k == 'arc']
r_a_asym = implied_radius(arcs_asym[0], 5.0)
r_b_asym = implied_radius(arcs_asym[1], 5.0)
check("corner A and corner B share the same radius even at different depths",
      abs(r_a_asym - r_b_asym) < 1e-6)
check("the shared radius is bound by the tighter (retract-side) constraint",
      abs(r_a_asym - (5.0 - (-2.0)) / 2.0) < 1e-6)

# --------------------------------------------------------------------------
# Corner arcs must never dip below the stock surface (z=0): their lateral
# sweep is only guaranteed collision-free at z >= 0 (this app's flat stock
# top), unlike a straight vertical retract/plunge at the *same* XY as the
# cut. A large configured arc radius combined with a small z_safe and a deep
# cut used to make retract_top = z_safe - r_a go negative.
# --------------------------------------------------------------------------
print("arc corners never dip below z=0")

low_zsafe_arc = ArcSettings(enabled=True, radius_mm=10.0, n_segments=8)
tp_deep_cut = ToolPath(segments=[
    [(0.0, 0.0, -20.0), (20.0, 0.0, -20.0)],
    [(50.0, 0.0, -20.0), (70.0, 0.0, -20.0)],
])
moves_low_zsafe = list(tp_deep_cut.iter_moves(z_safe=2.0, arc=low_zsafe_arc))
all_arc_z = [p[2] for kind, pts in moves_low_zsafe if kind == 'arc' for p in pts]
check("corner arcs stay at or above the stock surface even with a large radius",
      len(all_arc_z) == 0 or all(z >= -1e-6 for z in all_arc_z))
retract_end_z = [pts[-1][2] for kind, pts in moves_low_zsafe if kind == 'rapid_retract']
check("retract never ends below the stock surface (its start is the cut depth, which is fine)",
      all(z >= -1e-6 for z in retract_end_z))

# --------------------------------------------------------------------------
# Plunge rate, spindle speed and spin-up dwell
# --------------------------------------------------------------------------
print("plunge rate / spindle speed")

g_plain = tp.to_gcode(feed_rate=800, feed_rapid=1500, z_safe=5.0)
check("no S word or dwell when RPM is 0",
      not any(ln.startswith(('S', 'G4')) for ln in g_plain.splitlines()))
check("plunge runs at the cutting feed when plunge rate is 0",
      all('F800' in ln for ln in g_plain.splitlines() if '; plunge' in ln))

g_rpm = tp.to_gcode(feed_rate=800, feed_rapid=1500, z_safe=5.0,
                    plunge_rate=300, spindle_rpm=18000, spindle_dwell_s=2.5)
rpm_lines = g_rpm.splitlines()
s_idx = next(i for i, ln in enumerate(rpm_lines) if ln.startswith('S18000'))
check("S word follows the safe-height move",
      rpm_lines[s_idx - 1].startswith('G0 Z5.000'))
check("dwell follows the S word", rpm_lines[s_idx + 1].startswith('G4 P2.5'))
check("plunge lines use the plunge rate",
      all('F300' in ln for ln in rpm_lines if '; plunge' in ln))
step_down_line = next(ln for ln in rpm_lines if ln.startswith('G1 X20.000 Y0.000 Z-10.000'))
check("same-XY descent inside a cut segment uses the plunge rate", 'F300' in step_down_line)
check("horizontal cuts keep the cutting feed",
      any(ln.startswith('G1 X70.000') and 'F800' in ln for ln in rpm_lines))

g_no_dwell = tp.to_gcode(feed_rate=800, feed_rapid=1500, z_safe=5.0,
                         spindle_rpm=18000, spindle_dwell_s=0.0)
check("no G4 when dwell is 0",
      'S18000' in g_no_dwell and not any(ln.startswith('G4') for ln in g_no_dwell.splitlines()))

g_ramp = tp.to_gcode(feed_rate=800, feed_rapid=1500, z_safe=5.0,
                     ramp=ramp_s, plunge_rate=300)
check("ramps keep the cutting feed",
      all('F800' in ln for ln in g_ramp.splitlines() if '; ramp' in ln))

t_fast = tp.get_stats(feed_rate=800, feed_rapid=1500, z_safe=5.0)['est_minutes']
t_slow = tp.get_stats(feed_rate=800, feed_rapid=1500, z_safe=5.0,
                      plunge_rate=100)['est_minutes']
check("slower plunge rate increases the time estimate", t_slow > t_fast)
t_dwell = tp.get_stats(feed_rate=800, feed_rapid=1500, z_safe=5.0,
                       spindle_rpm=18000, spindle_dwell_s=6.0)['est_minutes']
check("dwell adds to the time estimate", abs((t_dwell - t_fast) * 60.0 - 6.0) < 1e-6)
t_dwell_no_rpm = tp.get_stats(feed_rate=800, feed_rapid=1500, z_safe=5.0,
                              spindle_dwell_s=6.0)['est_minutes']
check("dwell not counted when RPM is 0", abs(t_dwell_no_rpm - t_fast) < 1e-9)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
