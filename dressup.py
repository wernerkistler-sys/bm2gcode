"""Pure-geometry helpers for optional "dress-ups" applied to a toolpath.

Most of these (ramp-in, back-off, arc-rounded rapid corners) run at G-code
emission time on top of an already-computed ToolPath: they synthesize extra
travel geometry that toolpath.py's iter_moves() splices into the motion
stream at the gaps between segments, and never touch ToolPath.segments
itself.

apply_tabs() is the one exception. Leaving uncut "tabs" (bridges) in a
closed Cutout ring means raising Z for a stretch of the ring's *interior* --
that's a change to the cut geometry itself, not gap-filling travel, so it
has to run earlier: on the flat, full-depth ring points computed by
compute_cutout_toolpath(), before make_multipass() depth-slices them into
passes. See algorithms.py's compute_cutout_toolpath for the call site.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

Point = Tuple[float, float, float]


@dataclass
class RampSettings:
    enabled: bool = False
    angle_deg: float = 30.0
    max_length_mm: float = 15.0
    max_legs: int = 20            # internal safety cap, not exposed in UI


@dataclass
class BackoffSettings:
    enabled: bool = False
    distance_mm: float = 0.5


@dataclass
class ArcSettings:
    enabled: bool = False
    radius_mm: float = 3.0
    feed_mm_min: float = 1500.0   # literal G1 feedrate commanded on the arc segments
    n_segments: int = 8           # polyline segments per quarter-circle, internal


@dataclass
class TabSettings:
    enabled: bool = False
    count: int = 4
    width_mm: float = 5.0
    height_mm: float = 1.0
    phase_pct: float = 0.0        # 0-100, shifts all tabs together; wraps (100 == 0)
    tab_angle_deg: float = 60.0   # wall angle from horizontal; 90 = vertical/rectangular


def _xy_dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def generate_ramp(points: List[Point], z_start: float,
                   settings: Optional[RampSettings]) -> List[Point]:
    """Zig-zag descent from z_start down to points[0]'s Z, retracing the
    leading portion of `points` back and forth in XY while descending in Z.

    The XY footprint visited is always a literal subset of `points` (built by
    truncating/interpolating `points` itself, never a straight chord), and Z
    at every visited XY is clamped to never go below (more negative than) the
    true floor Z the final path leaves there — so the ramp can never gouge
    material that should remain, or cut outside the upcoming cut's own
    footprint.

    Returns a list of waypoints ending exactly at points[0] (its own Z).
    The caller already knows the starting position (x0, y0, z_start) and
    should NOT prepend it — it is intentionally omitted here. Returns
    [points[0]] (length 1) when ramping isn't applicable/worthwhile — the
    caller should treat that as "do a plain plunge instead".
    """
    if not settings or not settings.enabled or len(points) < 2:
        return [points[0]]

    x0, y0, z0 = points[0]
    depth = z_start - z0
    if depth <= 1e-9:
        return [points[0]]

    # Build the lookahead polyline: points[] truncated (with true, interpolated
    # Z carried at the cut point) to at most max_length_mm of XY arc length.
    lookahead = [points[0]]
    cum = [0.0]
    total_len = 0.0
    for i in range(1, len(points)):
        x1, y1, z1 = points[i - 1]
        x2, y2, z2 = points[i]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len < 1e-9:
            continue
        if total_len + seg_len >= settings.max_length_mm:
            remain = settings.max_length_mm - total_len
            t = max(0.0, min(1.0, remain / seg_len))
            lookahead.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1)))
            total_len += remain
            cum.append(total_len)
            break
        lookahead.append((x2, y2, z2))
        total_len += seg_len
        cum.append(total_len)

    L = total_len
    if L < 0.5 or len(lookahead) < 2:
        return [points[0]]

    tan_a = math.tan(math.radians(settings.angle_deg))
    if tan_a < 1e-9:
        return [points[0]]

    n_passes = max(1, math.ceil(depth / (2.0 * L * tan_a)))
    n_legs = min(max(2, 2 * n_passes), settings.max_legs)
    if n_legs % 2 == 1:
        n_legs -= 1
    n_legs = max(2, n_legs)
    leg_length = min(depth / (n_legs * tan_a), L)

    def sample(s: float) -> Point:
        """Point on `lookahead` at XY arc length s (0 <= s <= L)."""
        if s <= 0.0:
            return lookahead[0]
        for i in range(1, len(lookahead)):
            seg_len = cum[i] - cum[i - 1]
            if seg_len < 1e-12:
                continue
            if cum[i] >= s - 1e-9:
                t = max(0.0, min(1.0, (s - cum[i - 1]) / seg_len))
                x1, y1, z1 = lookahead[i - 1]
                x2, y2, z2 = lookahead[i]
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1))
        return lookahead[-1]

    knots = [s for s in cum if 0.0 < s < leg_length]

    result: List[Point] = []
    for k in range(1, n_legs + 1):
        forward = (k % 2 == 1)
        if forward:
            s_values = [0.0] + knots + [leg_length]
        else:
            s_values = [leg_length] + list(reversed(knots)) + [0.0]
        for s in s_values:
            if k == 1 and s == 0.0:
                continue  # equals the caller-known starting position
            x, y, floor_z = sample(s)
            t_in_leg = (s / leg_length) if leg_length > 1e-12 else 0.0
            progress = ((k - 1 + t_in_leg) if forward else (k - t_in_leg)) / n_legs
            naive_z = z_start - progress * depth
            z = max(naive_z, floor_z)
            pt = (x, y, z)
            if result and _xy_dist(result[-1], pt) < 1e-9 and abs(result[-1][2] - z) < 1e-9:
                continue
            result.append(pt)

    if not result:
        return [points[0]]
    result[-1] = points[0]
    return result


def compute_backoff_point(prev_points: List[Point],
                           settings: Optional[BackoffSettings]) -> Optional[Point]:
    """Point `distance_mm` back along the reverse of the last leg's travel
    direction, clamped to that leg's own length. None if the last move was
    Z-only (no XY direction to back off along), the leg is degenerate, or
    back-off is disabled/zero-distance."""
    if not settings or not settings.enabled or settings.distance_mm <= 0.0:
        return None
    if len(prev_points) < 2:
        return None
    x1, y1, _z1 = prev_points[-2]
    x2, y2, z2 = prev_points[-1]
    dx, dy = x2 - x1, y2 - y1
    horiz = math.hypot(dx, dy)
    if horiz < 1e-6:
        return None
    ux, uy = dx / horiz, dy / horiz
    dist = min(settings.distance_mm, horiz)
    return (x2 - ux * dist, y2 - uy * dist, z2 + dist)


def generate_corner_arc(corner: Point, dir_in: Tuple[float, float, float],
                         dir_out: Tuple[float, float, float],
                         radius_mm: float, n_segments: int) -> List[Point]:
    """Tangent-circle fillet polyline between two perpendicular unit
    directions dir_in (incoming, pointing into `corner`) and dir_out
    (outgoing, pointing away from `corner`).

    Returns the list of waypoints from the trim-in point to the trim-out
    point (both endpoints included), NOT including `corner` itself. Empty
    list if radius_mm is non-positive.
    """
    if radius_mm <= 1e-9 or n_segments < 1:
        return []
    cx, cy, cz = corner
    R = radius_mm
    p_in = (cx - R * dir_in[0], cy - R * dir_in[1], cz - R * dir_in[2])
    p_out = (cx + R * dir_out[0], cy + R * dir_out[1], cz + R * dir_out[2])
    o = (p_in[0] + R * dir_out[0], p_in[1] + R * dir_out[1], p_in[2] + R * dir_out[2])
    v1 = (p_in[0] - o[0], p_in[1] - o[1], p_in[2] - o[2])
    v2 = (p_out[0] - o[0], p_out[1] - o[1], p_out[2] - o[2])

    pts = []
    for i in range(n_segments + 1):
        theta = (math.pi / 2.0) * i / n_segments
        c, s = math.cos(theta), math.sin(theta)
        pts.append((
            o[0] + v1[0] * c + v2[0] * s,
            o[1] + v1[1] * c + v2[1] * s,
            o[2] + v1[2] * c + v2[2] * s,
        ))
    return pts


TabRegion = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


def _apply_tabs_to_ring(ring: List[Point], settings: TabSettings,
                         tool_diameter_mm: float, closed_tol_mm: float):
    """Raise Z to leave uncut, trapezoid-profiled tabs at evenly-spaced
    points around one closed, flat ring -- tapered walls (angle set by
    settings.tab_angle_deg, measured from horizontal like RampSettings'
    angle_deg; 90 = the old vertical-walled rectangular profile) so the
    tool ramps onto/off the tab instead of stepping straight into a wall.
    Returns (new_ring, tab_regions, warnings). `ring` is returned unchanged
    (and no regions/warnings) if it isn't closed, is degenerate, or tabs
    don't fit."""
    n = len(ring)
    if n < 3:
        return ring, [], []
    if _xy_dist(ring[0], ring[-1]) > closed_tol_mm:
        return ring, [], []  # open contour -- tabs need a closed loop

    cum = [0.0] * n
    for i in range(1, n):
        cum[i] = cum[i - 1] + _xy_dist(ring[i - 1], ring[i])
    L = cum[-1]
    if L < 1e-6:
        return ring, [], []

    z_full = ring[0][2]
    climb_mm = settings.height_mm
    tab_top_z = z_full + climb_mm
    # settings.width_mm is the tab's footprint left standing in the material
    # (README: "Each tab's footprint along the cut path"). The centerline
    # lift window has to be wider than that: right up to the edge of the
    # lift window the tool is still cutting at full engagement, and its
    # radius sweeps tool_diameter_mm/2 further into the would-be tab on
    # each side, eating into the bridge. Widening the window by the tool's
    # diameter cancels that bleed so the standing bridge comes out to the
    # requested width_mm.
    window_width_mm = settings.width_mm + tool_diameter_mm
    half_w = min(window_width_mm / 2.0, L / 2.0 - 1e-9)
    if half_w <= 1e-9:
        return ring, [], []

    # Ramp run: the arc-length distance needed to climb climb_mm at the
    # requested wall angle (from horizontal). 90 degrees (or anything close
    # enough that tan() blows up) means an instant vertical step -- the old
    # rectangular profile -- with no run at all. The run is clamped to
    # half_w so a shallow angle can't ask for more room than the tab's own
    # footprint; past that clamp the profile is just a full triangular
    # "roof" with no flat top, rather than encroaching on the neighbours
    # count/max_count already keep clear.
    angle_deg = min(max(settings.tab_angle_deg, 1.0), 90.0)
    if angle_deg >= 89.999:
        ramp_run = 0.0
    else:
        ramp_run = climb_mm / math.tan(math.radians(angle_deg))
    ramp_run = min(ramp_run, half_w)
    flat_half_w = half_w - ramp_run

    warnings: List[str] = []
    count = max(1, int(settings.count))
    max_count = max(1, int(L // max(window_width_mm, 1e-6)))
    if count > max_count:
        warnings.append(
            f"Tab count reduced from {count} to {max_count} "
            f"(ring only {L:.1f}mm long, too short for "
            f"{count} tabs of {settings.width_mm}mm)")
        count = max_count

    # phase is in units of "one inter-tab spacing" (not one full ring lap) --
    # shifting by a whole inter-tab spacing maps an evenly-spaced tab set
    # onto itself, so anchoring phase_pct's full 0-100 range to exactly one
    # spacing (rather than one full lap) means every value in that range
    # produces a genuinely distinct arrangement instead of repeating the
    # same pattern `count` times over.
    phase = (settings.phase_pct % 100.0) / 100.0
    centers = [(((i + phase) / count) % 1.0) * L for i in range(count)]

    def sample_at(s: float) -> Point:
        if s <= 0.0:
            return ring[0]
        if s >= L:
            return ring[-1]
        for i in range(1, n):
            if cum[i] >= s - 1e-9:
                seg_len = cum[i] - cum[i - 1]
                t = 0.0 if seg_len < 1e-12 else (s - cum[i - 1]) / seg_len
                x1, y1, z1 = ring[i - 1]
                x2, y2, z2 = ring[i]
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1))
        return ring[-1]

    def circ_dist(s: float, c: float) -> float:
        d = abs(s - c)
        return min(d, L - d)

    def tab_target_z(s: float):
        """Highest tab-raised Z any tab wants at arc length s, or None if no
        tab reaches this far (flat top for |s-center| <= flat_half_w,
        linear taper for flat_half_w < |s-center| <= half_w)."""
        best = None
        for c in centers:
            d = circ_dist(s, c)
            if d > half_w + 1e-9:
                continue
            if d <= flat_half_w + 1e-9:
                t = tab_top_z
            elif ramp_run > 1e-9:
                t = z_full + climb_mm * (half_w - d) / ramp_run
            else:
                t = tab_top_z
            if best is None or t > best:
                best = t
        return best

    # Guarantee an explicit vertex at each trapezoid corner (footprint edge
    # and flat-top edge), regardless of the ring's natural point density --
    # the sloped walls between these and whatever natural points fall
    # between them are already straight-line-correct without further
    # subdivision, since a taper is linear in (arc length, Z).
    extra_s = set()
    for c in centers:
        for s in (c - half_w, c - flat_half_w, c + flat_half_w, c + half_w):
            extra_s.add(s % L)

    events: List[Tuple[float, Point, bool]] = [
        (cum[i], ring[i], False) for i in range(n)
    ]
    for s in extra_s:
        events.append((s, sample_at(s), True))
    events.sort(key=lambda e: e[0])

    new_ring: List[Point] = []
    last_s = None
    for s, (x, y, z), is_boundary in events:
        if last_s is not None and abs(s - last_s) < 1e-9 and not is_boundary:
            continue  # coincides with an already-emitted boundary vertex
        target = tab_target_z(s)
        if target is not None:
            z = max(z, target)
        new_ring.append((x, y, z))
        last_s = s

    tab_regions: List[TabRegion] = []
    for sc in centers:
        cx, cy, _ = sample_at(sc)
        sx, sy, _ = sample_at((sc - half_w) % L)
        ex, ey, _ = sample_at((sc + half_w) % L)
        tab_regions.append(((cx, cy), (sx, sy), (ex, ey)))

    return new_ring, tab_regions, warnings


def apply_tabs(rings: List[List[Point]], settings: Optional[TabSettings],
                tool_diameter_mm: float, closed_tol_mm: float = 0.05):
    """Raise Z to leave uncut tabs at evenly-spaced points around each closed
    ring in `rings` (same shape as compute_cutout_toolpath's full_depth_segs
    -- one flat, full-depth closed polyline per silhouette shape).

    `tool_diameter_mm` widens the centerline Z-lift window beyond
    settings.width_mm so the tab standing in the material -- not just the
    toolpath's lift window -- comes out to the requested width; see
    _apply_tabs_to_ring.

    Returns (new_rings, tab_regions, warnings):
      new_rings   -- same length/order as `rings`, tab windows raised to
                     z_full + settings.height_mm
      tab_regions -- list of (center_xy, start_xy, end_xy) across all rings,
                     for 2-D preview markers
      warnings    -- human-readable strings (e.g. tab count clamped because
                     a ring was too short), for the caller to surface

    Rings that aren't closed (e.g. a silhouette touching the canvas edge)
    are returned unchanged and contribute no regions/warnings.
    """
    if not settings or not settings.enabled or settings.count < 1:
        return rings, [], []
    new_rings = []
    tab_regions: List[TabRegion] = []
    warnings: List[str] = []
    for ring in rings:
        new_ring, regions, warns = _apply_tabs_to_ring(
            ring, settings, tool_diameter_mm, closed_tol_mm)
        new_rings.append(new_ring)
        tab_regions.extend(regions)
        warnings.extend(warns)
    return new_rings, tab_regions, warnings
