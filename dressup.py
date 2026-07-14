"""Pure-geometry helpers for optional G-code "dress-ups" applied at emission
time on top of an already-computed ToolPath: ramp-in (replaces a vertical
plunge with a zig-zag descent), back-off (pulls the tool away from the wall
it just cut before retracting), and arc-rounded rapid corners (replaces the
sharp 90-degree turns of a retract/traverse/plunge rapid with a tangent-arc
polyline). None of these functions touch ToolPath.segments; they only
synthesize the extra travel geometry that toolpath.py's iter_moves() splices
into the motion stream.
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
    n_segments: int = 8           # polyline segments per quarter-circle, internal


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
    return (x2 - ux * dist, y2 - uy * dist, z2)


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
