import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from dressup import generate_ramp, compute_backoff_path, generate_corner_arc


def _ramp_plan(points, z_from, ramp):
    """Plan the descent from z_from down to points[0]'s Z (points[0][2]).

    Ramp-in only ever replaces the portion of a plunge below the ramp's
    start Z (``ramp.start_z_mm``, default 0.0 = the stock surface) and within
    ``ramp.max_depth_mm`` of the target (0 = no limit) -- everything above
    that is a plain vertical plunge, through the air and any depth earlier
    passes already cleared. Returns (split_z, ramp_pts):

    - split_z = min(z_from, start_z, z0 + max_depth), the Z at which the
      vertical leg (if any) hands off to the zig-zag (if any).
    - ramp_pts = generate_ramp(points, split_z, ramp) when there's actually
      material below split_z to ramp into (points[0][2] < split_z), else
      [points[0]] (nothing to ramp -- the caller should do a single plain
      plunge for the whole z_from -> points[0][2] descent).
    """
    z0 = points[0][2]
    start_z = ramp.start_z_mm if ramp is not None else 0.0
    split_z = min(z_from, start_z)
    if ramp is not None and ramp.max_depth_mm > 0.0:
        split_z = min(split_z, z0 + ramp.max_depth_mm)
    if z0 < split_z:
        return split_z, generate_ramp(points, split_z, ramp)
    return split_z, [points[0]]


def _is_vertical_descent(p0, p1):
    """True for a move that only goes down in Z (no XY travel) -- a plunge
    or a peck, fed at the plunge rate."""
    return (abs(p1[0] - p0[0]) < 1e-6 and abs(p1[1] - p0[1]) < 1e-6
            and p1[2] < p0[2] - 1e-9)


def _emit_plunge_ramp(x0, y0, z0, z_from, split_z, ramp_pts):
    """Yield the ('plunge', ...) / ('ramp', ...) move(s) for a descent from
    z_from to z0 at fixed XY, per the plan from _ramp_plan(): a plain
    vertical plunge down to split_z, followed by the zig-zag for the
    portion below it -- or a single plain plunge for the
    whole descent when there's nothing to ramp into or ramping isn't usable.
    """
    if len(ramp_pts) > 1:
        if split_z < z_from - 1e-9:
            yield ('plunge', [(x0, y0, z_from), (x0, y0, split_z)])
        yield ('ramp', [(x0, y0, split_z)] + ramp_pts)
    else:
        yield ('plunge', [(x0, y0, z_from), (x0, y0, z0)])


@dataclass
class ToolPath:
    """Structured representation of a computed tool path."""

    # Ordered list of motion segments. Each entry is a list of (x, y, z) mm
    # triples. Consecutive segments whose endpoints touch are bridged at feed
    # rate (no retract); a gap triggers retract → rapid → plunge.
    segments: List[List[Tuple[float, float, float]]] = field(default_factory=list)

    # 2-D float32 array (same shape as depth_map, values in [0, 1]): cumulative
    # depth achieved across all passes up to and including this one, in the same
    # normalised depth units as depth_map (1.0 = max_depth mm).  Equals
    # depth_map minus the remaining material.  Pass depth_map − result as
    # `prior_residual` / `work_surface` to the next compute_*_toolpath call.
    result: np.ndarray = None

    # 2-D float32 array (same shape as depth_map, values in [0, 1]): the
    # theoretical depth limit the tool can reach at each pixel given its geometry,
    # computed via grey-erosion of depth_map (Stage 2).  Equivalent to what the
    # tool would achieve with infinitely dense raster lines.  Used as the per-pixel
    # cut target in raster passes and to determine where the V-bit can engage in
    # V-carve passes.
    accessible: np.ndarray = None

    # 2-D float32 array (same shape as depth_map), populated only by the
    # Cutout algorithm: 1.0 = material still present, 0.0 = physically cut
    # all the way through the stock (a real hole).  None (the default, and
    # what every other algorithm leaves it as) means no step has flagged any
    # holes.  Deliberately orthogonal to `result`/`accessible` -- a cutout
    # kerf clips to 1.0 in those (same value ordinary relief carving reaches
    # at its deepest points), so this is the only place "physically removed"
    # is distinguishable from "carved to full depth". Never participates in
    # any depth/residual arithmetic; consumed only by the result preview to
    # render holes in a distinct colour.
    alpha: np.ndarray = None

    # 2-D bool array (same shape as depth_map), populated only by the
    # Undercut algorithm: pixels within the head radius of a position where
    # the tool fits under an overhang at some level -- a top-down outline of
    # the undercut material the step reaches. Consumed only by the 2D views'
    # overhang overlay.
    undercut_reach: np.ndarray = None

    metadata: dict = field(default_factory=dict)

    def resample(self, spacing_mm: float) -> 'np.ndarray':
        """Densely sample all segments at *spacing_mm* XY arc-length intervals.

        Returns a (N, 3) float64 array of (x, y, z) mm positions. Air moves
        between non-adjacent segments are not interpolated.
        """
        pts_out = []
        for seg in self.segments:
            if not seg:
                continue
            pts = np.asarray(seg, dtype=np.float64)
            pts_out.append(pts[0])
            carry = 0.0
            for i in range(1, len(pts)):
                diff = pts[i] - pts[i - 1]
                seg_len = float(np.hypot(diff[0], diff[1]))
                if seg_len < 1e-10:
                    continue
                pos = carry
                while pos < seg_len:
                    t = pos / seg_len
                    pts_out.append(pts[i - 1] + t * diff)
                    pos += spacing_mm
                carry = pos - seg_len
        if not pts_out:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(pts_out, dtype=np.float64)

    def iter_moves(self, segments=None, z_safe=5.0, ramp=None, backoff=None, arc=None):
        """Single source of truth for rapid/retract/plunge/ramp/back-off/arc
        geometry — shared by ``to_gcode``, ``get_stats``, and the 3D preview
        so they can never disagree about what the tool actually does between
        cuts.

        Yields ``(kind, pts)`` tuples where ``kind`` is one of
        ``'rapid_retract'``, ``'rapid_traverse'``, ``'backoff'``, ``'arc'``,
        ``'ramp'``, ``'plunge'``, ``'cut'``, and ``pts`` is an ordered list of
        two or more ``(x, y, z)`` mm waypoints — the first waypoint is always
        the position the tool is already at (i.e. the previous move's last
        point). This method only synthesizes geometry; feed rates and G-code
        formatting are the caller's concern.

        ``segments`` defaults to ``self.segments``; callers may pass a
        truncated list (e.g. for a progress-scrub preview) to only dress up
        a prefix of the path. ``ramp``/``backoff``/``arc`` are optional
        ``dressup.RampSettings`` / ``BackoffSettings`` / ``ArcSettings``
        instances (or ``None``/disabled, which reproduces the plain
        retract → traverse → plunge behaviour).

        Ramp-in only ever replaces the portion of a plunge below the ramp's
        start Z (``RampSettings.start_z_mm``, normally the stock top) and
        within ``RampSettings.max_depth_mm`` of the target with a zig-zag;
        anything above that is always a plain vertical plunge first (see
        ``_ramp_plan``/``_emit_plunge_ramp``).
        """
        segs = segments if segments is not None else self.segments
        prev_points = None

        for points in segs:
            if not points:
                continue
            x0, y0, z0 = points[0]
            gap = prev_points is None or (
                abs(prev_points[-1][0] - x0) > 1e-4
                or abs(prev_points[-1][1] - y0) > 1e-4
            )

            if gap:
                if prev_points is not None:
                    base = prev_points[-1]
                    bo = compute_backoff_path(prev_points, backoff)
                    if bo:
                        yield ('backoff', [base] + bo)
                        base = bo[-1]
                    px, py, pz = base
                    nx, ny = x0, y0
                    horiz = math.hypot(nx - px, ny - py)

                    # Both corners of this rapid (retract-side and
                    # plunge-side) share a single radius, clamped by
                    # whichever side has less room: the retract's climb
                    # (z_safe - pz), the plunge's descent (z_safe - z0), half
                    # the horizontal traverse (so the two arcs don't overlap),
                    # and z_safe itself -- each arc's Z sweeps between z_safe
                    # and (z_safe - r) while moving laterally, which is only
                    # guaranteed safe at z >= 0 (this app's flat stock top).
                    r = 0.0
                    u = None
                    if horiz > 1e-9:
                        u = ((nx - px) / horiz, (ny - py) / horiz, 0.0)
                        if arc is not None and arc.enabled:
                            r = min(arc.radius_mm, (z_safe - pz) / 2.0,
                                    (z_safe - z0) / 2.0, horiz / 2.0, z_safe)
                            if r < 0.1:
                                r = 0.0

                    if r > 0.0:
                        retract_top = z_safe - r
                        yield ('rapid_retract', [(px, py, pz), (px, py, retract_top)])
                        arc_a = generate_corner_arc((px, py, z_safe), (0.0, 0.0, 1.0),
                                                     u, r, arc.n_segments)
                        yield ('arc', [(px, py, retract_top)] + arc_a)
                        trav_start = arc_a[-1]
                    else:
                        yield ('rapid_retract', [(px, py, pz), (px, py, z_safe)])
                        trav_start = (px, py, z_safe)

                    if r > 0.0:
                        trav_end = (nx - u[0] * r, ny - u[1] * r, z_safe)
                        yield ('rapid_traverse', [trav_start, trav_end])
                        arc_b = generate_corner_arc((nx, ny, z_safe), u, (0.0, 0.0, -1.0),
                                                     r, arc.n_segments)
                        yield ('arc', [trav_end] + arc_b)
                        z_from = z_safe - r
                    else:
                        yield ('rapid_traverse', [trav_start, (nx, ny, z_safe)])
                        z_from = z_safe

                    # Ramp-in only ever replaces the lower portion of the
                    # descent that _ramp_plan picks; everything above it is
                    # a plain vertical plunge.
                    split_z, ramp_pts = _ramp_plan(points, z_from, ramp)
                    yield from _emit_plunge_ramp(x0, y0, z0, z_from, split_z, ramp_pts)
                else:
                    # First move: prior machine position is unknown, so there
                    # is no direction to fillet a corner against.
                    yield ('rapid_traverse', [(x0, y0, z_safe)])
                    split_z, ramp_pts = _ramp_plan(points, z_safe, ramp)
                    yield from _emit_plunge_ramp(x0, y0, z0, z_safe, split_z, ramp_pts)

            elif abs(prev_points[-1][2] - z0) > 1e-4:
                z_from = prev_points[-1][2]
                split_z, ramp_pts = _ramp_plan(points, z_from, ramp)
                yield from _emit_plunge_ramp(x0, y0, z0, z_from, split_z, ramp_pts)

            if len(points) >= 2:
                yield ('cut', points)

            prev_points = points

        if prev_points is not None:
            base = prev_points[-1]
            bo = compute_backoff_path(prev_points, backoff)
            if bo:
                yield ('backoff', [base] + bo)
                base = bo[-1]
            yield ('rapid_retract', [base, (base[0], base[1], z_safe)])

    def get_stats(self, feed_rate=800.0, feed_rapid=3000.0, z_safe=5.0,
                  accel_mm_s2=300.0, ramp=None, backoff=None, arc=None,
                  plunge_rate=0.0, spindle_rpm=0.0, spindle_dwell_s=0.0):
        """Compute statistics from the path segments.

        ``est_minutes`` models a trapezoidal (or triangular, for motions too
        short to reach cruise speed) velocity profile bounded by
        *accel_mm_s2*, and carries velocity through a corner only in
        proportion to how straight it is — full speed when collinear, down
        to a full stop for a ~90 deg-or-sharper turn. A plain
        distance/feed_rate estimate implicitly assumes infinite
        acceleration and can undershoot real run time by 2x or more on
        paths with many short, zig-zagging motions (dense raster/V-carve
        scans), since the tool never actually reaches the commanded feed
        rate on them.

        Geometry (rapid/retract/plunge/ramp/back-off/arc synthesis) comes
        from :meth:`iter_moves`, so these stats always match what
        :meth:`to_gcode` actually emits. Vertical descents run at
        *plunge_rate* when it is > 0, and the spindle spin-up dwell counts
        when *spindle_rpm* > 0, matching :meth:`to_gcode`.
        """
        cut_dist = rapid_dist = 0.0
        motion_count = 0

        # (length_mm, cruise_mm_s, unit_dir) per straight motion, in path
        # order, used below for the acceleration-aware time estimate.
        motions = []

        feed_rate_s  = max(feed_rate,  1.0) / 60.0
        feed_rapid_s = max(feed_rapid, 1.0) / 60.0
        plunge_s     = (max(plunge_rate, 1.0) / 60.0 if plunge_rate > 0
                        else feed_rate_s)

        def add_motion(p0, p1, vmax):
            dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            length = (dx*dx + dy*dy + dz*dz) ** 0.5
            if length < 1e-9:
                return
            motions.append((length, vmax, (dx/length, dy/length, dz/length)))

        rapid_kinds = ('rapid_retract', 'rapid_traverse', 'arc')
        arc_feed_s = max(arc.feed_mm_min, 1.0) / 60.0 if arc is not None else feed_rapid_s

        for kind, pts in self.iter_moves(z_safe=z_safe, ramp=ramp, backoff=backoff, arc=arc):
            if kind == 'arc':
                vmax = arc_feed_s
            elif kind in rapid_kinds:
                vmax = feed_rapid_s
            else:
                vmax = feed_rate_s
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                d = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2 + (p1[2]-p0[2])**2) ** 0.5
                if kind in rapid_kinds:
                    rapid_dist += d
                else:
                    cut_dist += d
                if kind in ('plunge', 'cut') and _is_vertical_descent(p0, p1):
                    add_motion(p0, p1, plunge_s)
                else:
                    add_motion(p0, p1, vmax)
                motion_count += 1

        # Junction (corner) velocities: the machine starts and ends the
        # whole path at rest, so junction_v[0] and junction_v[-1] stay 0.
        n = len(motions)
        junction_v = [0.0] * (n + 1)
        for i in range(1, n):
            _, vmax_prev, dir_prev = motions[i - 1]
            _, vmax_next, dir_next = motions[i]
            cos_theta = (dir_prev[0]*dir_next[0] + dir_prev[1]*dir_next[1]
                         + dir_prev[2]*dir_next[2])
            junction_v[i] = max(0.0, cos_theta) * min(vmax_prev, vmax_next)

        accel = max(accel_mm_s2, 1e-6)
        time_sec = 0.0
        for i, (length, vmax, _) in enumerate(motions):
            v_enter = min(junction_v[i],     vmax)
            v_exit  = min(junction_v[i + 1], vmax)
            d_acc = (vmax*vmax - v_enter*v_enter) / (2.0 * accel)
            d_dec = (vmax*vmax - v_exit *v_exit)  / (2.0 * accel)
            if d_acc + d_dec <= length:
                d_cruise = length - d_acc - d_dec
                time_sec += (((vmax - v_enter) + (vmax - v_exit)) / accel
                             + d_cruise / vmax)
            else:
                vp = (accel*length + 0.5*(v_enter*v_enter + v_exit*v_exit)) ** 0.5
                vp = min(vp, vmax)
                time_sec += (max(0.0, vp - v_enter) + max(0.0, vp - v_exit)) / accel

        if spindle_rpm > 0 and spindle_dwell_s > 0:
            time_sec += spindle_dwell_s

        return {
            'motion_count': motion_count,
            'cut_distance_mm': cut_dist,
            'rapid_distance_mm': rapid_dist,
            'est_minutes': time_sec / 60.0,
        }

    def to_gcode(self, feed_rate=800.0, feed_rapid=3000.0, z_safe=5.0,
                 accel_mm_s2=300.0, ramp=None, backoff=None, arc=None,
                 plunge_rate=0.0, spindle_rpm=0.0, spindle_dwell_s=0.0):
        """Generate G-code string for the motion segments.

        The output is self-contained: it starts with a move to z_safe and ends
        at z_safe.  Preamble and postamble (spindle on/off, units, etc.) are
        handled by the caller and are not included here.

        *ramp*, *backoff* and *arc* are optional ``dressup`` settings objects
        that replace vertical plunges with a zig-zag ramp, back the tool away
        from the wall before retracting, and round the rapid's sharp corners
        into tangent arcs, respectively. Left as ``None`` (or disabled),
        output is the plain retract/traverse/plunge behaviour.

        Vertical descents (plunges, and pecks inside cut segments) run at
        *plunge_rate* when it is > 0; ramps keep *feed_rate*. When
        *spindle_rpm* > 0, an ``S`` word follows the initial safe-height
        move, then a ``G4`` dwell of *spindle_dwell_s* seconds (if > 0) so
        the spindle reaches speed before cutting.
        """

        def fmt_time(minutes):
            if minutes < 1.0:
                return f"{minutes * 60:.0f} sec"
            if minutes < 60.0:
                return f"{minutes:.1f} min"
            h = int(minutes // 60)
            m = minutes % 60
            return f"{h}h {m:.0f}min"

        meta = self.metadata
        lines = [
            "; CNC Relief Carving — generated by CNC Relief Carving app",
            f"; Profile: {meta.get('profile', 'Unknown')}   "
            f"Radius: {meta.get('radius_mm', 0):.3f} mm",
            f"; Step-over: {meta.get('step_mm', 0):.3f} mm   "
            f"Step-down: {meta.get('stepdown', 0):.3f} mm",
            f"; Max depth: {meta.get('max_depth', 0):.3f} mm   "
            f"Passes: {meta.get('n_passes', 0)}",
            f"; Work area: {meta.get('width_mm', 0):.1f} mm × "
            f"{meta.get('height_mm', 0):.1f} mm",
        ]

        if 'rdp_tolerance' in meta:
            if 'rdp_z_tolerance' in meta:
                lines.append(f"; RDP path-simplification tolerance: "
                            f"{meta['rdp_tolerance']*1000:.1f} µm lateral, "
                            f"{meta['rdp_z_tolerance']*1000:.2f} µm vertical")
            else:
                lines.append(f"; RDP path-simplification tolerance: "
                            f"{meta['rdp_tolerance']*1000:.1f} µm")
        if 'link_threshold_mm' in meta:
            lines.append(f"; Link threshold (short-rapid replacement): "
                        f"{meta['link_threshold_mm']:.2f} mm")
        if 'refining_threshold_um' in meta:
            lines.append(f"; Refining threshold: "
                        f"{meta['refining_threshold_um']:.0f} µm "
                        f"(spans from prior residual only)")

        stats = self.get_stats(feed_rate, feed_rapid, z_safe, accel_mm_s2,
                                ramp, backoff, arc, plunge_rate,
                                spindle_rpm, spindle_dwell_s)
        plunge_feed = plunge_rate if plunge_rate > 0 else feed_rate

        lines += [
            f"; Total motion lines: {stats['motion_count']}",
            f"; Cut distance: {stats['cut_distance_mm']:.0f} mm",
            f"; Rapid distance: {stats['rapid_distance_mm']:.0f} mm",
            f"; Estimated time: {fmt_time(stats['est_minutes'])} "
            f"@ {feed_rate:.0f} mm/min",
        ]
        if plunge_rate > 0:
            lines.append(f"; Plunge rate: {plunge_rate:.0f} mm/min")
        lines.append("")

        # Explicit safe-height move at the start — body is self-contained and
        # does not rely on the preamble having done this.
        lines.append(f"G0 Z{z_safe:.3f}  ; move to safe height")
        if spindle_rpm > 0:
            lines.append(f"M3 S{spindle_rpm:.0f}  ; spindle on (clockwise) at speed")
            if spindle_dwell_s > 0:
                lines.append(f"G4 P{spindle_dwell_s:g}  ; wait for spindle to reach speed")

        for kind, pts in self.iter_moves(z_safe=z_safe, ramp=ramp, backoff=backoff, arc=arc):
            if kind == 'rapid_retract':
                lines.append(f"G0 Z{pts[-1][2]:.3f}  ; retract")
            elif kind == 'rapid_traverse':
                x, y, _ = pts[-1]
                lines.append(f"G0 X{x:.3f} Y{y:.3f}  ; rapid to span start")
            elif kind == 'backoff':
                for x, y, z in pts[1:]:
                    lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed_rate}  ; back off before retract")
            elif kind == 'arc':
                for x, y, z in pts[1:]:
                    lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{arc.feed_mm_min}  ; arc")
            elif kind == 'ramp':
                for x, y, z in pts[1:]:
                    lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed_rate}  ; ramp")
            elif kind == 'plunge':
                lines.append(f"G1 Z{pts[-1][2]:.3f} F{plunge_feed}  ; plunge")
            elif kind == 'cut':
                for p0, (x, y, z) in zip(pts, pts[1:]):
                    f = plunge_feed if _is_vertical_descent(p0, (x, y, z)) else feed_rate
                    lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{f}")

        return "\n".join(lines)
