import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


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

    def get_stats(self, feed_rate=800.0, feed_rapid=3000.0, z_safe=5.0,
                  accel_mm_s2=300.0):
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
        """
        cut_dist = rapid_dist = 0.0
        motion_count = 0
        prev_end = None

        # (length_mm, cruise_mm_s, unit_dir) per straight motion, in path
        # order, used below for the acceleration-aware time estimate.
        motions = []

        feed_rate_s  = max(feed_rate,  1.0) / 60.0
        feed_rapid_s = max(feed_rapid, 1.0) / 60.0

        def add_motion(p0, p1, vmax):
            dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            length = (dx*dx + dy*dy + dz*dz) ** 0.5
            if length < 1e-9:
                return
            motions.append((length, vmax, (dx/length, dy/length, dz/length)))

        for points in self.segments:
            if not points:
                continue

            # Rapid gap between consecutive segments: retract + traverse + plunge
            if prev_end is not None and (
                    abs(prev_end[0] - points[0][0]) > 1e-4
                    or abs(prev_end[1] - points[0][1]) > 1e-4):
                px, py, pz = prev_end
                nx, ny, nz = points[0]
                retract  = (px, py, z_safe)
                traverse = (nx, ny, z_safe)
                plunge   = (nx, ny, nz)
                rapid_dist += (abs(z_safe - pz)                              # retract
                               + ((nx-px)**2 + (ny-py)**2) ** 0.5           # traverse
                               + abs(z_safe - nz))                           # plunge
                add_motion(prev_end, retract,  feed_rapid_s)
                add_motion(retract,  traverse, feed_rapid_s)
                add_motion(traverse, plunge,   feed_rapid_s)
                motion_count += 3

            if len(points) >= 2:
                motion_count += len(points) - 1
                for i in range(len(points) - 1):
                    x0, y0, z0 = points[i]
                    x1, y1, z1 = points[i + 1]
                    cut_dist += ((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2) ** 0.5
                    add_motion(points[i], points[i + 1], feed_rate_s)

            prev_end = points[-1]

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

        return {
            'motion_count': motion_count,
            'cut_distance_mm': cut_dist,
            'rapid_distance_mm': rapid_dist,
            'est_minutes': time_sec / 60.0,
        }

    def to_gcode(self, feed_rate=800.0, feed_rapid=3000.0, z_safe=5.0,
                 accel_mm_s2=300.0):
        """Generate G-code string for the motion segments.

        The output is self-contained: it starts with a move to z_safe and ends
        at z_safe.  Preamble and postamble (spindle on/off, units, etc.) are
        handled by the caller and are not included here.
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
            f"; Work area: {meta.get('width_cm', 0):.1f} cm × "
            f"{meta.get('height_cm', 0):.1f} cm",
        ]

        if 'rdp_tolerance' in meta:
            lines.append(f"; RDP path-simplification tolerance: "
                        f"{meta['rdp_tolerance']*1000:.1f} µm")
        if 'link_threshold_mm' in meta:
            lines.append(f"; Link threshold (short-rapid replacement): "
                        f"{meta['link_threshold_mm']:.2f} mm")
        if 'refining_threshold_um' in meta:
            lines.append(f"; Refining threshold: "
                        f"{meta['refining_threshold_um']:.0f} µm "
                        f"(spans from prior residual only)")

        stats = self.get_stats(feed_rate, feed_rapid, z_safe, accel_mm_s2)

        lines += [
            f"; Total motion lines: {stats['motion_count']}",
            f"; Cut distance: {stats['cut_distance_mm']:.0f} mm",
            f"; Rapid distance: {stats['rapid_distance_mm']:.0f} mm",
            f"; Estimated time: {fmt_time(stats['est_minutes'])} "
            f"@ {feed_rate:.0f} mm/min",
            "",
        ]

        # Explicit safe-height move at the start — body is self-contained and
        # does not rely on the preamble having done this.
        lines.append(f"G0 Z{z_safe:.3f}  ; move to safe height")
        at_safe = True

        prev_end = None   # (x, y, z) of the last emitted endpoint

        for points in self.segments:
            if not points:
                continue

            x0, y0, z0 = points[0]

            needs_rapid = (
                prev_end is None
                or abs(prev_end[0] - x0) > 1e-4
                or abs(prev_end[1] - y0) > 1e-4
            )
            if needs_rapid:
                if not at_safe:
                    lines.append(f"G0 Z{z_safe:.3f}  ; retract")
                lines.append(f"G0 X{x0:.3f} Y{y0:.3f}  ; rapid to span start")
                lines.append(f"G1 Z{z0:.3f} F{feed_rate}  ; plunge")
                at_safe = False
            elif abs(prev_end[2] - z0) > 1e-4:
                lines.append(f"G1 Z{z0:.3f} F{feed_rate}")

            for x, y, z in points[1:]:
                lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed_rate}")

            prev_end = points[-1]

        if not at_safe:
            lines.append(f"G0 Z{z_safe:.3f}  ; retract")

        return "\n".join(lines)
