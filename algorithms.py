import dataclasses
import numpy as np
from tool import Tool
from toolpath import ToolPath
from dressup import apply_tabs


def _two_opt_reorder(seg_entries, seg_exits, origin=(0.0, 0.0), max_passes=5,
                     link_threshold=0.0, per_rapid_overhead=0.0,
                     seg_z=None, z_jump_weight=0.0):
    """Return (order, flipped) permutation that minimises rapid travel cost.

    seg_entries[i], seg_exits[i]: (x, y) G-code mm for the i-th segment's
    entry and exit points.  origin: tool XY position before the first segment.

    link_threshold: connections shorter than this distance are treated as free
        (they will be stitched at feed rate with no retract/plunge).
    per_rapid_overhead: fixed cost added to every connection that exceeds
        link_threshold, representing the retract + plunge distance overhead
        (typically 2 × z_safe).  Without this the optimiser may break several
        cheap stitches to save a few mm of traverse.
    seg_z: optional list of z-values (mm, negative) for each segment.  When
        provided, z_jump_weight * |Δz| is added to every transition cost,
        discouraging large depth jumps across z-levels.
    z_jump_weight: cost per mm of z-jump (applies regardless of link_threshold).

    Returns parallel lists — order (indices into the original lists) and
    flipped (bool: whether to reverse that segment's traversal direction).
    """
    n = len(seg_entries)
    if n < 3:
        return list(range(n)), [False] * n

    ent     = list(seg_entries)
    ext     = list(seg_exits)
    order   = list(range(n))
    flipped = [False] * n

    use_z = seg_z is not None and z_jump_weight > 0.0
    ent_z = list(seg_z) if use_z else None
    ext_z = list(seg_z) if use_z else None
    origin_z = 0.0

    def edge_cost(a, b, za=0.0, zb=0.0):
        d = ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5
        z_cost = z_jump_weight * abs(za - zb) if use_z else 0.0
        return z_cost if d < link_threshold else per_rapid_overhead + d + z_cost

    for _ in range(max_passes):
        improved = False
        for i in range(-1, n - 2):
            prev_ext = origin if i == -1 else ext[i]
            pz = (origin_z if i == -1 else ext_z[i]) if use_z else 0.0
            for j in range(i + 2, n):
                old = edge_cost(prev_ext, ent[i + 1], pz, ent_z[i + 1] if use_z else 0.0)
                new = edge_cost(prev_ext, ext[j],     pz, ext_z[j]     if use_z else 0.0)
                if j + 1 < n:
                    old += edge_cost(ext[j],     ent[j + 1],
                                     ext_z[j]     if use_z else 0.0,
                                     ent_z[j + 1] if use_z else 0.0)
                    new += edge_cost(ent[i + 1], ent[j + 1],
                                     ent_z[i + 1] if use_z else 0.0,
                                     ent_z[j + 1] if use_z else 0.0)
                if new < old - 1e-9:
                    lo, hi = i + 1, j + 1
                    block_e = ent[lo:hi]
                    block_x = ext[lo:hi]
                    ent[lo:hi]     = list(reversed(block_x))
                    ext[lo:hi]     = list(reversed(block_e))
                    order[lo:hi]   = list(reversed(order[lo:hi]))
                    flipped[lo:hi] = [not f for f in reversed(flipped[lo:hi])]
                    if use_z:
                        block_z = ent_z[lo:hi]
                        ent_z[lo:hi] = list(reversed(block_z))
                        ext_z[lo:hi] = list(reversed(block_z))
                    improved = True
        if not improved:
            break

    return order, flipped


def _bresenham_pixels(r0, c0, r1, c1):
    """Yield (row, col) pixels along the line (r0,c0)→(r1,c1)."""
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 >= r0 else -1), (1 if c1 >= c0 else -1)
    r, c = r0, c0
    if dc >= dr:
        err = 2 * dr - dc
        for _ in range(dc + 1):
            yield r, c
            if err > 0:
                r   += sr
                err -= 2 * dc
            err += 2 * dr
            c   += sc
    else:
        err = 2 * dc - dr
        for _ in range(dr + 1):
            yield r, c
            if err > 0:
                c   += sc
                err -= 2 * dr
            err += 2 * dc
            r   += sr


def _geodesic_link(r0, c0, r1, c1, ceil_norm,
                   center_depth_map, H, W, px_per_mm_x, px_per_mm_y, max_depth,
                   final_norm_val=0.0):
    """Dijkstra minimum-3D-arc-length path from (r0,c0) to (r1,c1).

    Uses the roughing surface (terrain − final_norm_val, capped at ceil_norm)
    for arc-length cost, so the path avoids edges in the surface being cut.
    """
    import heapq
    margin = max(2, max(abs(r1 - r0), abs(c1 - c0)) // 4 + 1)
    rlo = max(0, min(r0, r1) - margin)
    rhi = min(H - 1, max(r0, r1) + margin)
    clo = max(0, min(c0, c1) - margin)
    chi = min(W - 1, max(c0, c1) + margin)

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dist = {}
    prev = {}
    dist[(r0, c0)] = 0.0
    heap = [(0.0, r0, c0)]

    while heap:
        d, r, c = heapq.heappop(heap)
        if (r, c) == (r1, c1):
            break
        if d > dist.get((r, c), float('inf')):
            continue
        z_cur = min(max(float(center_depth_map[r, c]) - final_norm_val, 0.0), ceil_norm)
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (rlo <= nr <= rhi and clo <= nc <= chi):
                continue
            z_nb = min(max(float(center_depth_map[nr, nc]) - final_norm_val, 0.0),
                       ceil_norm)
            dxy2 = (dr / px_per_mm_y) ** 2 + (dc / px_per_mm_x) ** 2
            dz   = (z_nb - z_cur) * max_depth
            nd   = d + (dxy2 + dz * dz) ** 0.5
            if nd < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)] = nd
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (nd, nr, nc))

    if (r1, c1) not in prev and (r0, c0) != (r1, c1):
        return list(_bresenham_pixels(r0, c0, r1, c1))

    path, cur = [], (r1, c1)
    while cur != (r0, c0):
        path.append(cur)
        cur = prev[cur]
    path.append((r0, c0))
    path.reverse()
    return path


def mask_spans(mask_1d):
    """Return contiguous True-runs in a 1-D boolean array.

    Uses the same edge-detection trick as FDM slicer infill algorithms:
    pad with False on both ends, take np.diff, find +1 transitions
    (run starts) and -1 transitions (run ends).

    Parameters
    ----------
    mask_1d : 1-D bool array of length N

    Returns
    -------
    list of (start, end) index pairs, both inclusive, in ascending order.
    Empty list if no True pixels.
    """
    if not np.any(mask_1d):
        return []
    padded = np.concatenate(([False], mask_1d, [False])).astype(np.int8)
    diff   = np.diff(padded)
    starts = np.where(diff ==  1)[0]   # rising edges  → run begins here
    ends   = np.where(diff == -1)[0]   # falling edges → run ends before here
    return list(zip(starts.tolist(), (ends - 1).tolist()))


def simplify_line(points, tol):
    """Ramer-Douglas-Peucker path simplification (2-D or 3-D).

    Removes intermediate points whose perpendicular distance from the line
    segment connecting the current endpoints is below *tol*.  Collinear or
    near-collinear runs collapse to just their two endpoints.

    Parameters
    ----------
    points : list of (x, y, z) tuples  **or**  numpy array of shape (N, D)
    tol    : maximum allowed deviation (same units as the coordinates)

    Returns
    -------
    Same type as input: list of tuples when given a list, numpy (M, D) array
    when given an array.  Always includes first and last point.
    """
    is_array = isinstance(points, np.ndarray)
    if len(points) <= 2:
        return points.copy() if is_array else list(points)

    # Iterative RDP using an explicit stack to avoid Python recursion limits.
    # (Long branches / contours can have O(N) worst-case recursion depth when
    # the pivot is near one end, exceeding Python's ~1000-frame limit.)
    pts = np.asarray(points, dtype=np.float64)  # (N, D) — works for any D
    N   = len(pts)

    # Iterative RDP using an explicit stack to avoid Python recursion limits.
    # (Long V-carve skeleton branches can have O(N) worst-case recursion depth
    # when the pivot is consistently near one end, exceeding Python's ~1000
    # frame limit.)
    keep  = np.zeros(N, dtype=bool)
    keep[0]  = True
    keep[-1] = True

    stack = [(0, N - 1)]
    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue

        seg     = pts[end] - pts[start]
        seg_len = np.linalg.norm(seg)

        if seg_len < 1e-12:
            # Degenerate segment (start == end)
            dists = np.linalg.norm(pts[start+1:end] - pts[start], axis=1)
        else:
            diff  = pts[start+1:end] - pts[start]
            # Clamp to [0, 1] so points beyond either endpoint (e.g. a
            # backtracking spike that is collinear with start/end but not
            # between them) are measured against the nearest endpoint
            # instead of the infinite line, and so are never mistaken for
            # redundant collinear points.
            t     = np.clip(np.dot(diff, seg) / (seg_len * seg_len), 0.0, 1.0)
            proj  = pts[start] + t[:, None] * seg
            dists = np.linalg.norm(pts[start+1:end] - proj, axis=1)

        max_idx  = int(np.argmax(dists))
        max_dist = dists[max_idx]
        pivot    = start + 1 + max_idx

        if max_dist > tol:
            keep[pivot] = True
            stack.append((start, pivot))
            stack.append((pivot, end))

    return pts[keep] if is_array else [points[i] for i in range(N) if keep[i]]


def _build_kernel(tool, px_per_mm, max_depth):
    """Return (kernel, footprint) for morphological grey erosion.

    kernel    : float32 2-D array, depth offsets normalised to max_depth
    footprint : bool 2-D array, True where the tool can contact material
    """
    from math import radians, tan

    R_mm = tool.diameter_mm / 2.0

    if tool.profile == "Ball":
        k_rad = max(1, int(np.ceil(R_mm * px_per_mm)))
        ky, kx = np.mgrid[-k_rad:k_rad+1, -k_rad:k_rad+1]
        r_px = np.sqrt(kx**2 + ky**2)
        r_mm = r_px / px_per_mm
        with np.errstate(invalid="ignore"):
            cap = R_mm - np.sqrt(np.maximum(0.0, R_mm**2 - r_mm**2))
        cap[r_px > k_rad] = 1e9
        kernel    = (cap / max_depth).astype(np.float32)
        footprint = r_px <= k_rad
    elif tool.profile == "V-Bit":
        half_angle_rad = radians(tool.vbit_angle_deg / 2.0)
        tan_half = tan(half_angle_rad)
        cone_reach_mm = max_depth * tan_half
        if tool.diameter_mm > 0.0:
            cone_reach_mm = min(cone_reach_mm, R_mm)
        k_rad = max(1, int(np.ceil(cone_reach_mm * px_per_mm)))
        ky, kx = np.mgrid[-k_rad:k_rad+1, -k_rad:k_rad+1]
        r_px = np.sqrt(kx**2 + ky**2)
        r_mm = r_px / px_per_mm
        kernel    = (r_mm / (tan_half * max_depth)).astype(np.float32)
        footprint = r_px <= k_rad
    else:  # End Mill: flat bottom, vertical walls
        k_rad = max(1, int(np.ceil(R_mm * px_per_mm)))
        ky, kx = np.mgrid[-k_rad:k_rad+1, -k_rad:k_rad+1]
        r_px = np.sqrt(kx**2 + ky**2)
        cap = np.where(r_px <= k_rad, 0.0, 1e9).astype(np.float64)
        kernel    = (cap / max_depth).astype(np.float32)
        footprint = r_px <= k_rad

    return kernel, footprint


def _planning_kernel(tool, px_per_mm, max_depth, extra_offset_mm,
                      real_kernel, real_footprint):
    """Return (kernel, footprint) for the achievable-depth surface.

    Inflates the tool diameter by 2*extra_offset_mm so the erosion/dilation
    plans lateral clearance for a subsequent finishing step, without
    affecting the real tool geometry used elsewhere (material-removal
    simulation, link thresholds, metadata). Falls back to real_kernel/
    real_footprint when extra_offset_mm <= 0 (no behaviour change).
    """
    if extra_offset_mm <= 0.0:
        return real_kernel, real_footprint
    inflated = dataclasses.replace(
        tool, diameter_mm=max(tool.diameter_mm + 2.0 * extra_offset_mm, 0.01))
    return _build_kernel(inflated, px_per_mm, max_depth)


def simulate_toolpath_result(toolpath, shape, px_per_mm_x, px_per_mm_y, ox, oy,
                              max_depth_mm, kernel, footprint, resample_spacing_px=0.5):
    """Rasterize a toolpath's swept volume and return the material-removal surface.

    Densely resamples *toolpath* segments at sub-pixel XY spacing, paints each
    sample's depth into a pixel-space array, then grey-dilates with the tool
    geometry kernel to produce the cumulative material-removal map (float32,
    values in [0, 1] normalised depth units).
    """
    from scipy.ndimage import grey_dilation as _grey_dilation
    H, W = shape
    px_per_mm = (px_per_mm_x * px_per_mm_y) ** 0.5
    pts = toolpath.resample(resample_spacing_px / px_per_mm)
    achieved = np.zeros((H, W), dtype=np.float64)
    if len(pts):
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        cs     = (xs + ox) * px_per_mm_x
        rs     = (H - 1) - (ys + oy) * px_per_mm_y
        depths = np.clip(-zs / max_depth_mm, 0.0, 1.0)
        ris    = np.round(rs).astype(np.intp)
        cis    = np.round(cs).astype(np.intp)
        valid  = (ris >= 0) & (ris < H) & (cis >= 0) & (cis < W)
        np.maximum.at(achieved, (ris[valid], cis[valid]), depths[valid])
    return np.maximum(
        _grey_dilation(achieved, footprint=footprint,
                       structure=-kernel.astype(np.float64)).astype(np.float32),
        0.0)


def _plan_passes(max_depth_mm, stepdown_mm, final_stepdown_mm=0.0):
    """Return ordered list of cumulative cut-depth ceilings (mm).

    If final_stepdown_mm > 0, roughing stops at max_depth − final_stepdown_mm
    and a dedicated finishing pass reaches max_depth.
    stepdown_mm = 0 means one roughing step at full roughing depth (equivalent
    to stepdown_mm = max_depth_mm).
    """
    if stepdown_mm <= 0.0:
        if final_stepdown_mm <= 0.0:
            return [max_depth_mm]
        stepdown_mm = max_depth_mm   # one roughing step + one final pass
    if final_stepdown_mm > 0.0 and final_stepdown_mm < max_depth_mm:
        roughable = max_depth_mm - final_stepdown_mm
        n = max(1, int(np.ceil(roughable / stepdown_mm)))
        passes = [min((i + 1) * stepdown_mm, roughable) for i in range(n)]
        passes.append(max_depth_mm)
    else:
        n = max(1, int(np.ceil(max_depth_mm / stepdown_mm)))
        passes = [min((i + 1) * stepdown_mm, max_depth_mm) for i in range(n)]
    return passes


def make_multipass(segments, max_depth_mm, stepdown_mm, final_stepdown_mm=0.0):
    """Depth-slice full-depth segments into per-pass segments.

    Parameters
    ----------
    segments       : list of [(x, y, z_mm)…] where z_mm is negative (below surface)
    max_depth_mm   : total cut depth in mm
    stepdown_mm    : max depth increment per roughing pass (0 = single pass)
    final_stepdown_mm : depth reserved for a dedicated finishing pass (0 = none)

    Returns
    -------
    list of (ceil_mm, is_final_pass, pass_segments)
      ceil_mm       — this pass's depth ceiling in mm
      is_final_pass — True for the dedicated finishing pass
      pass_segments — list of [(x, y, z_mm)…] segments with z clipped to ceiling

    A point's depth after roughing to ceiling ceil_mm is
    max(min(z_full + final_stepdown_mm, 0.0), -ceil_mm), i.e. the full depth
    (minus the reserve left for the finishing pass) clipped to what this pass
    is allowed to remove. A roughing point is kept only if that pass's target
    is deeper than the previous roughing pass's target — otherwise the tool
    would repeat a move at a depth it has already cut, so a raster/contour
    line is only split once some points have nothing left to cut while others
    still do.

    If final_stepdown_mm > 0, roughing skips points whose full depth is
    within the reserve (nothing to rough there), and the dedicated finishing
    pass covers every point of the line, including those skipped points, at
    its true full depth. If final_stepdown_mm == 0, there is no finishing
    pass, so the first roughing pass covers every point of the line instead,
    including points whose full depth is 0 (no cut needed).
    """
    if stepdown_mm <= 0.0 and final_stepdown_mm <= 0.0:
        return [(max_depth_mm, True, list(segments))]

    pass_ceilings = _plan_passes(max_depth_mm, stepdown_mm, final_stepdown_mm)
    EPS = 1e-6
    n_passes = len(pass_ceilings)
    has_dedicated_final = final_stepdown_mm > 0.0

    def rough_target(z_full, ceil_mm):
        return max(min(z_full + final_stepdown_mm, 0.0), -ceil_mm)

    result = []
    for pi, ceil_mm in enumerate(pass_ceilings):
        is_final = (pi == n_passes - 1)
        is_dedicated_final = is_final and has_dedicated_final
        pass_segs = []
        for seg in segments:
            sub = []
            for x, y, z_full in seg:
                if is_dedicated_final:
                    z_target = z_full
                    keep = True
                else:
                    z_target = rough_target(z_full, ceil_mm)
                    if pi == 0:
                        keep = True if not has_dedicated_final else (z_target < -EPS)
                    else:
                        prev_target = rough_target(z_full, pass_ceilings[pi - 1])
                        keep = z_target < prev_target - EPS

                if keep:
                    sub.append((x, y, z_target))
                else:
                    if sub:
                        pass_segs.append(sub)
                        sub = []
            if sub:
                pass_segs.append(sub)
        result.append((ceil_mm, is_final, pass_segs))
    return result


def _order_free_raster_pass(pass_segs, origin, use_2opt, link_threshold_mm):
    """Greedy nearest-endpoint ordering with optional 2-opt refinement,
    for a raster pass with no forced cut direction — segments may be
    visited, and individually reversed, in whichever order minimises
    travel.
    """
    n_segs    = len(pass_segs)
    unvisited = list(range(n_segs))
    g_order   = []
    g_flips   = []
    cur       = origin
    while unvisited:
        best_i, best_rev, best_dist = 0, False, float('inf')
        for ui, si in enumerate(unvisited):
            seg = pass_segs[si]
            xs, ys = seg[0][0],  seg[0][1]
            xe, ye = seg[-1][0], seg[-1][1]
            ds = ((xs - cur[0])**2 + (ys - cur[1])**2) ** 0.5
            de = ((xe - cur[0])**2 + (ye - cur[1])**2) ** 0.5
            if ds <= de:
                d, rev = ds, False
            else:
                d, rev = de, True
            if d < best_dist:
                best_i, best_rev, best_dist = ui, rev, d
        si = unvisited.pop(best_i)
        g_order.append(si)
        g_flips.append(best_rev)
        seg = pass_segs[si]
        cur = (seg[0][0], seg[0][1]) if best_rev else (seg[-1][0], seg[-1][1])

    if use_2opt and n_segs >= 3:
        ent_xy = []
        ext_xy = []
        for si, rev in zip(g_order, g_flips):
            seg = pass_segs[si]
            if rev:
                ent_xy.append((seg[-1][0], seg[-1][1]))
                ext_xy.append((seg[0][0],  seg[0][1]))
            else:
                ent_xy.append((seg[0][0],  seg[0][1]))
                ext_xy.append((seg[-1][0], seg[-1][1]))
        order, flips = _two_opt_reorder(ent_xy, ext_xy,
                                        origin=origin,
                                        link_threshold=link_threshold_mm,
                                        per_rapid_overhead=10.0)
        return [
            list(reversed(pass_segs[g_order[i]])) if (g_flips[i] ^ flips[k])
            else pass_segs[g_order[i]]
            for k, i in enumerate(order)
        ]
    return [
        list(reversed(pass_segs[si])) if rev else pass_segs[si]
        for si, rev in zip(g_order, g_flips)
    ]


def _order_locked_raster_pass(pass_segs, direction, cur):
    """Order one depth-pass's segments for a locked Climb/Conventional
    direction, keeping consecutive cuts on the same or an adjacent raster
    line so the "uncut material on a fixed side" guarantee holds between
    segments, not just within one.

    pass_segs : list of [(x, y, z_mm)...], each already oriented for
                ascending line index ("mode A" — set by Phase 1's
                FORCE_REVERSE).
    direction : "X" (horizontal lines, grouped by their constant Y) or "Y"
                (vertical lines, grouped by their constant X).
    cur       : (x, y) starting tool position.

    Mode A visits lines in ascending line-index order using each segment's
    stored orientation. Mode B visits in descending order using each
    segment reversed. Reversing progression and per-segment travel
    together preserves which side carries uncut material, so mode B is
    equally valid — just the mirror image of mode A. Ascending line index
    is descending Y for horizontal lines but ascending X for vertical
    ones, so grouping keys on a direction-adjusted coordinate (-y or +x)
    makes "ascending key order" mean mode A either way.

    At each step the candidate pool is the current line's remaining
    segments plus the immediately next line's (in the active mode's
    direction) — never a line further away — so consecutive cuts always
    stay adjacent. A candidate from the next line becomes the new current
    line. When both are empty, the pointer advances; if it walks off the
    end of the map while segments remain elsewhere, direction flips (mode
    A <-> B) and the sweep resumes from that same end, oscillating until
    nothing is left.
    """
    def coord(seg):
        return -seg[0][1] if direction == "X" else seg[0][0]

    groups = {}
    for i, seg in enumerate(pass_segs):
        groups.setdefault(coord(seg), []).append(i)
    keys = sorted(groups)
    key_index = {k: idx for idx, k in enumerate(keys)}
    remaining = {k: set(v) for k, v in groups.items()}
    remaining_count = len(pass_segs)

    def orient(i, mode):
        seg = pass_segs[i]
        return seg if mode == 'A' else list(reversed(seg))

    def nearest(cands, mode, pos):
        best_i, best_d = None, float('inf')
        for i in cands:
            x, y = orient(i, mode)[0][:2]
            d = (x - pos[0])**2 + (y - pos[1])**2
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    # Choose the nearer of the two sweep endpoints as the starting mode.
    cand_a = nearest(remaining[keys[0]], 'A', cur)
    cand_b = nearest(remaining[keys[-1]], 'B', cur)
    xa, ya = orient(cand_a, 'A')[0][:2]
    xb, yb = orient(cand_b, 'B')[0][:2]
    da = (xa - cur[0])**2 + (ya - cur[1])**2
    db = (xb - cur[0])**2 + (yb - cur[1])**2
    mode, ptr = ('A', 0) if da <= db else ('B', len(keys) - 1)

    ordered = []
    while remaining_count > 0:
        step = 1 if mode == 'A' else -1
        nptr = ptr + step
        cands = list(remaining.get(keys[ptr], ())) if 0 <= ptr < len(keys) else []
        if 0 <= nptr < len(keys):
            cands += list(remaining[keys[nptr]])
        if cands:
            bi = nearest(cands, mode, cur)
            seg = orient(bi, mode)
            ordered.append(seg)
            key = coord(pass_segs[bi])
            remaining[key].discard(bi)
            remaining_count -= 1
            cur = seg[-1][:2]
            ptr = key_index[key]
        else:
            ptr = nptr
            if not (0 <= ptr < len(keys)):
                if remaining_count == 0:
                    break
                mode = 'B' if mode == 'A' else 'A'
                ptr = len(keys) - 1 if mode == 'B' else 0
    return ordered


def _grid_line_ids(boundary_px, px_per_mm, step_mm, raster_offset_pct):
    """Evenly spaced grid-line pixel indices spanning [0, boundary_px],
    step-adjusted to fit exactly (no partial leftover line) and shifted by
    raster_offset_pct of the adjusted step.

    Shared by Horizontal/Vertical raster lines (one axis) and both axes of
    the Spiral grid.

    Returns
    -------
    (line_ids, adj_step_mm) — line_ids is a list of pixel indices in
    [0, boundary_px]; adj_step_mm is the step size after rounding to fit.
    """
    span_mm = boundary_px / px_per_mm
    n = max(1, round(span_mm / step_mm))
    adj_step_mm = span_mm / n
    start_mm = (raster_offset_pct / 100.0) * adj_step_mm
    positions_mm = [start_mm + i * adj_step_mm
                    for i in range(n + 1)
                    if start_mm + i * adj_step_mm <= span_mm + 1e-9]
    line_ids = [min(int(round(m * px_per_mm)), boundary_px) for m in positions_mm]
    return line_ids, adj_step_mm


def _line_points(direction, fixed_idx, lo_idx, hi_idx, reverse,
                  center_depth_contour_raster, prior_mask, resid_tol,
                  px_per_mm, H, _ox, _oy, max_depth):
    """Build zero or more point-list segments along one raster line (or a
    sub-range of a line, for spiral legs), gated by prior_mask/mask_spans.

    direction : "X" (fixed row, points vary across columns lo_idx..hi_idx)
                or "Y" (fixed column, points vary across rows lo_idx..hi_idx)
    fixed_idx : the constant row (direction="X") or column (direction="Y")
                pixel index for this line/leg.
    lo_idx, hi_idx : inclusive pixel-index bounds along the varying axis;
                     always lo_idx <= hi_idx — use `reverse` for travel
                     direction, never swapped bounds.
    reverse   : if True, each resulting segment's points are reversed so
                the segment travels from hi_idx toward lo_idx instead of
                lo_idx toward hi_idx.

    Returns
    -------
    list of point-lists — [[(x_mm, y_mm, -z_mm), ...], ...] — normally one
    element, or several when prior_mask/mask_spans splits the range into
    disjoint active sub-spans.
    """
    if direction == "X":
        line_depth = center_depth_contour_raster[fixed_idx, lo_idx:hi_idx + 1]
    else:
        line_depth = center_depth_contour_raster[lo_idx:hi_idx + 1, fixed_idx]

    if prior_mask is not None:
        if direction == "X":
            line_mask = prior_mask[fixed_idx, lo_idx:hi_idx + 1]
        else:
            line_mask = prior_mask[lo_idx:hi_idx + 1, fixed_idx]
        effective = (line_depth * line_mask).astype(np.float32)
        spans = mask_spans(effective > resid_tol)
    else:
        spans = [(0, len(line_depth) - 1)]

    segs = []
    for s, e in spans:                       # s, e are LOCAL indices, 0-based within lo_idx..hi_idx
        points = []
        for local_px in range(s, e + 1):
            px = lo_idx + local_px
            row, col = (fixed_idx, px) if direction == "X" else (px, fixed_idx)
            z_mm = float(center_depth_contour_raster[row, col]) * max_depth
            x_mm = col / px_per_mm - _ox
            y_mm = (H - 1 - row) / px_per_mm - _oy
            points.append((x_mm, y_mm, -z_mm))
        if points:
            if reverse:
                points = list(reversed(points))
            segs.append(points)
    if reverse:
        # mask_spans always returns spans in ascending pixel-index order;
        # when the line travels hi_idx -> lo_idx, the span nearer hi_idx
        # is reached first, so the span LIST order must flip too, not
        # just each span's own point order.
        segs = list(reversed(segs))
    return segs


def _spiral_index_pairs_cw(ny, nx):
    """(row_idx, col_idx) ARRAY-position pairs in outside-in clockwise
    order: top row L→R, right col T→B, bottom row R→L, left col B→T,
    each ring shrinking inward. Standard 'spiral matrix' traversal,
    operating on grid-LINE counts (ny, nx), not raw image pixel counts.
    """
    top, bottom, left, right = 0, ny - 1, 0, nx - 1
    order = []
    while top <= bottom and left <= right:
        order += [(top, c) for c in range(left, right + 1)]
        top += 1
        order += [(r, right) for r in range(top, bottom + 1)]
        right -= 1
        if top <= bottom:
            order += [(bottom, c) for c in range(right, left - 1, -1)]
            bottom -= 1
        if left <= right:
            order += [(r, left) for r in range(bottom, top - 1, -1)]
            left += 1
    return order


def _spiral_index_pairs_ccw(ny, nx):
    """Mirror of _spiral_index_pairs_cw: left col T→B, bottom row L→R,
    right col B→T, top row R→L, each ring shrinking inward.
    """
    top, bottom, left, right = 0, ny - 1, 0, nx - 1
    order = []
    while top <= bottom and left <= right:
        order += [(r, left) for r in range(top, bottom + 1)]
        left += 1
        order += [(bottom, c) for c in range(left, right + 1)]
        bottom -= 1
        if left <= right:
            order += [(r, right) for r in range(bottom, top - 1, -1)]
            right -= 1
        if top <= bottom:
            order += [(top, c) for c in range(right, left - 1, -1)]
            top += 1
    return order


def _rle_into_legs(pts):
    """Run-length-encode a sequence of axis-aligned adjacent grid points
    (real pixel coordinates, as produced by _spiral_index_pairs_cw/ccw
    applied to row_ids/col_ids) into maximal straight legs.

    Two consecutive points always share either their row or their column
    (never both, except when the whole spiral is a single point). This
    automatically absorbs the "ring shrinks to 1 pixel wide" edge case
    into whichever neighbouring run the corner pixel belongs to, with no
    special-casing needed.

    Returns
    -------
    list of (direction, fixed_idx, start_idx, end_idx) — direction is "X"
    (fixed row, varying column) or "Y" (fixed column, varying row),
    matching compute_raster_toolpath's existing direction convention.
    start_idx/end_idx are actual pixel indices and may be given in either
    order: start_idx > end_idx means the leg travels toward a *decreasing*
    pixel index.
    """
    n = len(pts)
    if n == 1:
        r, c = pts[0]
        return [("X", r, c, c)]
    legs = []
    i = 0
    while i < n - 1:
        r0, c0 = pts[i]
        r1, c1 = pts[i + 1]
        if r1 == r0:
            k = i + 1
            while k + 1 < n and pts[k + 1][0] == r0:
                k += 1
            legs.append(("X", r0, c0, pts[k][1]))
        else:
            k = i + 1
            while k + 1 < n and pts[k + 1][1] == c0:
                k += 1
            legs.append(("Y", c0, r0, pts[k][0]))
        i = k
    return legs


def _spiral_legs(row_ids, col_ids, sense):
    """Perimeter-tracing spiral traversal of a (row_ids × col_ids) grid,
    outside-in, in the given rotational `sense` ("CW" or "CCW"). Handles
    ny==1, nx==1, ny==nx==1, and non-square ny/nx transparently — see
    _rle_into_legs.
    """
    ny, nx = len(row_ids), len(col_ids)
    index_order = (_spiral_index_pairs_cw(ny, nx) if sense == "CW"
                   else _spiral_index_pairs_ccw(ny, nx))
    pts = [(row_ids[r], col_ids[c]) for r, c in index_order]
    return _rle_into_legs(pts)


def _is_contiguous_chain(segs, tol_mm=1e-6):
    """True if segs (in list order) form one unbroken chain: each
    segment's last point is within tol_mm of the next segment's first
    point (in XY). Used to detect whether a pass's segments are still a
    single continuous path (e.g. an unfragmented spiral coil), in which
    case there is no travel to optimise and reordering only risks harm.
    """
    for i in range(len(segs) - 1):
        x0, y0 = segs[i][-1][0], segs[i][-1][1]
        x1, y1 = segs[i + 1][0][0], segs[i + 1][0][1]
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 > tol_mm * tol_mm:
            return False
    return True


def compute_raster_toolpath(depth_map, n_grey_levels, tool,
                            step_mm, stepdown_mm, max_depth,
                            width_mm, height_mm, raster_dir,
                            origin="Bottom-Left",
                            feed_rate=800.0,
                            work_surface_tol=None,
                            previous_result=None,
                            prior_reference=None,
                            subtract_unaccessible=False,
                            raster_offset_pct=0.0,
                            use_2opt=False,
                            cut_direction="Either",
                            final_stepdown_mm=0.0,
                            extra_offset_mm=0.0,
                            progress=None,
                            erosion_cache=None,
                            erosion_cache_key=None):
    """Compute a raster toolpath by scanning the depth map with the given tool.

    Parameters
    ----------
    depth_map         : 2-D float32 [0, 1]  — 1.0 = max cut depth (intaglio)
    n_grey_levels     : number of grey levels in source image (e.g. 256)
    tool              : Tool — encapsulates profile, diameter_mm, vbit_angle_deg
    step_mm           : lateral step-over in mm
    stepdown_mm       : max depth increment per roughing z-pass in mm
    max_depth         : total maximum cut depth in mm
    width_mm          : physical image width in mm
    height_mm         : physical image height in mm
    raster_dir        : "Horizontal" | "Vertical" | "Spiral (Outside-In)" |
                        "Spiral (Inside-Out)" — Horizontal/Vertical scan in
                        parallel lines; the Spiral patterns trace the depth
                        map's perimeter inward in concentric rectangular
                        rings (Outside-In) or the reverse, starting near the
                        center (Inside-Out), without lifting the tool.
                        Always starts at the top-left corner and uses
                        independent row/column step-overs, so non-square
                        depth maps are fully covered on both axes. See
                        cut_direction for how Climb/Conventional map onto
                        the spiral's clockwise/counter-clockwise sense.
    raster_offset_pct : percentage [0, 100) to shift raster lines relative to the
                        step-over distance; e.g. 50 shifts every line by half a step,
                        useful when combining with a previous raster step at 0%.
    cut_direction     : "Either" | "Climb" | "Conventional" — for a clockwise
                        spindle, forces every scan line to travel in the
                        direction that keeps uncut material on the tool's
                        right (Climb) or left (Conventional). "Either"
                        preserves the default behaviour: no forced direction,
                        and the ordering pass below is free to reverse
                        segments for shorter travel.
    origin            : G-code origin convention ("Bottom-Left", "Top-Left", …)
    feed_rate         : cutting feed rate in mm/min
    work_surface_tol  : threshold (normalised [0,1]) applied to the residual
                        (depth_map − prior_reference).  0.0 (default) visits
                        every pixel; higher values skip pixels where the
                        residual is shallower than the threshold.
    previous_result   : optional 2-D float32 [0,1] cumulative result from a
                        prior step (i.e. ToolPath.result).  Combined with the
                        raster's own removal surface to produce the cumulative
                        result of this step.  None treats the prior removal
                        as zero.
    prior_reference   : optional 2-D float32 [0,1] reference surface from a
                        prior step, subtracted from depth_map to obtain the
                        residual that gates which raster lines and spans are
                        cut (combined with work_surface_tol).  Pass
                        prior_step.result for residual-based gating,
                        prior_step.accessible for geometry-based gating, or
                        None to process the full depth_map.
    final_stepdown_mm : depth (mm) reserved for a dedicated finishing pass.
                        Roughing passes stop this far above each pixel's target;
                        a final pass then cuts to full depth.  0 = no reservation.
    extra_offset_mm   : stock-to-leave allowance (mm) applied laterally and
                        vertically, so a subsequent step (typically with
                        extra_offset_mm=0) can refine the result. Laterally,
                        the tool is treated as 2*extra_offset_mm larger when
                        computing the achievable-depth surface. Vertically,
                        that surface is lowered by extra_offset_mm. 0 = none.
    progress          : callable(pct, subtask) — thread-safe progress reporter
    erosion_cache     : optional dict used to memoise the grey-erosion/grey-dilation
                        achievable-depth surface (keyed by erosion_cache_key), so a
                        subsequent call with the same tool and physical scale can skip
                        recomputing it.  None disables caching.
    erosion_cache_key : hashable key identifying the (tool, px_per_mm, max_depth)
                        combination for erosion_cache lookups.

    Returns
    -------
    ToolPath
    """
    from scipy.ndimage import grey_erosion, grey_dilation


    if work_surface_tol is None:
        work_surface_tol = 0.0

    # ── Stage 0 · Resample to square pixels if needed (pre-pass) ──────────
    H, W = depth_map.shape
    px_per_mm_x = W / width_mm
    px_per_mm_y = H / height_mm
    if px_per_mm_x != px_per_mm_y:
        from scipy.ndimage import zoom as _zoom
        target = max(px_per_mm_x, px_per_mm_y)
        zy = target / px_per_mm_y
        zx = target / px_per_mm_x
        progress(0, f"Resampling depth map to square pixels "
                    f"({px_per_mm_x:.3f} × {px_per_mm_y:.3f} → {target:.3f} px/mm)…")
        depth_map = _zoom(depth_map, (zy, zx), order=1).astype(np.float32)
        if prior_reference is not None:
            prior_reference = _zoom(prior_reference, (zy, zx), order=1).astype(np.float32)
        if previous_result is not None:
            previous_result = _zoom(previous_result, (zy, zx), order=1).astype(np.float32)
        H, W = depth_map.shape
        px_per_mm_x = px_per_mm_y = target
    px_per_mm = px_per_mm_x

    residual = (np.clip(depth_map - prior_reference, 0.0, 1.0)
                if prior_reference is not None else depth_map)

    W_mm = (W - 1) / px_per_mm
    H_mm = (H - 1) / px_per_mm
    _ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
           "Top-Left":    0,       "Top-Right":    W_mm,
           "Center":      W_mm/2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm/2}.get(origin, 0)

    # ── Stage 1 · Build tool kernel (0–10 %) ─────────────────────────────
    progress(0, "Building tool geometry kernel…")
    kernel_raster, footprint_raster = _build_kernel(tool, px_per_mm, max_depth)
    kernel_plan, footprint_plan = _planning_kernel(
        tool, px_per_mm, max_depth, extra_offset_mm, kernel_raster, footprint_raster)
    ksize = kernel_raster.shape[0]
    progress(10, f"Kernel ready  ({ksize}×{ksize} px, profile: {tool.profile}).")

    # ── Stage 2 · Compute achievable depth surface (10–35 %) ─────────────
    cached = erosion_cache.get(erosion_cache_key) if erosion_cache is not None else None
    if cached is not None and cached[0].shape == depth_map.shape:
        center_depth_contour_raster, accessible_raster = cached
        progress(35, "Reusing achievable depth surface (same tool, unchanged)…")
    else:
        progress(10, "Computing achievable depth surface (morphological opening)…")
        structure = -kernel_plan.astype(np.float64)

        # Max depth the tool can reach when centred at each pixel (erosion).
        # Used internally as per-pixel depth targets during toolpath planning.
        center_depth_contour_raster = grey_erosion(depth_map.astype(np.float64),
                                  footprint=footprint_plan,
                                  structure=structure).astype(np.float32)
        center_depth_contour_raster = np.clip(center_depth_contour_raster, 0.0, 1.0)

        # Dilation of the erosion gives the true achievable depth at each material
        # point: the tool can carve (x,y) while centred at an offset position.
        # Erosion followed by dilation = morphological opening.
        accessible_raster = grey_dilation(center_depth_contour_raster.astype(np.float64),
                                   footprint=footprint_plan,
                                   structure=structure).astype(np.float32)
        accessible_raster = np.clip(accessible_raster, 0.0, 1.0)
        progress(35, "Achievable depth surface computed.")
        if erosion_cache is not None:
            erosion_cache[erosion_cache_key] = (center_depth_contour_raster, accessible_raster)

    if extra_offset_mm > 0.0:
        off_norm = extra_offset_mm / max_depth
        center_depth_contour_raster = np.clip(center_depth_contour_raster - off_norm, 0.0, 1.0)
        accessible_raster = np.clip(accessible_raster - off_norm, 0.0, 1.0)

    # ── Stage 3 · Plan passes (35–40 %) ───────────────────────────
    progress(35, "Planning depth passes…")
    pass_plan       = _plan_passes(max_depth, stepdown_mm, final_stepdown_mm)
    n_passes        = len(pass_plan)
    final_norm      = final_stepdown_mm / max_depth
    step_px = max(1, int(round(step_mm * px_per_mm)))   # for progress display only

    resid_tol_mm = max_depth / (n_grey_levels - 1)
    resid_tol    = 1.0 / (n_grey_levels - 1)   # normalised [0,1]

    IS_SPIRAL = raster_dir in ("Spiral (Outside-In)", "Spiral (Inside-Out)")
    direction = None if IS_SPIRAL else ("X" if raster_dir == "Horizontal" else "Y")

    # Forces every scan line/leg onto a single fixed travel direction (climb
    # keeps uncut material on the right of travel, conventional on the
    # left, for a clockwise spindle). "Either" leaves the default pixel-scan
    # order untouched and lets the ordering pass below reverse segments
    # freely for shorter travel. For Horizontal/Vertical this flips each
    # line's own point order (FORCE_REVERSE); for Spiral it instead selects
    # which rotational sense (CW/CCW) the whole spiral is generated in,
    # which achieves the same "uncut material on a fixed side" guarantee
    # for every leg without needing a per-leg reversal (see
    # SPIRAL_OUTSIDE_IN_SENSE and _spiral_legs() in Phase 1 below;
    # Inside-Out is a whole-path reversal on top of that).
    FORCE_REVERSE = {"Climb":        {"X": False, "Y": True},
                      "Conventional": {"X": True,  "Y": False}}
    SPIRAL_OUTSIDE_IN_SENSE = {"Climb": "CW", "Conventional": "CCW", "Either": "CW"}
    lock_direction = cut_direction in FORCE_REVERSE

    progress(40, f"{n_passes} depth pass{'es' if n_passes != 1 else ''}  "
                 f"({stepdown_mm:.2f} mm each"
                 + (f", final {final_stepdown_mm:.2f} mm" if final_stepdown_mm > 0 else "")
                 + f"), step-over {step_px} px = {step_mm:.2f} mm, "
                 + f"residual tol {resid_tol_mm*1000:.1f} µm.")

    # ── Stage 4 · 4-phase raster (40–80 %) ────────────────────────

    if work_surface_tol > 0.0:
        from scipy.ndimage import binary_dilation
        if subtract_unaccessible and prior_reference is not None:
            _gate = np.clip(accessible_raster - prior_reference, 0.0, 1.0)
        else:
            _gate = residual
        prior_mask = binary_dilation(
                         _gate >= work_surface_tol,
                         structure=footprint_raster,
                     ).astype(np.float32)
        progress(40, f"Threshold {work_surface_tol*max_depth*1000:.0f} µm: "
                     f"{int(np.sum(prior_mask))}/{H*W} pixels in mask.")
    else:
        prior_mask = None

    all_segments     = []
    tool_pos         = np.array([0.0, 0.0])
    total_segs       = 0
    n_links_inserted = 0
    n_links          = 0
    link_threshold_mm = 2.0 * step_mm

    # ── Phase 1: full-depth reference path (no terrain links) ────
    progress(40, "Phase 1: generating full-depth reference path…")
    full_depth_segs = []

    if IS_SPIRAL:
        row_ids, adj_step_y_mm = _grid_line_ids(H - 1, px_per_mm, step_mm, raster_offset_pct)
        col_ids, adj_step_x_mm = _grid_line_ids(W - 1, px_per_mm, step_mm, raster_offset_pct)
        sense = SPIRAL_OUTSIDE_IN_SENSE[cut_direction]
        legs = _spiral_legs(row_ids, col_ids, sense)
        if raster_dir == "Spiral (Inside-Out)":
            # Inside-Out is the point-reversal of Outside-In generated in
            # the sense that yields the same Climb/Conventional label:
            # reversing a path both reverses leg order and flips the
            # perceived rotational sense, so this reuses
            # SPIRAL_OUTSIDE_IN_SENSE instead of needing a second table.
            legs = [(axis, fixed, e, s) for axis, fixed, s, e in reversed(legs)]

        for axis, fixed_idx, start_idx, end_idx in legs:
            lo, hi = (start_idx, end_idx) if start_idx <= end_idx else (end_idx, start_idx)
            full_depth_segs.extend(
                _line_points(axis, fixed_idx, lo, hi, start_idx > end_idx,
                             center_depth_contour_raster, prior_mask, resid_tol,
                             px_per_mm, H, _ox, _oy, max_depth))
    else:
        spacing_boundary_px = H - 1 if direction == "X" else W - 1   # axis lines are spaced along
        line_span_px        = W - 1 if direction == "X" else H - 1   # each line's own full length
        all_line_ids, adj_step_mm = _grid_line_ids(spacing_boundary_px, px_per_mm, step_mm, raster_offset_pct)
        for line_idx in all_line_ids:
            reverse = lock_direction and FORCE_REVERSE[cut_direction][direction]
            full_depth_segs.extend(
                _line_points(direction, line_idx, 0, line_span_px, reverse,
                             center_depth_contour_raster, prior_mask, resid_tol,
                             px_per_mm, H, _ox, _oy, max_depth))

    progress(55, f"Phase 1 complete — {len(full_depth_segs)} full-depth span(s).")

    # ── Phase 2: depth-slice into per-pass segments ───────────────
    pass_list = make_multipass(full_depth_segs, max_depth, stepdown_mm,
                               final_stepdown_mm)

    # ── Phases 3–4: per-pass ordering + terrain link insertion ────
    for pi, (ceil_mm, is_final, pass_segs) in enumerate(pass_list):
        if not pass_segs:
            continue

        pct_base  = 55 + 25 * pi / n_passes
        ceil_norm = ceil_mm / max_depth
        progress(pct_base,
                 f"Pass {pi+1}/{n_passes} at z={ceil_mm:.3f} mm "
                 f"({'final' if is_final else 'roughing'}): "
                 f"{len(pass_segs)} segment(s).")

        # Phase 3: per-pass ordering (locked direction vs. free)
        cur = (float(tool_pos[0]), float(tool_pos[1]))
        if lock_direction and IS_SPIRAL:
            # Phase 1 already emits one continuous, correctly-sensed
            # perimeter walk (see SPIRAL_OUTSIDE_IN_SENSE above);
            # reordering would break that. _order_locked_raster_pass's
            # single-coordinate grouping assumes one axis-aligned line
            # family and doesn't generalize to a spiral's alternating
            # row/column legs, so pass make_multipass's output straight
            # through unreordered. Phase 4 below still bridges any gaps a
            # masked/gated refining pass introduces, same as it already
            # does for Horizontal/Vertical.
            ordered_segs = pass_segs
        elif lock_direction:
            ordered_segs = _order_locked_raster_pass(pass_segs, direction, cur)
        elif IS_SPIRAL and _is_contiguous_chain(pass_segs):
            # This pass is still one genuinely unbroken coil (the common
            # case: full-depth single pass, or a multi-pass/masked pass
            # that happened not to fragment) — skip reordering. This is
            # never a loss (an already-contiguous chain has zero
            # inter-segment travel to optimise) and avoids a real bug:
            # _order_free_raster_pass's greedy walk starts from the
            # tool's world origin (0,0), which can coincide exactly with
            # a NON-initial vertex of the spiral depending on the G-code
            # origin setting (e.g. Bottom-Left origin puts (0,0) on the
            # ring-0/ring-1 boundary corner). That exact-zero-distance tie
            # makes greedy pick the wrong leg first and chain backwards
            # through more zero-distance ties, closing each ring into a
            # rectangle linked to the next by a diagonal jump instead of
            # a spiral. When make_multipass or mask-based gating DOES
            # fragment a pass into disconnected pieces, this check is
            # False and we fall through to genuine nearest-neighbour
            # reordering below, same as any other pattern — there is no
            # single correct order left to preserve in that case, so
            # travel-optimizing among the fragments is the right thing
            # to do.
            ordered_segs = pass_segs
        else:
            ordered_segs = _order_free_raster_pass(pass_segs, cur, use_2opt,
                                                    link_threshold_mm)

        # Phase 4: terrain link insertion (respects final_stepdown_mm)
        for i, seg in enumerate(ordered_segs):
            all_segments.append(seg)
            total_segs += 1
            tool_pos = np.array([seg[-1][0], seg[-1][1]])

            if i + 1 < len(ordered_segs):
                nxt = ordered_segs[i + 1]
                x_entry, y_entry = nxt[0][0], nxt[0][1]
                nxt_dist = np.hypot(x_entry - tool_pos[0],
                                    y_entry - tool_pos[1])

                # Consecutive Spiral legs share their corner pixel exactly
                # (nxt_dist == 0.0); skip link insertion when segments are
                # already contiguous to avoid a spurious zero-length link
                # between every leg.
                if 1e-6 < nxt_dist < link_threshold_mm:
                    x_exit = float(tool_pos[0])
                    y_exit = float(tool_pos[1])
                    c0_px = int(round((x_exit  + _ox) * px_per_mm))
                    r0_px = int(round((H - 1) - (y_exit  + _oy) * px_per_mm))
                    c1_px = int(round((x_entry + _ox) * px_per_mm))
                    r1_px = int(round((H - 1) - (y_entry + _oy) * px_per_mm))

                    fn = 0.0 if is_final else final_norm
                    link_pixels = _geodesic_link(
                        r0_px, c0_px, r1_px, c1_px, ceil_norm,
                        center_depth_contour_raster, H, W,
                        px_per_mm, px_per_mm, max_depth, fn)
                    # Anchor both ends to the exact adjacent segment tuples so
                    # that the adjacency check in to_gcode / _render_3d sees
                    # bitwise-identical values and suppresses false retracts.
                    link_points = [seg[-1]]
                    for row_l, col_l in link_pixels[1:-1]:
                        terrain = float(center_depth_contour_raster[row_l, col_l])
                        if is_final:
                            z_link = min(terrain, ceil_norm)
                        else:
                            z_link = min(max(terrain - final_norm, 0.0), ceil_norm)
                        x_lmm = col_l / px_per_mm - _ox
                        y_lmm = (H - 1 - row_l) / px_per_mm - _oy
                        link_points.append((x_lmm, y_lmm, -z_link * max_depth))
                    link_points.append(nxt[0])

                    if len(link_points) >= 2:
                        all_segments.append(link_points)
                        n_links_inserted += 1

                n_links += 1

    progress(80, f"Raster complete — {total_segs} cut segments, "
                 f"{n_links_inserted} terrain links across all passes.")

    # ── Stage 6 · Simplify, simulate result, create ToolPath (88–100 %) ─────
    progress(80, "Simplifying tool path…")
    rdp_tol = max(
        1.0 / px_per_mm,
        max_depth / (n_grey_levels - 1),
    )
    simplified = [simplify_line(pts, rdp_tol) for pts in all_segments]

    progress(88, "Simulating tool-swept removal surface…")
    material_removed_raster = simulate_toolpath_result(
        ToolPath(segments=simplified),
        (H, W), px_per_mm, px_per_mm, _ox, _oy,
        max_depth, kernel_raster, footprint_raster,
    )
    raster_result = np.minimum(
        np.maximum(material_removed_raster,
                   previous_result if previous_result is not None else 0.0),
        depth_map).astype(np.float32)

    n_skip   = int(np.sum(material_removed_raster >= center_depth_contour_raster - resid_tol))
    pct_done = 100.0 * n_skip / max(1, H * W)
    label    = "post-refining" if previous_result is not None else "post-roughing"
    progress(92, f"{label.capitalize()} residual computed — "
                 f"{pct_done:.1f}% of pixels at target depth "
                 f"(within {resid_tol_mm*1000:.1f} µm).")

    progress(95, "Creating ToolPath object...")
    toolpath = ToolPath(
        segments=simplified,
        result=raster_result,
        accessible=accessible_raster,
        metadata={
            'profile': tool.profile,
            'radius_mm': tool.diameter_mm / 2,
            'step_mm': step_mm,
            'stepdown_mm': stepdown_mm,
            'final_stepdown_mm': final_stepdown_mm,
            'extra_offset_mm': extra_offset_mm,
            'max_depth': max_depth,
            'width_mm': width_mm,
            'height_mm': height_mm,
            'n_passes': n_passes,
            'rdp_tolerance': rdp_tol,
            'link_threshold_mm': link_threshold_mm,
            'n_links_inserted': n_links_inserted,
            'n_links_total': n_links,
        }
    )

    if previous_result is not None and work_surface_tol is not None:
        toolpath.metadata['refining_threshold_um'] = \
            work_surface_tol * max_depth * 1000

    progress(100, "Tool path computation complete!")

    return toolpath


def compute_vcarve_mask(ref, threshold_norm, min_feature_width_mm, px_per_mm,
                        max_feature_width_mm=0.0):
    """Return (filtered_ref, mask) for V-carve mask computation.

    Grey opening removes features narrower than min_feature_width_mm while
    leaving larger features at their original extent.  EDT-based filtering then
    removes pixels where the inscribed-circle radius exceeds max_feature_width_mm/2
    (the centre of features too wide for the tool to reach at the configured depth).

    filtered_ref is zeroed wherever mask is False so that downstream Voronoi
    contour extraction sees both the outer boundary and the inner hole boundary
    of wide features, not just the original unclipped disc shape.
    """
    filtered = ref
    if min_feature_width_mm > 0.0:
        from skimage.morphology import disk as _disk
        from scipy.ndimage import grey_opening as _grey_opening
        radius = max(1, int(np.ceil((min_feature_width_mm / 2.0) * px_per_mm)))
        filtered = _grey_opening(
            ref.astype(np.float64), footprint=_disk(radius)
        ).astype(np.float32)
    mask = filtered >= threshold_norm
    if max_feature_width_mm > 0.0:
        from scipy.ndimage import distance_transform_edt as _edt
        edt = _edt(mask)
        mask &= edt <= (max_feature_width_mm / 2.0) * px_per_mm
    filtered = np.where(mask, filtered, np.float32(0.0))
    return filtered, mask


def compute_vcarve_toolpath(depth_map, n_grey_levels,
                            tool, max_depth,
                            width_mm, height_mm,
                            origin="Bottom-Left",
                            feed_rate=800.0,
                            threshold_mm=0.0,
                            min_feature_width_mm=0.0,
                            max_feature_width_mm=0.0,
                            stepdown_mm=0.0, final_stepdown_mm=0.0,
                            skeleton_mode="medial_axis",
                            prior_reference=None, previous_result=None,
                            subtract_unaccessible=False,
                            use_2opt=False,
                            extra_offset_mm=0.0,
                            progress=None,
                            erosion_cache=None,
                            erosion_cache_key=None):
    """Compute a V-carving toolpath.

    Parameters
    ----------
    depth_map             : 2-D float32 [0, 1] — 1.0 = max cut depth
    n_grey_levels         : number of grey levels (for RDP tolerance)
    tool                  : Tool — encapsulates profile, diameter_mm, vbit_angle_deg
    max_depth             : total max cut depth in mm (physical scale)
    width_mm              : physical image width in mm
    height_mm             : physical image height in mm
    origin                : G-code origin convention ("Bottom-Left", "Top-Left", …)
    feed_rate             : cutting feed rate in mm/min
    threshold_mm          : minimum fine-detail depth in mm; regions shallower
                            than this are skipped
    min_feature_width_mm  : features narrower than this (mm) are skipped
    stepdown_mm           : max depth increment per z-pass in mm; 0 = single pass
    final_stepdown_mm     : depth increment for the finishing pass
                            (0 = same as stepdown_mm)
    extra_offset_mm       : stock-to-leave allowance (mm) applied laterally and
                            vertically, so a subsequent step (typically with
                            extra_offset_mm=0) can refine the result. Laterally,
                            the tool is treated as 2*extra_offset_mm larger when
                            computing the achievable-depth surface. Vertically,
                            that surface is lowered by extra_offset_mm. 0 = none.
    skeleton_mode         : "medial_axis" | "voronoi" | "inscribed"
    prior_reference       : optional 2-D float32 [0,1] reference surface from a
                            prior step, subtracted from depth_map to obtain the
                            region the V-carve operates on.  Pass
                            prior_step.accessible for geometry-based gating
                            (regions the prior tool couldn't reach), or
                            prior_step.result for residual-based gating
                            (material not yet removed).  None treats the full
                            depth_map as the region to V-carve.
    erosion_cache         : optional dict used to memoise the grey-erosion/grey-dilation
                            achievable-depth surface (keyed by erosion_cache_key), so a
                            subsequent call with the same tool and physical scale can
                            skip recomputing it.  None disables caching.
    erosion_cache_key     : hashable key identifying the (tool, px_per_mm, max_depth)
                            combination for erosion_cache lookups.
    previous_result       : optional 2-D float32 [0,1] cumulative result from a
                            prior step (i.e. ToolPath.result).  Combined with the
                            V-carve's own removal surface to produce the cumulative
                            result of this step.  Does not gate which regions are
                            visited (that is prior_reference's job).  None treats
                            the prior removal as zero.
    progress              : callable(pct, message) — thread-safe progress reporter

    Returns
    -------
    ToolPath
    """
    from scipy.ndimage import grey_erosion, grey_dilation, distance_transform_edt, label

    _alg_name = ('V-Carving (Sub-pixel)'    if skeleton_mode == 'voronoi'
                 else 'V-Carving (Feature Width)' if skeleton_mode == 'inscribed'
                 else 'V-Carving (Depth Map)')

    # ── Stage 0 · Resample to square pixels if needed (pre-pass) ──────────
    H, W = depth_map.shape
    px_per_mm_x = W / width_mm
    px_per_mm_y = H / height_mm
    if px_per_mm_x != px_per_mm_y:
        from scipy.ndimage import zoom as _zoom
        target = max(px_per_mm_x, px_per_mm_y)
        zy = target / px_per_mm_y
        zx = target / px_per_mm_x
        progress(0, f"Resampling depth map to square pixels "
                    f"({px_per_mm_x:.3f} × {px_per_mm_y:.3f} → {target:.3f} px/mm)…")
        depth_map = _zoom(depth_map, (zy, zx), order=1).astype(np.float32)
        if prior_reference is not None:
            prior_reference = _zoom(prior_reference, (zy, zx), order=1).astype(np.float32)
        if previous_result is not None:
            previous_result = _zoom(previous_result, (zy, zx), order=1).astype(np.float32)
        H, W = depth_map.shape
        px_per_mm_x = px_per_mm_y = target
    px_per_mm = px_per_mm_x

    W_mm = (W - 1) / px_per_mm
    H_mm = (H - 1) / px_per_mm
    _ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
           "Top-Left":    0,       "Top-Right":    W_mm,
           "Center":      W_mm/2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm/2}.get(origin, 0)

    # ── Stage 2a · Build tool kernel (0–10 %) ────────────────────
    progress(0, "Building tool geometry kernel…")
    if tool.profile == "Ball" and tool.diameter_mm <= 0.0:
        raise ValueError("Ball profile requires a positive tool diameter.")
    kernel_vcarve, footprint_vcarve = _build_kernel(tool, px_per_mm, max_depth)
    kernel_plan, footprint_plan = _planning_kernel(
        tool, px_per_mm, max_depth, extra_offset_mm, kernel_vcarve, footprint_vcarve)
    ksize = kernel_vcarve.shape[0]
    progress(10, f"Kernel ready  ({ksize}×{ksize} px, profile: {tool.profile}).")

    # ── Stage 2b · Achievable depth surface (10–35 %) ────────────
    cached = erosion_cache.get(erosion_cache_key) if erosion_cache is not None else None
    if cached is not None and cached[0].shape == depth_map.shape:
        center_depth_contour_vcarve, accessible_vcarve = cached
        progress(35, "Reusing achievable depth surface (same tool, unchanged)…")
    else:
        progress(10, "Computing achievable depth surface (morphological opening)…")
        center_depth_contour_vcarve = grey_erosion(
            depth_map.astype(np.float64),
            footprint=footprint_plan,
            structure=-kernel_plan.astype(np.float64),
        ).astype(np.float32)
        center_depth_contour_vcarve = np.clip(center_depth_contour_vcarve, 0.0, 1.0)

        accessible_vcarve = grey_dilation(
            center_depth_contour_vcarve.astype(np.float64),
            footprint=footprint_plan,
            structure=-kernel_plan.astype(np.float64),
        ).astype(np.float32)
        accessible_vcarve = np.clip(accessible_vcarve, 0.0, 1.0)
        progress(35, "Achievable depth surface computed.")
        if erosion_cache is not None:
            erosion_cache[erosion_cache_key] = (center_depth_contour_vcarve, accessible_vcarve)

    if extra_offset_mm > 0.0:
        off_norm = extra_offset_mm / max_depth
        center_depth_contour_vcarve = np.clip(center_depth_contour_vcarve - off_norm, 0.0, 1.0)
        accessible_vcarve = np.clip(accessible_vcarve - off_norm, 0.0, 1.0)

    # ── Stage 1 · V-carving mask (35–45 %) ────────────────
    progress(35, "Computing V-carving mask…")
    if prior_reference is None:
        uncut_depth = depth_map.copy()
    elif subtract_unaccessible:
        uncut_depth = np.clip(accessible_vcarve - prior_reference, 0.0, 1.0)
    else:
        uncut_depth = np.clip(depth_map - prior_reference, 0.0, 1.0)

    threshold_norm = threshold_mm / max_depth if max_depth else 0.0

    uncut_depth_filtered, mask_vcarve = compute_vcarve_mask(
        uncut_depth, threshold_norm, min_feature_width_mm, px_per_mm,
        max_feature_width_mm,
    )

    if not np.any(mask_vcarve) or np.all(mask_vcarve):
        progress(100, "No V-carving mask found — nothing to V-carve.")
        return ToolPath(
            segments=[],
            result=(previous_result.copy() if previous_result is not None
                    else np.zeros_like(depth_map, dtype=np.float32)),
            accessible=accessible_vcarve,
            metadata={'algorithm': _alg_name,
                      'vbit_angle_deg': tool.vbit_angle_deg,
                      'threshold_mm': threshold_mm,
                      'max_depth': max_depth}
        )

    progress(45, f"V-carving mask: {np.count_nonzero(mask_vcarve)} pixels.")

    # ── Stage 3 · Medial axis (45–60 %) ───────────────────────────
    if skeleton_mode in ('voronoi', 'inscribed'):
        progress(45, "Computing Voronoi skeleton…")
        from scipy.spatial import Voronoi as _Voronoi
        from skimage.measure import find_contours as _find_contours

        def _resample(pts, spacing):
            # Walk the polyline and emit a new point every `spacing` units of
            # arc length. `carry` tracks the fractional overshoot past the end
            # of each segment so spacing remains uniform across segment boundaries.
            out = [pts[0]]
            carry = 0.0
            for i in range(1, len(pts)):
                seg = pts[i] - pts[i - 1]
                seg_len = float(np.hypot(seg[0], seg[1]))
                if seg_len < 1e-10:
                    continue
                pos = carry
                while pos < seg_len:
                    out.append(pts[i - 1] + (pos / seg_len) * seg)
                    pos += spacing
                carry = pos - seg_len
            return np.array(out)

        # eps = sqrt(2)/2 removes all single-pixel staircase steps (worst case: 45° line)
        # while preserving corners whose deviation from a straight chord is >> 1 px.
        rdp_eps = np.sqrt(2.0) / 2.0
        raw_contours = _find_contours(uncut_depth_filtered, threshold_norm * (1.0 - 1e-6))
        pts_list = []
        for c in raw_contours:
            if len(c) < 4:
                continue
            simplified = simplify_line(c, rdp_eps)
            if len(simplified) < 2:
                continue
            resampled = _resample(simplified, spacing=1.0)
            if len(resampled) >= 2:
                pts_list.append(resampled)

        boundary_pts = np.vstack(pts_list) if pts_list else np.empty((0, 2))

        if len(boundary_pts) < 4:
            skeleton   = np.zeros_like(mask_vcarve, dtype=bool)
            radius_map = np.zeros((H, W), dtype=np.float32)
        else:
            vor = _Voronoi(boundary_pts)

            # Opening-angle criterion (Attali & Montanvert 1994):
            # Each Voronoi ridge separates the regions of two boundary
            # generators p_i, p_j.  At vertex v the inscribed-circle
            # radius r satisfies |v-p_i| = |v-p_j| = r, so the
            # opening angle θ obeys cos(θ) = 1 − d²/(2r²) where
            # d = |p_i-p_j|.
            #
            # Genuine medial-axis ridge: generators are from opposite
            # sides of the shape → d ≈ 2r → cos(θ) ≈ −1 → θ ≈ π.
            # Spurious same-side rib: d ≈ 1 pixel, r >> 1 → θ ≈ 0.
            # Threshold π/3 (cos = 0.5) cleanly separates the two
            # regardless of shape orientation or staircase boundaries.
            cos_thr = 0.5           # keep ridges with θ ≥ 60° at both ends
            verts   = vor.vertices  # float (row, col)

            rv = np.array(vor.ridge_vertices)          # (R, 2)
            rp = np.array(vor.ridge_points)            # (R, 2)
            finite = (rv[:, 0] >= 0) & (rv[:, 1] >= 0)
            rv = rv[finite];  rp = rp[finite]

            pi_arr = boundary_pts[rp[:, 0]]            # (R, 2)
            pj_arr = boundary_pts[rp[:, 1]]            # (R, 2)
            d2_arr = np.sum((pi_arr - pj_arr) ** 2, axis=1)   # (R,)

            va = verts[rv[:, 0]]                       # (R, 2)
            vb = verts[rv[:, 1]]                       # (R, 2)
            r0sq = np.sum((va - pi_arr) ** 2, axis=1)  # (R,)
            r1sq = np.sum((vb - pi_arr) ** 2, axis=1)  # (R,)
            cos0 = np.where(r0sq > 1e-12, 1.0 - d2_arr / (2.0 * r0sq), -1.0)
            cos1 = np.where(r1sq > 1e-12, 1.0 - d2_arr / (2.0 * r1sq), -1.0)

            keep = (cos0 <= cos_thr) & (cos1 <= cos_thr)
            va_keep = va[keep]
            vb_keep = vb[keep]

            r0_keep = np.sqrt(r0sq[keep])   # inscribed-circle radii at va_keep (px)
            r1_keep = np.sqrt(r1sq[keep])   # inscribed-circle radii at vb_keep (px)

            skeleton   = np.zeros((H, W), dtype=bool)
            radius_map = np.zeros((H, W), dtype=np.float32)
            for va_i, vb_i, r0_r, r1_r in zip(va_keep, vb_keep, r0_keep, r1_keep):
                r0f, c0f = va_i;  r1f, c1f = vb_i
                n = max(3, int(np.ceil(np.hypot(r1f-r0f, c1f-c0f) * 2)) + 1)
                rs = np.clip(
                    np.round(np.linspace(r0f, r1f, n)).astype(int), 0, H-1)
                cs = np.clip(
                    np.round(np.linspace(c0f, c1f, n)).astype(int), 0, W-1)
                skeleton[rs, cs] = True
                radius_map[rs, cs] = np.linspace(r0_r, r1_r, n)

            skeleton &= mask_vcarve & (center_depth_contour_vcarve > 0)
            from skimage.morphology import thin as _thin
            skeleton = _thin(skeleton)
    else:
        from skimage.morphology import medial_axis
        progress(45, "Computing medial axis…")
        skeleton, _ = medial_axis(mask_vcarve, return_distance=True)
        skeleton = skeleton & (center_depth_contour_vcarve > 0)

    # ── Inscribed-circle depth map (for 'inscribed' mode only) ───────────────
    if skeleton_mode == 'inscribed':
        from math import radians, tan
        if tool.profile == 'V-Bit':
            _tan_half = tan(radians(tool.vbit_angle_deg / 2.0))
            inscribed_depth_norm = (radius_map / px_per_mm) / (_tan_half * max_depth)
        elif tool.profile == 'Ball':
            _R_mm = tool.diameter_mm / 2.0
            _r_mm = radius_map / px_per_mm
            inscribed_depth_norm = (
                _R_mm - np.sqrt(np.maximum(0.0, _R_mm**2 - _r_mm**2))
            ) / max_depth
        else:  # End Mill — inscribed circle does not constrain depth
            progress(48, "Warning: 'inscribed' mode is not meaningful for End Mill; "
                         "falling back to grey-erosion depth.")
            inscribed_depth_norm = center_depth_contour_vcarve
        inscribed_depth_norm = np.clip(inscribed_depth_norm, 0.0, 1.0)

    _depth_norm = (inscribed_depth_norm if skeleton_mode == 'inscribed'
                   else center_depth_contour_vcarve)

    if not np.any(skeleton):
        progress(100, "Skeleton empty — no V-carve moves generated.")
        return ToolPath(
            segments=[],
            result=(previous_result.astype(np.float32) if previous_result is not None
                    else np.zeros_like(depth_map, dtype=np.float32)),
            metadata={'algorithm': _alg_name,
                      'vbit_angle_deg': tool.vbit_angle_deg,
                      'threshold_mm': threshold_mm,
                      'max_depth': max_depth}
        )

    n_skel_px = int(np.count_nonzero(skeleton))
    progress(60, f"Skeleton: {n_skel_px} pixels, decomposing into branches…")

    # ── Stage 4 · Skeleton branch decomposition + TSP (60–80 %) ───
    from scipy.ndimage import convolve as ndconvolve

    deg_kernel = np.ones((3, 3), dtype=np.int32)
    deg_kernel[1, 1] = 0
    degree_map = ndconvolve(skeleton.astype(np.int32), deg_kernel,
                            mode='constant', cval=0)
    degree_map[~skeleton] = 0

    def _nb8(r, c):
        """Return 8-connected skeleton neighbours of (r, c)."""
        nb = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and skeleton[nr, nc]:
                    nb.append((nr, nc))
        return nb

    skel_rows, skel_cols = np.where(skeleton)
    branch_starts = [(int(r), int(c))
                     for r, c in zip(skel_rows.tolist(), skel_cols.tolist())
                     if degree_map[r, c] != 2]

    skel_labeled, n_comp = label(skeleton, structure=np.ones((3, 3), dtype=bool))
    comp_ids_with_start  = {skel_labeled[r, c] for r, c in branch_starts}
    for comp_id in range(1, n_comp + 1):
        if comp_id not in comp_ids_with_start:
            rows_c, cols_c = np.where(skel_labeled == comp_id)
            branch_starts.append((int(rows_c[0]), int(cols_c[0])))

    branches      = []
    visited_edges = set()

    for start in branch_starts:
        for nb in _nb8(*start):
            edge = frozenset({start, nb})
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            branch = [start, nb]
            prev, cur = start, nb
            while degree_map[cur] == 2:
                nexts = [p for p in _nb8(*cur) if p != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                e2  = frozenset({cur, nxt})
                if e2 in visited_edges:
                    break
                visited_edges.add(e2)
                branch.append(nxt)
                prev, cur = cur, nxt
            branches.append(branch)

    # Degree-0 pixels (no 8-connected neighbours) are never reached by the
    # branch-start traversal above, so they must be collected separately.
    # Skip any that touch the mask boundary — those are Voronoi ridge clipping
    # artifacts produced when the skeleton is masked to the carve region.
    for r, c in zip(skel_rows.tolist(), skel_cols.tolist()):
        if degree_map[r, c] == 0:
            if not np.all(mask_vcarve[max(0, r-1):r+2, max(0, c-1):c+2]):
                continue  # boundary-clipping artifact
            branches.append([(int(r), int(c))])

    # Collapse junction pixel clusters
    junc_mask = degree_map >= 3
    if junc_mask.any():
        junc_labeled, n_junc = label(junc_mask,
                                     structure=np.ones((3, 3), dtype=bool))
        junc_remap = {}
        for jid in range(1, n_junc + 1):
            rows_j, cols_j = np.where(junc_labeled == jid)
            if len(rows_j) == 1:
                continue
            cr, cc = rows_j.mean(), cols_j.mean()
            dists   = (rows_j - cr) ** 2 + (cols_j - cc) ** 2
            best    = int(np.argmin(dists))
            rep     = (int(rows_j[best]), int(cols_j[best]))
            for r, c in zip(rows_j.tolist(), cols_j.tolist()):
                if (r, c) != rep:
                    junc_remap[(r, c)] = rep

        if junc_remap:
            remapped = []
            for br in branches:
                if len(br) == 1:
                    remapped.append([junc_remap.get(br[0], br[0])])
                else:
                    remapped.append(
                        [junc_remap.get(br[0], br[0])]
                        + br[1:-1]
                        + [junc_remap.get(br[-1], br[-1])]
                    )
            branches = [br for br in remapped if not (len(br) == 2 and br[0] == br[-1])]

    n_branches = len(branches)
    progress(75, f"Path ordering: {n_branches} branch(es).")

    # Greedy nearest-endpoint TSP ordering of branches.
    vcarve_link_mm  = 2.0 / px_per_mm
    ordered_branches = []
    tool_pos         = np.array([0.0, 0.0])
    remaining        = list(range(n_branches))

    while remaining:
        best_idx  = None
        best_dist = float('inf')
        best_rev  = False
        for i in remaining:
            path = branches[i]
            sr, sc = path[0]
            er, ec = path[-1]
            x_s = sc / px_per_mm - _ox
            y_s = (H - 1 - sr) / px_per_mm - _oy
            x_e = ec / px_per_mm - _ox
            y_e = (H - 1 - er) / px_per_mm - _oy
            d_s = float(np.hypot(x_s - tool_pos[0], y_s - tool_pos[1]))
            d_e = float(np.hypot(x_e - tool_pos[0], y_e - tool_pos[1]))
            if d_s <= d_e:
                d, rev = d_s, False
            else:
                d, rev = d_e, True
            if d < best_dist:
                best_dist, best_idx, best_rev = d, i, rev

        remaining.remove(best_idx)
        path = branches[best_idx]
        if best_rev:
            path = list(reversed(path))
        ordered_branches.append(path)

        er, ec = path[-1]
        tool_pos[0] = ec / px_per_mm - _ox
        tool_pos[1] = (H - 1 - er) / px_per_mm - _oy

    progress(80, f"Ordered {len(ordered_branches)} branches.")

    # ── Stage 5 · Multi-pass segment building (80–90 %) ──────────

    def _build_pass_segments(branches_ordered, depth_fn):
        """Build cut segments from ordered branches using depth_fn(r,c).

        depth_fn returns the Z cut depth in mm (negative) for a pixel, or
        None to skip it. Contiguous included pixels form sub-segments;
        adjacent sub-segments within vcarve_link_mm are chained together.
        """
        segs = []
        for branch in branches_ordered:
            sub = []
            for r, c in branch:
                z = depth_fn(r, c)
                if z is None:
                    if sub:
                        segs.append(sub)
                        sub = []
                    continue
                x = c / px_per_mm - _ox
                y = (H - 1 - r) / px_per_mm - _oy
                sub.append((x, y, z))
            if sub:
                if segs:
                    dx = sub[0][0] - segs[-1][-1][0]
                    dy = sub[0][1] - segs[-1][-1][1]
                    if (dx * dx + dy * dy) ** 0.5 < vcarve_link_mm:
                        segs[-1].extend(sub)
                        continue
                segs.append(sub)
        return segs

    def _two_opt_pass(segs, origin):
        """2-opt reorder of already-built segments for one pass.

        Uses actual segment endpoints (not branch endpoints) so that
        multi-pass skipping of skeleton pixels doesn't mislead the optimiser.
        """
        if not use_2opt or len(segs) < 3:
            return segs
        ent_xy = [(s[0][0],  s[0][1])  for s in segs]
        ext_xy = [(s[-1][0], s[-1][1]) for s in segs]
        order, flips = _two_opt_reorder(ent_xy, ext_xy, origin=origin,
                                        link_threshold=vcarve_link_mm,
                                        per_rapid_overhead=10.0)
        return [
            list(reversed(segs[i])) if flips[k] else segs[i]
            for k, i in enumerate(order)
        ]

    def _pass_origin(segs_so_far):
        if segs_so_far:
            last = segs_so_far[-1][-1]
            return (last[0], last[1])
        return (0.0, 0.0)

    all_segments = []

    # Phase 1: generate full-depth skeleton path
    def _depth_single(r, c):
        return -_depth_norm[r, c] * max_depth
    full_depth_segs = _build_pass_segments(ordered_branches, _depth_single)
    full_depth_segs = _two_opt_pass(full_depth_segs, _pass_origin(all_segments))

    # Phases 2–3: depth-slice + per-pass 2-opt reorder
    for _, _, pass_segs in make_multipass(
            full_depth_segs, max_depth, stepdown_mm, final_stepdown_mm):
        pass_segs = _two_opt_pass(pass_segs, _pass_origin(all_segments))
        all_segments.extend(pass_segs)

    progress(90, f"Generated {len(all_segments)} segments across all passes.")

    # ── Stage 6 · Simplify, simulate result, Simplify and assemble ToolPath (90–100 %) ───────
    progress(90, "Simplifying tool path (RDP)…")
    rdp_tol = max(
        1.0 / px_per_mm,
        max_depth / (n_grey_levels - 1),
    )
    simplified = [simplify_line(pts, rdp_tol) for pts in all_segments]

    progress(93, "Simulating tool-swept removal surface…")
    vcarve_result = simulate_toolpath_result(
        ToolPath(segments=simplified),
        (H, W), px_per_mm, px_per_mm, _ox, _oy,
        max_depth, kernel_vcarve, footprint_vcarve,
    )

    progress(97, "Creating ToolPath object…")
    toolpath = ToolPath(
        segments=simplified,
        result=np.maximum(
            previous_result if previous_result is not None else 0.0,
            vcarve_result).astype(np.float32),
        accessible=accessible_vcarve,
        metadata={
            'algorithm':       _alg_name,
            'skeleton_mode':   skeleton_mode,
            'profile':         tool.profile,
            'vbit_angle_deg':  tool.vbit_angle_deg,
            'min_feature_width_mm':  min_feature_width_mm,
            'threshold_mm':          threshold_mm,
            'stepdown_mm':           stepdown_mm,
            'final_stepdown_mm':     final_stepdown_mm,
            'extra_offset_mm':       extra_offset_mm,
            'max_depth':       max_depth,
            'width_mm':        width_mm,
            'height_mm':       height_mm,
            'n_branches':      n_branches,
            'rdp_tolerance':   rdp_tol,
        }
    )
    progress(100, "V-carve tool path computation complete!")
    return toolpath


def _prune_contours(all_z_contours, z_levels, lateral_tol_px, H, W,
                          px_per_mm=1.0, feed_rate=800.0, feed_rapid=3000.0,
                          z_safe=5.0, max_depth=1.0, max_step_down_mm=None):
    """Remove contour points at shallower z-levels that lie within lateral_tol_px
    of the nearest deeper (effective) contour — i.e. they are laterally
    redundant because the deeper pass will remove the same material.

    Processes deepest-to-shallowest.  The "effective next" reference for z_i is
    the nearest deeper level that was not itself fully pruned.  Because we
    always compare z_i *directly* against effective_next (never transitively),
    the lateral error is bounded to lateral_tol_px regardless of how many levels
    are skipped between z_i and effective_next.

    If max_step_down_mm is given, a point is pruned only if it is *both*
    laterally within lateral_tol_px of its nearest committed pixel *and* that
    same nearest pixel's recorded depth is within max_step_down_mm of the
    current level's depth. Otherwise it is kept even though laterally
    redundant, since relying on that deeper cut would require the tool to
    plunge more than one stepdown in a single move. A depth gap exactly equal
    to max_step_down_mm is treated as a legal single-pass stepdown and remains
    prunable. Note this only prevents pruning from *creating* an oversized
    gap — in adaptive-levels mode the programmed levels themselves can already
    be spaced further apart than max_step_down_mm in sparse regions.

    A pruned run is restored if its arc length is below the retract+plunge
    breakeven: feed_rate × 2 × (z_safe + z_level_mm) / feed_rapid.  The z_safe
    term accounts for travel from the surface to safe height; z_level_mm adds
    the depth of the current cut, since both contribute to retract travel.
    This threshold grows with depth, making deep-level splits harder to justify.

    Returns a dict with the same keys as all_z_contours but with redundant
    segments replaced by shorter open polylines (or removed entirely).
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.draw import line as draw_line

    def _rasterize(contours, depth_img=None, depth_mm=0.0):
        img = np.zeros((H, W), dtype=bool)
        for c in contours:
            rows = np.round(c[:, 0]).astype(int).clip(0, H - 1)
            cols = np.round(c[:, 1]).astype(int).clip(0, W - 1)
            img[rows, cols] = True
            for k in range(len(rows) - 1):
                rr, cc = draw_line(int(rows[k]), int(cols[k]),
                                   int(rows[k + 1]), int(cols[k + 1]))
                img[np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)] = True
        if depth_img is not None:
            depth_img[img] = depth_mm
        return img

    def _split_runs(c, keep_mask, min_pts=3, min_prune_arc_px=0.0):
        n = len(c)
        is_closed = np.linalg.norm(c[0] - c[-1]) < 1.5

        # Pass 1: restore pruned runs whose pixel-space arc length is below the
        # retract+plunge breakeven threshold.  Splitting there would cost more
        # in Z-travel overhead than the redundant cutting saves.
        mask = keep_mask.copy()
        if min_prune_arc_px > 0.0 and mask.any() and not mask.all():
            if is_closed:
                start = next((i for i in range(n)
                              if mask[i] and not mask[(i - 1) % n]), 0)
                seq = [(start + i) % n for i in range(n)]
            else:
                seq = list(range(n))
            i = 0
            while i < len(seq):
                if not mask[seq[i]]:
                    run_idx = []
                    while i < len(seq) and not mask[seq[i]]:
                        run_idx.append(seq[i])
                        i += 1
                    pts = c[run_idx]
                    arc = (float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                           if len(pts) > 1 else 0.0)
                    if arc < min_prune_arc_px:
                        for ri in run_idx:
                            mask[ri] = True
                else:
                    i += 1

        if not mask.any():
            return []
        if mask.all():
            return [c]

        # Pass 2: split on the (possibly restored) mask.
        if is_closed:
            start = next((i for i in range(n)
                          if mask[i] and not mask[(i - 1) % n]), 0)
            indices = [(start + i) % n for i in range(n)]
        else:
            indices = list(range(n))
        segments = []
        run = []
        for idx in indices:
            if mask[idx]:
                run.append(c[idx])
            else:
                if len(run) >= min_pts:
                    segments.append(np.array(run))
                run = []
        if len(run) >= min_pts:
            segments.append(np.array(run))
        return segments

    pruned = {}
    effective_next_img = None
    effective_next_depth = None

    for z_norm in reversed(z_levels):
        contours = all_z_contours.get(z_norm, [])
        z_level_mm = z_norm * max_depth
        if effective_next_img is None:
            # Deepest level — always kept
            pruned[z_norm] = contours
            if contours:
                effective_next_depth = np.full((H, W), np.inf, dtype=np.float32)
                effective_next_img = _rasterize(contours, effective_next_depth, z_level_mm)
            continue

        if not contours:
            pruned[z_norm] = []
            continue

        # Retract+plunge breakeven arc length for this depth level (in pixels).
        min_prune_arc_px = (feed_rate * 2.0 * (z_safe + z_level_mm)
                            / feed_rapid * px_per_mm)

        dt, nearest_idx = distance_transform_edt(~effective_next_img, return_indices=True)

        kept = []
        for c in contours:
            rows = np.round(c[:, 0]).astype(int).clip(0, H - 1)
            cols = np.round(c[:, 1]).astype(int).clip(0, W - 1)
            keep_mask = dt[rows, cols] >= lateral_tol_px
            if max_step_down_mm is not None:
                near_depth_mm = effective_next_depth[nearest_idx[0][rows, cols],
                                                      nearest_idx[1][rows, cols]]
                depth_gap_mm = near_depth_mm - z_level_mm
                keep_mask = keep_mask | (depth_gap_mm > max_step_down_mm + 1e-6)
            kept.extend(_split_runs(c, keep_mask,
                                    min_prune_arc_px=min_prune_arc_px))

        pruned[z_norm] = kept
        if kept:
            # Union with existing image so deeper coverage is never lost.
            # A shallower contour point is redundant if it falls within tol of
            # *any* committed cut at this level or deeper, not just the nearest
            # partially-pruned level's remaining segments.
            effective_next_img = effective_next_img | _rasterize(kept, effective_next_depth, z_level_mm)
        # else: fully pruned — effective_next_img unchanged, propagates upward

    return pruned


def compute_contour_toolpath(
    depth_map, n_grey_levels,
    tool, z_increment_mm, max_depth,
    width_mm, height_mm,
    origin="Bottom-Left",
    feed_rate=800.0,
    prior_accessible=None,
    previous_result=None,
    progress=None,
    adaptive_levels=False,
    lateral_tol_mm=0.0,
    z_safe=5.0,
    feed_rapid=3000.0,
    use_2opt=False,
    cut_direction="Either",
    max_step_down_mm=None,
    extra_offset_mm=0.0,
    erosion_cache=None,
    erosion_cache_key=None,
):
    """Compute a waterline (contour) tool path from a normalised depth map.

    The achievable depth surface is sliced at evenly-spaced Z-levels separated
    by *z_increment_mm*.  At each level the iso-depth contour lines are extracted
    and emitted as cut segments, giving a layer-by-layer waterline strategy.
    To account for tool geometry, the tool-centre depth limit is first computed
    by grey-eroding the depth map with the tool profile kernel; contours are
    extracted from this erosion.  The stored accessible surface is the full
    morphological opening (erosion followed by dilation), representing the true
    achievable depth at each material point.

    Parameters
    ----------
    depth_map : 2-D float32 array, values in [0, 1]
        Normalised depth map; 1.0 = maximum cut depth.
    n_grey_levels : int
        Number of grey levels in the source image (used for RDP tolerance).
    tool : Tool
        Encapsulates profile, diameter_mm, and vbit_angle_deg.
    z_increment_mm : float
        Z increment between contour levels in mm.  Also controls the number of
        passes in adaptive mode: n_passes = ceil(max_depth / z_increment_mm).
    adaptive_levels : bool
        When True, place the n_passes contour levels at equal-area (CDF-based)
        depth intervals rather than uniform Z spacing.  This concentrates passes
        where the depth map has the most pixels, avoiding redundant cuts between
        discrete grey-level transitions.
    max_depth : float
        Total maximum cut depth in mm.
    width_mm, height_mm : float
        Physical dimensions of the work area in mm.
    origin : str
        G-code origin convention ("Bottom-Left", "Top-Left", etc.).
    feed_rate : float
        Cutting feed rate in mm/min.
    prior_accessible : 2-D float32 array, values in [0, 1], or None
        Reserved for future use (analogous to vcarve's *prior_accessible*).
        Currently ignored; the accessible surface is always computed from
        *depth_map*.
    previous_result : 2-D float32 array, values in [0, 1], or None
        Cumulative result from a prior step (i.e. ToolPath.result).  Combined
        with the contour's own removal surface to produce the cumulative result
        of this step.  None treats the prior removal as zero.
    progress : callable(pct, message)
        Thread-safe progress reporter.
    lateral_tol_mm : float
        Lateral tolerance in mm for redundancy pruning.  Contour segments whose
        every point lies within this distance of the next deeper (effective)
        contour are considered redundant and omitted.  0.0 disables pruning.
    max_step_down_mm : float or None
        Maximum Z increment, in mm, that redundancy pruning is allowed to
        create between one retained cut and the next, and the window (in
        z-levels) that cross-level 2-opt is allowed to mix contours across.
        Both are derived from this single value: the pruning bound is used
        directly (see _prune_contours), and the mixing window is
        round(max_step_down_mm / z_increment_mm) levels.  None or <= 0 means
        unconstrained — pruning may collapse any number of redundant levels
        and 2-opt may mix across the full level range.
    extra_offset_mm : float
        Stock-to-leave allowance (mm) applied laterally and vertically, so a
        subsequent step (typically with extra_offset_mm=0) can refine the
        result. Laterally, the tool is treated as 2*extra_offset_mm larger
        when computing the achievable-depth surface. Vertically, that
        surface is lowered by extra_offset_mm. 0 = none.
    z_safe : float
        Safe Z height above the workpiece surface in mm.  Used with
        feed_rapid to compute the retract+plunge breakeven arc length per
        level (splits shorter than this are not worth the overhead).
    feed_rapid : float
        Rapid traverse rate in mm/min used for retract+plunge cost estimate.
    cut_direction : "Either" | "Climb" | "Conventional"
        For a clockwise spindle, forces every contour to travel in the
        direction that keeps uncut material on the tool's right (Climb) or
        left (Conventional). "Either" preserves the default behaviour: no
        forced direction, and the ordering/2-opt passes below are free to
        reverse contours for shorter travel.
    erosion_cache : dict, optional
        Used to memoise the grey-erosion/grey-dilation achievable-depth surface
        (keyed by erosion_cache_key), so a subsequent call with the same tool
        and physical scale can skip recomputing it.  None disables caching.
    erosion_cache_key : hashable, optional
        Key identifying the (tool, px_per_mm, max_depth) combination for
        erosion_cache lookups.

    Returns
    -------
    ToolPath
    """
    import math
    from scipy.ndimage import grey_erosion, grey_dilation
    from skimage.measure import find_contours

    # None/<=0 both mean "unconstrained" — normalise once, used for both the
    # pruning depth bound and the cross-level 2-opt mixing window below.
    _msd = max_step_down_mm if (max_step_down_mm is not None
                                 and max_step_down_mm > 0) else None

    # ── Stage 0 · Resample to square pixels if needed (pre-pass) ──────────
    H, W = depth_map.shape
    px_per_mm_x = W / width_mm
    px_per_mm_y = H / height_mm
    if px_per_mm_x != px_per_mm_y:
        from scipy.ndimage import zoom as _zoom
        target = max(px_per_mm_x, px_per_mm_y)
        zy = target / px_per_mm_y
        zx = target / px_per_mm_x
        progress(0, f"Resampling depth map to square pixels "
                    f"({px_per_mm_x:.3f} × {px_per_mm_y:.3f} → {target:.3f} px/mm)…")
        depth_map = _zoom(depth_map, (zy, zx), order=1).astype(np.float32)
        if previous_result is not None:
            previous_result = _zoom(previous_result, (zy, zx), order=1).astype(np.float32)
        H, W = depth_map.shape
        px_per_mm_x = px_per_mm_y = target
    px_per_mm = px_per_mm_x

    W_mm = (W - 1) / px_per_mm
    H_mm = (H - 1) / px_per_mm
    _ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
           "Top-Left":    0,       "Top-Right":    W_mm,
           "Center":      W_mm / 2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm / 2}.get(origin, 0)

    # ── Stage 1 · Build tool kernel (0–10 %) ──────────────────────────────
    progress(0, "Building tool geometry kernel…")
    kernel_contour, footprint_contour = _build_kernel(tool, px_per_mm, max_depth)
    kernel_plan, footprint_plan = _planning_kernel(
        tool, px_per_mm, max_depth, extra_offset_mm, kernel_contour, footprint_contour)
    ksize = kernel_contour.shape[0]
    progress(10, f"Kernel ready  ({ksize}×{ksize} px, profile: {tool.profile}).")

    # ── Stage 2 · Compute achievable depth surface (10–35 %) ──────────────
    cached = erosion_cache.get(erosion_cache_key) if erosion_cache is not None else None
    if cached is not None and cached[0].shape == depth_map.shape:
        center_depth_contour, accessible_contour = cached
        progress(35, "Reusing achievable depth surface (same tool, unchanged)…")
    else:
        progress(10, "Computing achievable depth surface (morphological opening)…")
        structure = -kernel_plan.astype(np.float64)

        # Max depth the tool can reach when centred at each pixel (erosion).
        # Used for contour extraction and adaptive level planning.
        center_depth_contour = grey_erosion(depth_map.astype(np.float64),
                                  footprint=footprint_plan,
                                  structure=structure).astype(np.float32)
        center_depth_contour = np.clip(center_depth_contour, 0.0, 1.0)

        # Dilation of the erosion gives the true achievable depth at each material
        # point: the tool can carve (x,y) while centred at an offset position.
        # Erosion followed by dilation = morphological opening.
        accessible_contour = grey_dilation(center_depth_contour.astype(np.float64),
                                           footprint=footprint_plan,
                                           structure=structure).astype(np.float32)
        accessible_contour = np.clip(accessible_contour, 0.0, 1.0)
        progress(35, "Achievable depth surface computed.")
        if erosion_cache is not None:
            erosion_cache[erosion_cache_key] = (center_depth_contour, accessible_contour)

    if extra_offset_mm > 0.0:
        off_norm = extra_offset_mm / max_depth
        center_depth_contour = np.clip(center_depth_contour - off_norm, 0.0, 1.0)
        accessible_contour = np.clip(accessible_contour - off_norm, 0.0, 1.0)

    # ── Stage 3 · Plan Z-levels (35–40 %) ─────────────────────────────────
    progress(35, "Planning contour levels…")
    n_passes = max(1, math.ceil(max_depth / z_increment_mm))
    if adaptive_levels:
        flat = center_depth_contour[center_depth_contour > 1e-6].ravel()
        if flat.size >= 2:
            quantiles = np.linspace(1.0 / n_passes, 1.0, n_passes)
            z_raw = np.quantile(flat, quantiles)
            min_gap = 0.5 / max(n_grey_levels - 1, 1)
            z_levels, prev = [], -1.0
            for z in np.clip(z_raw, 0.0, 1.0):
                if float(z) - prev >= min_gap:
                    z_levels.append(float(z))
                    prev = float(z)
            if not z_levels or z_levels[-1] < 1.0 - min_gap:
                z_levels.append(1.0)
        else:
            z_levels = [min(i * z_increment_mm / max_depth, 1.0)
                        for i in range(1, n_passes + 1)]
        progress(40, f"{len(z_levels)} contour level{'s' if len(z_levels) != 1 else ''}  "
                     f"(adaptive, target Z increment {z_increment_mm:.2f} mm).")
    else:
        z_levels = [min(i * z_increment_mm / max_depth, 1.0)
                    for i in range(1, n_passes + 1)]
        progress(40, f"{n_passes} contour level{'s' if n_passes != 1 else ''}  "
                     f"({z_increment_mm:.2f} mm each).")

    # ── Stage 4 · Extract contours and build segments (40–80 %) ───────────
    rdp_tol = max(1.0 / px_per_mm,
                  max_depth / (n_grey_levels - 1))

    # Half a grey-level step in normalised units.  Subtracting this from the
    # find_contours threshold keeps flat-region pixels firmly "inside" even
    # when z_norm lands at or just above their exact value (e.g. from CDF
    # quantile rounding).  Without it the contour jumps to the inner wall of
    # a flat step and the tool cuts air above the neighbouring deeper region.
    _contour_eps = 0.5 / max(n_grey_levels - 1, 1)

    # ── Stage 4a · Extract raw contours for every Z-level ─────────────────
    progress(40, "Extracting contours for all Z-levels…")
    # find_contours keeps uncut material consistently on the right of the
    # direction of travel (see compute_raster_toolpath's cut_direction docs
    # for the clockwise-spindle convention), so a single global reversal
    # rule suffices — no per-contour inside/outside analysis needed.
    reverse_contours = (cut_direction == "Conventional")
    lock_direction    = cut_direction in ("Climb", "Conventional")
    all_z_contours = {}
    for z_norm in z_levels:
        raw = find_contours(center_depth_contour, max(z_norm - _contour_eps, 0.0))
        kept = [c for c in raw if len(c) >= 3]
        if reverse_contours:
            kept = [c[::-1] for c in kept]
        all_z_contours[z_norm] = kept

    # ── Stage 4b · Prune laterally-redundant segments ─────────────────────
    if lateral_tol_mm > 0.0:
        progress(50, f"Pruning redundant segments (lateral tol={lateral_tol_mm:.3f} mm)…")
        lateral_tol_px = lateral_tol_mm * px_per_mm
        all_z_contours = _prune_contours(
            all_z_contours, z_levels, lateral_tol_px, H, W,
            px_per_mm=px_per_mm,
            feed_rate=feed_rate,
            feed_rapid=feed_rapid,
            z_safe=z_safe,
            max_depth=max_depth,
            max_step_down_mm=_msd)

    # ── Stage 4c · Ordering — greedy per max_step_down window, optional 2-opt
    # A contour's climb/conventional correctness comes entirely from
    # find_contours' own per-contour orientation guarantee (see the
    # cut_direction docs above), independent of visiting order or Z-level —
    # unlike raster, where correctness depends on visiting adjacent lines in
    # sequence. So the greedy step may freely draw candidates from anywhere
    # in the max_step_down_mm window regardless of lock state; only 2-opt
    # (which reverses segments/blocks) needs to stay gated by
    # not lock_direction.
    all_segments      = []
    current_pos       = np.array([0.0, 0.0])
    link_threshold_mm = tool.diameter_mm
    z_jump_weight_mm  = z_increment_mm * 1.5
    win_size     = (max(1, round(_msd / z_increment_mm))
                     if _msd and z_increment_mm > 0 else len(z_levels))
    n_windows    = max(1, -(-len(z_levels) // win_size))   # ceiling division

    for win_idx, win_start in enumerate(range(0, len(z_levels), win_size)):
        win_levels = z_levels[win_start : win_start + win_size]
        remaining  = [(c, -z_norm * max_depth)
                      for z_norm in win_levels
                      for c in all_z_contours.get(z_norm, [])]
        if not remaining:
            continue
        progress(55 + 25 * win_idx // n_windows,
                 f"Ordering window {win_idx + 1}/{n_windows} "
                 f"({len(remaining)} contour(s))…")

        # Shallowest Z in this window: once a candidate requires a real rapid
        # anyway (d >= link_threshold_mm), prefer whichever is closest to
        # this depth, so forced long jumps still trend top-to-bottom instead
        # of landing at an arbitrary depth. Short hops stay pure-XY.
        win_z_top = -win_levels[0] * max_depth

        # Greedy nearest-neighbour ordering across every contour in the
        # window. For closed contours any rotation of the loop is valid, so
        # we sample candidate start points; for open contours only the two
        # endpoints are considered (flip=True covers the other end).
        # Segments are collected into win_segs so that optional 2-opt can
        # reorder them before they are committed to all_segments.
        win_cur  = current_pos.copy()
        win_segs = []
        while remaining:
            best_idx  = 0
            best_flip = False
            best_rot  = 0
            best_cost = float('inf')

            for i, (c, z_mm) in enumerate(remaining):
                n         = len(c)
                is_closed = np.linalg.norm(c[0] - c[-1]) < 1.5
                rot_cands = range(0, n, max(1, n // 20)) if is_closed else [0]
                flip_cands = (False,) if lock_direction else (False, True)
                for flip in flip_cands:
                    orient = c[::-1] if flip else c
                    for j in rot_cands:
                        row0, col0 = orient[j]
                        x0 = col0 / px_per_mm - _ox
                        y0 = (H - 1 - row0) / px_per_mm - _oy
                        dx, dy = x0 - win_cur[0], y0 - win_cur[1]
                        d = (dx * dx + dy * dy) ** 0.5
                        cost = d if d < link_threshold_mm else (
                            d + z_jump_weight_mm * (win_z_top - z_mm))
                        if cost < best_cost:
                            best_cost = cost
                            best_idx  = i
                            best_flip = flip
                            best_rot  = j

            c_chosen, z_mm = remaining.pop(best_idx)
            is_closed = np.linalg.norm(c_chosen[0] - c_chosen[-1]) < 1.5
            if best_flip:
                c_chosen = c_chosen[::-1]
            if best_rot > 0:
                c_chosen = np.roll(c_chosen, -best_rot, axis=0)
            if is_closed:
                c_chosen = np.vstack([c_chosen, c_chosen[0:1]])

            points = [(col / px_per_mm - _ox,
                       (H - 1 - row) / px_per_mm - _oy,
                       z_mm)
                      for row, col in c_chosen]
            if len(points) < 2:
                continue

            win_segs.append(points)
            last = points[-1]
            win_cur[0] = last[0]
            win_cur[1] = last[1]

        if not win_segs:
            continue

        # Optional 2-opt reorder across the whole window, guided by a
        # z-jump penalty to discourage rapids that straddle levels more
        # than necessary. Skipped when locked, since its block-reversal
        # move would undo the forced direction.
        if use_2opt and len(win_segs) >= 3 and not lock_direction:
            win_z  = [s[0][2] for s in win_segs]
            ent_xy = [(s[0][0],  s[0][1])  for s in win_segs]
            ext_xy = [(s[-1][0], s[-1][1]) for s in win_segs]
            order, flips = _two_opt_reorder(
                ent_xy, ext_xy,
                origin=(float(current_pos[0]), float(current_pos[1])),
                link_threshold=link_threshold_mm,
                per_rapid_overhead=2.0 * z_safe,
                seg_z=win_z,
                z_jump_weight=z_jump_weight_mm)
            win_segs = [
                list(reversed(win_segs[i])) if flips[k] else win_segs[i]
                for k, i in enumerate(order)
            ]

        all_segments.extend(win_segs)
        last = win_segs[-1][-1]
        current_pos[0] = last[0]
        current_pos[1] = last[1]

    progress(80, f"Ordering complete ({len(all_segments)} segments).")

    # ── Stage 4d · Terrain-link insertion ──────────────────────────────────
    # Walk all_segments once and splice geodesic links wherever consecutive
    # segments are within link_threshold_mm of each other. Done as a separate
    # stage so cross-window boundaries are also covered.
    cut_segs    = all_segments
    all_segments = []
    n_links      = 0
    for i, seg in enumerate(cut_segs):
        all_segments.append(seg)
        if i + 1 < len(cut_segs):
            nxt      = cut_segs[i + 1]
            nxt_dist = ((nxt[0][0] - seg[-1][0])**2 +
                        (nxt[0][1] - seg[-1][1])**2) ** 0.5
            if nxt_dist < link_threshold_mm:
                ceil_norm = max(-seg[-1][2], -nxt[0][2]) / max_depth
                x_exit,  y_exit  = float(seg[-1][0]),  float(seg[-1][1])
                x_entry, y_entry = float(nxt[0][0]),   float(nxt[0][1])
                c0_px = int(round((x_exit  + _ox) * px_per_mm))
                r0_px = int(round((H - 1)  - (y_exit  + _oy) * px_per_mm))
                c1_px = int(round((x_entry + _ox) * px_per_mm))
                r1_px = int(round((H - 1)  - (y_entry + _oy) * px_per_mm))
                link_pixels = _geodesic_link(
                    r0_px, c0_px, r1_px, c1_px, ceil_norm,
                    center_depth_contour, H, W,
                    px_per_mm, px_per_mm, max_depth)
                link_points = [seg[-1]]
                for row_l, col_l in link_pixels[1:-1]:
                    z_link = min(float(center_depth_contour[row_l, col_l]),
                                 ceil_norm)
                    x_lmm = col_l / px_per_mm - _ox
                    y_lmm = (H - 1 - row_l) / px_per_mm - _oy
                    link_points.append((x_lmm, y_lmm, -z_link * max_depth))
                link_points.append(nxt[0])
                if len(link_points) >= 2:
                    all_segments.append(link_points)
                    n_links += 1

    progress(80, f"Assembly complete — {len(cut_segs)} cut segments, "
                 f"{n_links} terrain links.")

    # RDP-simplify every segment — cut contours and terrain links alike —
    # now that ordering is finalised. Simplification never moves a segment's
    # first/last point (see simplify_line), so deferring it to here rather
    # than per-contour during ordering has no effect on entry-point choice,
    # 2-opt, or terrain-link endpoints; it just needs to happen somewhere
    # after every segment (including links, built above) exists.
    all_segments = [simplify_line(pts, rdp_tol) for pts in all_segments]

    # ── Stage 5 · Simulate result (80–90 %) ───────────────────────────────
    progress(80, "Simulating tool-swept removal surface…")
    material_removed = simulate_toolpath_result(
        ToolPath(segments=all_segments),
        (H, W), px_per_mm, px_per_mm, _ox, _oy,
        max_depth, kernel_contour, footprint_contour,
    )
    result = np.maximum(
        previous_result if previous_result is not None else 0.0,
        material_removed).astype(np.float32)
    progress(90, "Result computed.")

    # ── Stage 6 · Assemble ToolPath (90–100 %) ────────────────────────────
    progress(90, "Assembling tool path…")
    tp = ToolPath(
        segments=all_segments,
        result=result,
        accessible=accessible_contour,
        metadata={
            'algorithm':       'Contour',
            'profile':         tool.profile,
            'max_depth':       max_depth,
            'radius_mm':       tool.diameter_mm / 2,
            'z_increment_mm':  z_increment_mm,
            'adaptive_levels': adaptive_levels,
            'use_2opt':           use_2opt,
            'max_step_down_mm':   _msd,
            'extra_offset_mm':    extra_offset_mm,
            'lateral_tol_mm':     lateral_tol_mm,
            'n_passes':        len(z_levels),
            'width_mm':        width_mm,
            'height_mm':       height_mm,
            'rdp_tolerance':   rdp_tol,
        }
    )
    progress(100, "Contour tool path computation complete!")
    return tp


def compute_cutout_toolpath(
    depth_map, n_grey_levels,
    tool, cutout_depth_mm, max_depth,
    width_mm, height_mm,
    origin="Bottom-Left",
    feed_rate=800.0,
    threshold_mm=0.0,
    stepdown_mm=0.0,
    final_stepdown_mm=0.0,
    extra_offset_mm=0.0,
    cut_direction="Either",
    prior_accessible=None,
    previous_result=None,
    previous_alpha=None,
    use_2opt=False,
    z_safe=5.0,
    feed_rapid=3000.0,
    tab_settings=None,
    progress=None,
):
    """Compute an outer-silhouette cutout tool path.

    Thin wrapper over _compute_cutout_rings (Phase A: silhouette mask +
    contour extraction -- independent of cutout depth/stepdown/tab settings)
    and _compute_cutout_from_rings (Phase B: depth-slicing, tabs, ordering,
    simulation). Split so a tab-settings-only change can re-run just Phase B
    -- see app.py's per-step rings cache.

    Traces the outer perimeter of every connected region of *depth_map* at
    or above *threshold_mm*, offsets it outward by the tool radius (plus
    *extra_offset_mm*), and cuts a uniform-depth closed loop through the
    full stock thickness (*cutout_depth_mm*), stepping down in Z across
    however many passes *stepdown_mm*/*final_stepdown_mm* require.

    Unlike compute_raster_toolpath/compute_vcarve_toolpath/
    compute_contour_toolpath this is a flat, straight-wall cut -- tool.profile
    is expected to be "End Mill" (enforced by the UI, not re-validated here)
    -- with no depth-shaping kernel: only the tool's lateral radius matters,
    so there is no grey_erosion/grey_dilation achievable-depth surface stage
    and no erosion_cache.

    Interior holes in the silhouette (e.g. a letter's counter) are always
    filled before contour tracing -- v1 cuts the outer perimeter of each
    shape only; cutting interior holes is not supported yet.

    Known limitation: if the silhouette touches the edge of the canvas,
    find_contours may return an open polyline running along the border
    rather than a closed ring (marching squares does not synthesize a
    boundary at the array edge). Keep a margin between artwork and the
    canvas edge to avoid this.

    Parameters
    ----------
    depth_map : 2-D float32 array, values in [0, 1]
        Normalised depth map; 1.0 = canvas max_depth.
    n_grey_levels : int
        Unused -- present only for call-signature parity with the other
        compute_*_toolpath functions (their RDP tolerance uses it; a flat
        boundary loop has no grey-level concept).
    tool : Tool
        Expected profile "End Mill".
    cutout_depth_mm : float
        Total absolute cut depth in mm, independent of and typically larger
        than *max_depth* (must cut through the full stock thickness). Not
        used to renormalise depth_map or any kernel; flows only into
        _plan_passes/make_multipass (which take plain mm).
    max_depth : float
        Canvas's shared max cut depth in mm. Used only to (a) interpret
        threshold_mm as a normalised fraction of depth_map and (b)
        normalise the simulated result/accessible arrays into the canvas's
        shared [0, 1] space.
    width_mm, height_mm : float
        Physical dimensions of the work area in mm.
    origin : str
        G-code origin convention ("Bottom-Left", "Top-Left", etc.).
    feed_rate : float
        Cutting feed rate in mm/min.
    threshold_mm : float
        Depth (mm, as a fraction of max_depth) used to split "inside the
        object silhouette" from "background"; threshold_norm = threshold_mm
        / max_depth. Which side of the threshold is "the object" depends on
        how the source was authored (a carved object against a shallow
        background vs. an uncarved object standing proud of a deeply
        recessed background) and isn't knowable from depth_map alone, so
        both directions are tried -- see Stage 2 below and the
        'mask_direction' metadata key ('<=' or '>=') for which one won.
    stepdown_mm, final_stepdown_mm : float
        As in the sibling algorithms -- passed straight to
        make_multipass(segments, cutout_depth_mm, stepdown_mm,
        final_stepdown_mm).
    extra_offset_mm : float
        Lateral-only outward push of the tool-centre path beyond the
        minimum tool-radius clearance; 0 = tool tangent to the (filled)
        silhouette edge. Implemented as a single binary_dilation of the
        filled silhouette mask by a disk of radius
        (tool.diameter_mm/2 + extra_offset_mm), in pixels.
    cut_direction : "Either" | "Climb" | "Conventional"
        Same find_contours-orientation convention as
        compute_contour_toolpath (reverse_contours =
        cut_direction == "Conventional").
    prior_accessible, previous_result : 2-D float32 array or None
        Cumulative accessible/result surfaces from the previous step
        (ToolPath.accessible/.result). Unconditionally unioned into this
        step's own accessible/result.
    previous_alpha : 2-D float32 array or None
        Cumulative hole-visibility surface from the previous step
        (ToolPath.alpha). 1.0 = material present, 0.0 = a real hole. None
        means no prior step has flagged any holes.
    use_2opt : bool
        Optional 2-opt travel-order refinement across the loops within
        each depth pass.
    z_safe, feed_rapid : float
        Used only to seed per-rapid overhead in the optional 2-opt cost
        model.
    tab_settings : dressup.TabSettings or None
        When enabled, leaves evenly-spaced uncut bridges ("tabs") in every
        closed ring, applied to the full-depth ring geometry before
        make_multipass() depth-slices it -- see dressup.apply_tabs.
    progress : callable(pct, message)
        Thread-safe progress reporter.

    Returns
    -------
    ToolPath
    """
    rings_cache = _compute_cutout_rings(
        depth_map, tool, max_depth, width_mm, height_mm,
        origin=origin, threshold_mm=threshold_mm, extra_offset_mm=extra_offset_mm,
        cut_direction=cut_direction,
        prior_accessible=prior_accessible, previous_result=previous_result,
        previous_alpha=previous_alpha, progress=progress,
    )
    return _compute_cutout_from_rings(
        rings_cache, tool, cutout_depth_mm, max_depth, width_mm, height_mm,
        threshold_mm=threshold_mm, stepdown_mm=stepdown_mm,
        final_stepdown_mm=final_stepdown_mm, extra_offset_mm=extra_offset_mm,
        cut_direction=cut_direction, use_2opt=use_2opt, z_safe=z_safe,
        feed_rapid=feed_rapid, tab_settings=tab_settings, progress=progress,
    )


@dataclasses.dataclass
class _CutoutRings:
    """Phase-A output of compute_cutout_toolpath: everything Phase B
    (_compute_cutout_from_rings) needs to depth-slice and assemble the final
    ToolPath. Independent of cutout depth/stepdown/tab settings, so it can
    be cached and reused across tab-only settings changes -- see app.py's
    per-step rings cache."""
    rings: list                # raw pixel-space contours; [] if none found
    message: str                # status, or the reason `rings` is empty
    px_per_mm: float
    H: int
    W: int
    ox: float
    oy: float
    kernel_cutout: np.ndarray
    footprint_cutout: np.ndarray
    mask_direction: str
    depth_map: np.ndarray               # possibly resampled (Stage 0)
    previous_result: np.ndarray
    prior_accessible: np.ndarray
    previous_alpha: np.ndarray


def _compute_cutout_rings(
    depth_map, tool, max_depth, width_mm, height_mm,
    origin="Bottom-Left",
    threshold_mm=0.0,
    extra_offset_mm=0.0,
    cut_direction="Either",
    prior_accessible=None,
    previous_result=None,
    previous_alpha=None,
    progress=None,
):
    """Phase A of compute_cutout_toolpath: silhouette mask + outer contour
    extraction. See compute_cutout_toolpath for full parameter docs.
    """
    from scipy.ndimage import binary_fill_holes, binary_dilation
    from skimage.morphology import disk
    from skimage.measure import find_contours

    # ── Stage 0 · Resample to square pixels if needed (pre-pass) ──────────
    H, W = depth_map.shape
    px_per_mm_x = W / width_mm
    px_per_mm_y = H / height_mm
    if px_per_mm_x != px_per_mm_y:
        from scipy.ndimage import zoom as _zoom
        target = max(px_per_mm_x, px_per_mm_y)
        zy = target / px_per_mm_y
        zx = target / px_per_mm_x
        progress(0, f"Resampling depth map to square pixels "
                    f"({px_per_mm_x:.3f} × {px_per_mm_y:.3f} → {target:.3f} px/mm)…")
        depth_map = _zoom(depth_map, (zy, zx), order=1).astype(np.float32)
        if previous_result is not None:
            previous_result = _zoom(previous_result, (zy, zx), order=1).astype(np.float32)
        if prior_accessible is not None:
            prior_accessible = _zoom(prior_accessible, (zy, zx), order=1).astype(np.float32)
        if previous_alpha is not None:
            previous_alpha = _zoom(previous_alpha, (zy, zx), order=1).astype(np.float32)
        H, W = depth_map.shape
        px_per_mm_x = px_per_mm_y = target
    px_per_mm = px_per_mm_x

    W_mm = (W - 1) / px_per_mm
    H_mm = (H - 1) / px_per_mm
    _ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
           "Top-Left":    0,       "Top-Right":    W_mm,
           "Center":      W_mm / 2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm / 2}.get(origin, 0)

    # ── Stage 1 · Build tool kernel (0–10 %) ──────────────────────────────
    # Only used for the final material-removal simulation (Stage 5) -- a
    # straight-wall cutout has no depth-shaping kernel of its own, so
    # _planning_kernel/erosion_cache (used by the other algorithms to plan a
    # depth-shaped achievable surface) don't apply here.
    progress(0, "Building tool geometry kernel…")
    kernel_cutout, footprint_cutout = _build_kernel(tool, px_per_mm, max_depth)
    ksize = kernel_cutout.shape[0]
    progress(10, f"Kernel ready  ({ksize}×{ksize} px, profile: {tool.profile}).")

    mask_direction = None

    def _empty(message):
        progress(45, message)
        return _CutoutRings(
            rings=[], message=message, px_per_mm=px_per_mm, H=H, W=W,
            ox=_ox, oy=_oy, kernel_cutout=kernel_cutout, footprint_cutout=footprint_cutout,
            mask_direction=mask_direction, depth_map=depth_map,
            previous_result=previous_result, prior_accessible=prior_accessible,
            previous_alpha=previous_alpha,
        )

    # ── Stage 2 · Silhouette mask: threshold → fill holes → dilate (10–35 %)
    # Whether "the object" is the deeply-cut region (dark-background source,
    # background left at/near the surface) or the shallow/uncarved region
    # (background carved away deeply so the object stands proud in relief)
    # depends on how the source was authored -- depth_map alone doesn't say
    # which. Try object = "at or below threshold" first (an object standing
    # proud of a deeply-recessed background); if that yields nothing usable
    # (empty, or -- once holes are filled -- the entire canvas, e.g. because
    # it actually picked out the *background* and binary_fill_holes closed
    # over the object as an enclosed "hole" in it), retry with object = "at
    # or above threshold" instead (a carved object against a shallow/uncut
    # background). This only disambiguates correctly when the object doesn't
    # itself touch the canvas edge -- same margin caveat as the open-contour
    # limitation noted above.
    progress(10, "Building silhouette mask…")
    threshold_norm = threshold_mm / max_depth if max_depth else 0.0

    r_px = max(1, int(round((tool.diameter_mm / 2.0 + extra_offset_mm) * px_per_mm)))

    def _try_mask(cmp):
        raw = (depth_map < threshold_norm) if cmp == '<=' else (depth_map >= threshold_norm)
        if not np.any(raw):
            return None
        # v1: interior holes are always filled before contour tracing --
        # only the outer perimeter of each connected shape is ever cut.
        filled = binary_fill_holes(raw)
        dilated = binary_dilation(filled, structure=disk(r_px))
        if np.all(dilated):
            return None
        return dilated

    mask_cutout = _try_mask('<=')
    mask_direction = '<='
    if mask_cutout is None:
        mask_cutout = _try_mask('>=')
        mask_direction = '>='
    if mask_cutout is None:
        return _empty(
            "No usable silhouette boundary found (tried both depth-threshold "
            "directions) — nothing to cut.")
    progress(35, f"Silhouette mask ready ({np.count_nonzero(mask_cutout)} px, "
                 f"object = depth {mask_direction} threshold).")

    # ── Stage 3 · Contour extraction (35–45 %) ────────────────────────────
    progress(35, "Extracting outer silhouette contour…")
    # Same global winding-direction convention as compute_contour_toolpath --
    # find_contours' marching-squares orientation guarantee already keeps
    # uncut material consistently on one side, so no per-contour
    # inside/outside analysis is needed.
    reverse_contours = (cut_direction == "Conventional")
    rings = [c for c in find_contours(mask_cutout.astype(np.float64), 0.5) if len(c) >= 3]
    if reverse_contours:
        rings = [c[::-1] for c in rings]
    if not rings:
        return _empty("No closed silhouette boundary found — nothing to cut.")
    progress(45, f"{len(rings)} outer boundary loop{'s' if len(rings) != 1 else ''} found.")

    return _CutoutRings(
        rings=rings, message="", px_per_mm=px_per_mm, H=H, W=W, ox=_ox, oy=_oy,
        kernel_cutout=kernel_cutout, footprint_cutout=footprint_cutout,
        mask_direction=mask_direction, depth_map=depth_map,
        previous_result=previous_result, prior_accessible=prior_accessible,
        previous_alpha=previous_alpha,
    )


def _compute_cutout_from_rings(
    rings_cache, tool, cutout_depth_mm, max_depth, width_mm, height_mm,
    threshold_mm=0.0,
    stepdown_mm=0.0,
    final_stepdown_mm=0.0,
    extra_offset_mm=0.0,
    cut_direction="Either",
    use_2opt=False,
    z_safe=5.0,
    feed_rapid=3000.0,
    tab_settings=None,
    progress=None,
):
    """Phase B of compute_cutout_toolpath: apply tabs, depth-slice into
    passes, order/simplify them, and simulate the result. This is the part
    that changes when only stepdown/tab settings change, so it's the part
    worth recomputing on its own -- see compute_cutout_toolpath for full
    parameter docs.
    """
    px_per_mm = rings_cache.px_per_mm
    H, W = rings_cache.H, rings_cache.W
    _ox, _oy = rings_cache.ox, rings_cache.oy
    depth_map = rings_cache.depth_map
    previous_result = rings_cache.previous_result
    prior_accessible = rings_cache.prior_accessible
    previous_alpha = rings_cache.previous_alpha

    def _empty_result(message):
        progress(100, message)
        return ToolPath(
            segments=[],
            result=(previous_result.copy() if previous_result is not None
                    else np.zeros_like(depth_map, dtype=np.float32)),
            accessible=(prior_accessible.copy() if prior_accessible is not None
                        else np.zeros_like(depth_map, dtype=np.float32)),
            alpha=(previous_alpha.copy() if previous_alpha is not None else None),
            metadata={'algorithm': 'Cutout', 'profile': tool.profile,
                      'max_depth': max_depth, 'cutout_depth_mm': cutout_depth_mm,
                      'threshold_mm': threshold_mm}
        )

    if not rings_cache.rings:
        return _empty_result(rings_cache.message or "Nothing to cut.")

    lock_direction = cut_direction in ("Climb", "Conventional")

    # ── Stage 4 · Tabs + multipass + ordering + simplify (45–75 %) ────────
    # Every ring gets the exact same full-depth Z -- there is only one
    # cutout_depth_mm for the whole tool path, so (unlike compute_contour_
    # toolpath's per-level Z jumps) no z-jump weighting is ever needed
    # below: every point of every ring in a given pass shares one Z.
    full_depth_segs = [
        [(col / px_per_mm - _ox, (H - 1 - row) / px_per_mm - _oy, -cutout_depth_mm)
         for row, col in ring]
        for ring in rings_cache.rings
    ]

    tab_regions = []
    if tab_settings is not None and tab_settings.enabled:
        full_depth_segs, tab_regions, tab_warnings = apply_tabs(
            full_depth_segs, tab_settings, closed_tol_mm=1.5 / px_per_mm)
        for warning in tab_warnings:
            progress(45, warning)

    pass_list = make_multipass(full_depth_segs, cutout_depth_mm, stepdown_mm, final_stepdown_mm)
    n_passes = len(pass_list)

    all_segments = []
    current_pos  = np.array([0.0, 0.0])
    link_threshold_mm = tool.diameter_mm
    closed_tol_mm = 1.5 / px_per_mm

    for pi, (ceil_mm, is_final, pass_segs) in enumerate(pass_list):
        if not pass_segs:
            continue
        progress(45 + 30 * pi // max(n_passes, 1),
                 f"Ordering pass {pi + 1}/{n_passes} ({len(pass_segs)} loop(s))…")

        remaining = [np.asarray(seg, dtype=np.float64) for seg in pass_segs]
        pass_ordered = []
        while remaining:
            best_idx  = 0
            best_flip = False
            best_rot  = 0
            best_cost = float('inf')
            for i, c in enumerate(remaining):
                n = len(c)
                is_closed = np.hypot(c[0, 0] - c[-1, 0], c[0, 1] - c[-1, 1]) < closed_tol_mm
                rot_cands = range(0, n, max(1, n // 20)) if is_closed else [0]
                flip_cands = (False,) if lock_direction else (False, True)
                for flip in flip_cands:
                    orient = c[::-1] if flip else c
                    for j in rot_cands:
                        dx = orient[j, 0] - current_pos[0]
                        dy = orient[j, 1] - current_pos[1]
                        d = (dx * dx + dy * dy) ** 0.5
                        if d < best_cost:
                            best_cost = d
                            best_idx, best_flip, best_rot = i, flip, j

            c_chosen = remaining.pop(best_idx)
            is_closed = np.hypot(c_chosen[0, 0] - c_chosen[-1, 0],
                                 c_chosen[0, 1] - c_chosen[-1, 1]) < closed_tol_mm
            if best_flip:
                c_chosen = c_chosen[::-1]
            if best_rot > 0:
                c_chosen = np.roll(c_chosen, -best_rot, axis=0)
            if is_closed:
                c_chosen = np.vstack([c_chosen, c_chosen[0:1]])

            points = [tuple(p) for p in c_chosen]
            if len(points) < 2:
                continue
            pass_ordered.append(points)
            current_pos[0], current_pos[1] = points[-1][0], points[-1][1]

        if not pass_ordered:
            continue

        # Optional 2-opt reorder within the pass. Skipped when locked, since
        # its block-reversal move would undo the forced climb/conventional
        # direction. No z_jump term -- every loop in this pass shares one Z.
        if use_2opt and len(pass_ordered) >= 3 and not lock_direction:
            ent_xy = [(s[0][0],  s[0][1])  for s in pass_ordered]
            ext_xy = [(s[-1][0], s[-1][1]) for s in pass_ordered]
            order, flips = _two_opt_reorder(
                ent_xy, ext_xy,
                origin=(float(all_segments[-1][-1][0]) if all_segments else 0.0,
                        float(all_segments[-1][-1][1]) if all_segments else 0.0),
                link_threshold=link_threshold_mm,
                per_rapid_overhead=2.0 * z_safe)
            pass_ordered = [
                list(reversed(pass_ordered[i])) if flips[k] else pass_ordered[i]
                for k, i in enumerate(order)
            ]

        all_segments.extend(pass_ordered)
        last = pass_ordered[-1][-1]
        current_pos[0], current_pos[1] = last[0], last[1]

    progress(75, f"Ordering complete ({len(all_segments)} segments across {n_passes} pass(es)).")

    # No terrain-link stage -- unlike compute_contour_toolpath's
    # _geodesic_link (which bridges gaps along the surface being cut),
    # Cutout has no terrain (flat stock); gaps between loops/passes become
    # plain retract/rapid/plunge via ToolPath.iter_moves's own gap
    # detection. This is an intentional simplification.
    rdp_tol = 1.0 / px_per_mm
    all_segments = [simplify_line(pts, rdp_tol) for pts in all_segments]

    # ── Stage 5 · Simulate result/accessible/alpha (75–90 %) ──────────────
    progress(75, "Simulating tool-swept removal surface…")
    material_removed_cutout = simulate_toolpath_result(
        ToolPath(segments=all_segments),
        (H, W), px_per_mm, px_per_mm, _ox, _oy,
        max_depth, rings_cache.kernel_cutout, rings_cache.footprint_cutout,
    )
    result = np.maximum(
        previous_result if previous_result is not None else 0.0,
        material_removed_cutout).astype(np.float32)
    accessible = np.maximum(
        prior_accessible if prior_accessible is not None else 0.0,
        material_removed_cutout).astype(np.float32)

    # alpha: the new hole-visibility channel. Never participates in any
    # depth/residual arithmetic -- it exists solely so the result preview
    # can tell "reached full relief depth" apart from "this is a real hole".
    # Only actually a hole when cutout_depth_mm exceeds max_depth -- if the
    # user set a shallower cut depth (<= max_depth), this pass is just an
    # ordinary partial-depth cut like any other algorithm's, and result's
    # own grey level (< 1.0) already represents it correctly; no pixel here
    # has been clipped to 1.0 by simulate_toolpath_result, so there is
    # nothing to flag.
    #
    # kerf_mask requires the achieved depth to have actually SATURATED the
    # max_depth-normalised scale (>= 1.0), not merely to be non-zero: with
    # tabs, a touched pixel can fall well short of full depth (result's own
    # grey level already reflects that correctly there), so "touched at
    # all" is no longer equivalent to "reached full depth" the way it was
    # before tabs existed (every kerf pixel used to share the exact same
    # full-depth Z). A tab whose depth still exceeds max_depth but stays
    # short of cutout_depth_mm remains indistinguishable from a real hole
    # here -- an existing limitation of this shared max_depth scale, not
    # something tabs introduce.
    if cutout_depth_mm > max_depth:
        kerf_mask = material_removed_cutout >= 1.0 - 1e-3
        alpha = np.minimum(
            previous_alpha if previous_alpha is not None else 1.0,
            np.where(kerf_mask, 0.0, 1.0)).astype(np.float32)
    else:
        alpha = previous_alpha.copy() if previous_alpha is not None else None
    progress(90, "Result computed.")

    # ── Stage 6 · Assemble ToolPath (90–100 %) ────────────────────────────
    progress(90, "Assembling tool path…")
    metadata = {
        'algorithm':         'Cutout',
        'profile':           tool.profile,
        'radius_mm':         tool.diameter_mm / 2,
        'max_depth':         max_depth,
        'cutout_depth_mm':   cutout_depth_mm,
        'stepdown_mm':       stepdown_mm,
        'final_stepdown_mm': final_stepdown_mm,
        'extra_offset_mm':   extra_offset_mm,
        'threshold_mm':      threshold_mm,
        'mask_direction':    rings_cache.mask_direction,
        'cut_direction':     cut_direction,
        'use_2opt':          use_2opt,
        'n_passes':          n_passes,
        'width_mm':          width_mm,
        'height_mm':         height_mm,
        'rdp_tolerance':     rdp_tol,
    }
    if tab_regions:
        metadata['tab_regions'] = tab_regions
    tp = ToolPath(
        segments=all_segments,
        result=result,
        accessible=accessible,
        alpha=alpha,
        metadata=metadata,
    )
    progress(100, "Cutout tool path computation complete!")
    return tp
