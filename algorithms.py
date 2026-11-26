import dataclasses
import numpy as np
from tool import Tool
from toolpath import ToolPath
from dressup import apply_tabs
import tool_profile


def _two_opt_reorder(seg_entries, seg_exits, origin=(0.0, 0.0), max_passes=5,
                     link_threshold=0.0, per_rapid_overhead=0.0):
    """Return (order, flipped) permutation that minimises rapid travel cost.

    seg_entries[i], seg_exits[i]: (x, y) G-code mm for the i-th segment's
    entry and exit points.  origin: tool XY position before the first segment.

    link_threshold: connections shorter than this distance are treated as free
        (they will be stitched at feed rate with no retract/plunge).
    per_rapid_overhead: fixed cost added to every connection that exceeds
        link_threshold, representing the retract + plunge distance overhead
        (typically 2 × z_safe).  Without this the optimiser may break several
        cheap stitches to save a few mm of traverse.

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

    def edge_cost(a, b):
        d = ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5
        return 0.0 if d < link_threshold else per_rapid_overhead + d

    for _ in range(max_passes):
        improved = False
        for i in range(-1, n - 2):
            prev_ext = origin if i == -1 else ext[i]
            for j in range(i + 2, n):
                old = edge_cost(prev_ext, ent[i + 1])
                new = edge_cost(prev_ext, ext[j])
                if j + 1 < n:
                    old += edge_cost(ext[j],     ent[j + 1])
                    new += edge_cost(ent[i + 1], ent[j + 1])
                if new < old - 1e-9:
                    lo, hi = i + 1, j + 1
                    block_e = ent[lo:hi]
                    block_x = ext[lo:hi]
                    ent[lo:hi]     = list(reversed(block_x))
                    ext[lo:hi]     = list(reversed(block_e))
                    order[lo:hi]   = list(reversed(order[lo:hi]))
                    flipped[lo:hi] = [not f for f in reversed(flipped[lo:hi])]
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
    The search runs on the 8-connected pixel grid of a window around both
    ends (a margin of a quarter of their distance, at least 2 px), each
    move costing its 3-D length; scipy.sparse.csgraph.dijkstra solves it.
    Returns the (row, col) pixels from (r0,c0) to (r1,c1), or the
    Bresenham line between them when no path is found.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    margin = max(2, max(abs(r1 - r0), abs(c1 - c0)) // 4 + 1)
    rlo = max(0, min(r0, r1) - margin)
    rhi = min(H - 1, max(r0, r1) + margin)
    clo = max(0, min(c0, c1) - margin)
    chi = min(W - 1, max(c0, c1) + margin)

    z = np.clip(center_depth_map[rlo:rhi + 1, clo:chi + 1].astype(np.float64)
                - final_norm_val, 0.0, ceil_norm)
    h, w = z.shape
    idx = np.arange(h * w).reshape(h, w)
    # Each undirected edge once: right, down, down-right, down-left.
    src, dst, cost = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cols_a = slice(max(0, -dc), w - max(0, dc))
        cols_b = slice(max(0, dc), w - max(0, -dc))
        dz = (z[dr:, cols_b] - z[:h - dr, cols_a]).ravel() * max_depth
        dxy2 = (dr / px_per_mm_y) ** 2 + (dc / px_per_mm_x) ** 2
        src.append(idx[:h - dr, cols_a].ravel())
        dst.append(idx[dr:, cols_b].ravel())
        cost.append(np.sqrt(dxy2 + dz * dz))
    graph = csr_matrix((np.concatenate(cost),
                        (np.concatenate(src), np.concatenate(dst))),
                       shape=(h * w, h * w))
    start = (r0 - rlo) * w + (c0 - clo)
    target = (r1 - rlo) * w + (c1 - clo)
    _, pred = dijkstra(graph, directed=False, indices=start,
                       return_predecessors=True)

    if target != start and pred[target] < 0:
        return list(_bresenham_pixels(r0, c0, r1, c1))

    path, cur = [], target
    while cur != start:
        path.append((rlo + cur // w, clo + cur % w))
        cur = int(pred[cur])
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


def simplify_line(points, tol, z_tol=None):
    """Ramer-Douglas-Peucker path simplification (2-D or 3-D).

    Removes intermediate points whose perpendicular distance from the line
    segment connecting the current endpoints is below *tol*.  Collinear or
    near-collinear runs collapse to just their two endpoints.

    With *z_tol* given (3-D points only), the deviation test is anisotropic:
    Z is scaled by tol / z_tol before measuring distances, so a removed
    point lies within an ellipsoid of lateral semi-axis *tol* and vertical
    semi-axis *z_tol* around its nearest point on the simplified path.  This
    keeps the vertical error bounded independently of the XY resolution.

    Parameters
    ----------
    points : list of (x, y, z) tuples  **or**  numpy array of shape (N, D)
    tol    : maximum allowed deviation (same units as the coordinates);
             the lateral deviation when *z_tol* is given
    z_tol  : maximum allowed vertical deviation, or None to measure Z with
             the same *tol* as X/Y

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
    pts = np.array(points, dtype=np.float64)  # (N, D) — works for any D
    N   = len(pts)
    if z_tol is not None and pts.shape[1] >= 3:
        pts[:, 2] *= tol / max(z_tol, 1e-12)

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

    if is_array:
        return np.asarray(points, dtype=np.float64)[keep]
    return [points[i] for i in range(N) if keep[i]]


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
    elif tool.profile == "Custom":
        if not tool.profile_curve:
            raise ValueError("Custom profile has no curve.")
        r_s, h_s = tool_profile.curve_arrays(tool.profile_curve)
        k_rad = max(1, int(np.ceil(r_s[-1] * px_per_mm)))
        ky, kx = np.mgrid[-k_rad:k_rad+1, -k_rad:k_rad+1]
        r_px = np.sqrt(kx**2 + ky**2)
        cap = np.interp(r_px / px_per_mm, r_s, h_s)
        cap[r_px > k_rad] = 1e9
        kernel    = (cap / max_depth).astype(np.float32)
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
    if tool.profile == "Custom":
        inflated.profile_curve = tool_profile.offset_curve(tool.profile_curve,
                                                           extra_offset_mm)
    return _build_kernel(inflated, px_per_mm, max_depth)


def _vbit_cone_reach_check(tool, stepdown_mm):
    """Return a warning string if a V-bit's cone can't reach one stepdown_mm
    increment before its own diameter clips it, else None.

    Only meaningful for tool.profile == 'V-Bit'. The hybrid skeleton_mode's
    skeleton pass assigns depth from the inscribed-circle formula alone
    (independent of the real tool's radius — see the 'inscribed'-mode depth
    calc above), so in wide areas it can call for a deeper cut than the REAL
    tool's cone can deliver before its own diameter/shank takes over —  a
    radius limitation that make_multipass's step-down splitting doesn't fix,
    since it only limits how much depth is removed per pass, not how far
    the tool can reach sideways at any given depth.
    """
    if tool.profile != "V-Bit":
        return None
    from math import radians, tan
    tan_half = tan(radians(tool.vbit_angle_deg / 2.0))
    required_diam_mm = 2.0 * stepdown_mm * tan_half
    if tool.diameter_mm < required_diam_mm:
        return (f"Warning: V-bit diameter ({tool.diameter_mm:.2f} mm) is smaller "
                f"than the {required_diam_mm:.2f} mm needed for its "
                f"{tool.vbit_angle_deg:.0f}° cone to reach a single {stepdown_mm:.2f} mm "
                f"step-down without the shank interfering. Deep skeleton cuts in "
                f"wide areas may not be a clean V-taper past that depth.")
    return None


def _find_compatible_vbit_erosion(erosion_cache, base_key, min_diameter_mm):
    """Search erosion_cache for a cached V-Bit achievable-depth surface with
    base_key's angle/max_depth/dims/extra_offset/source_visibility but a
    diameter_mm >= min_diameter_mm.

    Such an entry's grey-erosion result is byte-identical to one computed
    with exactly min_diameter_mm, since _build_kernel's cone_reach_mm
    saturates at max_depth*tan_half once R_mm reaches that value — the
    kernel/footprint stop changing for any bigger diameter. Returns the
    cached center_depth_contour array, or None if nothing qualifies.

    base_key : the (profile, diameter_mm, vbit_angle_deg, max_depth, ...)
               erosion_cache_key tuple for the CURRENT call, used to match
               every field except diameter (which is relaxed to >=).
    """
    if erosion_cache is None or base_key is None or base_key[0] != "V-Bit":
        return None
    _, _, angle, *rest = base_key
    candidates = [(key, val) for key, val in erosion_cache.items()
                  if key[0] == "V-Bit" and key[2] == angle
                  and list(key[3:]) == rest and key[1] >= min_diameter_mm]
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[0][1])   # prefer the smallest sufficient match
    return candidates[0][1][0]                  # (center_depth_contour, accessible) -> center_depth_contour


def simulate_toolpath_result(toolpath, shape, px_per_mm_x, px_per_mm_y, ox, oy,
                              max_depth_mm, kernel, footprint, resample_spacing_px=0.5):
    """Rasterize a toolpath's swept volume and return the material-removal surface.

    Densely resamples *toolpath* segments at sub-pixel XY spacing, paints each
    sample's depth into a pixel-space array, then grey-dilates with the tool
    geometry kernel to produce the cumulative material-removal map (float32,
    values in [0, 1] normalised depth units). Samples are painted chunk by
    chunk as ToolPath.iter_resampled produces them.
    """
    from grey_morphology import grey_dilation as _grey_dilation
    H, W = shape
    px_per_mm = (px_per_mm_x * px_per_mm_y) ** 0.5
    achieved = np.zeros((H, W), dtype=np.float32)
    for pts in toolpath.iter_resampled(resample_spacing_px / px_per_mm):
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
                       structure=-kernel.astype(np.float64)).astype(np.float32, copy=False),
        0.0)


def _plan_passes(max_depth_mm, stepdown_mm, final_stepdown_mm=0.0, top_mm=0.0):
    """Return ordered list of cumulative cut-depth ceilings (mm).

    The passes step down from top_mm (0 = the stock surface; deeper when the
    material above it is already removed) to max_depth_mm.
    If final_stepdown_mm > 0, roughing stops at max_depth − final_stepdown_mm
    and a dedicated finishing pass reaches max_depth.
    stepdown_mm = 0 means one roughing step at full roughing depth (equivalent
    to stepdown_mm = max_depth_mm − top_mm).
    """
    band_mm = max_depth_mm - top_mm
    if band_mm <= 0.0:
        return [max_depth_mm]
    if stepdown_mm <= 0.0:
        if final_stepdown_mm <= 0.0:
            return [max_depth_mm]
        stepdown_mm = band_mm   # one roughing step + one final pass
    if final_stepdown_mm > 0.0 and final_stepdown_mm < band_mm:
        roughable = band_mm - final_stepdown_mm
        n = max(1, int(np.ceil(roughable / stepdown_mm)))
        passes = [top_mm + min((i + 1) * stepdown_mm, roughable) for i in range(n)]
        passes.append(max_depth_mm)
    else:
        n = max(1, int(np.ceil(band_mm / stepdown_mm)))
        passes = [top_mm + min((i + 1) * stepdown_mm, band_mm) for i in range(n)]
    return passes


def _insert_ceiling_crossings(seg, thresholds):
    """Insert points where seg's z crosses each (t_mm, pass_idx) in
    *thresholds*, tagging the inserted point with the pass index whose
    entry it marks. Original points are tagged None.

    Without this, a pass whose sub-segment begins mid-line (the terrain
    dips below that pass's ceiling somewhere between two stored points)
    would start at the next stored point instead of at the true crossing,
    leaving a gap between where the previous pass's cut actually reached
    its ceiling and where the next pass picks up.
    """
    if not thresholds or len(seg) < 2:
        return [(x, y, z, None) for x, y, z in seg]
    out = [(seg[0][0], seg[0][1], seg[0][2], None)]
    for (x0, y0, z0), (x1, y1, z1) in zip(seg, seg[1:]):
        crossings = []
        if z1 != z0:
            for t_mm, pass_idx in thresholds:
                lo, hi = (z0, z1) if z0 < z1 else (z1, z0)
                if lo < t_mm < hi:
                    frac = (t_mm - z0) / (z1 - z0)
                    crossings.append((frac, t_mm, pass_idx))
        crossings.sort(key=lambda c: c[0])
        for frac, t_mm, pass_idx in crossings:
            out.append((x0 + frac * (x1 - x0), y0 + frac * (y1 - y0), t_mm, pass_idx))
        out.append((x1, y1, z1, None))
    return out


def make_multipass(segments, max_depth_mm, stepdown_mm, final_stepdown_mm=0.0,
                   top_mm=0.0):
    """Depth-slice full-depth segments into per-pass segments.

    Parameters
    ----------
    segments       : list of [(x, y, z_mm)…] where z_mm is negative (below surface)
    max_depth_mm   : total cut depth in mm
    stepdown_mm    : max depth increment per roughing pass (0 = single pass)
    final_stepdown_mm : depth reserved for a dedicated finishing pass (0 = none)
    top_mm         : depth the first pass steps down from (0 = the stock
                     surface); material above it counts as already removed

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

    A roughing pass's set of kept points can start or end mid-segment,
    wherever the line's z crosses that pass's entry depth (the previous
    pass's ceiling, offset by the reserve). Since segments may be RDP-
    simplified before reaching here, that crossing can fall well short of
    the next stored point, so each segment is pre-split with an exact
    interpolated point at every such crossing (see
    _insert_ceiling_crossings) before the per-pass keep/drop logic runs.
    """
    if stepdown_mm <= 0.0 and final_stepdown_mm <= 0.0:
        return [(max_depth_mm, True, list(segments))]

    pass_ceilings = _plan_passes(max_depth_mm, stepdown_mm, final_stepdown_mm,
                                 top_mm)
    EPS = 1e-6
    n_passes = len(pass_ceilings)
    has_dedicated_final = final_stepdown_mm > 0.0
    n_rough = n_passes - 1 if has_dedicated_final else n_passes

    def rough_target(z_full, ceil_mm):
        return max(min(z_full + final_stepdown_mm, 0.0), -ceil_mm)

    thresholds = []
    for k in range(n_rough):
        if k == 0 and not has_dedicated_final:
            continue   # first roughing pass with no reserve keeps every point already
        prev_ceil = pass_ceilings[k - 1] if k > 0 else top_mm
        thresholds.append((-prev_ceil - final_stepdown_mm, k))

    aug_segments = [_insert_ceiling_crossings(seg, thresholds) for seg in segments]

    result = []
    for pi, ceil_mm in enumerate(pass_ceilings):
        is_final = (pi == n_passes - 1)
        is_dedicated_final = is_final and has_dedicated_final
        pass_segs = []
        for seg in aug_segments:
            sub = []
            for x, y, z_full, boundary_pi in seg:
                if is_dedicated_final:
                    z_target = z_full
                    keep = True
                else:
                    z_target = rough_target(z_full, ceil_mm)
                    if boundary_pi == pi:
                        keep = True   # exact entry crossing — force it in despite the EPS margin below
                    elif pi == 0:
                        keep = (True if not has_dedicated_final
                                else z_target < -top_mm - EPS)
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


def compute_air_free_depth(previous_result, kernel, footprint, shape,
                           previous_alpha=None):
    """Return the deepest normalised tool-tip depth at each centre pixel at
    which the tool touches no material left by previous_result.

    A tool centred at c with its tip at depth z removes material at c+d
    down to z - kernel(d); it cuts only air there iff
    z <= previous_result(c+d) + kernel(d) for every d in the footprint. The
    minimum of that bound over the footprint is a grey erosion of
    previous_result with structure -kernel -- the same operation that builds
    the achievable-depth surface from the depth map. kernel/footprint must
    be the real tool's (_build_kernel), not the stock-to-leave-inflated
    planning kernel, since this is about physical contact.

    previous_result None (no prior step) means untouched stock, i.e. zeros.
    previous_alpha (ToolPath.alpha, 0.0 = a hole cut through the stock), if
    given, marks pixels with no material at any depth: they never constrain
    the tool, however deep it goes.
    """
    if previous_result is None and previous_alpha is None:
        return np.zeros(shape, dtype=np.float32)
    from grey_morphology import grey_erosion
    removed = (previous_result.astype(np.float32, copy=False) if previous_result is not None
               else np.zeros(shape, dtype=np.float32))
    if previous_alpha is not None:
        removed = np.where(previous_alpha < 0.5, 1e6, removed)
    return grey_erosion(removed, footprint=footprint,
                        structure=-kernel.astype(np.float64)).astype(np.float32, copy=False)


def _retract_breakeven_mm(feed_rate, feed_rapid, z_safe, depth_mm):
    """Return the feed length (mm) whose cutting time equals the extra
    retract + plunge a split introduces at depth_mm: an air run shorter than
    this is cheaper to feed through than to lift over."""
    return feed_rate * 2.0 * (z_safe + depth_mm) / feed_rapid


def _prune_air_passes(pass_list, previous_result, kernel, footprint, shape,
                      px_per_mm, ox, oy, max_depth, tol_norm,
                      feed_rate, feed_rapid, z_safe, parent_ids=None,
                      previous_alpha=None):
    """Apply prune_air_segments to every pass of a make_multipass pass list,
    against compute_air_free_depth(previous_result, kernel, footprint, shape)
    (optionally previous_alpha) and the retract/plunge breakeven for
    feed_rate/feed_rapid/z_safe.

    parent_ids, if given, holds one id per segment of every non-empty pass
    (callers use it where make_multipass keeps a pass's segments one-to-one
    with the full-depth segments, as in Pocket). Returns
    (pruned_pass_list, per_pass_ids, air_mm): per_pass_ids[pi] holds one id
    per segment of pruned pass pi, or is None when parent_ids is None;
    air_mm is the total feed length removed across all passes.
    """
    free = compute_air_free_depth(previous_result, kernel, footprint, shape,
                                  previous_alpha=previous_alpha)

    def breakeven(depth_mm):
        return _retract_breakeven_mm(feed_rate, feed_rapid, z_safe, depth_mm)

    pruned, per_pass_ids, air_mm = [], [], 0.0
    for ceil_mm, is_final, pass_segs in pass_list:
        segs, ids, air = prune_air_segments(
            pass_segs, free, px_per_mm, shape[0], ox, oy, max_depth, tol_norm,
            breakeven_fn=breakeven,
            parent_ids=parent_ids if (parent_ids is not None and pass_segs) else None)
        pruned.append((ceil_mm, is_final, segs))
        per_pass_ids.append(ids if ids is not None else ([] if parent_ids is not None else None))
        air_mm += air
    return pruned, per_pass_ids, air_mm


def _true_runs(mask):
    """Return [(start, end_exclusive), ...] for each run of True in a 1-D
    bool array."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def prune_air_segments(segments, free_depth, px_per_mm, H, ox, oy, max_depth,
                       tol_norm, breakeven_fn=None, parent_ids=None,
                       sample_px=0.5):
    """Remove the portions of mm-space segments where the tool cuts only air.

    Each segment [(x, y, z_mm), ...] is sampled along its edges every
    sample_px pixels; a sample cuts material iff its depth (-z / max_depth)
    exceeds free_depth (see compute_air_free_depth) at its nearest pixel by
    more than tol_norm. Air runs at a segment's start or end are always
    trimmed (that costs no extra retract). An interior air run is removed,
    splitting the segment in two, only if its length is at least
    breakeven_fn(deepest depth in mm along the run) -- see
    _retract_breakeven_mm -- otherwise the tool feeds straight through it.
    breakeven_fn None removes every interior air run.

    A closed loop (first and last points within 1.5 px) is first rotated to
    start inside its longest air run, so that run becomes the trimmed
    start/end rather than an interior split -- the loop is entered once
    either way.

    Each kept run is widened by one sample into the adjacent air on both
    sides, so its start/end lie at most sample_px short of the true
    air/material crossing on the air side and every material-cutting sample
    stays covered. Kept runs contain the segment's own vertices plus those
    two boundary samples, never the intermediate samples, so the output
    has the same vertex density as the input.

    parent_ids, if given (one per segment), is mapped onto the output: each
    output segment carries its source segment's id.

    Returns (pruned_segments, pruned_parent_ids, air_mm) -- pruned_parent_ids
    is None when parent_ids is None; air_mm is the total feed length removed.
    """
    Hf, Wf = free_depth.shape
    out, out_ids = [], []
    air_mm = 0.0
    closed_tol_mm = 1.5 / px_per_mm
    for si, seg in enumerate(segments):
        pid = parent_ids[si] if parent_ids is not None else None
        pts = np.asarray(seg, dtype=np.float64)
        if len(pts) < 2:
            out.append(seg)
            out_ids.append(pid)
            continue
        is_closed = (len(pts) > 2 and
                     np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1]) < closed_tol_mm)
        if is_closed and not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[:1]])

        # Densify: n_e samples per edge, t = 0, 1/n_e, ... plus the last vertex.
        d = np.diff(pts, axis=0)
        edge_px = np.hypot(d[:, 0], d[:, 1]) * px_per_mm
        n_e = np.maximum(1, np.ceil(edge_px / sample_px)).astype(np.intp)
        starts = np.cumsum(n_e) - n_e
        total = int(n_e.sum())
        edge_of = np.repeat(np.arange(len(d)), n_e)
        t = (np.arange(total) - np.repeat(starts, n_e)) / np.repeat(n_e, n_e)
        dense = np.vstack([pts[edge_of] + t[:, None] * d[edge_of], pts[-1:]])
        is_vertex = np.zeros(len(dense), dtype=bool)
        is_vertex[starts] = True
        is_vertex[-1] = True

        cols = np.clip(np.round((dense[:, 0] + ox) * px_per_mm).astype(np.intp), 0, Wf - 1)
        rows = np.clip(np.round((H - 1) - (dense[:, 1] + oy) * px_per_mm).astype(np.intp), 0, Hf - 1)
        depth_norm = -dense[:, 2] / max_depth
        cut = depth_norm > free_depth[rows, cols] + tol_norm

        if cut.all():
            out.append(seg)
            out_ids.append(pid)
            continue
        step = np.hypot(np.diff(dense[:, 0]), np.diff(dense[:, 1]))
        if not cut.any():
            air_mm += float(step.sum())
            continue

        if is_closed:
            # Drop the duplicated closing sample, rotate so the sequence
            # starts at the first sample of the longest air run, then close
            # it again by repeating that first sample at the end. The run
            # then sits at the sequence's start, where it is trimmed.
            dense, is_vertex, cut = dense[:-1], is_vertex[:-1], cut[:-1]
            n = len(cut)
            first_cut = int(np.argmax(cut))
            order = (np.arange(n) + first_cut) % n       # starts on a cut sample
            air_runs = _true_runs(~cut[order])
            longest = max(air_runs, key=lambda r: r[1] - r[0])
            order = np.roll(order, -longest[0])
            order = np.append(order, order[0])
            dense, cut, is_vertex = dense[order], cut[order], is_vertex[order]
            step = np.hypot(np.diff(dense[:, 0]), np.diff(dense[:, 1]))

        n = len(cut)
        cum = np.concatenate([[0.0], np.cumsum(step)])
        keep = cut.copy()
        # Restore interior air runs shorter than the retract/plunge breakeven.
        if breakeven_fn is not None:
            for i, j in _true_runs(~cut):
                if i > 0 and j < n:
                    run_depth_mm = max(float(np.max(-dense[i:j, 2])), 0.0)
                    if cum[j] - cum[i - 1] < breakeven_fn(run_depth_mm):
                        keep[i:j] = True

        # Widen every kept run by one sample on each side.
        widened = keep.copy()
        widened[1:] |= keep[:-1]
        widened[:-1] |= keep[1:]

        kept_len = 0.0
        for i, j in _true_runs(widened):
            idx = np.concatenate([[i], i + 1 + np.flatnonzero(is_vertex[i + 1:j - 1]), [j - 1]])
            out.append([tuple(float(v) for v in dense[k]) for k in idx])
            out_ids.append(pid)
            kept_len += float(cum[j - 1] - cum[i])
        air_mm += float(cum[-1]) - kept_len
    return out, (out_ids if parent_ids is not None else None), air_mm


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


def _order_by_precedence(polylines, piece_node, parents, current_pos, *,
                         closed=None, allow_flip=False, progress=None,
                         progress_lo=0, progress_hi=100, what="pieces"):
    """Order pieces along a precedence graph, nearest-first among those
    that may be cut.

    Every piece belongs to one node, and a node may hold any number of
    pieces, none included. A node is complete once all of its pieces are
    cut, and its pieces may be cut once every node in parents[node] is
    complete; a node without pieces completes as soon as its parents do.
    Among the pieces that may be cut, greedy nearest-neighbour picks the
    one whose start lies nearest the tool. Contour (_order_contours_graph)
    uses this with one node per hill or crater at each level, Onion Skin
    (_order_onion_pass) with the level graph of each region's ring chains.

    polylines  : per piece, an (N, 2) array of (x, y) mm.
    piece_node : per piece, its node.
    parents    : per node, the nodes that must be complete before it.
    current_pos: (x, y) tool position before the first piece.
    closed     : per piece, whether it is a closed ring that may start at
                 any of about 20 points along itself (None: none is).
    allow_flip : pieces may also run backwards.
    what       : what the pieces are, for the progress messages.

    Returns (order, end_pos): order holds (piece, flip, start) per piece
    in cutting order -- flip whether it runs backwards, start the index
    into the (flipped) polyline it starts at, after which a closed piece
    runs round back to that point -- and end_pos is the (x, y) tool
    position after the last piece.
    """
    if progress is None:
        progress = lambda pct, msg: None
    n_nodes = len(parents)
    children = [[] for _ in range(n_nodes)]
    for n, ps in enumerate(parents):
        for p in ps:
            children[p].append(n)
    waiting = [len(ps) for ps in parents]
    by_node = [[] for _ in range(n_nodes)]
    for i, n in enumerate(piece_node):
        by_node[n].append(i)
    left = [len(b) for b in by_node]
    if closed is None:
        closed = [False] * len(polylines)

    # Start candidates of every piece: (piece, flip, index into the
    # oriented polyline) and their position in mm.
    cand_piece, cand_flip, cand_idx, cand_xy = [], [], [], []
    for i, (c, is_closed) in enumerate(zip(polylines, closed)):
        n = len(c)
        idx = np.arange(0, n, max(1, n // 20)) if is_closed else np.array([0])
        for flip in ((False, True) if allow_flip else (False,)):
            rc = c[::-1] if flip else c
            cand_piece.append(np.full(len(idx), i))
            cand_flip.append(np.full(len(idx), flip))
            cand_idx.append(idx)
            cand_xy.append(rc[idx])
    if polylines:
        cand_piece = np.concatenate(cand_piece)
        cand_flip = np.concatenate(cand_flip)
        cand_idx = np.concatenate(cand_idx)
        cand_xy = np.vstack(cand_xy)
    available = np.zeros(len(polylines), bool)

    def release(n):
        stack = [n]
        while stack:
            n = stack.pop()
            if left[n]:
                available[by_node[n]] = True
                continue
            for ch in children[n]:
                waiting[ch] -= 1
                if waiting[ch] == 0:
                    stack.append(ch)

    for n in range(n_nodes):
        if waiting[n] == 0:
            release(n)

    order = []
    cur = np.array(current_pos, dtype=float)
    n_pieces = len(polylines)
    n_report = max(1, n_pieces // 20)
    for step in range(n_pieces):
        if step % n_report == 0:
            progress(progress_lo + (progress_hi - progress_lo) * step // n_pieces,
                     f"Ordering {what} ({step}/{n_pieces})…")
        d2 = np.sum((cand_xy - cur) ** 2, axis=1)
        d2[~available[cand_piece]] = np.inf
        k = int(np.argmin(d2))
        i = int(cand_piece[k])
        flip, start = bool(cand_flip[k]), int(cand_idx[k])
        order.append((i, flip, start))
        rc = polylines[i][::-1] if flip else polylines[i]
        cur = np.array(rc[start] if closed[i] else rc[-1], dtype=float)
        available[i] = False
        n = piece_node[i]
        left[n] -= 1
        if left[n] == 0:
            release(n)
    return order, (float(cur[0]), float(cur[1]))


def _order_onion_pass(pass_segs, pass_chains, chain_parents, origin):
    """Order one pass of Onion Skin segments along the level graph of
    their ring chains (_order_by_precedence), never reversing a segment.

    pass_chains   : per segment, the chain it was cut from (air pruning
                    can split a chain into several segments or drop it).
    chain_parents : per chain, the chains that must be cut before it
                    (_pocket_infill: for Outside-In the chains one step
                    further out it follows, for Inside-Out the chains
                    that follow it).

    A chain's segments are cut in their own order, one after the other,
    so a chain keeps its rings' outside-in or inside-out sequence. Chains
    whose predecessors are all cut are then picked nearest-first, the way
    Contour picks among its features. Chain c is node c, which holds no
    segment; segment j is node n_chains + j, after the segment before it
    in the same chain or, for a chain's first segment, after node c.
    Node c waits for the last segment of each of its parent chains, or
    for the parent's own node where none of its segments is left.
    """
    n_chains = len(chain_parents)
    piece_node = [n_chains + j for j in range(len(pass_segs))]
    seg_parents = []
    last = list(range(n_chains))   # the node a chain's next segment follows
    for c in pass_chains:
        seg_parents.append([last[c]])
        last[c] = n_chains + len(seg_parents) - 1
    parents = [[last[p] for p in ps] for ps in chain_parents] + seg_parents
    polylines = [np.asarray(seg, dtype=float)[:, :2] for seg in pass_segs]
    order, _ = _order_by_precedence(polylines, piece_node, parents, origin)
    return [pass_segs[i] for i, _, _ in order]


def _order_locked_raster_pass(pass_segs, direction, cur):
    """Order one depth-pass's segments for a locked Climb/Conventional
    direction, keeping consecutive cuts on the same or an adjacent raster
    line so the "uncut material on a fixed side" guarantee holds between
    segments, not just within one.

    pass_segs : list of [(x, y, z_mm)...], each already oriented for
                ascending line index ("mode A" — set by Phase 1's
                FORCE_REVERSE).
    direction : "X" (horizontal lines, grouped by their constant Y), "Y"
                (vertical lines, grouped by their constant X), "D45"
                (45-degree lines, grouped by their constant x-y), or "D135"
                (135-degree lines, grouped by their constant x+y).
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
        x, y = seg[0][0], seg[0][1]
        return {"X": -y, "Y": x, "D45": x - y, "D135": x + y}[direction]

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
        # The first cut is taken from the chosen end line alone: an adjacent
        # line's entry is often nearer cur, and taking it would strand the
        # end line until the sweep flips back for it.
        if ordered and 0 <= nptr < len(keys):
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


def _order_free_region_pass(pass_segs, direction, cur):
    """Order one pocket region's own Zig-Zag segments for an unlocked
    (Either) cut direction: the first line entered is whichever of the
    region's own two extreme lines (by the same coord() key
    _order_locked_raster_pass uses) is nearer cur, entered from whichever
    of its own two ends is nearer -- never an arbitrary interior line.
    Every remaining line is then ordered by plain unrestricted greedy
    nearest-endpoint search (_order_free_raster_pass), which is free to
    enter each line from whichever end is closer. Unlike a locked sweep,
    which must keep every line travelling the same physical direction to
    preserve which side carries uncut material, Either has no such
    constraint, so letting each line's own entry end flip freely is what
    lets consecutive lines actually stay close together -- fixing every
    line to the same direction here would mean returning to the same
    starting side every line regardless of how close the opposite ends
    of two consecutive lines are.
    """
    def coord(seg):
        x, y = seg[0][0], seg[0][1]
        return {"X": -y, "Y": x, "D45": x - y, "D135": x + y}[direction]

    keys = sorted(set(coord(seg) for seg in pass_segs))
    extremes = [i for i, seg in enumerate(pass_segs) if coord(seg) in (keys[0], keys[-1])]

    best_i, best_rev, best_dist = extremes[0], False, float('inf')
    for i in extremes:
        seg = pass_segs[i]
        for rev in (False, True):
            x, y = (seg[-1] if rev else seg[0])[:2]
            d = (x - cur[0]) ** 2 + (y - cur[1]) ** 2
            if d < best_dist:
                best_i, best_rev, best_dist = i, rev, d

    first = list(reversed(pass_segs[best_i])) if best_rev else pass_segs[best_i]
    rest = [seg for j, seg in enumerate(pass_segs) if j != best_i]
    return [first] + _order_free_raster_pass(rest, first[-1][:2], False, 0.0)


def _order_zigzag_pass(pass_segs, seg_region_ids, direction, cur, lock_direction):
    """Order one Pocket pass's Zig-Zag segments so that one pocket region's
    own sweep completes before the next region's begins -- two different
    regions' segments are never interleaved in the visiting order, even
    when that would shorten travel.

    Each region's own segments (grouped by seg_region_ids, one entry per
    pass_segs element -- see full_depth_segs/seg_region_ids alignment in
    compute_pocket_toolpath's Stage 4) are ordered by
    _order_locked_raster_pass when cut direction is locked (Climb or
    Conventional -- needed to keep uncut material on one consistent side
    between lines, the same guarantee it already provides for a single
    scan area), or by _order_free_region_pass when it's Either (no such
    correctness constraint, so lines can flip freely for shorter travel).
    Both start from whichever of the region's own two extreme lines is
    nearer the running tool position, rather than an arbitrary interior
    line -- the only reason that doesn't already happen today is that
    ordering runs across every region's segments pooled together, so a
    single region's own two extremes aren't necessarily the pool's global
    sweep ends. Calling either per region's own segments here is what
    fixes that.

    At each step, every not-yet-visited region's own best order (and
    entry point) is (re)computed against the current tool position, and
    whichever region's entry point is nearest is visited next -- so a
    region's chosen entry corner can differ from step to step depending on
    where the tool ends up after the previous region, same as which of a
    single region's own two ends is nearer can differ by incoming
    position.
    """
    groups = {}
    for seg, rid in zip(pass_segs, seg_region_ids):
        groups.setdefault(rid, []).append(seg)
    remaining = list(groups.values())

    ordered = []
    while remaining:
        best_i, best_order, best_dist = 0, None, float('inf')
        for i, group_segs in enumerate(remaining):
            if lock_direction:
                candidate = _order_locked_raster_pass(group_segs, direction, cur)
            else:
                candidate = _order_free_region_pass(group_segs, direction, cur)
            x, y = candidate[0][0][0], candidate[0][0][1]
            d = (x - cur[0]) ** 2 + (y - cur[1]) ** 2
            if d < best_dist:
                best_i, best_order, best_dist = i, candidate, d
        ordered.extend(best_order)
        cur = best_order[-1][-1][:2]
        remaining.pop(best_i)
    return ordered


def _grid_line_ids(boundary_px, px_per_mm, step_mm, raster_offset_pct):
    """Evenly spaced grid-line pixel indices spanning [0, boundary_px],
    step-adjusted to fit exactly (no partial leftover line) and shifted by
    raster_offset_pct of the adjusted step.

    Shared by Horizontal/Vertical raster lines (one axis) and both axes of
    the Coil grid.

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
                  px_per_mm, H, _ox, _oy, max_depth, offset=(0, 0)):
    """Build zero or more point-list segments along one raster line (or a
    sub-range of a line, for coil legs), gated by prior_mask/mask_spans.

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
    offset    : the canvas (row, col) of center_depth_contour_raster's and
                prior_mask's [0, 0], for arrays cropped to a window of the
                canvas; fixed_idx, lo_idx and hi_idx stay canvas indices.

    Returns
    -------
    list of point-lists — [[(x_mm, y_mm, -z_mm), ...], ...] — normally one
    element, or several when prior_mask/mask_spans splits the range into
    disjoint active sub-spans.
    """
    r0, c0 = offset
    if direction == "X":
        line_sel = np.s_[fixed_idx - r0, lo_idx - c0:hi_idx - c0 + 1]
    else:
        line_sel = np.s_[lo_idx - r0:hi_idx - r0 + 1, fixed_idx - c0]
    line_depth = center_depth_contour_raster[line_sel]

    if prior_mask is not None:
        effective = (line_depth * prior_mask[line_sel]).astype(np.float32)
        spans = mask_spans(effective > resid_tol)
    else:
        spans = [(0, len(line_depth) - 1)]

    segs = []
    for s, e in spans:                       # s, e are LOCAL indices, 0-based within lo_idx..hi_idx
        points = []
        for local_px in range(s, e + 1):
            px = lo_idx + local_px
            row, col = (fixed_idx, px) if direction == "X" else (px, fixed_idx)
            z_mm = float(line_depth[local_px]) * max_depth
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


def _diag_grid_line_ids(H, W, family, px_per_mm, step_mm, raster_offset_pct):
    """Evenly spaced diagonal grid-line ids spanning an H x W canvas, for the
    45-degree ("D45": row + col = k) or 135-degree ("D135": row - col = k)
    line family, step-adjusted to fit exactly and shifted by
    raster_offset_pct -- the diagonal counterpart of _grid_line_ids.

    Consecutive diagonal lines of either family are spaced 1/sqrt(2) px
    apart per unit of k, so this reuses _grid_line_ids unmodified: passing
    an effective px_per_mm of px_per_mm*sqrt(2) turns its ordinary
    pixel-index spacing into the correct real-world mm spacing between
    diagonal lines.

    Returns
    -------
    (k_ids, adj_step_mm) -- k_ids is a list of diagonal line constants (k =
    row+col for D45, k = row-col for D135); adj_step_mm is the step size
    after rounding to fit.
    """
    boundary_px = (H - 1) + (W - 1)
    offset = 0 if family == "D45" else -(W - 1)
    line_ids, adj_step_mm = _grid_line_ids(boundary_px, px_per_mm * (2 ** 0.5),
                                           step_mm, raster_offset_pct)
    return [i + offset for i in line_ids], adj_step_mm


def _diag_line_bounds(family, k, r_lo, r_hi, c_lo, c_hi):
    """Inclusive row-parameter bounds (t_lo, t_hi) for the portion of one
    diagonal line (row+col=k for "D45", row-col=k for "D135") that lies
    within [r_lo,r_hi] x [c_lo,c_hi], or None if the line misses the box.
    """
    if family == "D45":
        t_lo, t_hi = max(r_lo, k - c_hi), min(r_hi, k - c_lo)
    else:
        t_lo, t_hi = max(r_lo, k + c_lo), min(r_hi, k + c_hi)
    return (t_lo, t_hi) if t_lo <= t_hi else None


def _diag_line_points(family, k, t_lo, t_hi, reverse,
                      center_depth_contour_raster, prior_mask, resid_tol,
                      px_per_mm, H, _ox, _oy, max_depth, offset=(0, 0)):
    """Diagonal counterpart to _line_points: samples the 45-degree
    ("D45": row+col=k) or 135-degree ("D135": row-col=k) line between row
    parameters t_lo..t_hi (inclusive, t_lo <= t_hi), gated by
    prior_mask/mask_spans exactly as _line_points, using fancy indexing
    since both row and column vary per sample. Point spacing along the line
    is sqrt(2) px, vs. 1 px for an axis-aligned _line_points line.

    reverse   : if True, each resulting segment's points are reversed so
                the segment travels from t_hi toward t_lo instead of
                t_lo toward t_hi.
    offset    : the canvas (row, col) of center_depth_contour_raster's and
                prior_mask's [0, 0], as in _line_points.

    Returns
    -------
    list of point-lists -- [[(x_mm, y_mm, -z_mm), ...], ...] -- normally one
    element, or several when prior_mask/mask_spans splits the range into
    disjoint active sub-spans.
    """
    rows = np.arange(t_lo, t_hi + 1)
    cols = (k - rows) if family == "D45" else (rows - k)

    r0, c0 = offset
    line_depth = center_depth_contour_raster[rows - r0, cols - c0]

    if prior_mask is not None:
        line_mask = prior_mask[rows - r0, cols - c0]
        effective = (line_depth * line_mask).astype(np.float32)
        spans = mask_spans(effective > resid_tol)
    else:
        spans = [(0, len(line_depth) - 1)]

    segs = []
    for s, e in spans:                       # s, e are LOCAL indices, 0-based within t_lo..t_hi
        points = []
        for local_px in range(s, e + 1):
            row, col = int(rows[local_px]), int(cols[local_px])
            z_mm = float(line_depth[local_px]) * max_depth
            x_mm = col / px_per_mm - _ox
            y_mm = (H - 1 - row) / px_per_mm - _oy
            points.append((x_mm, y_mm, -z_mm))
        if points:
            if reverse:
                points = list(reversed(points))
            segs.append(points)
    if reverse:
        segs = list(reversed(segs))
    return segs


def _coil_index_pairs_cw(ny, nx):
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


def _coil_index_pairs_ccw(ny, nx):
    """Mirror of _coil_index_pairs_cw: left col T→B, bottom row L→R,
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
    (real pixel coordinates, as produced by _coil_index_pairs_cw/ccw
    applied to row_ids/col_ids) into maximal straight legs.

    Two consecutive points always share either their row or their column
    (never both, except when the whole coil is a single point). This
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


def _coil_legs(row_ids, col_ids, sense):
    """Perimeter-tracing coil traversal of a (row_ids × col_ids) grid,
    outside-in, in the given rotational `sense` ("CW" or "CCW"). Handles
    ny==1, nx==1, ny==nx==1, and non-square ny/nx transparently — see
    _rle_into_legs.
    """
    ny, nx = len(row_ids), len(col_ids)
    index_order = (_coil_index_pairs_cw(ny, nx) if sense == "CW"
                   else _coil_index_pairs_ccw(ny, nx))
    pts = [(row_ids[r], col_ids[c]) for r, c in index_order]
    return _rle_into_legs(pts)


# Largest corner gap the rectangular Coil leaves between two of its
# own sharp-cornered rings one step apart (the diameter of the biggest
# circle fitting in the L-shaped band at a corner), as a multiple of the
# step: the Spiral's ring spacing never leaves a wider one.
_SPIRAL_CORNER_GAP_FACTOR = 2.0 * np.sqrt(2.0) / (1.0 + np.sqrt(2.0))


def _spiral_corner_radius(t, m):
    """Corner radius of the Spiral ring at level t (0..m): equal
    to t near the centre (circles/stadiums, a linear spiral) and falling
    back to 0 at t = m (the work box's own sharp-cornered rectangle)."""
    if m <= 0.0:
        return np.zeros_like(np.asarray(t, dtype=np.float64))
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, m)
    return t * (1.0 - t / m)


def _spiral_ring_points(ex, ey, t, rho, u):
    """Points on Spiral rings: rounded rectangles of half-extents
    (ex + t, ey + t) and corner radius rho, at arc-length fraction u in
    [0, 1) of each ring's perimeter. u = 0 is the midpoint of the top-left
    corner arc and u increases clockwise as displayed (top side, right
    side, bottom side, left side). t, rho and u are broadcast together.

    Returns
    -------
    (dy, dx) — pixel offsets from the work box's centre, row axis pointing
    down; arrays of the broadcast shape.
    """
    t, rho, u = np.broadcast_arrays(np.asarray(t, dtype=np.float64),
                                    np.asarray(rho, dtype=np.float64),
                                    np.asarray(u, dtype=np.float64))
    A, B = ex + t, ey + t
    a, b = A - rho, B - rho                  # straight half-lengths
    q = 0.25 * np.pi * rho                   # arc length of 45 degrees
    # (length, point-at-fraction) per piece, clockwise from the TL arc's
    # midpoint; arc angles are in the y-down frame, where increasing angle
    # is clockwise on screen.
    def arc(cx, cy, th0, th1):
        return lambda f: (cy + rho * np.sin(th0 + f * (th1 - th0)),
                          cx + rho * np.cos(th0 + f * (th1 - th0)))
    d = np.pi / 180.0
    pieces = [
        (q,      arc(-a, -b, 225 * d, 270 * d)),
        (2 * a,  lambda f: (-B + 0 * f, -a + 2 * a * f)),
        (2 * q,  arc(a, -b, 270 * d, 360 * d)),
        (2 * b,  lambda f: (-b + 2 * b * f, A + 0 * f)),
        (2 * q,  arc(a, b, 0 * d, 90 * d)),
        (2 * a,  lambda f: (B + 0 * f, a - 2 * a * f)),
        (2 * q,  arc(-a, b, 90 * d, 180 * d)),
        (2 * b,  lambda f: (b - 2 * b * f, -A + 0 * f)),
        (q,      arc(-a, -b, 180 * d, 225 * d)),
    ]
    perim = sum(length for length, _ in pieces)
    s = u * perim
    dy, dx = pieces[0][1](np.zeros_like(u))
    dy, dx = dy.copy(), dx.copy()
    start = np.zeros_like(u)
    for length, point in pieces:
        safe = np.where(length > 0.0, length, 1.0)
        f = np.clip((s - start) / safe, 0.0, 1.0)
        py, px = point(f)
        take = (s >= start) & (length > 0.0)
        dy = np.where(take, py, dy)
        dx = np.where(take, px, dx)
        start = start + length
    return dy, dx


def _spiral_corner_gap(t_out, delta, m):
    """Width of the band between the Spiral rings at levels t_out
    and t_out - delta: the larger of delta (the gap along the straight
    sides) and the diameter of the largest circle fitting between the two
    rings at a corner."""
    t_in = np.maximum(t_out - delta, 0.0)
    rho_o = _spiral_corner_radius(t_out, m)
    rho_i = _spiral_corner_radius(t_in, m)
    s2 = np.sqrt(2.0)
    # Circle touching both corner arcs on the diagonal ...
    r1 = (s2 * delta - (s2 - 1.0) * (rho_o - rho_i)) / 2.0
    # ... or, once that circle is wider than the outer arc, sitting in the
    # outer corner against both outer straight sides and the inner arc.
    r2 = (s2 * delta + (s2 - 1.0) * rho_i) / (1.0 + s2)
    r = np.where(r1 <= rho_o, r1, r2)
    return np.maximum(delta, 2.0 * r)


def _spiral_max_spacing(t, m, step_px, iters=40):
    """Largest per-turn level increment at each level t such that the gap
    to the next ring inward is at most step_px along the sides and at most
    _SPIRAL_CORNER_GAP_FACTOR * step_px at the corners (bisection)."""
    t = np.asarray(t, dtype=np.float64)
    corner_cap = _SPIRAL_CORNER_GAP_FACTOR * step_px
    lo = np.zeros_like(t)
    hi = np.full_like(t, float(step_px))
    ok_full = _spiral_corner_gap(t, hi, m) <= corner_cap + 1e-12
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = _spiral_corner_gap(t, mid, m) <= corner_cap
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return np.where(ok_full, float(step_px), np.maximum(lo, 1e-3 * step_px))


def _spiral_px(r_lo, r_hi, c_lo, c_hi, step_px, sense,
               raster_offset_pct=0.0, sample_px=1.0):
    """Outside-in Spiral over the pixel box [r_lo, r_hi] x [c_lo, c_hi]:
    one continuous path inward to the box's centre point (square box) or
    centre segment (oblong box), whose rings morph from the box's own
    rectangle to circles/stadiums -- see _spiral_corner_radius. Turn
    spacing follows _spiral_max_spacing, rescaled so a whole number of
    turns n ends exactly at the centre.

    The path is P(level(s), u) for ring position u (see
    _spiral_ring_points) and a turn coordinate s that runs 0 -> n, level(s)
    falling from the box's outer level at s = 0 to the centre at s = n.
    raster_offset_pct = 100*o shifts the turn coordinate to s = turn + u + o,
    so at every ring position the path sits the fraction o of the way
    between two neighbouring turns of the unshifted path (o = 0.5: midway,
    the same as rotating a linear spiral by 180 degrees). The first turn
    instead uses s = o*u, running from the outer level to level(o): for
    o = 0 that is one closed ring around the outer level, so the box's
    edge is cut completely, and for o > 0 it sits between that ring and
    the unshifted path's first turn, the same way.

    sense : "CW" or "CCW" as displayed, both starting at the top-left
            (CW along the top side first, CCW down the left side first).
            Paths interleave under raster_offset_pct only when they share
            a sense; opposite senses wind in opposite directions and cross.

    Returns
    -------
    (N, 2) float array of (row, col) pixel coordinates, sampled about
    sample_px apart along the path.
    """
    cr, cc = 0.5 * (r_lo + r_hi), 0.5 * (c_lo + c_hi)
    hy, hx = 0.5 * (r_hi - r_lo), 0.5 * (c_hi - c_lo)
    m = min(hx, hy)
    ex, ey = hx - m, hy - m
    step_px = max(float(step_px), 1e-6)
    o = (float(raster_offset_pct) / 100.0) % 1.0

    def perimeter(t):
        rho = _spiral_corner_radius(t, m)
        return 4.0 * (ex + ey + 2.0 * t) - (8.0 - 2.0 * np.pi) * rho

    if m > 0.0:
        t_grid = np.linspace(0.0, m, 2001)
        inv = 1.0 / _spiral_max_spacing(t_grid, m, step_px)
        phi = np.concatenate(([0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1])
                                               * np.diff(t_grid))))
        total = phi[-1]
        n_turns = max(1, int(np.ceil(total - 1e-6)))

        def level(s):
            return np.interp(total * (1.0 - s / n_turns), phi, t_grid)
    else:
        n_turns = 1

        def level(s):
            return np.zeros_like(np.asarray(s, dtype=np.float64))

    # Path coordinate q: the first turn is q in [0, 1), every later turn
    # q in [k, k + 1); u = frac(q) throughout, ending where s reaches n.
    def turn_coord(q):
        return np.where(q < 1.0, o * q, q - 1.0 + o)

    q_end = n_turns + 1.0 - o
    q_parts = []
    for k in range(int(np.ceil(q_end - 1e-9))):
        nk = max(8, int(np.ceil(float(perimeter(level(turn_coord(float(k)))))
                                / sample_px)))
        qk = k + np.arange(nk) / nk
        q_parts.append(qk[qk < q_end - 1e-9])
    q_parts.append(np.array([q_end]))
    q = np.concatenate(q_parts)
    t_all = level(np.minimum(turn_coord(q), n_turns))
    u_all = q - np.floor(q)

    if sense == "CCW":
        u_all = (1.0 - u_all) % 1.0
    dy, dx = _spiral_ring_points(ex, ey, t_all,
                                 _spiral_corner_radius(t_all, m), u_all)
    rows = np.clip(cr + dy, r_lo, r_hi)
    cols = np.clip(cc + dx, c_lo, c_hi)
    return np.column_stack((rows, cols))


def _curve_points(rc, depth, prior_mask, resid_tol,
                  px_per_mm, H, _ox, _oy, max_depth):
    """Sub-pixel counterpart to _line_points for an arbitrary polyline rc
    ((N, 2) float array of (row, col)), gated by prior_mask/mask_spans at
    each sample's nearest pixel exactly as _line_points gates a line. Z is
    sampled bilinearly from depth, the same linear interpolation a G1 move
    between two neighbouring raster samples already makes.

    Returns
    -------
    list of point-lists — [[(x_mm, y_mm, -z_mm), ...], ...] — one element
    for an ungated curve, or several when prior_mask/mask_spans splits it.
    """
    from scipy.ndimage import map_coordinates
    if len(rc) == 0:
        return []
    Hd, Wd = depth.shape
    ri = np.clip(np.rint(rc[:, 0]).astype(np.intp), 0, Hd - 1)
    ci = np.clip(np.rint(rc[:, 1]).astype(np.intp), 0, Wd - 1)
    if prior_mask is not None:
        effective = (depth[ri, ci] * prior_mask[ri, ci]).astype(np.float32)
        spans = mask_spans(effective > resid_tol)
    else:
        spans = [(0, len(rc) - 1)]

    z = map_coordinates(depth.astype(np.float64, copy=False), rc.T,
                        order=1, mode="nearest") * max_depth
    x = rc[:, 1] / px_per_mm - _ox
    y = (H - 1 - rc[:, 0]) / px_per_mm - _oy
    return [list(zip(x[s:e + 1].tolist(), y[s:e + 1].tolist(),
                     (-z[s:e + 1]).tolist()))
            for s, e in spans]


def _is_contiguous_chain(segs, tol_mm=1e-6):
    """True if segs (in list order) form one unbroken chain: each
    segment's last point is within tol_mm of the next segment's first
    point (in XY). Used to detect whether a pass's segments are still a
    single continuous path (e.g. an unfragmented coil or spiral), in which
    case there is no travel to optimise and reordering only risks harm.
    """
    for i in range(len(segs) - 1):
        x0, y0 = segs[i][-1][0], segs[i][-1][1]
        x1, y1 = segs[i + 1][0][0], segs[i + 1][0][1]
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 > tol_mm * tol_mm:
            return False
    return True


def _polylines_inside(pts, safe_mask_f, tol=0.1):
    """(N,) bool: whether each of N sampled polylines, pts of shape
    (N, S, 2) in (row, col) with consecutive samples at most 0.5 px apart,
    stays inside safe_mask_f -- every sample has a bilinearly-interpolated
    mask value of at least 0.5 - tol, the same 0.5 iso-level
    find_contours traces (tol absorbs marching squares' own straight-chord
    approximation of that level set, so a ring vertex and its neighbouring
    edge still count as inside).
    """
    from scipy.ndimage import map_coordinates
    n, n_samples = pts.shape[0], pts.shape[1]
    vals = map_coordinates(safe_mask_f, pts.reshape(-1, 2).T, order=1,
                           mode="constant", cval=0.0)
    return np.all(vals.reshape(n, n_samples) >= 0.5 - tol, axis=1)


def _link_is_inside(p, q, safe_mask_f, tol=0.1):
    """True if the straight segment p -> q (row, col) stays inside
    safe_mask_f, by _polylines_inside's test on samples at most 0.5 px
    apart.

    p and q may also be (N, 2) arrays, checking N segments at once;
    returns an (N,) bool array in that case.
    """
    p = np.atleast_2d(np.asarray(p, dtype=float))
    q = np.atleast_2d(np.asarray(q, dtype=float))
    length = np.hypot(q[:, 0] - p[:, 0], q[:, 1] - p[:, 1])
    n_samples = max(2, int(np.ceil(length.max() / 0.5)) + 1)
    t = np.linspace(0.0, 1.0, n_samples)
    pts = p[:, None, :] + t[None, :, None] * (q - p)[:, None, :]
    inside = _polylines_inside(pts, safe_mask_f, tol)
    return inside if len(inside) > 1 else bool(inside[0])


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _biarc(p0, t0, p1, t1, min_radius, max_sweep=np.radians(120.0)):
    """Tangent-continuous two-arc link from p0 (leaving along unit tangent
    t0) to p1 (arriving along unit tangent t1), vectorized over N pairs
    ((N, 2) arrays, (row, col)).

    Uses the equal-control-length biarc: with d the solution of
    2(1 - t0.t1) d^2 + 2 (v.(t0 + t1)) d - v.v = 0 (v = p1 - p0), the two
    arcs meet at J = ((p0 + d t0) + (p1 - d t1)) / 2, where both share the
    tangent along (p1 - d t1) - (p0 + d t0). Arcs turning the opposite way
    give an S, the same way a C; both are smooth.

    A pair is valid when d > 0, the link heads forward (v.t0 > 0), each
    arc turns by at most max_sweep, and each arc's radius is at least
    min_radius (a straight arc, of infinite radius, always qualifies).

    Returns
    -------
    dict of (N,) arrays: 'valid', 'length', 'joint' (N, 2), 'tj' (N, 2)
    (the shared tangent at the joint), 'half1'/'half2' (each arc's signed
    half sweep angle).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return _biarc_unchecked(p0, t0, p1, t1, min_radius, max_sweep)


def _biarc_unchecked(p0, t0, p1, t1, min_radius, max_sweep):
    v = p1 - p0
    c = np.sum(t0 * t1, axis=-1)
    vt = np.sum(v * (t0 + t1), axis=-1)
    vv = np.sum(v * v, axis=-1)
    a = 2.0 * (1.0 - c)
    parallel = a < 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        d_quad = (-2.0 * vt + np.sqrt(4.0 * vt * vt + 4.0 * a * vv)) / (2.0 * a)
        d_par = vv / (4.0 * np.sum(v * t0, axis=-1))
    d = np.where(parallel, d_par, d_quad)
    q0 = p0 + d[:, None] * t0
    q1 = p1 - d[:, None] * t1
    joint = 0.5 * (q0 + q1)
    tj = q1 - q0
    tj_len = np.hypot(tj[:, 0], tj[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        tj = tj / tj_len[:, None]

    def arc(a_pt, a_tan, b_pt):
        chord = b_pt - a_pt
        half = np.arctan2(_cross2(a_tan, chord), np.sum(a_tan * chord, axis=-1))
        clen = np.hypot(chord[:, 0], chord[:, 1])
        sin_h = np.abs(np.sin(half))
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.where(sin_h > 1e-9, clen / (2.0 * sin_h), np.inf)
            length = np.where(sin_h > 1e-9, radius * 2.0 * np.abs(half), clen)
        return half, radius, length

    half1, r1, len1 = arc(p0, t0, joint)
    half2, r2, len2 = arc(joint, tj, p1)
    valid = (np.isfinite(d) & (d > 1e-9) & (tj_len > 1e-9)
             & (np.sum(v * t0, axis=-1) > 0.0)
             & (2.0 * np.abs(half1) <= max_sweep) & (2.0 * np.abs(half2) <= max_sweep)
             & (r1 >= min_radius) & (r2 >= min_radius))
    return {'valid': valid, 'length': np.where(valid, len1 + len2, np.inf),
            'joint': joint, 'tj': tj, 'half1': half1, 'half2': half2}


def _sample_biarc(p0, t0, p1, arc, spacing=0.5):
    """Sample the biarcs described by _biarc's result `arc` (restricted to
    the same N pairs as p0/t0/p1) into (N, S, 2) points at most `spacing`
    px apart, from p0 through the joint to p1 inclusive."""
    def sample_arc(a_pt, a_tan, b_pt, half, n):
        u = np.linspace(0.0, 1.0, n)[None, :]
        chord = b_pt - a_pt
        clen = np.hypot(chord[:, 0], chord[:, 1])
        normal = np.stack([-a_tan[:, 1], a_tan[:, 0]], axis=-1)
        straight = np.abs(half) < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            r_signed = np.where(straight, 0.0, clen / (2.0 * np.sin(half)))
        phi = 2.0 * half[:, None] * u
        curved = (a_pt[:, None, :]
                  + r_signed[:, None, None] * (np.sin(phi)[..., None] * a_tan[:, None, :]
                                               + (1.0 - np.cos(phi))[..., None] * normal[:, None, :]))
        linear = a_pt[:, None, :] + u[..., None] * chord[:, None, :]
        return np.where(straight[:, None, None], linear, curved)

    total = np.max(np.where(np.isfinite(arc['length']), arc['length'], 0.0))
    n = max(2, int(np.ceil(total / spacing)) + 1)
    first = sample_arc(p0, t0, arc['joint'], arc['half1'], n)
    second = sample_arc(arc['joint'], arc['tj'], p1, arc['half2'], n)
    return np.concatenate([first, second[:, 1:, :]], axis=1)


def _ring_seam_candidates(ring, spacing_px, max_cands):
    """Seam candidates for one onion-skin ring, as (body, entry_idx,
    exit_idx): body is the ring's own vertex array (a closed ring's
    duplicate closing vertex dropped), and entry_idx[i]/exit_idx[i] are
    the body indices the chain enters and leaves the ring on for its i-th
    candidate seam.

    A closed ring is entered and left on the same vertex, so its
    candidates are vertices spaced about spacing_px apart in arc length
    (widened so there are at most max_cands of them). An open ring can
    only be traversed end to end, so its two candidates are its two ends,
    each leaving on the opposite end.
    """
    is_closed = np.linalg.norm(ring[0] - ring[-1]) < 1.5
    if not is_closed:
        return ring, np.array([0, len(ring) - 1]), np.array([len(ring) - 1, 0])
    body = ring[:-1] if np.hypot(*(ring[0] - ring[-1])) < 1e-6 else ring
    edge = np.hypot(*np.diff(body, axis=0).T)
    cum = np.concatenate(([0.0], np.cumsum(edge)))
    spacing = max(spacing_px, cum[-1] / max_cands, 1e-9)
    idx = np.unique(np.searchsorted(cum, np.arange(0.0, cum[-1] + 1e-9, spacing)))
    idx = idx[idx < len(body)]
    return body, idx, idx


def _ring_tangents(body, idx, window_px):
    """Incoming and outgoing unit tangents of a closed ring at vertices
    idx, as two (len(idx), 2) arrays: incoming is the chord from the
    vertex window_px of arc length behind idx to idx (the direction the
    chain arrives in after a full lap ending there), outgoing the chord
    from idx to the vertex window_px ahead (the direction it sets off in).
    Measuring over window_px rather than a single edge smooths out
    find_contours' pixel stair-steps, and the two one-sided chords keep a
    seam on a ring corner well defined."""
    n = len(body)
    nxt = np.roll(body, -1, axis=0)
    edge = np.hypot(nxt[:, 0] - body[:, 0], nxt[:, 1] - body[:, 1])
    perim = float(np.sum(edge))
    window = min(window_px, 0.45 * perim)
    cum = np.concatenate(([0.0], np.cumsum(edge)))           # len n + 1
    tiled = np.concatenate([cum[:-1] - perim, cum[:-1], cum[:-1] + perim])
    s = cum[idx]
    ahead = np.searchsorted(tiled, s + window, side="left") % n
    behind = (np.searchsorted(tiled, s - window, side="right") - 1) % n

    def unit(vec):
        length = np.hypot(vec[:, 0], vec[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            return vec / length[:, None]
    return unit(body[idx] - body[behind]), unit(body[ahead] - body[idx])


def _spine_detour(walk, i):
    """The out-and-back path along a spine walk (_spine_walks) starting
    and ending at its point i: on along the walk to its end, back along
    it to its start, and on to point i again."""
    walk = np.asarray(walk, dtype=float)
    loop = np.vstack([walk, walk[-2:0:-1]])
    return np.vstack([np.roll(loop, -i, axis=0), loop[i:i + 1]])


def _nearest_inside_link(walk, rings, safe_mask_f, max_len=np.inf, n_checked=32):
    """(ring, ring vertex index, walk index) of the shortest straight link
    between a point of walk and a vertex of one of rings that stays
    inside safe_mask_f and is at most max_len long, or None. The
    n_checked shortest of each walk point's nearest-vertex links are
    checked."""
    from scipy.spatial import cKDTree
    pairs = []
    for ri, ring in enumerate(rings):
        d, vi = cKDTree(ring).query(walk)
        pairs.extend(zip(d, [ri] * len(walk), vi, range(len(walk))))
    pairs = sorted((p for p in pairs if p[0] <= max_len), key=lambda p: p[0])[:n_checked]
    if not pairs:
        return None
    ok = np.atleast_1d(_link_is_inside(
        np.array([walk[wi] for _, _, _, wi in pairs]),
        np.array([rings[ri][vi] for _, ri, vi, _ in pairs]), safe_mask_f))
    for (_, ri, vi, wi), good in zip(pairs, ok):
        if good:
            return rings[ri], int(vi), int(wi)
    return None


def _splice_detours(lap, at, ring_detours):
    """lap, a ring's traversal, with every detour cut in at its ring
    vertex: (vertex, loop) pairs, the vertex as an index into the ring's
    own vertex array (at maps it to its position in lap), the loop the
    out-and-back path along a spine from its attachment point
    (_spine_detour), reached from the vertex and left back to it by a
    straight move each."""
    if not ring_detours:
        return lap
    pieces, start = [], 0
    for pos, loop in sorted(((at(v), loop) for v, loop in ring_detours),
                            key=lambda item: item[0]):
        pieces.append(lap[start:pos + 1])
        pieces.append(loop)
        start = pos
    pieces.append(lap[start:])
    return np.vstack(pieces)


def _stitch_onion_rings(rings, safe_mask_f, step_px, max_cands=200,
                        max_checked_pairs=1024, s_links_per_exit=3, detours=None):
    """Stitch one onion-skin track's rings (one per level, already in
    visiting order and winding sense; see _chain_onion_rings) into
    continuous chain(s), choosing every ring's seam -- the vertex it is
    entered and left on -- jointly over the whole track.

    Every ring is traversed in full, from its seam back round to it, and
    consecutive rings are joined by a link from one ring's exit to the
    next ring's entry. A link is preferably an S-shaped biarc (_biarc):
    one arc leaving the finished ring tangentially, one arriving
    tangentially on the next, each of radius at least step_px, so the
    tool changes ring without the two sharp turns of a straight chord.
    Rings about one step_px apart need the entry to sit roughly
    sqrt(3) * step_px further along the next ring than the exit, for arcs
    of that radius. Where no such biarc fits (tangents pointing apart, a
    ring too small for the radius), a straight link is used instead, at a cost
    penalty of 8 * step_px so a biarc is always preferred where one
    exists.

    The seams minimise the track's total link cost, solved exactly over
    each ring's seam candidates (_ring_seam_candidates) as a shortest path
    through a layered graph -- one layer per ring, one node per
    candidate. A globally chosen seam line routes itself along the
    region's straight flanks, where the gap between consecutive rings is
    about one step_px, and away from acute tips and vanished protrusions,
    where it opens up much wider; a greedy nearest-point choice made one
    ring at a time cannot see that coming.

    Only links that stay inside safe_mask_f (_polylines_inside) are
    allowed. For each pair of consecutive rings, biarcs are considered
    for pairs within 4 * step_px of each other, and the s_links_per_exit
    shortest valid ones per exit candidate are checked; straight links
    are considered for the max_checked_pairs shortest pairs, all checked.
    Every other pair is excluded. If no pair between two rings is allowed,
    the track is split there into separate chains, left for Stage 6
    ordering and linking to join.

    detours maps id(ring) to (ring, detours of that ring): (vertex, loop)
    pairs that _splice_detours cuts into the ring's traversal. Each entry
    is taken out of detours as its ring is cut. A detour leaves the ring
    and returns to it at the same vertex, so detours play no part in
    choosing seams.

    Returns
    -------
    list of Nx2 (row, col) chains, in visiting order.
    """
    if not rings:
        return []
    spacing_px = max(1.0, step_px / 4.0)
    window_px = max(2.0, step_px / 2.0)
    min_radius = float(step_px)
    straight_penalty = 8.0 * step_px
    cands = [_ring_seam_candidates(r, spacing_px, max_cands) for r in rings]

    def tangents(j):
        body, entry_idx, exit_idx = cands[j]
        if not np.array_equal(entry_idx, exit_idx):   # open ring: straight links only
            nan = np.full((len(entry_idx), 2), np.nan)
            return nan, nan
        return _ring_tangents(body, entry_idx, window_px)

    tans = [tangents(j) for j in range(len(rings))]

    def s_link_pairs(j, d):
        """Candidate (ia, ib) index pairs, and their biarc, for a biarc
        link from ring j to ring j+1: the s_links_per_exit shortest valid
        biarcs per exit candidate."""
        body_a, _, exit_a = cands[j]
        body_b, entry_b, _ = cands[j + 1]
        ia, ib = np.nonzero(d <= 4.0 * step_px)
        if len(ia) == 0:
            return None
        t_out = tans[j][0][ia]            # incoming tangent at ring j's exit
        t_in = tans[j + 1][1][ib]         # outgoing tangent at ring j+1's entry
        arc = _biarc(body_a[exit_a][ia], t_out, body_b[entry_b][ib], t_in, min_radius)
        keep = np.nonzero(arc['valid'])[0]
        if len(keep) == 0:
            return None
        # s_links_per_exit shortest per exit candidate ia
        order = np.lexsort((arc['length'][keep], ia[keep]))
        keep = keep[order]
        rank = np.arange(len(keep)) - np.searchsorted(ia[keep], ia[keep], side="left")
        keep = keep[rank < s_links_per_exit]
        sub = {k: v[keep] for k, v in arc.items()}
        return ia[keep], ib[keep], t_out[keep], sub

    def link_costs(j):
        body_a, _, exit_a = cands[j]
        body_b, entry_b, _ = cands[j + 1]
        pa, pb = body_a[exit_a], body_b[entry_b]
        d = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
        cost = np.full(d.shape, np.inf)
        is_s = np.zeros(d.shape, dtype=bool)

        flat = d.ravel()
        k = min(max_checked_pairs, flat.size)
        shortest = np.argpartition(flat, k - 1)[:k]
        ia, ib = np.unravel_index(shortest, d.shape)
        ok = np.atleast_1d(_link_is_inside(pa[ia], pb[ib], safe_mask_f))
        cost[ia[ok], ib[ok]] = d[ia[ok], ib[ok]] + straight_penalty

        s_pairs = s_link_pairs(j, d)
        if s_pairs is not None:
            ia, ib, t_out, arc = s_pairs
            pts = _sample_biarc(pa[ia], t_out, pb[ib], arc)
            ok = _polylines_inside(pts, safe_mask_f)
            better = ok & (arc['length'] < cost[ia, ib])
            cost[ia[better], ib[better]] = arc['length'][better]
            is_s[ia[better], ib[better]] = True
        return cost, is_s

    def link_points(j, a, b):
        """Interior points of the chosen biarc from ring j's exit
        candidate a to ring j+1's entry candidate b."""
        body_a, _, exit_a = cands[j]
        body_b, entry_b, _ = cands[j + 1]
        p0 = body_a[exit_a[a]][None, :]
        p1 = body_b[entry_b[b]][None, :]
        t0 = tans[j][0][a][None, :]
        arc = _biarc(p0, t0, p1, tans[j + 1][1][b][None, :], min_radius)
        return _sample_biarc(p0, t0, p1, arc)[0, 1:-1]

    # Forward pass. segments collects (first_ring, last_ring, backpointers,
    # per-link biarc flags, final accumulated costs) for each run of rings
    # that stays linkable end to end.
    segments = []
    seg_start = 0
    acc = np.zeros(len(cands[0][1]))
    back, kinds = [], []
    for j in range(len(rings) - 1):
        cost, is_s = link_costs(j)
        tot = acc[:, None] + cost
        best = np.argmin(tot, axis=0)
        nxt = tot[best, np.arange(tot.shape[1])]
        if not np.any(np.isfinite(nxt)):
            segments.append((seg_start, j, back, kinds, acc))
            seg_start, back, kinds = j + 1, [], []
            acc = np.zeros(len(cands[j + 1][1]))
        else:
            back.append(best)
            kinds.append(is_s)
            acc = nxt
    segments.append((seg_start, len(rings) - 1, back, kinds, acc))

    chains = []
    for first, last, back, kinds, acc in segments:
        seam = [int(np.argmin(acc))]
        for bp in reversed(back):
            seam.append(int(bp[seam[-1]]))
        seam.reverse()
        parts = []
        for j, c in zip(range(first, last + 1), seam):
            if j > first:
                a = seam[j - first - 1]
                if kinds[j - first - 1][a, c]:
                    parts.append(link_points(j - 1, a, c))
            body, entry_idx, exit_idx = cands[j]
            e = int(entry_idx[c])
            if exit_idx[c] == e:
                lap = np.vstack([np.roll(body, -e, axis=0), body[e:e + 1]])
                at = lambda v: (v % len(body) - e) % len(body)
            else:
                lap = body if e == 0 else body[::-1]
                at = lambda v: v if e == 0 else len(body) - 1 - v
            entry = detours.pop(id(rings[j]), None) if detours else None
            ring_detours = entry[1] if entry is not None and entry[0] is rings[j] else []
            parts.append(_splice_detours(lap, at, ring_detours))
        chains.append(np.vstack(parts))
    return chains


def _spine_walks(spine_mask, min_spur_px=3.0, smooth_iters=3):
    """One continuous open walk per 8-connected component of spine_mask, as
    Nx2 (row, col) float polylines through its pixel centres.

    A component's pixels are joined into a tree (the minimum spanning tree
    of their 8-neighbour graph, edges weighted by length, which also breaks
    any loop open). The walk runs along the tree's longest path, from one
    end to the other, and detours out and back along every side branch on
    the way, each side branch visited before the path continues. Side
    branches reaching no more than min_spur_px from the longest path are
    left out: they are pixel noise of the medial axis, and the longest path
    clears everything they would within that distance anyway. The tree's
    nodes are then smoothed smooth_iters times, each node except the
    tree's ends moving halfway to the mean of its neighbours, which keeps
    every point well within a pixel of the medial axis. A single-pixel
    component gives a two-point walk on that pixel.
    """
    from scipy.ndimage import label
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra

    labeled, n = label(spine_mask, structure=np.ones((3, 3), dtype=bool))
    walks = []
    for cid in range(1, n + 1):
        rows, cols = np.nonzero(labeled == cid)
        pts = np.column_stack([rows, cols]).astype(float)
        m = len(pts)
        if m == 1:
            walks.append(np.vstack([pts, pts]))
            continue
        index = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
        ii, jj, ww = [], [], []
        for i, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
            for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
                j = index.get((r + dr, c + dc))
                if j is not None:
                    ii.append(i)
                    jj.append(j)
                    ww.append(np.hypot(dr, dc))
        graph = coo_matrix((ww, (ii, jj)), shape=(m, m)).tocsr()
        tree = minimum_spanning_tree(graph)
        tree = (tree + tree.T).tocsr()

        dist0 = dijkstra(tree, directed=False, indices=0)
        a = int(np.argmax(dist0))
        dist_a, pred = dijkstra(tree, directed=False, indices=a,
                                return_predecessors=True)
        b = int(np.argmax(dist_a))
        main = [b]
        while main[-1] != a:
            main.append(int(pred[main[-1]]))
        main.reverse()                                # a ... b
        on_main = np.zeros(m, dtype=bool)
        on_main[main] = True

        # Each node's reach: how far its subtree (hanging off the main
        # path, away from it) extends below it.
        children = [[] for _ in range(m)]
        order = []
        stack = list(main)
        seen = on_main.copy()
        while stack:
            u = stack.pop()
            order.append(u)
            for v, w in zip(tree.indices[tree.indptr[u]:tree.indptr[u + 1]],
                            tree.data[tree.indptr[u]:tree.indptr[u + 1]]):
                if not seen[v]:
                    seen[v] = True
                    children[u].append((int(v), float(w)))
                    stack.append(int(v))
        reach = np.zeros(m)
        for u in reversed(order):
            for v, w in children[u]:
                reach[u] = max(reach[u], reach[v] + w)

        walk = []

        def visit_side(u):
            # Iterative out-and-back traversal of u's kept side branches.
            todo = [(u, iter(children[u]))]
            while todo:
                node, it = todo[-1]
                for v, w in it:
                    if reach[v] + w > min_spur_px:
                        walk.append(v)
                        todo.append((v, iter(children[v])))
                        break
                else:
                    todo.pop()
                    if todo:
                        walk.append(todo[-1][0])

        for u in main:
            walk.append(u)
            visit_side(u)

        # Laplacian smoothing over the tree, its ends pinned, takes out the
        # one-pixel zig-zag of a ridge that runs between two pixel rows.
        deg = np.diff(tree.indptr)
        free = deg > 1
        inv_deg = 1.0 / np.maximum(deg, 1)
        for _ in range(smooth_iters):
            avg = (tree.astype(bool).astype(float) @ pts) * inv_deg[:, None]
            pts[free] = 0.5 * (pts[free] + avg[free])
        walks.append(pts[walk])
    return walks


def _chain_onion_rings(rings, ring_nodes, node_parents, loose, safe_mask_f,
                       step_px, detours):
    """Join an onion-skin region's rings into chains along its level graph
    (_LevelGraph), and say which chain must be cut after which.

    rings        : every ring, level by level, outermost level first.
    ring_nodes   : each ring's node in the level graph.
    node_parents : the level graph's parents.
    loose        : (spine chain, rings of its own level in its branch) for
                   each spine that is a chain of its own.

    A ring's parent rings are the rings of its node's parent nodes, looking
    through nodes that hold none. A track is a run of rings, one per level,
    each the only child of the ring before it, which is its only parent:
    the outer boundary of an undivided region shrinking inward, or a hole's
    boundary growing outward. A track ends where material merges (two
    holes, or a hole and the outside, meeting: the merged ring has several
    parents) or air splits (the region forking: the ring has several
    children), and the rings after that start tracks of their own. Each
    track is stitched into chain(s) by _stitch_onion_rings, each cut after
    the one before it where the stitcher split it; a track's first chain
    is cut after the last chain of each of its parent tracks. A loose
    spine is cut after the last chain of each track holding a ring of its
    own level in its branch.

    Returns (chains, parents): (row, col) polylines in the padded frame,
    each running outside-in, and per chain the chains it is cut after.
    """
    node_rings = [[] for _ in node_parents]
    for i, n in enumerate(ring_nodes):
        node_rings[n].append(i)
    memo = {}

    def rings_above(n):
        if n not in memo:
            above = set()
            for p in node_parents[n]:
                above.update(node_rings[p] if node_rings[p] else rings_above(p))
            memo[n] = sorted(above)
        return memo[n]

    ring_parents = [rings_above(n) for n in ring_nodes]
    n_children = [0] * len(rings)
    for ps in ring_parents:
        for p in ps:
            n_children[p] += 1
    # Rings come level by level, so a ring's parents precede it.
    track_of = [0] * len(rings)
    tracks, track_parents = [], []
    for i, ps in enumerate(ring_parents):
        if len(ps) == 1 and n_children[ps[0]] == 1:
            tracks[track_of[ps[0]]].append(i)
            track_of[i] = track_of[ps[0]]
        else:
            track_of[i] = len(tracks)
            tracks.append([i])
            track_parents.append(sorted({track_of[p] for p in ps}))

    chains, parents = [], []
    track_end = []   # per track: the chains a chain after it is cut after
    for t, ids in enumerate(tracks):
        after = sorted({c for p in track_parents[t] for c in track_end[p]})
        for chain in _stitch_onion_rings([rings[i] for i in ids], safe_mask_f,
                                         step_px, detours=detours):
            chains.append(chain)
            parents.append(after)
            after = [len(chains) - 1]
        track_end.append(after)

    index = {id(r): i for i, r in enumerate(rings)}
    for chain, own in loose:
        chains.append(chain)
        parents.append(sorted({c for r in own
                               for c in track_end[track_of[index[id(r)]]]}))
    return chains, parents


def _onion_skin_chains(region_mask, step_px, min_area_px, reverse_rings=False,
                       tool_radius_px=None, return_parents=False):
    """Concentric-ring toolpath chain(s) clearing one connected pocket
    region: erosion level k is the set of region pixels farther than
    k * step_px from the region's outside (one Euclidean distance transform
    of region_mask, thresholded at each level), and each level's
    boundary is extracted via find_contours as its rings: its outer
    boundary and one ring around each hole. The rings are joined level by
    level into tracks along the level graph (_chain_onion_rings): the
    outer boundary shrinking inward is one track, each hole's boundary
    growing outward another, until boundaries meet or the region forks.
    Each track's consecutive rings are stitched into one continuous chain
    with _stitch_onion_rings, which picks every ring's seam jointly for
    the cheapest total
    ring-to-ring links, joins rings with a tangent-continuous S-shaped
    biarc wherever one fits (a straight link otherwise), and only ever
    uses a link that stays inside region_mask. Every pixel on erosion
    level k+1 has its disk(step_px) inside level k (its distance to the
    outside exceeds (k+1) * step_px, so every pixel of that disk lies
    farther than k * step_px from the outside), so from every point of a level k+1
    ring a link of about step_px back to level k's ring exists and lies
    within that disk, i.e. inside the region. Level 1 is exactly
    region_mask eroded by disk(step_px), and every level is the region's
    Euclidean inward offset by k * step_px to within a pixel. The same
    holds for a hole's ring at level k+1 and its ring at level k. Every
    ring is emitted the moment its mask is non-empty (before the area/erosion check below),
    so a region smaller than one step_px still gets its own single
    perimeter ring rather than nothing.

    reverse_rings reverses every ring's own point order right after
    find_contours (same "uncut material stays on the tool's right of
    travel" convention as _extract_z_contours' reverse_contours), so the
    climb/conventional winding sense of the whole chain is decided once
    here, before stitching -- seam selection never changes a ring's
    winding, only which of its points the chain enters on.

    If an erosion iteration's connected-component count rises above 1, any
    component smaller than min_area_px is first discarded as a stray
    fragment -- erosion detaching a sliver at a concave cusp of the
    boundary rather than a genuine second lobe -- since it would otherwise
    become its own tiny standalone toolpath segment. Only once that leaves
    more than one substantial component does this count as a genuine fork
    (a dumbbell-shaped neck fully eroded away): each surviving component
    continues as its own independent branch, and its rings start tracks of
    their own -- no attempt is made to bridge across the split, since that
    would re-cross material the tool has already cleared. If filtering
    leaves at most one substantial component, the branch simply continues
    with it: a dropped fragment takes no ring, so it never splits an
    otherwise-ordinary lobe's track in two. All branches are advanced one
    erosion level at a time, so each level's rings go into the level graph
    together. The chains are listed in no cutting order of their own;
    return_parents gives the order they must be cut in.

    With tool_radius_px given, rings that leave material standing get
    spines. A ring at level j cuts everything within the tool radius R of
    it, and the next level's ring everything within R of that, so the
    rings alone clear the region wherever every point lies within R of
    one of them. Where the next level doesn't reach -- the inside of a
    branch's last ring, a finger of a ring that the next level stops
    short of, or a stray fragment discarded above -- points deeper than
    j * step_px + R stay uncut. With a step-over wider than R that band
    is a strip along the region's medial axis; _level_spines finds it and
    covers it with medial axis walks (_spine_walks). Each walk is spliced
    into a ring as an out-and-back detour (attach_spine), joined to it
    by one short straight move each way: into the next level's ring,
    whose cusp the medial axis runs into at the spine's root, or, where
    there is none within two tool radii, into its own level's ring. A
    ring carrying several spines simply makes several detours, and the
    rings' seams are chosen without regard to them. Without
    tool_radius_px there are no spines.

    region_mask is padded by one False pixel on every side before any
    processing (compensated for in the returned coordinates), since the
    caller crops each region to its tight bounding box, which touches the
    mask on every side by definition: find_contours needs at least a 2x2
    array (a region only 1 pixel tall or wide would otherwise crash it),
    and even for a larger region, without the margin the outermost ring
    would be clipped into an open contour at the crop edge instead of the
    true closed boundary.

    Returns
    -------
    list of Nx2 (row, col) float pixel-space polylines, each running
    outside-in, in region_mask's own (unpadded) coordinate frame. An
    undivided island without holes is one chain unless _stitch_onion_rings
    had to split it where no inside link exists. With return_parents,
    (chains, parents) instead: parents[i] lists the chains that chain i
    must be cut after, cutting outside-in (see _chain_onion_rings). A
    chain where boundaries merge has several parents.
    """
    from scipy.ndimage import label, distance_transform_edt, binary_erosion
    from skimage.measure import find_contours

    padded = np.pad(region_mask, 1, mode="constant", constant_values=False)
    safe_mask_f = padded.astype(np.float32)
    # Squared distance of every pixel to the nearest outside pixel, rounded
    # to the exact integer it is; level k is where it exceeds (k*step_px)^2.
    dist2 = np.rint(np.square(distance_transform_edt(padded)))
    branches = [{'mask': padded, 'level': 0}]
    # Every ring of every level goes into one level graph: air is the
    # level's mask, material everything outside it, holes included.
    graph = _LevelGraph()
    all_rings, ring_nodes = [], []
    # Spines that are chains of their own, each with its own level's rings
    # in its branch.
    loose = []
    detours = {}

    def extract_rings(mask):
        level = [r for r in find_contours(mask.astype(np.float32), 0.5) if len(r) >= 3]
        return [r[::-1] for r in level] if reverse_rings else level

    # Spines clear what the rings leave standing: with a step-over wider
    # than the tool radius, the ridge of an area the next level doesn't
    # reach; at any step-over, a stray fragment the next level drops.
    spine_r = float(tool_radius_px) if tool_radius_px is not None else None
    dist = np.sqrt(dist2) if spine_r is not None else None
    skeleton, nearest_out = [], []

    def attach_spine(walk, inner_rings, own_rings):
        """Splice walk into a ring as an out-and-back detour
        (_spine_detour), linked at the ring vertex and spine point nearest
        each other by a straight move that stays inside the region:
        preferably on the next level's rings (inner_rings), where the
        medial axis the spine follows runs into a cusp of the ring, within
        two tool radii; otherwise on its own level's rings (own_rings),
        which every point of the spine lies within one step-over of. With
        no inside link to either, the spine is a chain of its own, added
        to loose with own_rings."""
        found = _nearest_inside_link(walk, inner_rings, safe_mask_f,
                                     max_len=2.0 * spine_r + 1.0)
        if found is None:
            found = _nearest_inside_link(walk, own_rings, safe_mask_f)
        if found is None:
            loose.append((_spine_detour(walk, 0), own_rings))
            return
        ring, vi, wi = found
        # Each entry holds its ring, so that the ring stays alive (and its
        # id unused by any other ring) until the stitcher takes the entry.
        detours.setdefault(id(ring), (ring, []))[1].append((vi, _spine_detour(walk, wi)))

    def _reaches(cand, inner, j, reach_r):
        """The pixels of cand within reach_r of inner, the next level's
        kept components (whose rings lie within half a pixel of their
        pixels, which reach_r allows for). Most are found along their own
        ray: from a pixel's nearest outside point through the pixel, the
        distance to the outside grows at unit rate, so the ray enters the
        next level t = (j + 1) * step_px - dist past the pixel; where the
        pixel one further along lies in inner and that is within reach_r,
        so is inner. The rest -- near the ridge, where the ray stops
        rising before that level -- are measured exactly, against inner's
        boundary pixels."""
        from scipy.spatial import cKDTree
        if not nearest_out:
            nearest_out.append(distance_transform_edt(
                padded, return_distances=False, return_indices=True))
        pr, pc = np.nonzero(cand)
        nr, nc = nearest_out[0][0][pr, pc], nearest_out[0][1][pr, pc]
        d = dist[pr, pc]
        t = (j + 1) * step_px - d + 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            qr = np.rint(pr + (pr - nr) * t / d).astype(np.intp)
            qc = np.rint(pc + (pc - nc) * t / d).astype(np.intp)
        ok = ((t <= reach_r) & (qr >= 0) & (qr < inner.shape[0])
              & (qc >= 0) & (qc < inner.shape[1]))
        hit = np.zeros(len(pr), dtype=bool)
        hit[ok] = inner[qr[ok], qc[ok]]
        rest = np.nonzero(~hit)[0]
        if len(rest):
            edge = inner & ~binary_erosion(inner)
            tree = cKDTree(np.column_stack(np.nonzero(edge)))
            near, _ = tree.query(np.column_stack([pr[rest], pc[rest]]),
                                 distance_upper_bound=reach_r)
            hit[rest] = near <= reach_r
        out = np.zeros_like(cand)
        out[pr[hit], pc[hit]] = True
        return out

    def _level_spines(cur, inner_masks, j):
        """Spine walks clearing what the rings of cur (level j, at distance
        j * step_px from the outside) and of inner_masks (the next level's
        kept components) leave standing in cur. A pixel of cur is
        dist - j * step_px from cur's own ring (along its ray to the nearest
        outside point, which stays in cur). Every deeper ring lies inside
        the next level's kept components, beyond their own rings, so a
        pixel outside those components stays uncut where both its distance
        to cur's ring and to theirs exceed spine_r. Above a step-over of
        spine_r, that band runs along the region's medial
        axis, and the medial axis pixels inside it are its spine. Pixels
        of the band farther than spine_r from those (beside a spine's end)
        add the nearest medial axis pixel, or, where none is within reach,
        their own ridge point: their component's deepest pixel."""
        # Rings are traced to within half a pixel of their level, so a
        # pixel only counts as uncut beyond spine_r plus that half pixel.
        reach_r = spine_r + 0.5
        cand = cur & (dist - j * step_px > reach_r)
        for m in inner_masks:
            cand &= ~m
        if not np.any(cand):
            return []
        if inner_masks:
            inner = np.logical_or.reduce(inner_masks)
            cand &= ~_reaches(cand, inner, j, reach_r)
            if not np.any(cand):
                return []
        # Everything below runs in a window around what is left uncut,
        # wide enough to hold every medial axis pixel that could reach it.
        rows, cols = np.nonzero(cand)
        margin = int(np.ceil(reach_r)) + 2
        r0, c0 = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
        win = (slice(r0, rows.max() + margin + 1), slice(c0, cols.max() + margin + 1))
        uncut = cand[win]
        if not skeleton:
            from skimage.morphology import medial_axis
            # A fixed rng: medial_axis breaks ties in random order.
            skeleton.append(medial_axis(padded, rng=0))
        skel = skeleton[0][win] & cur[win]
        spine = skel & uncut
        resid = uncut
        if np.any(spine):
            resid = uncut & (distance_transform_edt(~spine) > spine_r)
        if np.any(resid):
            if np.any(skel):
                d_skel, (sr, sc) = distance_transform_edt(~skel, return_indices=True)
                reach = resid & (d_skel <= spine_r)
                spine[sr[reach], sc[reach]] = True
            else:
                reach = np.zeros_like(resid)
            far = resid & ~reach
            if np.any(far):
                far_lab, n_far = label(far, structure=np.ones((3, 3), dtype=bool))
                win_dist = dist[win]
                for cid in range(1, n_far + 1):
                    comp_d = np.where(far_lab == cid, win_dist, -1.0)
                    spine[np.unravel_index(np.argmax(comp_d), comp_d.shape)] = True
        return [w + (r0, c0) for w in _spine_walks(spine)]

    while branches:
        next_round = []
        level_mask = np.zeros_like(padded)
        level_rings = []
        for br in branches:
            cur, k = br['mask'], br['level'] + 1

            level = br['level_rings'] if 'level_rings' in br else extract_rings(cur)
            level_mask |= cur
            level_rings.extend(level)
            if not level:
                continue

            if np.sum(cur) < min_area_px:
                surviving = []
            else:
                # Level k within this branch: a pixel's disk(step_px) is
                # 4-connected and lies inside level k-1, so it lies inside
                # the pixel's own component there, the branch's mask.
                nxt = cur & (dist2 > float(k * step_px) ** 2)
                labeled, n_comp = label(nxt)
                if n_comp <= 1:
                    surviving = [nxt] if n_comp else []
                else:
                    # A split can throw off a stray fragment of just a few
                    # pixels -- erosion detaching a sliver at a concave
                    # cusp of the boundary rather than a genuine second
                    # lobe. Filter those out *before* deciding whether this
                    # erosion step is a genuine fork: discarding noise must
                    # not, by itself, force this branch's chain to close
                    # off and restart, or a single ordinary lobe that
                    # merely sheds a stray pixel would get split into two
                    # chains for no real reason.
                    surviving = [labeled == cid for cid in range(1, n_comp + 1)
                                 if np.sum(labeled == cid) >= min_area_px]

            nexts = [{'mask': comp_mask, 'level': k} for comp_mask in surviving]
            if spine_r is not None:
                # The next level's rings are extracted here, so that this
                # level's spines can be attached to them.
                inner = []
                for nx in nexts:
                    nx['level_rings'] = extract_rings(nx['mask'])
                    inner.extend(nx['level_rings'])
                for walk in _level_spines(cur, surviving, k - 1):
                    attach_spine(walk, inner, level)
            next_round.extend(nexts)
        all_rings.extend(level_rings)
        ring_nodes.extend(graph.add_level(~level_mask, level_rings))
        branches = next_round

    chains, parents = _chain_onion_rings(all_rings, ring_nodes, graph.parents,
                                         loose, safe_mask_f, step_px, detours)
    chains = [c - 1.0 for c in chains]
    return (chains, parents) if return_parents else chains


def _ring_chain_to_points(chain_px, z_mm, px_per_mm, H, _ox, _oy):
    """Convert one onion-skin pixel-space chain (row, col) to
    [(x_mm, y_mm, -z_mm), ...] at the pocket's flat floor depth. Every
    point of an onion-skin chain already lies inside the pocket mask by
    construction (it comes from repeatedly eroding that same mask), so no
    mask_spans clipping is needed the way _line_points needs it for
    straight raster/coil lines.
    """
    return [(col / px_per_mm - _ox, (H - 1 - row) / px_per_mm - _oy, -z_mm)
            for row, col in chain_px]


def _region_contour_rings(region_mask_local):
    """One pocket region's own boundary contour ring(s) -- one outer ring,
    plus one ring per interior hole/island, with no erosion (unlike
    _onion_skin_chains, which extracts a fresh ring at every erosion step).

    Same padding technique _onion_skin_chains uses (algorithms.py): pad by
    one False pixel on every side before find_contours, since the caller
    always crops region_mask_local to its own tight bounding box, which
    touches the mask on every side by definition -- without the margin the
    boundary would be clipped into an open contour at the crop edge instead
    of the true closed ring.

    Returns
    -------
    list of Nx2 (row, col) float pixel-space rings, in region_mask_local's
    own (unpadded) coordinate frame.
    """
    from skimage.measure import find_contours
    padded = np.pad(region_mask_local, 1, mode="constant", constant_values=False)
    rings = [r - 1.0 for r in find_contours(padded.astype(np.float32), 0.5)
             if len(r) >= 3]
    return rings


@dataclasses.dataclass
class _ContourRing:
    """One pocket region's boundary ring, ready for arc-length lookups
    (Stage 6's Zig-Zag contour-linking)."""
    pts_px: np.ndarray   # Nx2 (row, col) float, full-canvas coordinates
    cum: np.ndarray      # cum[i] = arc length from vertex 0 to vertex i, walking
                          # forward (increasing index); len == pts_px
    L: float             # ring's total perimeter (px), including the closing
                          # edge from the last vertex back to vertex 0


def _make_contour_ring(pts_px):
    """Build a _ContourRing from a closed (row, col) polyline as returned by
    find_contours (ring[0] == ring[-1]): drops that duplicate closing vertex
    so pts_px becomes a proper N-vertex cyclic array, then precomputes the
    per-vertex forward arc length and total perimeter (including the
    closing edge back to vertex 0) that _nearest_ring_vertex/
    _ring_arc_between's modular arithmetic relies on."""
    if len(pts_px) > 1 and np.hypot(*(pts_px[0] - pts_px[-1])) < 1e-6:
        pts_px = pts_px[:-1]
    next_pts = np.roll(pts_px, -1, axis=0)
    edge_len = np.hypot(next_pts[:, 0] - pts_px[:, 0], next_pts[:, 1] - pts_px[:, 1])
    cum = np.concatenate(([0.0], np.cumsum(edge_len)[:-1]))
    return _ContourRing(pts_px=pts_px, cum=cum, L=float(np.sum(edge_len)))


def _nearest_ring_vertex(ring, row, col, tol_px):
    """Index of the ring vertex nearest (row, col), or None if every vertex
    is further than tol_px away.

    Snaps to the nearest existing vertex rather than projecting onto the
    nearest edge: find_contours already emits vertices roughly 1 px apart,
    the same resolution the rest of the pocket pipeline already works at,
    and RDP simplification downstream absorbs a deviation of that size.
    """
    d2 = (ring.pts_px[:, 0] - row) ** 2 + (ring.pts_px[:, 1] - col) ** 2
    i = int(np.argmin(d2))
    return i if d2[i] <= tol_px * tol_px else None


def _ring_arc_between(ring, i0, i1):
    """The sub-arc of ring.pts_px from vertex i0 to i1, walking whichever of
    the ring's two directions is shorter (by ring.cum/ring.L).

    Every returned point is one of the ring's own vertices, so the arc can
    never leave the region regardless of how concave its boundary is -- an
    unbroken walk along a region's own boundary stays on that boundary
    either direction, which is why "shorter arc" alone is a safe choice
    even where a scan line has been split into multiple segments by a
    concave notch.
    """
    n = len(ring.pts_px)
    fwd_len = (ring.cum[i1] - ring.cum[i0]) % ring.L
    bwd_len = (ring.cum[i0] - ring.cum[i1]) % ring.L
    if fwd_len <= bwd_len:
        idx = [i0]
        i = i0
        while i != i1:
            i = (i + 1) % n
            idx.append(i)
    else:
        idx = [i0]
        i = i0
        while i != i1:
            i = (i - 1) % n
            idx.append(i)
    return ring.pts_px[idx]


# Sideways tolerance (px) of the "Depth map change" Process Region: two
# rasterizations of the same mesh can place a sharp wall up to this far
# apart, which must not count as a change.
DEPTH_MAP_CHANGE_TOL_PX = 2


def depth_map_change_reference(prior_depth_map, tol_px=DEPTH_MAP_CHANGE_TOL_PX):
    """The prior step's depth map, grey-dilated by a disk of *tol_px*: the
    deepest it gets within *tol_px* of each pixel. Used as prior_reference,
    clip(depth_map − this, 0, 1) is the depth this step asks for beyond
    anything the prior depth map has within *tol_px*, so a wall shifted by
    up to *tol_px* yields no difference while thin new features keep theirs."""
    from grey_morphology import grey_dilation
    from skimage.morphology import disk
    return grey_dilation(np.asarray(prior_depth_map, dtype=np.float32),
                         footprint=disk(tol_px)).astype(np.float32, copy=False)


def depth_map_change(depth_map, prior_depth_map, tol_px=DEPTH_MAP_CHANGE_TOL_PX):
    """Signed depth map change with the sideways tolerance of
    depth_map_change_reference: positive where *depth_map* is deeper than
    the prior depth map anywhere within *tol_px* (the part the "Depth map
    change" Process Region processes), negative where it is shallower than
    the prior depth map everywhere within *tol_px*, 0 otherwise."""
    from grey_morphology import grey_erosion
    from skimage.morphology import disk
    prior = np.asarray(prior_depth_map, dtype=np.float32)
    deeper = np.clip(depth_map - depth_map_change_reference(prior, tol_px), 0.0, 1.0)
    shallower = np.clip(grey_erosion(prior, footprint=disk(tol_px)) - depth_map, 0.0, 1.0)
    return (deeper - shallower).astype(np.float32)


def compute_raster_gate_mask(residual, work_surface_tol, prior_reference,
                              subtract_unaccessible, accessible, footprint):
    """Build the Process Region gate mask for a raster-style pass, or None
    if ungated.

    work_surface_tol <= 0 (the default) means "no gate" -- every pixel is
    visited regardless of residual/accessible depth, so this returns None
    (the caller then treats the full line/leg as active, unmasked).
    Otherwise the gate is dilated by *footprint* (the tool's own kernel
    footprint) so a masked line's active span extends a full tool-radius
    past the qualifying pixels, same as the achievable-depth surface itself.
    """
    if work_surface_tol <= 0.0:
        return None
    from grey_morphology import binary_dilation
    if subtract_unaccessible and prior_reference is not None:
        gate = np.clip(accessible - prior_reference, 0.0, 1.0)
    else:
        gate = residual
    return binary_dilation(gate >= work_surface_tol, structure=footprint).astype(np.float32)


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
                            cut_direction="Climb",
                            final_stepdown_mm=0.0,
                            extra_offset_mm=0.0,
                            avoid_air=False,
                            z_safe=5.0,
                            feed_rapid=3000.0,
                            progress=None,
                            erosion_cache=None,
                            erosion_cache_key=None,
                            z_range=(0.0, 1.0)):
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
    raster_dir        : "Horizontal" | "Vertical" | "Diagonal (45°)" |
                        "Diagonal (135°)" | "Coil (Outside-In)" |
                        "Coil (Inside-Out)" | "Spiral (Outside-In)" |
                        "Spiral (Inside-Out)" — Horizontal/Vertical/Diagonal
                        scan in parallel lines at 0°/90°/45°/135°; the Coil
                        patterns trace the depth map's perimeter inward in
                        concentric rectangular rings (Outside-In) or the
                        reverse, starting near the center (Inside-Out),
                        without lifting the tool. Spiral is one continuous
                        curve whose rings morph from the rectangular
                        perimeter into a linear spiral at the center, with
                        no sharp corners inside the outermost rings (see
                        _spiral_px). Horizontal/
                        Vertical always start at the top-left corner and use
                        independent row/column step-overs, so non-square
                        depth maps are fully covered on both axes; Diagonal
                        lines are similarly spaced and clipped to cover the
                        full canvas along both diagonals. See cut_direction
                        for how Climb/Conventional map onto each scan
                        direction's travel sense (or the coil's/spiral's
                        clockwise/counter-clockwise sense).
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
    avoid_air         : if True, remove the portions of every pass where the
                        tool would cut only air -- material previous_result
                        (or earlier stock surface) has already cleared -- see
                        prune_air_segments.
    z_safe, feed_rapid : safe height (mm) and rapid rate (mm/min), used by
                        avoid_air's retract/plunge breakeven.
    progress          : callable(pct, subtask) — thread-safe progress reporter
    erosion_cache     : optional dict used to memoise the grey-erosion/grey-dilation
                        achievable-depth surface (keyed by erosion_cache_key), so a
                        subsequent call with the same tool and physical scale can skip
                        recomputing it.  None disables caching.
    erosion_cache_key : hashable key identifying the (tool, px_per_mm, max_depth)
                        combination for erosion_cache lookups.
    z_range           : (lo, hi), normalised, 0 <= lo < hi <= 1 -- depth
                        band of the cut. The tool never goes below hi (it
                        clears flat at hi where the relief is deeper), lines
                        are cut only where the tool-centre depth is deeper
                        than lo, and the passes step down from lo, so material
                        above lo counts as already removed. The stored
                        accessible surface is capped at hi.

    Returns
    -------
    ToolPath
    """
    from grey_morphology import grey_erosion, grey_dilation

    z_lo, z_hi = check_z_range(z_range)
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
        center_depth_contour_raster = grey_erosion(depth_map.astype(np.float32, copy=False),
                                  footprint=footprint_plan,
                                  structure=structure).astype(np.float32)
        center_depth_contour_raster = np.clip(center_depth_contour_raster, 0.0, 1.0)

        # Dilation of the erosion gives the true achievable depth at each material
        # point: the tool can carve (x,y) while centred at an offset position.
        # Erosion followed by dilation = morphological opening.
        accessible_raster = grey_dilation(center_depth_contour_raster,
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
    resid_tol_mm = max_depth / (n_grey_levels - 1)
    resid_tol    = 1.0 / (n_grey_levels - 1)   # normalised [0,1]
    band = None
    if z_lo > 0.0:
        # Tool-centre criterion, so not widened by the footprint like the
        # Process Region gate: spans start and end where the tool tip
        # crosses lo. Taken before the cap at hi, so with lo == hi the
        # positions deeper than lo are cut flat at it.
        band = (center_depth_contour_raster > z_lo + 0.5 * resid_tol).astype(np.float32)
    if z_hi < 1.0:
        # New arrays: the cached surfaces stay uncapped for other steps.
        center_depth_contour_raster = np.minimum(center_depth_contour_raster, z_hi)
        accessible_raster = np.minimum(accessible_raster, z_hi)
    band_top_mm    = z_lo * max_depth
    band_bottom_mm = z_hi * max_depth

    # ── Stage 3 · Plan passes (35–40 %) ───────────────────────────
    progress(35, "Planning depth passes…")
    pass_plan       = _plan_passes(band_bottom_mm, stepdown_mm, final_stepdown_mm,
                                   band_top_mm)
    n_passes        = len(pass_plan)
    final_norm      = final_stepdown_mm / max_depth
    step_px = max(1, int(round(step_mm * px_per_mm)))   # for progress display only

    RASTER_ANGLE_FAMILY = {"Horizontal": "X", "Vertical": "Y",
                           "Diagonal (45°)": "D45", "Diagonal (135°)": "D135"}
    IS_COIL   = raster_dir in ("Coil (Outside-In)", "Coil (Inside-Out)")
    IS_SPIRAL = raster_dir in ("Spiral (Outside-In)", "Spiral (Inside-Out)")
    IS_WOUND  = IS_COIL or IS_SPIRAL      # one continuous wound path per area
    direction = None if IS_WOUND else RASTER_ANGLE_FAMILY[raster_dir]

    # Forces every scan line/leg onto a single fixed travel direction (climb
    # keeps uncut material on the right of travel, conventional on the
    # left, for a clockwise spindle). "Either" leaves the default pixel-scan
    # order untouched and lets the ordering pass below reverse segments
    # freely for shorter travel. For Horizontal/Vertical/Diagonal this flips
    # each line's own point order (FORCE_REVERSE); for Coil and Spiral it
    # instead selects which rotational sense (CW/CCW) the whole path is
    # generated in, which achieves the same "uncut material on a fixed side" guarantee
    # for every leg without needing a per-leg reversal (see
    # WOUND_OUTSIDE_IN_SENSE, _coil_legs() and _spiral_px() in Phase 1 below;
    # Inside-Out is a whole-path reversal on top of that).
    FORCE_REVERSE = {"Climb":        {"X": False, "Y": True, "D45": True,  "D135": True},
                      "Conventional": {"X": True,  "Y": False, "D45": False, "D135": False}}
    WOUND_OUTSIDE_IN_SENSE = {"Climb": "CW", "Conventional": "CCW", "Either": "CW"}
    lock_direction = cut_direction in FORCE_REVERSE

    progress(40, f"{n_passes} depth pass{'es' if n_passes != 1 else ''}  "
                 f"({stepdown_mm:.2f} mm each"
                 + (f", final {final_stepdown_mm:.2f} mm" if final_stepdown_mm > 0 else "")
                 + f"), step-over {step_px} px = {step_mm:.2f} mm, "
                 + f"residual tol {resid_tol_mm*1000:.1f} µm.")

    # ── Stage 4 · 4-phase raster (40–80 %) ────────────────────────

    prior_mask = compute_raster_gate_mask(
        residual, work_surface_tol, prior_reference, subtract_unaccessible,
        accessible_raster, footprint_raster)
    if prior_mask is not None:
        progress(40, f"Threshold {work_surface_tol*max_depth*1000:.0f} µm: "
                     f"{int(np.sum(prior_mask))}/{H*W} pixels in mask.")
    if band is not None:
        prior_mask = band if prior_mask is None else prior_mask * band

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
        sense = WOUND_OUTSIDE_IN_SENSE[cut_direction]
        rc = _spiral_px(0, H - 1, 0, W - 1, step_mm * px_per_mm, sense,
                              raster_offset_pct)
        if raster_dir == "Spiral (Inside-Out)":
            rc = rc[::-1]       # same whole-path reversal as Coil's Inside-Out
        full_depth_segs.extend(
            _curve_points(rc, center_depth_contour_raster, prior_mask, resid_tol,
                          px_per_mm, H, _ox, _oy, max_depth))
    elif IS_COIL:
        row_ids, adj_step_y_mm = _grid_line_ids(H - 1, px_per_mm, step_mm, raster_offset_pct)
        col_ids, adj_step_x_mm = _grid_line_ids(W - 1, px_per_mm, step_mm, raster_offset_pct)
        sense = WOUND_OUTSIDE_IN_SENSE[cut_direction]
        legs = _coil_legs(row_ids, col_ids, sense)
        if raster_dir == "Coil (Inside-Out)":
            # Inside-Out is the point-reversal of Outside-In generated in
            # the sense that yields the same Climb/Conventional label:
            # reversing a path both reverses leg order and flips the
            # perceived rotational sense, so this reuses
            # WOUND_OUTSIDE_IN_SENSE instead of needing a second table.
            legs = [(axis, fixed, e, s) for axis, fixed, s, e in reversed(legs)]

        for axis, fixed_idx, start_idx, end_idx in legs:
            lo, hi = (start_idx, end_idx) if start_idx <= end_idx else (end_idx, start_idx)
            full_depth_segs.extend(
                _line_points(axis, fixed_idx, lo, hi, start_idx > end_idx,
                             center_depth_contour_raster, prior_mask, resid_tol,
                             px_per_mm, H, _ox, _oy, max_depth))
    elif direction in ("X", "Y"):
        spacing_boundary_px = H - 1 if direction == "X" else W - 1   # axis lines are spaced along
        line_span_px        = W - 1 if direction == "X" else H - 1   # each line's own full length
        all_line_ids, adj_step_mm = _grid_line_ids(spacing_boundary_px, px_per_mm, step_mm, raster_offset_pct)
        for line_idx in all_line_ids:
            reverse = lock_direction and FORCE_REVERSE[cut_direction][direction]
            full_depth_segs.extend(
                _line_points(direction, line_idx, 0, line_span_px, reverse,
                             center_depth_contour_raster, prior_mask, resid_tol,
                             px_per_mm, H, _ox, _oy, max_depth))
    else:
        k_ids, adj_step_mm = _diag_grid_line_ids(H, W, direction, px_per_mm,
                                                 step_mm, raster_offset_pct)
        for k in k_ids:
            bounds = _diag_line_bounds(direction, k, 0, H - 1, 0, W - 1)
            if bounds is None:
                continue
            t_lo, t_hi = bounds
            reverse = lock_direction and FORCE_REVERSE[cut_direction][direction]
            full_depth_segs.extend(
                _diag_line_points(direction, k, t_lo, t_hi, reverse,
                                  center_depth_contour_raster, prior_mask, resid_tol,
                                  px_per_mm, H, _ox, _oy, max_depth))

    progress(55, f"Phase 1 complete — {len(full_depth_segs)} full-depth span(s).")

    # ── Phase 2: depth-slice into per-pass segments ───────────────
    pass_list = make_multipass(full_depth_segs, band_bottom_mm, stepdown_mm,
                               final_stepdown_mm, top_mm=band_top_mm)

    # ── Phase 2b: drop air-cutting runs (optional) ────────────────
    air_pruned_mm = 0.0
    if avoid_air:
        progress(55, "Removing air-cutting portions of each pass…")
        pass_list, _, air_pruned_mm = _prune_air_passes(
            pass_list, previous_result, kernel_raster, footprint_raster, (H, W),
            px_per_mm, _ox, _oy, max_depth, resid_tol,
            feed_rate, feed_rapid, z_safe)
        progress(55, f"Removed {air_pruned_mm:.0f} mm of air cutting.")

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
        if lock_direction and IS_WOUND:
            # Phase 1 already emits one continuous, correctly-sensed
            # perimeter walk (see WOUND_OUTSIDE_IN_SENSE above);
            # reordering would break that. _order_locked_raster_pass's
            # single-coordinate grouping assumes one axis-aligned line
            # family and doesn't generalize to a coil's alternating
            # row/column legs or a spiral's curve, so pass
            # make_multipass's output straight through unreordered. Phase 4 below still bridges any gaps a
            # masked/gated refining pass introduces, same as it already
            # does for Horizontal/Vertical.
            ordered_segs = pass_segs
        elif lock_direction:
            ordered_segs = _order_locked_raster_pass(pass_segs, direction, cur)
        elif IS_WOUND and _is_contiguous_chain(pass_segs):
            # This pass is still one genuinely unbroken coil/spiral (the common
            # case: full-depth single pass, or a multi-pass/masked pass
            # that happened not to fragment) — skip reordering. This is
            # never a loss (an already-contiguous chain has zero
            # inter-segment travel to optimise) and avoids a real bug:
            # _order_free_raster_pass's greedy walk starts from the
            # tool's world origin (0,0), which can coincide exactly with
            # a NON-initial vertex of the coil depending on the G-code
            # origin setting (e.g. Bottom-Left origin puts (0,0) on the
            # ring-0/ring-1 boundary corner). That exact-zero-distance tie
            # makes greedy pick the wrong leg first and chain backwards
            # through more zero-distance ties, closing each ring into a
            # rectangle linked to the next by a diagonal jump instead of
            # a coil. When make_multipass or mask-based gating DOES
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

                # Consecutive Coil legs share their corner pixel exactly
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
    rdp_tol   = 1.0 / px_per_mm
    rdp_z_tol = max_depth / (n_grey_levels - 1)
    simplified = [simplify_line(pts, rdp_tol, rdp_z_tol) for pts in all_segments]

    progress(88, "Simulating tool-swept removal surface…")
    material_removed_raster = simulate_toolpath_result(
        ToolPath(segments=simplified),
        (H, W), px_per_mm, px_per_mm, _ox, _oy,
        max_depth, kernel_raster, footprint_raster,
    )
    raster_result = np.maximum(
        material_removed_raster,
        previous_result if previous_result is not None else 0.0).astype(np.float32)

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
            'z_range': (z_lo, z_hi),
            'max_depth': max_depth,
            'width_mm': width_mm,
            'height_mm': height_mm,
            'n_passes': n_passes,
            'rdp_tolerance': rdp_tol,
            'rdp_z_tolerance': rdp_z_tol,
            'link_threshold_mm': link_threshold_mm,
            'n_links_inserted': n_links_inserted,
            'n_links_total': n_links,
            'air_pruned_mm': air_pruned_mm,
        }
    )

    if previous_result is not None and work_surface_tol is not None:
        toolpath.metadata['refining_threshold_um'] = \
            work_surface_tol * max_depth * 1000

    progress(100, "Tool path computation complete!")

    return toolpath


class _PocketPattern:
    """Which infill family a Pocket Scan direction selects, and how Cut
    direction applies to it."""

    ZIGZAG_FAMILY = {"Zig-Zag (0°)": "X", "Zig-Zag (45°)": "D45",
                     "Zig-Zag (90°)": "Y", "Zig-Zag (135°)": "D135"}
    FORCE_REVERSE = {"Climb":        {"X": False, "Y": True, "D45": True,  "D135": True},
                     "Conventional": {"X": True,  "Y": False, "D45": False, "D135": False}}
    ONION_SKIN = ("Onion Skin (Inside-Out)", "Onion Skin (Outside-In)")

    def __init__(self, raster_dir, cut_direction):
        if raster_dir not in self.ONION_SKIN and raster_dir not in self.ZIGZAG_FAMILY:
            raise ValueError(f"unknown Pocket scan direction: {raster_dir!r}")
        self.raster_dir = raster_dir
        self.is_onion = raster_dir in self.ONION_SKIN
        self.is_zigzag = raster_dir in self.ZIGZAG_FAMILY
        self.direction = self.ZIGZAG_FAMILY.get(raster_dir)   # None for Onion Skin
        self.lock_direction = cut_direction in self.FORCE_REVERSE


def _pocket_pattern(raster_dir, cut_direction):
    return _PocketPattern(raster_dir, cut_direction)


class _RegionRunner:
    """Runs pocket regions' infill jobs (_start_pocket_infill) on a thread
    pool: the regions are independent, and most of their array work
    (labelling, distance transforms, contour tracing) releases the GIL.

    A job starts once its estimated peak memory fits the budget beside the
    jobs already running, or when no other job runs. The budget is
    AVAILABLE_FRACTION of the memory available to the pool, read again
    each time a job is about to start: the memory available now plus the
    estimates of the running jobs (which the memory available now already
    lacks), but never more than was available when the runner was made.
    So memory taken by another program while the jobs run lowers the
    budget, down to one job at a time. A job waiting for room checks again
    every WAIT_S seconds, since memory freed elsewhere sends no signal.
    Where memory can't be read, the budget is erosion_cache.FALLBACK_BYTES.
    Use it as a context manager: leaving it cancels the jobs not yet
    started and waits for the running ones.

    memory : a function returning (total, available) bytes or None, as
             erosion_cache.system_memory (the default).
    """

    AVAILABLE_FRACTION = 0.5
    WAIT_S = 0.1

    def __init__(self, max_workers=None, memory=None):
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from erosion_cache import system_memory
        self._memory = memory or system_memory
        mem = self._memory()
        self._available0 = None if mem is None else mem[1]
        self._pool = ThreadPoolExecutor(max_workers=max_workers or os.cpu_count() or 1)
        self._cond = threading.Condition()
        self._running = 0

    def _budget(self):
        """The bytes the running jobs may hold together, from the memory
        available now (called with the condition held)."""
        from erosion_cache import FALLBACK_BYTES
        mem = self._memory() if self._available0 is not None else None
        if mem is None:
            return FALLBACK_BYTES
        return self.AVAILABLE_FRACTION * min(self._available0, mem[1] + self._running)

    def submit(self, fn, nbytes):
        """Run fn() on the pool once nbytes fit the budget; returns its
        Future."""
        def run():
            with self._cond:
                while self._running and self._running + nbytes > self._budget():
                    self._cond.wait(self.WAIT_S)
                self._running += nbytes
            try:
                return fn()
            finally:
                with self._cond:
                    self._running -= nbytes
                    self._cond.notify_all()
        return self._pool.submit(run)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._pool.shutdown(wait=True, cancel_futures=True)


# Peak memory of one region's infill job per pixel of its bounding box
# (measured: about 64 for _onion_skin_chains, 15 for a Zig-Zag region's
# boundary rings plus its cropped masks).
_ONION_REGION_BYTES_PER_PX = 80
_ZIGZAG_REGION_BYTES_PER_PX = 32


def _pocket_infill(pocket_mask, level_norm, level_mm, pattern, step_mm,
                   raster_offset_pct, cut_direction, px_per_mm, H, _ox, _oy,
                   max_depth, tool_radius_mm=None):
    """Pocket's per-region infill of *pocket_mask* at one flat depth.

    level_norm / level_mm : the floor depth, normalised and in mm.
    pattern               : _pocket_pattern(raster_dir, cut_direction).
    tool_radius_mm        : the tool's radius, for Onion Skin's spines
                            (_onion_skin_chains); None for none.

    Returns (segs, seg_region_ids, chain_parents, region_contours,
    labeled_regions, flat_depth, n_regions): the full-depth segments, the
    pocket region (label) each segment belongs to, for Onion Skin the
    segments (ring chains) each segment must be cut after (None for other
    patterns; see _order_onion_pass), the Zig-Zag boundary rings per region
    (empty for other patterns), the region labels, the flat floor surface
    (level_norm inside the mask, 0 outside) and the number of regions.

    The regions are computed in parallel on a _RegionRunner of their own;
    _start_pocket_infill shares one runner between several calls.
    """
    with _RegionRunner() as runner:
        return _start_pocket_infill(
            pocket_mask, level_norm, level_mm, pattern, step_mm,
            raster_offset_pct, cut_direction, px_per_mm, H, _ox, _oy,
            max_depth, runner, tool_radius_mm)()


def _start_pocket_infill(pocket_mask, level_norm, level_mm, pattern, step_mm,
                         raster_offset_pct, cut_direction, px_per_mm, H, _ox, _oy,
                         max_depth, runner, tool_radius_mm=None):
    """Label *pocket_mask*'s regions and submit each region's infill to
    *runner* (a _RegionRunner). Returns a function that waits for them and
    returns _pocket_infill's result, the regions in label order whatever
    order they finish in.
    """
    from scipy.ndimage import label, find_objects

    # Each connected pocket region gets its own infill, generated within
    # that region's own bounding box from its own label mask (not the
    # pooled pocket_mask), so two disjoint regions with overlapping
    # bounding boxes never double-generate a line over each other's pixels.
    # Zig-Zag's scan lines gate on flat_depth * region_mask > mask_resid_tol,
    # which can't tell inside from outside at a level of 0 (the stock
    # surface), so the floor depth is kept at least 1e-6 as in
    # pocket_surface_and_mask.
    level_norm = max(level_norm, 1e-6)
    mask_resid_tol = level_norm * 0.5   # (flat_depth * region_mask) is 0 or level_norm

    raster_dir = pattern.raster_dir
    IS_ONION = pattern.is_onion
    direction = pattern.direction
    FORCE_REVERSE = _PocketPattern.FORCE_REVERSE
    lock_direction = pattern.lock_direction

    labeled_regions, n_regions = label(pocket_mask)
    region_boxes = find_objects(labeled_regions)

    def onion_region(local_mask, offset):
        step_px = max(1, int(round(step_mm * px_per_mm)))
        min_area_px = max(4, int(round(np.pi * (0.5 * step_px) ** 2)))
        chains, parents = _onion_skin_chains(
            local_mask, step_px, min_area_px,
            reverse_rings=(cut_direction == "Conventional"),
            tool_radius_px=(None if tool_radius_mm is None
                            else tool_radius_mm * px_per_mm),
            return_parents=True)
        if raster_dir == "Onion Skin (Inside-Out)":
            # Each chain runs backwards, from its deepest ring out, and
            # the graph is inverted: a chain is cut after every chain
            # that follows it rather than after the ones it follows.
            chains = [c[::-1] for c in chains]
            inverted = [[] for _ in chains]
            for i, ps in enumerate(parents):
                for p in ps:
                    inverted[p].append(i)
            parents = inverted
        segs = [_ring_chain_to_points(chain + offset, level_mm,
                                      px_per_mm, H, _ox, _oy)
                for chain in chains]
        return segs, parents, None

    def zigzag_region(local_mask, offset, box):
        r_lo, r_hi = box[0].start, box[0].stop - 1
        c_lo, c_hi = box[1].start, box[1].stop - 1
        # Each region's own boundary ring(s) (one outer, plus one per
        # interior hole/island), used by _order_and_link_pocket_pass to
        # bridge gaps by walking the region's actual contour instead of a
        # geodesic search -- see _region_contour_rings/_ContourRing.
        contours = [_make_contour_ring(ring + offset)
                    for ring in _region_contour_rings(local_mask)]
        # The lines read the region's mask and floor cropped to its box.
        region_mask_f = local_mask.astype(np.float32)
        flat_local = np.where(local_mask, level_norm, 0.0).astype(np.float32)
        box_origin = (r_lo, c_lo)
        reverse = lock_direction and FORCE_REVERSE[cut_direction][direction]
        segs = []
        if direction in ("X", "Y"):
            if direction == "X":
                spacing_lo, spacing_hi = r_lo, r_hi
                span_lo,    span_hi    = c_lo, c_hi
            else:
                spacing_lo, spacing_hi = c_lo, c_hi
                span_lo,    span_hi    = r_lo, r_hi
            local_line_ids, _ = _grid_line_ids(spacing_hi - spacing_lo, px_per_mm,
                                               step_mm, raster_offset_pct)
            for line_idx in (i + spacing_lo for i in local_line_ids):
                segs.extend(_line_points(direction, line_idx, span_lo, span_hi, reverse,
                                         flat_local, region_mask_f, mask_resid_tol,
                                         px_per_mm, H, _ox, _oy, max_depth,
                                         offset=box_origin))
        else:
            k_offset = (r_lo + c_lo) if direction == "D45" else (r_lo - c_lo)
            local_k_ids, _ = _diag_grid_line_ids(r_hi - r_lo + 1, c_hi - c_lo + 1,
                                                 direction, px_per_mm, step_mm,
                                                 raster_offset_pct)
            for k in (k + k_offset for k in local_k_ids):
                bounds = _diag_line_bounds(direction, k, r_lo, r_hi, c_lo, c_hi)
                if bounds is None:
                    continue
                t_lo, t_hi = bounds
                segs.extend(_diag_line_points(direction, k, t_lo, t_hi, reverse,
                                              flat_local, region_mask_f, mask_resid_tol,
                                              px_per_mm, H, _ox, _oy, max_depth,
                                              offset=box_origin))
        return segs, None, contours

    def region_job(region_id, box):
        local_mask = labeled_regions[box] == region_id
        offset = np.array([box[0].start, box[1].start])
        if IS_ONION:
            return onion_region(local_mask, offset)
        return zigzag_region(local_mask, offset, box)

    bytes_per_px = _ONION_REGION_BYTES_PER_PX if IS_ONION else _ZIGZAG_REGION_BYTES_PER_PX
    jobs = []
    for region_id, box in enumerate(region_boxes, start=1):
        if box is None:
            continue
        nbytes = bytes_per_px * (box[0].stop - box[0].start) * (box[1].stop - box[1].start)
        jobs.append((region_id, runner.submit(
            lambda region_id=region_id, box=box: region_job(region_id, box), nbytes)))

    def finish():
        full_depth_segs = []
        # Parallel to full_depth_segs, one entry per element, recording
        # which pocket region produced it -- Zig-Zag orders whole regions
        # by it.
        seg_region_ids = []
        # Onion Skin only, parallel to full_depth_segs: the chains each
        # chain must be cut after, along its region's level graph.
        chain_parents = [] if IS_ONION else None
        # Zig-Zag only: each region's own boundary ring(s).
        region_contours = {}
        for region_id, future in jobs:
            segs, parents, contours = future.result()
            base = len(full_depth_segs)
            full_depth_segs.extend(segs)
            seg_region_ids.extend([region_id] * len(segs))
            if IS_ONION:
                chain_parents.extend([base + p for p in ps] for ps in parents)
            else:
                region_contours[region_id] = contours
        flat_depth = np.where(labeled_regions > 0, level_norm, 0.0).astype(np.float32)
        return (full_depth_segs, seg_region_ids, chain_parents, region_contours,
                labeled_regions, flat_depth, n_regions)

    return finish


def _order_and_link_pocket_pass(pass_segs, pass_ids, ceil_mm, is_final,
                                tool_pos, pattern, link_threshold_mm,
                                labeled_regions, region_contours, link_surface,
                                final_norm, px_per_mm, H, W, _ox, _oy, max_depth,
                                chain_parents=None):
    """Order one pass of Pocket infill segments and bridge the short gaps
    between consecutive ones.

    pass_ids        : per segment of pass_segs, for Zig-Zag the pocket
                      region it belongs to, for Onion Skin the chain it
                      was cut from (an index into chain_parents).
    chain_parents   : Onion Skin only: _pocket_infill's chain_parents,
                      which _order_onion_pass orders the chains along.
    tool_pos        : (x, y) mm where the tool is before the pass.
    labeled_regions, region_contours : _pocket_infill's region labels and
                      Zig-Zag boundary rings, for contour-walk links.
    link_surface    : normalised depth a geodesic link follows (capped at
                      the pass ceiling); Pocket passes its flat floor.

    Returns (segments, tool_pos, (n_cut, n_links_inserted,
    n_links_contour, n_links)): the ordered cut segments with the link
    segments between them, and the tool position after the last cut.
    """
    IS_ONION, IS_ZIGZAG = pattern.is_onion, pattern.is_zigzag
    direction, lock_direction = pattern.direction, pattern.lock_direction
    RING_SNAP_TOL_PX = 2.0   # how close a segment endpoint must sit to a region's own contour ring
    ceil_norm = ceil_mm / max_depth
    cur = (float(tool_pos[0]), float(tool_pos[1]))
    # Onion Skin's winding sense and its Outside-In vs. Inside-Out sequence
    # are decided at generation time (_pocket_infill) -- each chain's own
    # direction and the level graph it is ordered along -- so its ordering
    # never reverses a chain, whatever lock_direction says.
    if IS_ONION:
        ordered_segs = _order_onion_pass(pass_segs, pass_ids, chain_parents, cur)
    else:
        ordered_segs = _order_zigzag_pass(pass_segs, pass_ids, direction, cur,
                                          lock_direction)

    out = []
    n_cut = n_links_inserted = n_links_contour = n_links = 0
    for i, seg in enumerate(ordered_segs):
        out.append(seg)
        n_cut += 1
        tool_pos = (seg[-1][0], seg[-1][1])

        if i + 1 < len(ordered_segs):
            nxt = ordered_segs[i + 1]
            x_entry, y_entry = nxt[0][0], nxt[0][1]
            nxt_dist = np.hypot(x_entry - tool_pos[0],
                                y_entry - tool_pos[1])

            if 1e-6 < nxt_dist < link_threshold_mm:
                x_exit = float(tool_pos[0])
                y_exit = float(tool_pos[1])
                c0_px = int(round((x_exit  + _ox) * px_per_mm))
                r0_px = int(round((H - 1) - (y_exit  + _oy) * px_per_mm))
                c1_px = int(round((x_entry + _ox) * px_per_mm))
                r1_px = int(round((H - 1) - (y_entry + _oy) * px_per_mm))

                contour_link_points = None
                if IS_ZIGZAG:
                    rid0 = labeled_regions[r0_px, c0_px]
                    rid1 = labeled_regions[r1_px, c1_px]
                    if rid0 == rid1 and rid0 != 0:
                        for ring in region_contours.get(int(rid0), []):
                            i0 = _nearest_ring_vertex(ring, r0_px, c0_px, RING_SNAP_TOL_PX)
                            i1 = _nearest_ring_vertex(ring, r1_px, c1_px, RING_SNAP_TOL_PX)
                            if i0 is not None and i1 is not None:
                                z = seg[-1][2]
                                arc = _ring_arc_between(ring, i0, i1)
                                contour_link_points = [seg[-1]] + [
                                    (col / px_per_mm - _ox, (H - 1 - row) / px_per_mm - _oy, z)
                                    for row, col in arc[1:-1]
                                ] + [nxt[0]]
                                break

                if contour_link_points is not None:
                    link_points = contour_link_points
                    if len(link_points) >= 2:
                        out.append(link_points)
                        n_links_inserted += 1
                        n_links_contour += 1
                else:
                    fn = 0.0 if is_final else final_norm
                    link_pixels = _geodesic_link(
                        r0_px, c0_px, r1_px, c1_px, ceil_norm,
                        link_surface, H, W,
                        px_per_mm, px_per_mm, max_depth, fn)
                    link_points = [seg[-1]]
                    for row_l, col_l in link_pixels[1:-1]:
                        terrain = float(link_surface[row_l, col_l])
                        if is_final:
                            z_link = min(terrain, ceil_norm)
                        else:
                            z_link = min(max(terrain - final_norm, 0.0), ceil_norm)
                        x_lmm = col_l / px_per_mm - _ox
                        y_lmm = (H - 1 - row_l) / px_per_mm - _oy
                        link_points.append((x_lmm, y_lmm, -z_link * max_depth))
                    link_points.append(nxt[0])

                    if len(link_points) >= 2:
                        out.append(link_points)
                        n_links_inserted += 1

            n_links += 1

    return out, tool_pos, (n_cut, n_links_inserted, n_links_contour, n_links)


def compute_pocket_mask(depth_map, tool, px_per_mm, max_depth, target_z_mm,
                         n_grey_levels, boundary_offset_mm=0.0,
                         reference_surface=None,
                         erosion_cache=None, erosion_cache_key=None,
                         progress=None):
    """Resolve target_z_mm and build the pocket-interior mask: every
    position where *tool*, centred there, can still reach target_z_mm
    without its footprint overhanging shallower material.

    reference_surface, if given, replaces depth_map as the surface the
    tool's achievable-depth erosion is computed against (depth_map itself
    is otherwise unused by this function) -- used by V-Carving Clean-up to
    erode V-Carving (Hybrid)'s own idealized-V-bit achievable-depth surface
    a second time by the clearing tool's kernel, so the resulting mask
    boundary retreats exactly as far as needed to not gouge the V-bit's
    actual sloped wall, for any clearing tool profile (see
    compute_vcarve_synth_achievable_depth). None (the default) uses
    depth_map directly, i.e. ordinary Pocket behaviour.

    target_z_mm is used literally, including 0 -- a 0 target depth makes
    the pocket interior mask cover every tool-centre position the tool can
    occupy at all (the erosion is always >= 0), so the whole reachable
    surface is cleared flat at (effectively) Z=0: a full-stock flattening/
    facing pass. target_z_norm is floored just above exact 0 rather than
    landing on it, so the resulting floor is negligibly, not exactly, 0 --
    see the comment at its computation below. Only None (no value given)
    falls back to max_depth (the deepest region of the depth map).

    Gates on the EROSION (tool-centre-safe depth), not the dilated/opened
    accessible surface: the accessible surface tells us which MATERIAL
    points can be reached by *some* nearby tool-centre position, which is
    the right question for "did this get cut" (ToolPath.accessible), but
    the wrong one for "where can the tool be centred" -- the toolpath
    itself must only visit centre positions whose full footprint clears
    shallower material, exactly what the erosion already encodes (the same
    surface Contour extracts its iso-depth contours from). Using the
    dilated surface here would let the tool centre travel right up to (or
    past) a boundary that's only reachable via a NEARBY centre position,
    cutting full depth where the tool's own edge overhangs past where
    material should remain.

    boundary_offset_mm retreats this tool-centre-safe mask further away
    from the pocket wall by a plain lateral erosion -- the same
    lateral-only push compute_cutout_toolpath applies (there as a dilation
    of the object silhouette, since positive there also means "away from
    the boundary", just on the opposite side of it) -- rather than reusing
    the depth-shaped kernel inflation the other algorithms' extra_offset_mm
    relies on, since Pocket's cut depth is already fixed independently of
    the achievable-depth surface by target_z_mm.

    Returns (pocket_mask, center_depth_contour_pocket, accessible_pocket,
    target_z_norm, kernel_pocket, footprint_pocket).
    """
    from grey_morphology import binary_erosion, grey_erosion, grey_dilation
    from skimage.morphology import disk

    if progress is None:
        progress = lambda pct, msg: None

    source_surface = depth_map if reference_surface is None else reference_surface

    # ── Build tool kernel ─────────────────────────────────────────────────
    progress(0, "Building tool geometry kernel…")
    kernel_pocket, footprint_pocket = _build_kernel(tool, px_per_mm, max_depth)
    ksize = kernel_pocket.shape[0]
    progress(10, f"Kernel ready  ({ksize}×{ksize} px, profile: {tool.profile}).")

    # ── Compute achievable depth surface ───────────────────────────────────
    cached = erosion_cache.get(erosion_cache_key) if erosion_cache is not None else None
    if cached is not None and cached[0].shape == source_surface.shape:
        center_depth_contour_pocket, accessible_pocket = cached
        progress(35, "Reusing achievable depth surface (same tool, unchanged)…")
    else:
        progress(10, "Computing achievable depth surface (morphological opening)…")
        structure = -kernel_pocket.astype(np.float64)
        center_depth_contour_pocket = grey_erosion(source_surface.astype(np.float32, copy=False),
                                  footprint=footprint_pocket,
                                  structure=structure).astype(np.float32)
        center_depth_contour_pocket = np.clip(center_depth_contour_pocket, 0.0, 1.0)
        accessible_pocket = grey_dilation(center_depth_contour_pocket,
                                   footprint=footprint_pocket,
                                   structure=structure).astype(np.float32)
        accessible_pocket = np.clip(accessible_pocket, 0.0, 1.0)
        progress(35, "Achievable depth surface computed.")
        if erosion_cache is not None:
            erosion_cache[erosion_cache_key] = (center_depth_contour_pocket, accessible_pocket)

    # ── Target-Z interior mask ──────────────────────────────────────────────
    progress(35, "Computing pocket interior mask…")
    if target_z_mm is None:
        target_z_mm = max_depth
    # Floored just above 0 rather than allowed to land exactly there: Zig-Zag
    # (see compute_pocket_toolpath) reuses compute_raster_toolpath's
    # own _line_points/_diag_line_points to build scan lines, which decide
    # which pixels are active by multiplying this normalised depth by the
    # region mask and comparing against a tolerance -- the same real
    # depth*mask test Raster's own Z-threshold/residual gating depends on, so
    # it can't be changed here. At target_z_norm == 0.0 that product is
    # identically 0 both inside and outside the pocket (flat_depth is 0
    # everywhere a flat target depth of 0 implies), so no tolerance can tell
    # them apart and every scan line comes back empty. A 1e-6 floor keeps
    # in-pocket pixels distinguishably (if negligibly) nonzero -- Onion Skin
    # is unaffected, since it gates on the region mask directly rather than
    # through this product.
    target_z_norm = float(np.clip(target_z_mm / max_depth, 1e-6, 1.0))
    contour_eps = 0.5 / max(n_grey_levels - 1, 1)
    pocket_mask = center_depth_contour_pocket >= (target_z_norm - contour_eps)

    if boundary_offset_mm > 0.0:
        boundary_px = max(1, int(round(boundary_offset_mm * px_per_mm)))
        pocket_mask = binary_erosion(pocket_mask, structure=disk(boundary_px))

    return (pocket_mask, center_depth_contour_pocket, accessible_pocket,
            target_z_norm, kernel_pocket, footprint_pocket)


def compute_pocket_toolpath(depth_map, n_grey_levels, tool,
                             step_mm, stepdown_mm, max_depth,
                             width_mm, height_mm, raster_dir,
                             target_z_mm=None,
                             origin="Bottom-Left",
                             feed_rate=800.0,
                             previous_result=None,
                             raster_offset_pct=0.0,
                             cut_direction="Climb",
                             final_stepdown_mm=0.0,
                             boundary_offset_mm=0.0,
                             reference_surface=None,
                             avoid_air=False,
                             z_safe=5.0,
                             feed_rapid=3000.0,
                             progress=None,
                             erosion_cache=None,
                             erosion_cache_key=None):
    """Compute a flat-bottomed pocket toolpath: clear the interior of the
    depth map's target-Z iso-depth region with a raster-style infill.

    Every position where the tool can be centred and still reach target_z_mm
    without its footprint overhanging shallower material (i.e. the tool-
    centre erosion, not the dilated/opened accessible surface) is treated as
    pocket interior; that interior is cleared flat at target_z_mm with
    concentric rings following each pocket's own outline (Onion Skin) or
    Zig-Zag scan lines, one tool-accessibility-bounded connected region at a
    time.

    Parameters
    ----------
    depth_map         : 2-D float32 [0, 1]  — 1.0 = max cut depth (intaglio)
    n_grey_levels     : number of grey levels in source image (e.g. 256)
    tool              : Tool — encapsulates profile, diameter_mm, vbit_angle_deg
    step_mm           : lateral step-over in mm
    stepdown_mm       : max depth increment per roughing z-pass in mm
    max_depth         : total maximum cut depth in mm (depth map's own scale)
    width_mm          : physical image width in mm
    height_mm         : physical image height in mm
    raster_dir        : "Onion Skin (Inside-Out)" | "Onion Skin (Outside-In)" |
                        "Zig-Zag (0°)" | "Zig-Zag (45°)" | "Zig-Zag (90°)" |
                        "Zig-Zag (135°)". Zig-Zag scans parallel lines at the
                        given angle within each region, same line-generation
                        machinery as compute_raster_toolpath's Horizontal/
                        Vertical/Diagonal directions (applied independently
                        within each pocket region), but bridges gaps between
                        neighbouring lines by walking the region's own
                        boundary contour wherever both gap endpoints lie on
                        it, falling back to the same geodesic search Raster
                        uses otherwise (e.g. a gap spanning two different
                        regions). Onion Skin
                        offsets concentric rings inward from each region's
                        own boundary by step_mm, stitched into one
                        continuous chain per undivided island, so the
                        pattern follows the pocket's actual silhouette.
    target_z_mm       : depth (mm) defining the pocket floor and which tool-
                        centre positions count as pocket interior (tool-centre
                        erosion >= this, i.e. the tool's own footprint must
                        clear shallower material at that depth). Used
                        literally, including 0 -- a 0 target depth clears
                        the entire reachable surface flat at Z=0 (a full-
                        stock flattening/facing pass), since every tool-
                        centre erosion is >= 0. None defaults to max_depth
                        (the deepest region of the depth map). Clipped to
                        [0, max_depth].
    raster_offset_pct : percentage [0, 100) to shift infill lines relative to
                        the step-over distance, same meaning as Raster.
    cut_direction     : "Either" | "Climb" | "Conventional" — forces every
                        infill line/leg onto a fixed travel direction, same
                        semantics as compute_raster_toolpath.
    origin            : G-code origin convention ("Bottom-Left", "Top-Left", …)
    feed_rate         : cutting feed rate in mm/min
    previous_result   : optional 2-D float32 [0,1] cumulative result from a
                        prior step (ToolPath.result), unioned into this
                        step's result.
    final_stepdown_mm : depth (mm) reserved for a dedicated finishing pass at
                        the pocket floor. 0 = no reservation.
    boundary_offset_mm : lateral-only push of the tool-centre path away from
                        the pocket wall, beyond the minimum tool-radius
                        clearance, in mm; 0 = tool tangent to the wall (the
                        plain tool-centre-erosion boundary). Implemented as
                        a binary erosion of the pocket interior mask, so
                        positive values retreat the tool centre further from
                        the wall (leaving more material uncut near the
                        boundary) rather than pushing it toward/into the
                        wall. Same "away from the boundary" direction as
                        compute_cutout_toolpath's own boundary_offset_mm,
                        which dilates the silhouette mask instead since
                        there the boundary sits on the opposite side (the
                        object) of the mask. Unlike compute_raster_toolpath's
                        extra_offset_mm, this has no vertical component:
                        Pocket's cut depth is controlled independently by
                        target_z_mm, so there is nothing for a Z-offset to
                        mean here.
    reference_surface : optional 2-D float32 [0,1] surface, same shape as
                        depth_map after Stage 0's own resampling, replacing
                        depth_map as the input to the tool's achievable-
                        depth erosion (Stages 1-3 only — Stage 4 onward,
                        including result simulation, still work from the
                        real depth_map/tool). Used by V-Carving Clean-up to
                        erode V-Carving (Hybrid)'s idealized-V-bit
                        achievable-depth surface a second time by this
                        step's own clearing tool, so the pocket boundary
                        retreats exactly as far as needed to not gouge the
                        V-bit's actual sloped wall, for any clearing tool
                        profile — see compute_vcarve_synth_achievable_depth.
                        None (the default) uses depth_map directly, i.e.
                        ordinary Pocket behaviour. Must already be resampled
                        to match depth_map's own post-Stage-0 grid — it is
                        not resampled again here.
    avoid_air         : if True, remove the portions of every pass where the
                        tool would cut only air -- see prune_air_segments.
                        Splitting a segment remaps its region id so Zig-Zag/
                        Onion Skin region grouping still holds.
    z_safe, feed_rapid : safe height (mm) and rapid rate (mm/min), used by
                        avoid_air's retract/plunge breakeven.
    progress          : callable(pct, subtask) — thread-safe progress reporter
    erosion_cache     : optional dict memoising the grey-erosion/grey-dilation
                        achievable-depth surface, keyed by erosion_cache_key.
    erosion_cache_key : hashable key identifying the (tool, px_per_mm,
                        max_depth) combination for erosion_cache lookups.

    Returns
    -------
    ToolPath

    Note: this does not cut a wall-finishing pass around the pocket
    perimeter — only the interior infill described above. The infill's
    outermost line leaves the same stair-stepped boundary edge Raster
    already leaves at any masked boundary.
    """

    if progress is None:
        progress = lambda pct, msg: None

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
           "Center":      W_mm/2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm/2}.get(origin, 0)

    # ── Stages 1–3 · Kernel, achievable depth surface, pocket interior mask ─
    (pocket_mask, center_depth_contour_pocket, accessible_pocket, target_z_norm,
     kernel_pocket, footprint_pocket) = compute_pocket_mask(
        depth_map, tool, px_per_mm, max_depth, target_z_mm, n_grey_levels,
        boundary_offset_mm=boundary_offset_mm,
        reference_surface=reference_surface,
        erosion_cache=erosion_cache, erosion_cache_key=erosion_cache_key,
        progress=progress,
    )
    pocket_depth_mm = target_z_norm * max_depth

    n_pocket_px = int(np.sum(pocket_mask))
    progress(40, f"Target depth {pocket_depth_mm:.2f} mm: "
                 f"{n_pocket_px}/{H*W} pixels in pocket interior.")

    if n_pocket_px == 0:
        progress(100, "No pixels reach the target depth — nothing to cut.")
        return ToolPath(
            segments=[],
            result=(previous_result.copy() if previous_result is not None
                     else np.zeros((H, W), dtype=np.float32)),
            accessible=accessible_pocket,
            metadata={'algorithm': 'Pocket', 'target_z_mm': pocket_depth_mm,
                      'step_mm': step_mm, 'n_regions': 0},
        )

    # ── Stage 4 · Per-region infill generation (40–70 %) ─────────────────
    pattern = _pocket_pattern(raster_dir, cut_direction)
    (full_depth_segs, seg_region_ids, chain_parents, region_contours,
     labeled_regions, flat_depth, n_regions) = _pocket_infill(
        pocket_mask, target_z_norm, pocket_depth_mm, pattern, step_mm,
        raster_offset_pct, cut_direction, px_per_mm, H, _ox, _oy, max_depth,
        tool_radius_mm=0.5 * tool.diameter_mm)
    progress(40, f"Phase 1: generated infill for {n_regions} pocket region(s).")
    IS_ZIGZAG, IS_ONION = pattern.is_zigzag, pattern.is_onion

    progress(70, f"Phase 1 complete — {len(full_depth_segs)} full-depth span(s) "
                 f"across {n_regions} region(s).")

    # ── Stage 5 · Depth-slice the flat floor into per-pass segments ──────
    pass_list = make_multipass(full_depth_segs, pocket_depth_mm, stepdown_mm,
                               final_stepdown_mm)
    n_passes = len(pass_list)

    # Every point of every region shares the one uniform target_z_norm, so
    # make_multipass's per-pass keep/drop decision is identical for every
    # segment in a given pass: a non-empty pass reproduces full_depth_segs
    # unchanged in count and order. pass_ids holds each pass's own
    # per-segment ids -- the region for Zig-Zag, the chain (the segment's
    # own index in full_depth_segs) for Onion Skin -- which air pruning
    # (splitting or dropping segments) remaps.
    seg_ids = list(range(len(full_depth_segs))) if IS_ONION else seg_region_ids
    pass_ids = [seg_ids] * n_passes
    air_pruned_mm = 0.0
    if avoid_air:
        progress(70, "Removing air-cutting portions of each pass…")
        pass_list, pruned_ids, air_pruned_mm = _prune_air_passes(
            pass_list, previous_result, kernel_pocket, footprint_pocket, (H, W),
            px_per_mm, _ox, _oy, max_depth, 1.0 / (n_grey_levels - 1),
            feed_rate, feed_rapid, z_safe,
            parent_ids=seg_ids if (IS_ZIGZAG or IS_ONION) else None)
        if IS_ZIGZAG or IS_ONION:
            pass_ids = pruned_ids
        progress(70, f"Removed {air_pruned_mm:.0f} mm of air cutting.")
    link_threshold_mm = 2.0 * step_mm   # longest gap bridged by a link instead of a retract

    # ── Stage 6 · Per-pass ordering + terrain link insertion ─────────────
    # For Zig-Zag, a short gap between two segments both belonging to the
    # same region is bridged by walking that region's own contour ring
    # (region_contours), whenever both gap endpoints snap onto the same
    # ring within RING_SNAP_TOL_PX -- e.g. a scan line split by a concave
    # notch resumes by tracing the notch's own boundary rather than cutting
    # across it. Every other gap (any raster_dir other than Zig-Zag, or a
    # Zig-Zag gap that doesn't resolve to one shared ring -- crossing
    # between two different regions, or between a region's outer boundary
    # and an interior hole/island) falls back to the same terrain linking
    # compute_raster_toolpath uses (_geodesic_link), but against flat_depth
    # -- the flat pocket-floor abstraction built in Stage 4 -- rather than
    # the real achievable-depth surface. Within the mask flat_depth is
    # constant, so a short gap between two infill lines/legs is bridged by
    # a level G1 move at the current pass's ceiling instead of a full
    # retract/rapid/plunge; a gap that must cross outside the mask rises
    # only to the surface (Z=0), it never dips to match whatever relief the
    # real depth map happens to have there, since the pocket floor isn't
    # cutting to that relief.
    final_norm = final_stepdown_mm / max_depth
    all_segments = []
    tool_pos = (0.0, 0.0)
    total_segs = 0
    n_links_inserted = 0
    n_links_contour = 0
    n_links = 0
    for pi, (ceil_mm, is_final, pass_segs) in enumerate(pass_list):
        if not pass_segs:
            continue
        progress(70 + 15 * pi / n_passes,
                 f"Pass {pi+1}/{n_passes} at z={ceil_mm:.3f} mm "
                 f"({'final' if is_final else 'roughing'}): "
                 f"{len(pass_segs)} segment(s).")
        # pass_ids[pi] is aligned one-to-one with pass_segs (see Stage 5).
        out, tool_pos, counts = _order_and_link_pocket_pass(
            pass_segs, pass_ids[pi], ceil_mm, is_final, tool_pos,
            pattern, link_threshold_mm, labeled_regions,
            region_contours, flat_depth, final_norm,
            px_per_mm, H, W, _ox, _oy, max_depth,
            chain_parents=chain_parents)
        all_segments.extend(out)
        total_segs += counts[0]
        n_links_inserted += counts[1]
        n_links_contour += counts[2]
        n_links += counts[3]

    progress(85, f"Pocket complete — {total_segs} cut segments, "
                 f"{n_links_inserted} terrain links across all passes.")

    # ── Stage 7 · Simplify, simulate result, create ToolPath (85–100 %) ──
    progress(85, "Simplifying tool path…")
    rdp_tol   = 1.0 / px_per_mm
    rdp_z_tol = max_depth / (n_grey_levels - 1)
    simplified = [simplify_line(pts, rdp_tol, rdp_z_tol) for pts in all_segments]

    progress(90, "Simulating tool-swept removal surface…")
    material_removed_pocket = simulate_toolpath_result(
        ToolPath(segments=simplified),
        (H, W), px_per_mm, px_per_mm, _ox, _oy,
        max_depth, kernel_pocket, footprint_pocket,
    )
    pocket_result = np.maximum(
        material_removed_pocket,
        previous_result if previous_result is not None else 0.0).astype(np.float32)

    progress(95, "Creating ToolPath object...")
    toolpath = ToolPath(
        segments=simplified,
        result=pocket_result,
        accessible=accessible_pocket,
        metadata={
            'algorithm': 'Pocket',
            'profile': tool.profile,
            'radius_mm': tool.diameter_mm / 2,
            'step_mm': step_mm,
            'stepdown_mm': stepdown_mm,
            'final_stepdown_mm': final_stepdown_mm,
            'boundary_offset_mm': boundary_offset_mm,
            'target_z_mm': pocket_depth_mm,
            'max_depth': max_depth,
            'width_mm': width_mm,
            'height_mm': height_mm,
            'n_passes': n_passes,
            'n_regions': n_regions,
            'rdp_tolerance': rdp_tol,
            'rdp_z_tolerance': rdp_z_tol,
            'link_threshold_mm': link_threshold_mm,
            'n_links_inserted': n_links_inserted,
            'n_links_contour': n_links_contour,
            'n_links_total': n_links,
            'air_pruned_mm': air_pruned_mm,
        }
    )

    progress(100, "Tool path computation complete!")

    return toolpath


def level_clearing_surface(depth_map, tool, px_per_mm, max_depth,
                           extra_offset_mm=0.0, erosion_cache=None,
                           erosion_cache_key=None, progress=None):
    """Level Clearing's tool-centre surface: Contour's grey erosion of
    *depth_map* with the stock-to-leave planning kernel, cached under the
    same key as Contour's, minus Extra offset (normalised).

    Returns (center_depth, kernel, footprint, kernel_plan, footprint_plan):
    the clipped erosion, the real tool's kernel (contact and result
    simulation) and the planning kernel.
    """
    from grey_morphology import grey_erosion, grey_dilation

    if progress is None:
        progress = lambda pct, msg: None
    progress(0, "Building tool geometry kernel…")
    kernel, footprint = _build_kernel(tool, px_per_mm, max_depth)
    kernel_plan, footprint_plan = _planning_kernel(
        tool, px_per_mm, max_depth, extra_offset_mm, kernel, footprint)
    progress(10, f"Kernel ready  ({kernel.shape[0]}×{kernel.shape[0]} px, "
                 f"profile: {tool.profile}).")

    cached = erosion_cache.get(erosion_cache_key) if erosion_cache is not None else None
    if cached is not None and cached[0].shape == depth_map.shape:
        center_depth = cached[0]
        progress(35, "Reusing achievable depth surface (same tool, unchanged)…")
    else:
        progress(10, "Computing achievable depth surface (morphological opening)…")
        structure = -kernel_plan.astype(np.float64)
        center_depth = np.clip(grey_erosion(depth_map.astype(np.float32, copy=False),
                                            footprint=footprint_plan,
                                            structure=structure).astype(np.float32),
                               0.0, 1.0)
        # The opening is cached whole, in the same form Contour stores it.
        accessible = np.clip(grey_dilation(center_depth, footprint=footprint_plan,
                                           structure=structure).astype(np.float32),
                             0.0, 1.0)
        progress(35, "Achievable depth surface computed.")
        if erosion_cache is not None:
            erosion_cache[erosion_cache_key] = (center_depth, accessible)
    if extra_offset_mm > 0.0:
        center_depth = np.clip(center_depth - extra_offset_mm / max_depth, 0.0, 1.0)
    return center_depth, kernel, footprint, kernel_plan, footprint_plan


def plan_clearing_levels(center_depth, z_range, z_increment_mm, max_depth,
                         n_grey_levels, max_step_down_mm=0.0,
                         adaptive_levels=False, min_area_px=1):
    """Level Clearing's normalised level depths, top-down.

    The band is z_range with its bottom raised to the deepest tool-centre
    depth, so the relief's deepest floor is always a level. Its uniform
    levels are band_levels, Z increment apart; Adaptive moves them onto
    flat floors with plateau_levels, which counts a floor once its
    tool-centre positions cover min_area_px (callers pass a quarter of the
    tool footprint: a floor about 1.5 tool diameters across). Max step-down
    > 0 then fills every gap wider than it (_fill_level_gaps). No levels
    when the surface never goes deeper than the band's top. A band with
    lo == hi is that single level, when the surface reaches it.
    """
    lo, hi = check_z_range(z_range)
    eps = 0.5 / max(n_grey_levels - 1, 1)
    if lo == hi:
        return [hi] if float(center_depth.max()) >= hi - eps else []
    hi = min(hi, float(center_depth.max()))
    if hi <= lo + eps:
        return []
    levels = band_levels((lo, hi), z_increment_mm, max_depth)
    if adaptive_levels:
        levels = plateau_levels(center_depth, levels, lo, n_grey_levels, min_area_px)
    if max_step_down_mm and max_step_down_mm > 0.0:
        levels = _fill_level_gaps(levels, max_step_down_mm / max_depth, hi, start=lo)
    return levels


def level_clearing_mask(center_depth, level_norm, n_grey_levels):
    """Tool-centre positions cleared at level_norm: the erosion reaches it
    (with Pocket's half-grey-level tolerance)."""
    return center_depth >= (level_norm - 0.5 / max(n_grey_levels - 1, 1))


def compute_level_clearing_toolpath(depth_map, n_grey_levels, tool,
                                    step_mm, z_increment_mm, max_depth,
                                    width_mm, height_mm, raster_dir,
                                    origin="Bottom-Left",
                                    feed_rate=800.0,
                                    previous_result=None,
                                    raster_offset_pct=0.0,
                                    cut_direction="Climb",
                                    max_step_down_mm=0.0,
                                    adaptive_levels=False,
                                    z_range=(0.0, 1.0),
                                    extra_offset_mm=0.0,
                                    avoid_air=False,
                                    z_safe=5.0,
                                    feed_rapid=3000.0,
                                    progress=None,
                                    erosion_cache=None,
                                    erosion_cache_key=None):
    """Z-level roughing: a flat Pocket clearing at every level, top-down.

    Levels are planned like Contour's (plan_clearing_levels: Z increment,
    Adaptive, Z range, Max step-down). At level z_k the tool-centre
    positions whose erosion reaches z_k (level_clearing_mask) are cleared
    flat at z_k with Pocket's infill (_pocket_infill, raster_dir any of
    Pocket's Scan directions). These sets nest -- a deeper threshold only
    shrinks them -- so every region at level k lies inside one region at
    level k-1, which was cleared down to z_(k-1) before it: each level
    removes at most its own spacing, in one pass.

    Regions form a forest (each region's parent is the region one level up
    containing it) and are cut depth-first: after a region, the nearest
    uncut child of the deepest region still having one, so one cavity is
    cleared to the bottom before the next. Within a region, Pocket's own
    ordering and linking (_order_and_link_pocket_pass) applies; its
    geodesic links follow the floor already cleared by the regions cut so
    far rather than rising to the stock surface. Moves between regions are
    retract/rapid/plunge.

    Extra offset leaves stock on walls and floors, as in Contour. With
    avoid_air, segment portions where previous_result already cleared the
    material are removed (prune_air_segments), so a second step refines
    the first one's terraces and floors.

    Returns a ToolPath. Its accessible surface is the tool's reach from the
    erosion rounded down to the deepest level at or above it -- the
    terraces this step leaves.
    """
    from scipy.ndimage import find_objects
    from grey_morphology import grey_dilation

    if progress is None:
        progress = lambda pct, msg: None
    z_lo, z_hi = check_z_range(z_range)

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
           "Center":      W_mm/2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm/2}.get(origin, 0)

    # ── Stages 1–2 · Kernel and tool-centre surface (0–35 %) ──────────────
    center_depth, kernel, footprint, kernel_plan, footprint_plan = level_clearing_surface(
        depth_map, tool, px_per_mm, max_depth, extra_offset_mm,
        erosion_cache, erosion_cache_key, progress)

    # ── Stage 3 · Level planning (35–40 %) ────────────────────────────────
    progress(35, "Planning clearing levels…")
    levels = plan_clearing_levels(
        center_depth, (z_lo, z_hi), z_increment_mm, max_depth, n_grey_levels,
        max_step_down_mm, adaptive_levels, max(4, int(footprint_plan.sum()) // 4))
    levels_mm = [z * max_depth for z in levels]
    progress(40, f"{len(levels)} clearing level{'s' if len(levels) != 1 else ''}"
                 f"{' (adaptive)' if adaptive_levels else ''}.")

    pattern = _pocket_pattern(raster_dir, cut_direction)
    base_meta = {'algorithm': 'Level Clearing', 'profile': tool.profile,
                 'radius_mm': tool.diameter_mm / 2, 'step_mm': step_mm,
                 'z_increment_mm': z_increment_mm,
                 'max_step_down_mm': max_step_down_mm,
                 'adaptive_levels': adaptive_levels, 'z_range': (z_lo, z_hi),
                 'levels_mm': levels_mm, 'max_depth': max_depth,
                 'width_mm': width_mm, 'height_mm': height_mm}

    # ── Stage 4 · Per-level infill + air pruning (40–60 %) ────────────────
    free = None
    if avoid_air:
        # A position within one pixel of an air position counts as air: the
        # earlier step's path was simplified by up to a pixel (RDP), so a
        # level retracing one of its levels would otherwise keep slivers
        # along every steep wall.
        from scipy.ndimage import maximum_filter
        free = maximum_filter(
            compute_air_free_depth(previous_result, kernel, footprint, (H, W)), size=3)

        def breakeven(depth_mm):
            return _retract_breakeven_mm(feed_rate, feed_rapid, z_safe, depth_mm)
    tol_norm = 1.0 / (n_grey_levels - 1)
    # One entry per level: (labeled_regions, region_contours, region boxes,
    # {region id: ([segment, ...], [its infill segment index, ...])},
    # chain_parents).
    level_data = []
    air_pruned_mm = 0.0
    n_regions_per_level = []
    n_lv = max(len(levels), 1)
    # Every level's regions are submitted to one runner before any is
    # collected, so that the regions of all levels compute in parallel.
    with _RegionRunner() as runner:
        pending = []
        for k, (z, z_mm) in enumerate(zip(levels, levels_mm)):
            progress(40 + 5 * k / n_lv,
                     f"Level {k + 1}/{len(levels)} at z={z_mm:.3f} mm: regions…")
            mask = level_clearing_mask(center_depth, z, n_grey_levels)
            pending.append(_start_pocket_infill(
                mask, z, z_mm, pattern, step_mm, raster_offset_pct,
                cut_direction, px_per_mm, H, _ox, _oy, max_depth, runner,
                tool_radius_mm=0.5 * tool.diameter_mm))
            del mask
        for k, z_mm in enumerate(levels_mm):
            progress(45 + 15 * k / n_lv,
                     f"Level {k + 1}/{len(levels)} at z={z_mm:.3f} mm: infill…")
            (segs, seg_region_ids, chain_parents, region_contours, labeled, _flat,
             n_regions) = pending[k]()
            pending[k] = None
            # src: the infill segment (for Onion Skin, the chain) each
            # segment comes from.
            src = list(range(len(segs)))
            if avoid_air and segs:
                segs, src, air = prune_air_segments(
                    segs, free, px_per_mm, H, _ox, _oy, max_depth, tol_norm,
                    breakeven_fn=breakeven, parent_ids=src)
                air_pruned_mm += air
            by_region = {}
            for seg, si in zip(segs, src):
                entry = by_region.setdefault(int(seg_region_ids[si]), ([], []))
                entry[0].append(seg)
                entry[1].append(si)
            level_data.append((labeled, region_contours, find_objects(labeled),
                               by_region, chain_parents))
            n_regions_per_level.append(n_regions)

    # ── Stage 5 · Region forest (60 %) ────────────────────────────────────
    # A region's parent is the region one level up holding any of its
    # pixels (the level masks nest, so all of them).
    children = {}   # (k, rid) -> [(k + 1, child rid), ...]; (-1, 0) is the root
    for k, (labeled, _rc, boxes, _br, _cp) in enumerate(level_data):
        for rid, box in enumerate(boxes, start=1):
            if box is None:
                continue
            if k == 0:
                parent = (-1, 0)
            else:
                local = labeled[box] == rid
                r, c = np.argwhere(local)[0]
                prev_labeled = level_data[k - 1][0]
                parent = (k - 1, int(prev_labeled[box[0].start + r, box[1].start + c]))
            children.setdefault(parent, []).append((k, rid))

    # ── Stage 6 · Depth-first cutting order + linking (60–85 %) ───────────
    # cleared: the normalised floor the regions cut so far leave at every
    # tool-centre pixel, which geodesic links follow.
    cleared = np.zeros((H, W), dtype=np.float32)
    link_threshold_mm = 2.0 * step_mm
    all_segments = []
    tool_pos = (0.0, 0.0)
    n_cut = n_links_inserted = n_links_contour = n_links = 0
    n_nodes = sum(len(v) for v in children.values())
    n_done = 0

    def node_segments(node):
        k, rid = node
        return level_data[k][3].get(rid, ([], []))[0]

    def node_distance(node):
        segs = node_segments(node)
        if not segs:
            return 0.0   # nothing to cut: visited at once, releasing its children
        return min(min(np.hypot(s[0][0] - tool_pos[0], s[0][1] - tool_pos[1]),
                       np.hypot(s[-1][0] - tool_pos[0], s[-1][1] - tool_pos[1]))
                   for s in segs)

    stack = [(-1, 0)]
    pending = {node: list(kids) for node, kids in children.items()}
    while stack:
        kids = pending.get(stack[-1])
        if not kids:
            stack.pop()
            continue
        node = min(kids, key=node_distance)
        kids.remove(node)
        k, rid = node
        labeled, region_contours, boxes, by_region, chain_parents = level_data[k]
        box = boxes[rid - 1]
        in_region = labeled[box] == rid
        cleared[box][in_region] = np.maximum(cleared[box][in_region], levels[k])
        segs, src = by_region.get(rid, ([], []))
        if segs:
            out, tool_pos, counts = _order_and_link_pocket_pass(
                segs, src if pattern.is_onion else [rid] * len(segs),
                levels_mm[k], True, tool_pos,
                pattern, link_threshold_mm, labeled,
                region_contours, cleared, 0.0,
                px_per_mm, H, W, _ox, _oy, max_depth,
                chain_parents=chain_parents)
            all_segments.extend(out)
            n_cut += counts[0]
            n_links_inserted += counts[1]
            n_links_contour += counts[2]
            n_links += counts[3]
        n_done += 1
        progress(60 + 25 * n_done / max(n_nodes, 1),
                 f"Region {n_done}/{n_nodes} (level {k + 1}) ordered.")
        stack.append(node)

    progress(85, f"Level Clearing complete — {n_cut} cut segments, "
                 f"{n_links_inserted} links across {len(levels)} level(s).")

    # ── Stage 7 · Simplify, simulate result, accessible (85–100 %) ────────
    progress(85, "Simplifying tool path…")
    rdp_tol   = 1.0 / px_per_mm
    rdp_z_tol = max_depth / (n_grey_levels - 1)
    simplified = [simplify_line(pts, rdp_tol, rdp_z_tol) for pts in all_segments]

    progress(90, "Simulating tool-swept removal surface…")
    removed = simulate_toolpath_result(
        ToolPath(segments=simplified), (H, W), px_per_mm, px_per_mm, _ox, _oy,
        max_depth, kernel, footprint)
    result = np.maximum(
        removed, previous_result if previous_result is not None else 0.0).astype(np.float32)

    # Each tool centre reaches the deepest level at or above its erosion
    # (positions above the first level stay at most at the band's top).
    progress(95, "Computing accessible surface…")
    if levels:
        eps = 0.5 / max(n_grey_levels - 1, 1)
        lv = np.asarray(levels, dtype=np.float32)
        n_reached = np.searchsorted(lv - eps, center_depth, side='right')
        reached = np.where(n_reached > 0, lv[np.maximum(n_reached - 1, 0)],
                           np.minimum(center_depth, z_lo)).astype(np.float32)
    else:
        reached = np.minimum(center_depth, z_lo).astype(np.float32)
    accessible = np.clip(grey_dilation(reached, footprint=footprint_plan,
                                       structure=-kernel_plan.astype(np.float64)),
                         0.0, 1.0).astype(np.float32)

    toolpath = ToolPath(
        segments=simplified, result=result, accessible=accessible,
        metadata={**base_meta,
                  'n_levels': len(levels),
                  'n_regions_per_level': n_regions_per_level,
                  'n_regions': sum(n_regions_per_level),
                  'rdp_tolerance': rdp_tol, 'rdp_z_tolerance': rdp_z_tol,
                  'link_threshold_mm': link_threshold_mm,
                  'n_links_inserted': n_links_inserted,
                  'n_links_contour': n_links_contour,
                  'n_links_total': n_links,
                  'air_pruned_mm': air_pruned_mm})
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
        from grey_morphology import grey_opening as _grey_opening
        radius = max(1, int(np.ceil((min_feature_width_mm / 2.0) * px_per_mm)))
        filtered = _grey_opening(
            ref.astype(np.float32, copy=False), footprint=_disk(radius)
        ).astype(np.float32, copy=False)
    mask = filtered >= threshold_norm
    if max_feature_width_mm > 0.0:
        from scipy.ndimage import distance_transform_edt as _edt
        edt = _edt(mask)
        mask &= edt <= (max_feature_width_mm / 2.0) * px_per_mm
    filtered = np.where(mask, filtered, np.float32(0.0))
    return filtered, mask


def resolve_vcarve_threshold_norm(ref, threshold_mm, max_depth):
    """Resolve threshold_mm (mm) to a normalised [0,1] split point against
    *ref*'s own value range.

    threshold_mm > 0 is used literally; 0 (or unset) means "auto": split at
    the midpoint of ref's own value range rather than at depth 0. ref's
    floor is always 0 (background/already-cut material), so a literal 0
    threshold would keep every pixel -- including the flat background --
    turning the V-carving skeleton into the medial axis of the whole canvas
    rectangle instead of the actual artwork.
    """
    if threshold_mm > 0.0:
        return threshold_mm / max_depth if max_depth else 0.0
    return float(ref.min() + ref.max()) / 2.0


def compute_vcarve_synth_achievable_depth(depth_map, tool, max_depth,
                                           width_mm, height_mm,
                                           extra_offset_mm=0.0,
                                           erosion_cache=None,
                                           erosion_cache_key=None,
                                           progress=None):
    """Achievable-depth surface for an idealized ("just large enough") V-bit
    of tool.vbit_angle_deg that always tapers to a point exactly at
    max_depth. Used by V-Carving (Hybrid)'s own deep waterline contours, and
    by V-Carving Clean-up, which erodes this surface a second time by its
    own clearing tool's kernel to find the boundary it must retreat to so
    it doesn't gouge the V-bit's actual sloped wall.

    depth_map/width_mm/height_mm are the raw (not pre-resampled) inputs --
    resampled to square pixels here exactly as every compute_*_toolpath
    function's own Stage 0 does, so this is safe to call standalone.
    """
    from math import radians, tan
    from grey_morphology import grey_erosion

    if progress is None:
        progress = lambda pct, msg: None

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
        H, W = depth_map.shape
        px_per_mm_x = px_per_mm_y = target
    px_per_mm = px_per_mm_x

    tan_half   = tan(radians(tool.vbit_angle_deg / 2.0))
    synth_diam = 2.0 * max_depth * tan_half * (1.0 + 1e-6)

    # The deep contour rings need an achievable-depth surface that always
    # tapers to a point at max_depth (see _find_compatible_vbit_erosion's
    # docstring) — build it with an idealized, "just large enough" V-bit
    # rather than the real tool, reusing any compatible cached erosion
    # first since it can be expensive to recompute.
    center_depth_contour_synth = _find_compatible_vbit_erosion(
        erosion_cache, erosion_cache_key, synth_diam)
    if center_depth_contour_synth is None:
        synth_tool = dataclasses.replace(tool, diameter_mm=synth_diam)
        kernel_synth, footprint_synth = _build_kernel(
            synth_tool, px_per_mm, max_depth)
        kernel_synth_plan, footprint_synth_plan = _planning_kernel(
            synth_tool, px_per_mm, max_depth, extra_offset_mm,
            kernel_synth, footprint_synth)
        center_depth_contour_synth = grey_erosion(
            depth_map.astype(np.float32, copy=False),
            footprint=footprint_synth_plan,
            structure=-kernel_synth_plan.astype(np.float64),
        ).astype(np.float32)
        center_depth_contour_synth = np.clip(center_depth_contour_synth, 0.0, 1.0)
        if erosion_cache is not None and erosion_cache_key is not None:
            synth_key = (tool.profile, synth_diam, *erosion_cache_key[2:])
            erosion_cache[synth_key] = (center_depth_contour_synth,
                                         center_depth_contour_synth)

    if extra_offset_mm > 0.0:
        # Mirror Stage 2b's vertical stock-to-leave adjustment, applied
        # here regardless of whether the surface above came from cache or
        # was just computed (cache always stores the pre-adjustment
        # surface).
        off_norm = extra_offset_mm / max_depth
        center_depth_contour_synth = np.clip(
            center_depth_contour_synth - off_norm, 0.0, 1.0)

    return center_depth_contour_synth


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
                            cut_direction="Climb",
                            extra_offset_mm=0.0,
                            avoid_air=False,
                            z_safe=5.0,
                            feed_rapid=3000.0,
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
                            than this are skipped. 0 (the default) means
                            "auto": split at the midpoint of the uncut
                            depth's own value range instead, since that
                            depth's floor is always 0 and a literal 0
                            threshold would keep the flat background in the
                            mask along with the artwork.
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
    skeleton_mode         : "medial_axis" | "voronoi" | "inscribed" | "hybrid"
                            "hybrid" combines the full, unpruned "inscribed"
                            skeleton (capturing sharp corners/fine detail at
                            every depth) with waterline contour rings from
                            stepdown_mm to max_depth, added alongside it to
                            also clear mask regions wider than the tool's
                            cone can reach in a single skeleton pass.
                            stepdown_mm=0 collapses this to a single ring
                            at max_depth (see _plan_passes). Requires
                            tool.profile == 'V-Bit'.
    cut_direction         : "Either" | "Climb" | "Conventional" — only used by
                            "hybrid" mode's deep-layer contour rings (see
                            compute_contour_toolpath's cut_direction docs);
                            ignored by the other skeleton_mode values, whose
                            single medial-axis engraving line has no
                            inside/outside to keep on a particular side.
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
    avoid_air             : if True, remove the portions of every pass where the
                            tool would cut only air -- see prune_air_segments.
    z_safe, feed_rapid    : safe height (mm) and rapid rate (mm/min), used by
                            avoid_air's retract/plunge breakeven.
    progress              : callable(pct, message) — thread-safe progress reporter

    Returns
    -------
    ToolPath
    """
    from scipy.ndimage import distance_transform_edt, label
    from grey_morphology import grey_erosion, grey_dilation

    _alg_name = ('V-Carving (Sub-pixel)'    if skeleton_mode == 'voronoi'
                 else 'V-Carving (Feature Width)' if skeleton_mode == 'inscribed'
                 else 'V-Carving (Hybrid)' if skeleton_mode == 'hybrid'
                 else 'V-Carving (Depth Map)')

    _cone_reach_warning = None
    if skeleton_mode == 'hybrid':
        if tool.profile != 'V-Bit':
            raise ValueError("skeleton_mode='hybrid' currently supports only "
                             "V-Bit tools.")
        # stepdown_mm=0 means a single roughing step at full depth (see
        # _plan_passes), so the cone only needs to reach max_depth in that
        # one step rather than the (nonexistent) 0 mm increment.
        _cone_reach_warning = _vbit_cone_reach_check(
            tool, stepdown_mm if stepdown_mm > 0.0 else max_depth)
        if _cone_reach_warning:
            progress(0, _cone_reach_warning)

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
    if tool.profile == "Custom" and not tool.profile_curve:
        raise ValueError("Custom profile has no curve.")
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
            depth_map.astype(np.float32, copy=False),
            footprint=footprint_plan,
            structure=-kernel_plan.astype(np.float64),
        ).astype(np.float32)
        center_depth_contour_vcarve = np.clip(center_depth_contour_vcarve, 0.0, 1.0)

        accessible_vcarve = grey_dilation(
            center_depth_contour_vcarve,
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

    threshold_norm = resolve_vcarve_threshold_norm(uncut_depth, threshold_mm, max_depth)

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
    if skeleton_mode in ('voronoi', 'inscribed', 'hybrid'):
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

    # ── Inscribed-circle depth map (for 'inscribed' and 'hybrid' modes) ──────
    if skeleton_mode in ('inscribed', 'hybrid'):
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
        elif tool.profile == 'Custom':
            inscribed_depth_norm = tool_profile.inscribed_depth(
                tool.profile_curve, radius_map / px_per_mm) / max_depth
        else:  # End Mill — inscribed circle does not constrain depth
            progress(48, "Warning: 'inscribed' mode is not meaningful for End Mill; "
                         "falling back to grey-erosion depth.")
            inscribed_depth_norm = center_depth_contour_vcarve
        inscribed_depth_norm = np.clip(inscribed_depth_norm, 0.0, 1.0)

    _depth_norm = (inscribed_depth_norm if skeleton_mode in ('inscribed', 'hybrid')
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

    if skeleton_mode == 'hybrid':
        # Skeleton: the full, unpruned inscribed-depth path (capturing sharp
        # corners/fine detail at every depth, not just a shallow top layer).
        # 2-opt is applied to it alone (rather than to the combined list
        # below) since a combined-list 2-opt works purely on XY proximity
        # and would freely interleave skeleton strokes with deep contour
        # rings, undoing the top-down order of the contour graph that the
        # rings get from _order_contours_graph.
        skeleton_segs = _two_opt_pass(full_depth_segs, _pass_origin(all_segments))

        # Waterline contour rings from stepdown_mm to max_depth, added
        # alongside the skeleton to also clear mask regions wider than the
        # tool's cone can reach in a single skeleton pass —
        # reusing compute_contour_toolpath's extraction/ordering helpers.
        center_depth_contour_synth = compute_vcarve_synth_achievable_depth(
            depth_map, tool, max_depth, width_mm, height_mm,
            extra_offset_mm=extra_offset_mm,
            erosion_cache=erosion_cache, erosion_cache_key=erosion_cache_key,
            progress=progress)

        deep_levels_norm = [c / max_depth for c in
                             _plan_passes(max_depth, stepdown_mm, final_stepdown_mm)]
        all_z_contours = _extract_z_contours(
            center_depth_contour_synth, deep_levels_norm, n_grey_levels,
            cut_direction=cut_direction)
        all_z_contours = _clip_contours_to_mask(all_z_contours, mask_vcarve)
        contour_segs, _ = _order_contours_graph(
            all_z_contours, deep_levels_norm, center_depth_contour_synth, n_grey_levels,
            max_depth=max_depth,
            px_per_mm=px_per_mm, H=H, ox=_ox, oy=_oy,
            lock_direction=cut_direction in ("Climb", "Conventional"),
            current_pos=(0.0, 0.0),
            progress=progress, progress_lo=80, progress_hi=88)

        full_depth_segs = skeleton_segs + contour_segs
    else:
        full_depth_segs = _two_opt_pass(full_depth_segs, _pass_origin(all_segments))

    # Simplify the full-depth path before multipass depth-slicing. RDP's
    # "distance from chord" is measured in the full (x, y, z) space, so
    # simplifying beforehand means every pass make_multipass produces below
    # shares the same XY vertex sequence — only Z differs between passes —
    # keeping their footprints perfectly aligned when viewed from above.
    progress(88, "Simplifying tool path (RDP)…")
    rdp_tol   = 1.0 / px_per_mm
    rdp_z_tol = max_depth / (n_grey_levels - 1)
    full_depth_segs = [simplify_line(pts, rdp_tol, rdp_z_tol) for pts in full_depth_segs]

    # Phases 2–3: depth-slice, drop air-cutting runs (optional), per-pass
    # 2-opt reorder
    pass_list = make_multipass(full_depth_segs, max_depth, stepdown_mm, final_stepdown_mm)
    air_pruned_mm = 0.0
    if avoid_air:
        progress(89, "Removing air-cutting portions of each pass…")
        pass_list, _, air_pruned_mm = _prune_air_passes(
            pass_list, previous_result, kernel_vcarve, footprint_vcarve, (H, W),
            px_per_mm, _ox, _oy, max_depth, 1.0 / (n_grey_levels - 1),
            feed_rate, feed_rapid, z_safe)
        progress(89, f"Removed {air_pruned_mm:.0f} mm of air cutting.")
    for _, _, pass_segs in pass_list:
        pass_segs = _two_opt_pass(pass_segs, _pass_origin(all_segments))
        all_segments.extend(pass_segs)

    progress(90, f"Generated {len(all_segments)} segments across all passes.")

    # ── Stage 6 · Simulate result, assemble ToolPath (90–100 %) ───────
    progress(93, "Simulating tool-swept removal surface…")
    vcarve_result = simulate_toolpath_result(
        ToolPath(segments=all_segments),
        (H, W), px_per_mm, px_per_mm, _ox, _oy,
        max_depth, kernel_vcarve, footprint_vcarve,
    )

    progress(97, "Creating ToolPath object…")
    toolpath = ToolPath(
        segments=all_segments,
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
            'rdp_z_tolerance': rdp_z_tol,
            'cone_reach_warning': _cone_reach_warning,
            'cut_direction':   cut_direction,
            'air_pruned_mm':   air_pruned_mm,
        }
    )
    progress(100, "V-carve tool path computation complete!")
    return toolpath


def _split_runs(c, keep_mask, min_pts=3, min_prune_arc_px=0.0):
    """Split a pixel-space (row, col) polyline c into the runs where
    keep_mask is True, returning a list of arrays of at least min_pts
    points. A closed polyline (ends within 1.5 px) is split starting at
    its first kept run, so a dropped run that wraps across the ends stays
    one run. Dropped runs shorter than min_prune_arc_px in arc length are
    restored before splitting (retract+plunge breakeven)."""
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
    level above (the stock surface, for the shallowest level). Otherwise it
    is kept even though laterally redundant: without it, the cut at the level
    above would be followed by that deeper cut, more than one stepdown lower.
    A depth gap exactly equal to max_step_down_mm is treated as a legal
    single-pass stepdown and remains prunable.

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

    pruned = {}
    effective_next_img = None
    effective_next_depth = None

    for i, z_norm in reversed(list(enumerate(z_levels))):
        contours = all_z_contours.get(z_norm, [])
        z_level_mm = z_norm * max_depth
        z_above_mm = z_levels[i - 1] * max_depth if i > 0 else 0.0
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
        min_prune_arc_px = (_retract_breakeven_mm(feed_rate, feed_rapid, z_safe, z_level_mm)
                            * px_per_mm)

        dt, nearest_idx = distance_transform_edt(~effective_next_img, return_indices=True)

        kept = []
        for c in contours:
            rows = np.round(c[:, 0]).astype(int).clip(0, H - 1)
            cols = np.round(c[:, 1]).astype(int).clip(0, W - 1)
            keep_mask = dt[rows, cols] >= lateral_tol_px
            if max_step_down_mm is not None:
                near_depth_mm = effective_next_depth[nearest_idx[0][rows, cols],
                                                      nearest_idx[1][rows, cols]]
                depth_gap_mm = near_depth_mm - z_above_mm
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


def _prune_air_contours(all_z_contours, free_depth, tol_norm, px_per_mm,
                        max_depth, breakeven_fn):
    """Air-cutting counterpart to _prune_contours for Contour's pixel-space
    (row, col) rings: at each z-level z_norm, a point cuts material iff
    z_norm > free_depth[row, col] + tol_norm (see compute_air_free_depth).
    Kept runs are widened by one point into the adjacent air on both sides
    (wrapping around a closed ring), then split with _split_runs, which
    restores dropped runs shorter than breakeven_fn(level depth in mm).

    Returns (pruned dict with the same keys, total removed length in mm).
    """
    Hf, Wf = free_depth.shape
    pruned, air_px = {}, 0.0
    for z_norm, contours in all_z_contours.items():
        min_arc_px = breakeven_fn(z_norm * max_depth) * px_per_mm
        kept = []
        for c in contours:
            rows = np.round(c[:, 0]).astype(int).clip(0, Hf - 1)
            cols = np.round(c[:, 1]).astype(int).clip(0, Wf - 1)
            cut = z_norm > free_depth[rows, cols] + tol_norm
            if np.linalg.norm(c[0] - c[-1]) < 1.5:
                keep = cut | np.roll(cut, 1) | np.roll(cut, -1)
            else:
                keep = cut.copy()
                keep[1:] |= cut[:-1]
                keep[:-1] |= cut[1:]
            parts = _split_runs(c, keep, min_prune_arc_px=min_arc_px)
            before = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
            after = sum(float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
                        for p in parts)
            air_px += max(before - after, 0.0)
            kept.extend(parts)
        pruned[z_norm] = kept
    return pruned, air_px / px_per_mm


def _clip_contours_to_mask(all_z_contours, mask, min_pts=3):
    """Restrict each contour ring to the portion inside `mask`.

    Splits each (row, col) pixel-space polyline at mask-boundary crossings,
    dropping runs entirely outside the mask and runs shorter than min_pts
    points. Used to restrict hybrid V-carving's deep-layer waterline
    contours — extracted from an unmasked achievable-depth surface, see
    compute_vcarve_toolpath's 'hybrid' branch — to the region
    compute_vcarve_mask actually flagged as needing V-carving, without
    inventing spurious geometry at the mask boundary (which zeroing the
    surface outside the mask before extraction would do).

    Returns a dict with the same keys as all_z_contours.
    """
    H, W = mask.shape
    clipped = {}
    for z_norm, contours in all_z_contours.items():
        kept = []
        for c in contours:
            n = len(c)
            rows = np.round(c[:, 0]).astype(int).clip(0, H - 1)
            cols = np.round(c[:, 1]).astype(int).clip(0, W - 1)
            keep_mask = mask[rows, cols]
            if not keep_mask.any():
                continue
            if keep_mask.all():
                kept.append(c)
                continue

            is_closed = np.linalg.norm(c[0] - c[-1]) < 1.5
            if is_closed:
                start = next((i for i in range(n)
                              if keep_mask[i] and not keep_mask[(i - 1) % n]), 0)
                indices = [(start + i) % n for i in range(n)]
            else:
                indices = list(range(n))

            run = []
            for idx in indices:
                if keep_mask[idx]:
                    run.append(c[idx])
                else:
                    if len(run) >= min_pts:
                        kept.append(np.array(run))
                    run = []
            if len(run) >= min_pts:
                kept.append(np.array(run))
        clipped[z_norm] = kept
    return clipped


def _contour_threshold(z_norm, n_grey_levels):
    """find_contours threshold of the contours at normalised level z_norm.

    Half a grey-level step below z_norm: that keeps flat-region pixels
    firmly "inside" even when z_norm lands at or just above their exact
    value (e.g. from CDF quantile rounding). Without it the contour jumps to
    the inner wall of a flat step and the tool cuts air above the
    neighbouring deeper region.
    """
    return max(z_norm - 0.5 / max(n_grey_levels - 1, 1), 0.0)


def check_z_range(z_range):
    """Return z_range as a (lo, hi) float pair, or raise ValueError unless
    0 <= lo <= hi <= 1. lo == hi is the single level hi (0: the stock
    surface)."""
    lo, hi = float(z_range[0]), float(z_range[1])
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"Z range must satisfy 0 % <= min <= max <= 100 % "
                         f"(got {lo * 100:g} % – {hi * 100:g} %).")
    return lo, hi


def band_levels(z_range, z_increment_mm, max_depth):
    """Uniform normalised depth levels lo + i·z_increment_mm (i >= 1) inside
    z_range = (lo, hi), the last one capped at hi, so hi itself is always
    the deepest level and lo is one only when lo == hi (the single level
    hi)."""
    import math
    lo, hi = z_range
    n = max(1, math.ceil((hi - lo) * max_depth / z_increment_mm))
    levels = []
    for i in range(1, n + 1):
        z = min(lo + i * z_increment_mm / max_depth, hi)
        if not levels or z > levels[-1]:
            levels.append(z)
    return levels


def _fill_level_gaps(z_levels, max_gap, deepest=1.0, start=0.0):
    """Insert evenly spaced levels wherever two consecutive levels, or the
    start depth and the first level, are more than max_gap apart.

    z_levels : ascending normalised depths, all kept as they are
    max_gap  : largest allowed spacing, normalised (a gap exactly this large
               stays unfilled)
    deepest  : deepest point of the surface; a gap starting at or below it
               holds no contours, so it stays unfilled
    start    : depth the first gap starts at (0 = the stock surface)

    Returns the combined ascending list.
    """
    filled, prev = [], start
    for z in z_levels:
        if prev < deepest:
            n = int(np.ceil((z - prev) / max_gap - 1e-9))
            filled.extend(prev + (z - prev) * k / n for k in range(1, n))
        filled.append(z)
        prev = z
    return filled


def plateau_levels(surface, grid, lo, n_grey_levels, min_area_px):
    """Move the uniform levels *grid* (ascending normalised depths below
    lo, the last one the band's deepest) onto the flat floors of *surface*
    (a tool-centre erosion).

    Only flat pixels count: those whose 3x3 neighbourhood spans less than
    half a grey level, so the narrow strips of a steep ramp never do. A
    plateau is a depth window one grey level wide (two adjacent bins of
    half a grey level) that holds at least min_area_px flat pixels, at
    least as many as either overlapping window, and at least twice as many
    as any window one to one and a half grey levels away. The last rule
    passes a flat floor surrounded by slopes but not the evenly filled
    terraces of a gently sloping quantised ramp. A plateau's depth is the
    shallowest value in its window, so the whole floor lies at or below it.

    Every grid level but the deepest owns the depths from halfway to the
    level above (lo, for the first) to halfway to the level below, and
    moves to the plateau there with the most flat pixels, if any. So the
    level count stays len(grid), a cluster of shelves close in depth moves
    one level only, and levels away from any plateau stay on the grid,
    where an earlier step with the same grid has already cut. Returns the
    ascending levels.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    grid = [float(z) for z in grid]
    if len(grid) < 2:
        return grid
    half = 0.5 / max(n_grey_levels - 1, 1)
    hi = grid[-1]
    flat = (maximum_filter(surface, size=3) - minimum_filter(surface, size=3)) < half
    vals = surface[flat & (surface > lo + half) & (surface < hi - half)].astype(np.float64)
    if not vals.size:
        return grid
    idx = np.floor(vals / half).astype(np.int64)
    idx -= idx.min()
    counts = np.bincount(idx)
    mins = np.full(counts.size, np.inf)
    np.minimum.at(mins, idx, vals)
    pad = 3
    pair = np.zeros(counts.size + 2 * pad + 1, dtype=np.int64)
    pair[pad:pad + counts.size] = counts
    pair[pad:pad + counts.size] += np.append(counts[1:], 0)
    mins_p = np.full(pair.size, np.inf)
    mins_p[pad:pad + counts.size] = mins
    plateaus = []   # (flat pixel count, depth)
    for i in np.nonzero(pair >= max(1, min_area_px))[0]:
        c = pair[i]
        if c < pair[i - 1] or c <= pair[i + 1]:
            continue
        if c < 2 * max(pair[i - 3], pair[i - 2], pair[i + 2], pair[i + 3]):
            continue
        plateaus.append((int(c), float(min(mins_p[i], mins_p[i + 1]))))

    levels = list(grid)
    bounds = [lo] + [0.5 * (a + b) for a, b in zip(grid, grid[1:])]
    for j in range(len(grid) - 1):
        top, bottom = bounds[j], bounds[j + 1]
        inside = [p for p in plateaus if top < p[1] <= bottom]
        if inside:
            levels[j] = max(inside)[1]
    return sorted(levels)


def _extract_z_contours(surface, z_levels, n_grey_levels, cut_direction="Climb"):
    """Extract raw iso-depth contour polylines at each Z-level.

    surface       : 2-D float32 achievable-depth array, normalised [0, 1]
    z_levels      : iterable of normalised depths (ascending, in (0, 1])
    n_grey_levels : sizes the half-grey-level find_contours threshold epsilon
    cut_direction : "Either" | "Climb" | "Conventional" — "Conventional"
                    reverses every contour's vertex order (see
                    compute_contour_toolpath's cut_direction docs)

    Returns dict[z_norm] -> list of (row, col) Nx2 pixel-space polylines
    (>= 3 points each), in find_contours' native orientation (or reversed
    if cut_direction == "Conventional").
    """
    from skimage.measure import find_contours

    # find_contours' native (unreversed) direction keeps higher surface
    # values -- i.e. deeper-cut (valley) pixels, since surface is a depth
    # map where 1.0 = max cut depth -- consistently on the right of the
    # direction of travel, so uncut (mountain) pixels are on the left.
    # Climb requires uncut material on the right (see
    # compute_raster_toolpath's cut_direction docs for the clockwise-spindle
    # convention), so Climb is the case that needs reversing, not
    # Conventional. A single global reversal rule suffices — no per-contour
    # inside/outside analysis needed.
    reverse_contours = (cut_direction == "Climb")
    all_z_contours = {}
    for z_norm in z_levels:
        raw = find_contours(surface, _contour_threshold(z_norm, n_grey_levels))
        kept = [c for c in raw if len(c) >= 3]
        if reverse_contours:
            kept = [c[::-1] for c in kept]
        all_z_contours[z_norm] = kept
    return all_z_contours


class _LevelGraph:
    """Which pieces of a stack of nested levels must be cut before which,
    built one level at a time (add_level), shallowest first.

    At level k, the material L_k and the air D_k (its complement) are each
    split into 8-connected components. Every boundary at level k runs
    between one L_k component and one D_k component, so a node is such a
    pair (k, l, d) of components meeting across a pixel edge. Material only
    grows from level to level (L_k is inside L_{k+1}) and air only shrinks
    (D_{k+1} is inside D_k), so every L_k component lies in one L_{k+1}
    component and every D_{k+1} component in one D_k component. Node
    (k, l, d) must be cut before node (k+1, l', d') when l lies in l' and d'
    lies in d: separate pieces of material are told apart by their material,
    separate pieces of air by their air, and no boundary of either level
    separates two nodes so joined. Material merging gives a node several
    parents, air splitting gives a node several children. Where the
    boundaries of several levels coincide, the same pixel pair carries
    every level's labels, so those boundaries are joined too.
    8-connectivity can only merge components, i.e. add precedence, never
    drop it.

    Contour (_contour_graph) stacks its depth thresholds: material is what
    stands above the level, air what lies below it. Onion Skin
    (_onion_skin_chains) stacks its erosion levels: air is the shrunk
    region, material everything outside it, holes included.

    parents[n] lists the nodes that must be complete before node n.
    Nodes without pieces stay in the graph, so precedence passes through
    them.
    """

    _EIGHT = np.ones((3, 3), bool)

    def __init__(self):
        self.parents = []
        self._prev = None   # (material, L labels, D labels, n_D, nodes by key) one level up

    def add_level(self, mat, pieces):
        """Add the next level, its material mask mat and its pieces (each
        an Nx2 (row, col) polyline on the boundary). Returns each piece's
        node.

        Each piece (a whole ring, or a run left by pruning or clipping) is
        assigned from the labels of the pixels around its own points: the
        pair most of its points lie between. A piece whose points match no
        node (which a boundary of this same material can't produce) waits
        for every node of a shallower level.
        """
        from scipy.ndimage import label

        H, W = mat.shape
        parents = self.parents
        L, _ = label(mat, structure=self._EIGHT)
        D, n_d = label(~mat, structure=self._EIGHT)
        pairs = []
        for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
            for la, db in ((L[a], D[b]), (L[b], D[a])):
                m = (la > 0) & (db > 0)
                pairs.append(la[m].astype(np.int64) * (n_d + 1) + db[m])
        keys = np.unique(np.concatenate(pairs))
        first = len(parents)
        nodes = {int(key): first + i for i, key in enumerate(keys)}
        parents.extend([] for _ in range(len(keys)))

        if self._prev is not None:
            mat_p, L_p, D_p, n_d_p, nodes_p = self._prev
            parent_l = np.zeros(L_p.max() + 1, np.int64)
            parent_l[L_p[mat_p]] = L[mat_p]
            parent_d = np.zeros(n_d + 1, np.int64)
            parent_d[D[~mat]] = D_p[~mat]
            above = {}
            for key, n in nodes_p.items():
                l, d = divmod(key, n_d_p + 1)
                above.setdefault((int(parent_l[l]), d), []).append(n)
            for key, n in nodes.items():
                l, d = divmod(key, n_d + 1)
                parents[n] = above.get((l, int(parent_d[d])), [])

        assigned = []
        for c in pieces:
            rows = (np.clip(np.floor(c[:, 0]), 0, H - 1).astype(np.int64),
                    np.clip(np.ceil(c[:, 0]), 0, H - 1).astype(np.int64))
            cols = (np.clip(np.floor(c[:, 1]), 0, W - 1).astype(np.int64),
                    np.clip(np.ceil(c[:, 1]), 0, W - 1).astype(np.int64))
            # The material (and the air) pixels of a 2x2 cell are 8-connected,
            # so they carry one label each and the maximum picks it.
            l = np.max([L[r, cc] for r in rows for cc in cols], axis=0).astype(np.int64)
            d = np.max([D[r, cc] for r in rows for cc in cols], axis=0)
            ok = (l > 0) & (d > 0)
            vals, counts = np.unique(l[ok] * (n_d + 1) + d[ok], return_counts=True)
            node = None
            for k in np.argsort(-counts):
                node = nodes.get(int(vals[k]))
                if node is not None:
                    break
            if node is None:
                node = len(parents)
                parents.append(list(range(first)))
            assigned.append(node)
        self._prev = (mat, L, D, n_d, nodes)
        return assigned


def _contour_graph(surface, z_levels, n_grey_levels, all_z_contours):
    """The contour graph (_LevelGraph): which contours must be cut before
    which. At each level the material is {surface < threshold_k}, the air
    {surface >= threshold_k}, so separate hills are told apart by their
    material, separate craters by their air; hills merging below a saddle
    give a node several parents, a crater floor splitting gives several
    children. On a vertical wall, the contours of several levels coincide
    and are joined.

    Returns (parents, piece_node): parents[n] lists the nodes that must be
    complete before node n is cut; piece_node[z_norm][i] is the node of
    all_z_contours[z_norm][i].
    """
    graph = _LevelGraph()
    piece_node = {}
    for z_norm in z_levels:
        piece_node[z_norm] = graph.add_level(
            surface < _contour_threshold(z_norm, n_grey_levels),
            all_z_contours.get(z_norm, []))
    return graph.parents, piece_node


def _order_contours_graph(all_z_contours, z_levels, surface, n_grey_levels, *,
                          max_depth, px_per_mm, H, ox, oy,
                          lock_direction, current_pos,
                          progress=None, progress_lo=55, progress_hi=80):
    """Order contour pieces top-down per feature, nearest-first across features.

    A piece may be cut once every node its node depends on in the contour
    graph (_contour_graph) is complete, i.e. all of its pieces are cut: each
    hill and each crater is cut strictly top-down, its next ring only after
    the ring above it, while separate features are independent. Among the
    pieces that may be cut, greedy nearest-neighbour (_order_by_precedence)
    picks the next one, so the tool follows one feature down until it
    reaches a join (hills merging below a saddle), then the nearest other
    branch. A closed piece may start at any of about 20 points along
    itself. Open pieces start at either end and closed ones run either way
    unless lock_direction is set: a contour's climb/conventional sense
    comes entirely from its own orientation.

    surface is the achievable-depth surface the contours were extracted
    from, at the thresholds of _contour_threshold.

    current_pos : (x, y) tool position before the first segment.

    Returns
    -------
    (segments, new_pos)
      segments : List[List[(x_mm, y_mm, z_mm)]] in visiting order
      new_pos  : (x, y) tool position after the last segment
    """
    if progress is None:
        progress = lambda pct, msg: None

    progress(progress_lo, "Building the contour graph…")
    parents, piece_node = _contour_graph(surface, z_levels, n_grey_levels, all_z_contours)

    pieces = []        # (Nx2 pixel polyline, z_mm, closed)
    nodes = []
    for z_norm in z_levels:
        for c, n in zip(all_z_contours.get(z_norm, []), piece_node.get(z_norm, [])):
            if len(c) < 2:
                continue
            nodes.append(n)
            pieces.append((c, -z_norm * max_depth,
                           bool(np.linalg.norm(c[0] - c[-1]) < 1.5)))

    progress(progress_lo, f"Ordering {len(pieces)} contour piece(s)…")
    polylines = [np.column_stack([c[:, 1] / px_per_mm - ox,
                                  (H - 1 - c[:, 0]) / px_per_mm - oy])
                 for c, _, _ in pieces]
    order, new_pos = _order_by_precedence(
        polylines, nodes, parents, current_pos,
        closed=[closed for _, _, closed in pieces],
        allow_flip=not lock_direction, progress=progress,
        progress_lo=progress_lo, progress_hi=progress_hi, what="contours")

    segments = []
    for i, flip, start in order:
        c, z_mm, closed = pieces[i]
        if flip:
            c = c[::-1]
        if start > 0:
            c = np.roll(c, -start, axis=0)
        if closed:
            c = np.vstack([c, c[:1]])
        segments.append([(col / px_per_mm - ox, (H - 1 - row) / px_per_mm - oy, z_mm)
                         for row, col in c])

    progress(progress_hi, f"Ordering complete ({len(segments)} segments).")
    return segments, new_pos


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
    cut_direction="Climb",
    max_step_down_mm=None,
    extra_offset_mm=0.0,
    avoid_air=False,
    erosion_cache=None,
    erosion_cache_key=None,
    z_range=(0.0, 1.0),
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
        discrete grey-level transitions.  Gaps wider than max_step_down_mm are
        then filled with evenly spaced extra levels (see _fill_level_gaps).
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
        Maximum Z increment, in mm, between one cut and the next: the bound
        on adaptive level spacing, and on what redundancy pruning may create
        between one retained cut and the next (see _prune_contours).
        None or <= 0 means unconstrained — adaptive gaps stay unfilled and
        pruning may collapse any number of redundant levels.
    extra_offset_mm : float
        Stock-to-leave allowance (mm) applied laterally and vertically, so a
        subsequent step (typically with extra_offset_mm=0) can refine the
        result. Laterally, the tool is treated as 2*extra_offset_mm larger
        when computing the achievable-depth surface. Vertically, that
        surface is lowered by extra_offset_mm. 0 = none.
    avoid_air : bool
        If True, remove the portions of every z-level's contours where the
        tool would cut only air -- material previous_result has already
        cleared -- see _prune_air_contours. Uses z_safe/feed_rapid for the
        same retract/plunge breakeven as lateral pruning.
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
        forced direction, and the ordering below is free to reverse
        contours for shorter travel.
    erosion_cache : dict, optional
        Used to memoise the grey-erosion/grey-dilation achievable-depth surface
        (keyed by erosion_cache_key), so a subsequent call with the same tool
        and physical scale can skip recomputing it.  None disables caching.
    erosion_cache_key : hashable, optional
        Key identifying the (tool, px_per_mm, max_depth) combination for
        erosion_cache lookups.
    z_range : (lo, hi), normalised, 0 <= lo < hi <= 1
        Depth band of the contour levels: every level lies in (lo, hi] and
        hi is the deepest one. Material above lo counts as already removed.
        The stored accessible surface is capped at hi.

    Returns
    -------
    ToolPath
    """
    import math
    from grey_morphology import grey_erosion, grey_dilation
    from skimage.measure import find_contours

    z_lo, z_hi = check_z_range(z_range)

    # None/<=0 both mean "unconstrained" for the depth bound.
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
        center_depth_contour = grey_erosion(depth_map.astype(np.float32, copy=False),
                                  footprint=footprint_plan,
                                  structure=structure).astype(np.float32)
        center_depth_contour = np.clip(center_depth_contour, 0.0, 1.0)

        # Dilation of the erosion gives the true achievable depth at each material
        # point: the tool can carve (x,y) while centred at an offset position.
        # Erosion followed by dilation = morphological opening.
        accessible_contour = grey_dilation(center_depth_contour,
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
    n_passes = max(1, math.ceil((z_hi - z_lo) * max_depth / z_increment_mm))
    if adaptive_levels:
        # Quantiles over the pixels inside the band; deeper ones all count
        # towards the band's floor hi.
        flat = np.minimum(
            center_depth_contour[center_depth_contour > max(z_lo, 1e-6)], z_hi).ravel()
        if flat.size >= 2:
            quantiles = np.linspace(1.0 / n_passes, 1.0, n_passes)
            z_raw = np.quantile(flat, quantiles)
            min_gap = 0.5 / max(n_grey_levels - 1, 1)
            # A level at lo itself would repeat the step above's deepest cut.
            z_levels, prev = [], (z_lo if z_lo > 0.0 else -1.0)
            for z in np.clip(z_raw, 0.0, z_hi):
                if float(z) - prev >= min_gap:
                    z_levels.append(float(z))
                    prev = float(z)
            if not z_levels or z_levels[-1] < z_hi - min_gap:
                z_levels.append(z_hi)
        else:
            z_levels = band_levels((z_lo, z_hi), z_increment_mm, max_depth)
        if _msd is not None:
            z_levels = _fill_level_gaps(z_levels, _msd / max_depth,
                                        float(center_depth_contour.max()),
                                        start=z_lo)
        progress(40, f"{len(z_levels)} contour level{'s' if len(z_levels) != 1 else ''}  "
                     f"(adaptive, target Z increment {z_increment_mm:.2f} mm).")
    else:
        z_levels = band_levels((z_lo, z_hi), z_increment_mm, max_depth)
        progress(40, f"{len(z_levels)} contour level{'s' if len(z_levels) != 1 else ''}  "
                     f"({z_increment_mm:.2f} mm each).")

    # ── Stage 4 · Extract contours and build segments (40–80 %) ───────────
    rdp_tol   = 1.0 / px_per_mm
    rdp_z_tol = max_depth / (n_grey_levels - 1)

    # ── Stage 4a · Extract raw contours for every Z-level ─────────────────
    progress(40, "Extracting contours for all Z-levels…")
    lock_direction = cut_direction in ("Climb", "Conventional")
    all_z_contours = _extract_z_contours(center_depth_contour, z_levels,
                                          n_grey_levels, cut_direction)

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

    # ── Stage 4b′ · Drop air-cutting runs (optional) ──────────────────────
    air_pruned_mm = 0.0
    if avoid_air:
        progress(52, "Removing air-cutting portions of each Z-level…")
        free = compute_air_free_depth(previous_result, kernel_contour,
                                      footprint_contour, (H, W))
        all_z_contours, air_pruned_mm = _prune_air_contours(
            all_z_contours, free, 1.0 / (n_grey_levels - 1), px_per_mm, max_depth,
            lambda depth_mm: _retract_breakeven_mm(feed_rate, feed_rapid, z_safe, depth_mm))
        progress(52, f"Removed {air_pruned_mm:.0f} mm of air cutting.")

    # ── Stage 4c · Ordering — top-down per feature along the contour graph
    link_threshold_mm = tool.diameter_mm
    all_segments, _ = _order_contours_graph(
        all_z_contours, z_levels, center_depth_contour, n_grey_levels,
        max_depth=max_depth,
        px_per_mm=px_per_mm, H=H, ox=_ox, oy=_oy,
        lock_direction=lock_direction,
        current_pos=(0.0, 0.0),
        progress=progress, progress_lo=55, progress_hi=80)

    # ── Stage 4d · Terrain-link insertion ──────────────────────────────────
    # Walk all_segments once and splice geodesic links wherever consecutive
    # segments are within link_threshold_mm of each other.
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
    # than per-contour during ordering has no effect on entry-point choice
    # or terrain-link endpoints; it just needs to happen somewhere
    # after every segment (including links, built above) exists.
    all_segments = [simplify_line(pts, rdp_tol, rdp_z_tol) for pts in all_segments]

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
        accessible=(np.minimum(accessible_contour, z_hi) if z_hi < 1.0
                    else accessible_contour),
        metadata={
            'algorithm':       'Contour',
            'z_range':         (z_lo, z_hi),
            'profile':         tool.profile,
            'max_depth':       max_depth,
            'radius_mm':       tool.diameter_mm / 2,
            'z_increment_mm':  z_increment_mm,
            'adaptive_levels': adaptive_levels,
            'max_step_down_mm':   _msd,
            'extra_offset_mm':    extra_offset_mm,
            'lateral_tol_mm':     lateral_tol_mm,
            'air_pruned_mm':      air_pruned_mm,
            'n_passes':        len(z_levels),
            'width_mm':        width_mm,
            'height_mm':       height_mm,
            'rdp_tolerance':   rdp_tol,
            'rdp_z_tolerance': rdp_z_tol,
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
    boundary_offset_mm=0.0,
    cut_direction="Climb",
    prior_accessible=None,
    previous_result=None,
    previous_alpha=None,
    use_2opt=False,
    z_safe=5.0,
    feed_rapid=3000.0,
    tab_settings=None,
    avoid_air=False,
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
    *boundary_offset_mm*), and cuts a uniform-depth closed loop through the
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
        / max_depth. 0 (the default) means "auto": split at the midpoint of
        depth_map's own value range instead, since depth_map's floor is
        always 0 and a literal 0 threshold would put every pixel on the same
        side of the split. Which side of the threshold is "the object"
        depends on how the source was authored (a carved object against a
        shallow background vs. an uncarved object standing proud of a deeply
        recessed background) and isn't knowable from depth_map alone, so
        both directions are tried -- see Stage 2 below and the
        'mask_direction' metadata key ('<=' or '>=') for which one won.
    stepdown_mm, final_stepdown_mm : float
        As in the sibling algorithms -- passed straight to
        make_multipass(segments, cutout_depth_mm, stepdown_mm,
        final_stepdown_mm).
    boundary_offset_mm : float
        Lateral-only outward push of the tool-centre path beyond the
        minimum tool-radius clearance; 0 = tool tangent to the (filled)
        silhouette edge. Implemented as a single binary_dilation of the
        filled silhouette mask by a disk of radius
        (tool.diameter_mm/2 + boundary_offset_mm), in pixels.
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
        Seed per-rapid overhead in the optional 2-opt cost model, and
        avoid_air's retract/plunge breakeven.
    tab_settings : dressup.TabSettings or None
        When enabled, leaves evenly-spaced uncut bridges ("tabs") in every
        closed ring, applied to the full-depth ring geometry before
        make_multipass() depth-slices it -- see dressup.apply_tabs.
    avoid_air : bool
        If True, remove the portions of every pass where the tool would cut
        only air -- material previous_result has already cleared, or a hole
        previous_alpha marks as cut through -- see prune_air_segments. A
        pruned closed loop becomes an open segment.
    progress : callable(pct, message)
        Thread-safe progress reporter.

    Returns
    -------
    ToolPath
    """
    rings_cache = _compute_cutout_rings(
        depth_map, tool, max_depth, width_mm, height_mm,
        origin=origin, threshold_mm=threshold_mm, boundary_offset_mm=boundary_offset_mm,
        cut_direction=cut_direction,
        prior_accessible=prior_accessible, previous_result=previous_result,
        previous_alpha=previous_alpha, progress=progress,
    )
    return _compute_cutout_from_rings(
        rings_cache, tool, cutout_depth_mm, max_depth, width_mm, height_mm,
        threshold_mm=threshold_mm, stepdown_mm=stepdown_mm,
        final_stepdown_mm=final_stepdown_mm, boundary_offset_mm=boundary_offset_mm,
        cut_direction=cut_direction, use_2opt=use_2opt, z_safe=z_safe,
        feed_rapid=feed_rapid, tab_settings=tab_settings,
        avoid_air=avoid_air, feed_rate=feed_rate, n_grey_levels=n_grey_levels,
        progress=progress,
    )


def compute_cutout_silhouette_mask(depth_map, threshold_mm, max_depth, r_px):
    """Resolve threshold_mm to a silhouette mask of "the object".

    threshold_mm > 0 is used literally, as a normalised split point;
    0 (or unset) means "auto": split at the midpoint of the depth map's own
    value range rather than at depth 0, since the depth map's floor is
    always 0 -- a literal 0 threshold would put every pixel on the same
    side of the split, both directions, guaranteeing "no usable silhouette
    boundary" below.

    Whether "the object" is the deeply-cut region (dark-background source,
    background left at/near the surface) or the shallow/uncarved region
    (background carved away deeply so the object stands proud in relief)
    depends on how the source was authored -- depth_map alone doesn't say
    which. Try object = "at or below threshold" first (an object standing
    proud of a deeply-recessed background); if that yields nothing usable
    (empty, or -- once holes are filled -- the entire canvas, e.g. because
    it actually picked out the *background* and binary_fill_holes closed
    over the object as an enclosed "hole" in it), retry with object = "at
    or above threshold" instead (a carved object against a shallow/uncut
    background). This only disambiguates correctly when the object doesn't
    itself touch the canvas edge.

    Returns (mask, mask_direction, threshold_norm); mask is None (with
    mask_direction None) if neither direction yields a usable silhouette.
    """
    from scipy.ndimage import binary_fill_holes
    from grey_morphology import binary_dilation
    from skimage.morphology import disk

    if threshold_mm > 0.0:
        threshold_norm = threshold_mm / max_depth if max_depth else 0.0
    else:
        threshold_norm = float(depth_map.min() + depth_map.max()) / 2.0

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

    mask = _try_mask('<=')
    mask_direction = '<='
    if mask is None:
        mask = _try_mask('>=')
        mask_direction = '>='
    if mask is None:
        mask_direction = None
    return mask, mask_direction, threshold_norm


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
    boundary_offset_mm=0.0,
    cut_direction="Climb",
    prior_accessible=None,
    previous_result=None,
    previous_alpha=None,
    progress=None,
):
    """Phase A of compute_cutout_toolpath: silhouette mask + outer contour
    extraction. See compute_cutout_toolpath for full parameter docs.
    """
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
    progress(10, "Building silhouette mask…")
    r_px = max(1, int(round((tool.diameter_mm / 2.0 + boundary_offset_mm) * px_per_mm)))
    mask_cutout, mask_direction, threshold_norm = compute_cutout_silhouette_mask(
        depth_map, threshold_mm, max_depth, r_px)
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
    boundary_offset_mm=0.0,
    cut_direction="Climb",
    use_2opt=False,
    z_safe=5.0,
    feed_rapid=3000.0,
    tab_settings=None,
    avoid_air=False,
    feed_rate=800.0,
    n_grey_levels=256,
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
            full_depth_segs, tab_settings, tool.diameter_mm,
            closed_tol_mm=1.5 / px_per_mm)
        for warning in tab_warnings:
            progress(45, warning)

    pass_list = make_multipass(full_depth_segs, cutout_depth_mm, stepdown_mm, final_stepdown_mm)
    n_passes = len(pass_list)
    air_pruned_mm = 0.0
    if avoid_air:
        progress(45, "Removing air-cutting portions of each pass…")
        pass_list, _, air_pruned_mm = _prune_air_passes(
            pass_list, previous_result, rings_cache.kernel_cutout,
            rings_cache.footprint_cutout, (H, W),
            px_per_mm, _ox, _oy, max_depth, 1.0 / (n_grey_levels - 1),
            feed_rate, feed_rapid, z_safe, previous_alpha=previous_alpha)
        progress(45, f"Removed {air_pruned_mm:.0f} mm of air cutting.")

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
        # direction.
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
        'boundary_offset_mm': boundary_offset_mm,
        'threshold_mm':      threshold_mm,
        'mask_direction':    rings_cache.mask_direction,
        'cut_direction':     cut_direction,
        'use_2opt':          use_2opt,
        'n_passes':          n_passes,
        'air_pruned_mm':     air_pruned_mm,
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


# ── Registration holes ─────────────────────────────────────────────────────
# Stock-space coordinates throughout: (0, 0) is the stock's bottom-left
# corner (margin included), x to the right, y up, in mm. Conversion to G-code
# coordinates happens only in compute_registration_holes_toolpath, via the
# same origin offset table every other algorithm uses.

_ORIGIN_CORNERS = {"Bottom-Left": (0, 0), "Bottom-Right": (1, 0),
                   "Top-Left":    (0, 1), "Top-Right":    (1, 1)}

# Hole diameters within this of the tool diameter are drilled with a straight
# (pecked) plunge instead of a helix -- a helix of near-zero radius would just
# be a slow plunge with extra points.
_PLUNGE_RADIUS_TOL_MM = 0.02


def _registration_diag_frame(W_mm, H_mm, origin, diag_dir):
    """Return (anchor, u, v) for a Diagonal registration pattern.

    anchor is the point of the flip line the pattern hangs off: the origin
    corner, or the stock centre for a Center origin. u and v are the unit x/y
    directions the L's arms point along (into the stock). The flip line runs
    through anchor along u + v, so reflecting across it swaps u and v.
    """
    corner = _ORIGIN_CORNERS.get(origin)
    if corner is None:   # Center
        u = (1.0, 0.0)
        v = (0.0, 1.0) if diag_dir == "+45°" else (0.0, -1.0)
        return (W_mm / 2.0, H_mm / 2.0), u, v
    fx, fy = corner
    u = (1.0, 0.0) if fx == 0 else (-1.0, 0.0)
    v = (0.0, 1.0) if fy == 0 else (0.0, -1.0)
    return (fx * W_mm, fy * H_mm), u, v


def registration_flip_axis(pattern, W_mm, H_mm, origin="Bottom-Left", diag_dir="+45°"):
    """Return the flip axis of a registration pattern as (point, direction).

    point is any (x, y) on the axis in stock space; direction is a unit
    vector along it.

    - "Vertical axis": x = W/2.
    - "Horizontal axis": y = H/2.
    - "Diagonal": the 45° line through the G-code origin -- the only line a
      diagonal flip may use without re-zeroing X/Y, since a diagonal flip
      transposes the stock's W×H footprint and only points on the flip line
      stay put. The line's direction follows from the origin corner (+45°
      through Bottom-Left/Top-Right, -45° through Top-Left/Bottom-Right); for
      a Center origin, diag_dir ("+45°" or "−45°") picks it.
    """
    if pattern == "Vertical axis":
        return (W_mm / 2.0, 0.0), (0.0, 1.0)
    if pattern == "Horizontal axis":
        return (0.0, H_mm / 2.0), (1.0, 0.0)
    anchor, u, v = _registration_diag_frame(W_mm, H_mm, origin, diag_dir)
    s = 1.0 / np.sqrt(2.0)
    return anchor, ((u[0] + v[0]) * s, (u[1] + v[1]) * s)


def registration_auto_spacing(W_mm, H_mm, inset_mm):
    """Arm length of the Diagonal L pattern that keeps both arm holes the
    same inset from the stock edges as the corner hole."""
    return min(W_mm, H_mm) - 2.0 * inset_mm


def registration_hole_centers(pattern, W_mm, H_mm, origin="Bottom-Left",
                              diag_dir="+45°", inset_mm=10.0, spacing_mm=None):
    """Return the registration hole centres as a list of (x, y) in stock space.

    - "Vertical axis": two holes on x = W/2, inset_mm from the bottom and top
      edges.
    - "Horizontal axis": two holes on y = H/2, inset_mm from the left and
      right edges.
    - "Diagonal": an L of three holes -- a corner hole c on the flip line
      (see registration_flip_axis) plus c + spacing along each of the two
      edge directions into the stock. Reflecting across the flip line fixes
      c and swaps the two arm holes. With a corner origin, c sits inset_mm
      from both edges at that corner; with a Center origin, c sits on the
      flip line at the far end of the line's centred min(W, H) square, again
      inset_mm in from its edges. spacing_mm None means
      registration_auto_spacing.

    Every pattern maps onto itself under its own flip, so the pins fit the
    same holes after the stock is turned over. No attempt is made to keep
    holes inside the stock or out of the artwork: the placement is the
    user's call.
    """
    if pattern == "Vertical axis":
        return [(W_mm / 2.0, inset_mm), (W_mm / 2.0, H_mm - inset_mm)]
    if pattern == "Horizontal axis":
        return [(inset_mm, H_mm / 2.0), (W_mm - inset_mm, H_mm / 2.0)]

    anchor, u, v = _registration_diag_frame(W_mm, H_mm, origin, diag_dir)
    if origin in _ORIGIN_CORNERS:
        back = -inset_mm
    else:
        back = min(W_mm, H_mm) / 2.0 - inset_mm
    c = (anchor[0] - back * (u[0] + v[0]), anchor[1] - back * (u[1] + v[1]))
    s = registration_auto_spacing(W_mm, H_mm, inset_mm) if spacing_mm is None else spacing_mm
    return [c,
            (c[0] + s * u[0], c[1] + s * u[1]),
            (c[0] + s * v[0], c[1] + s * v[1])]


def bore_hole_segments(cx, cy, r_path, depth_mm, pitch_mm, climb=True):
    """Return the cutting segments that bore one hole centred at (cx, cy).

    r_path is the tool-centre path radius: (hole diameter - tool diameter)/2.

    - r_path below _PLUNGE_RADIUS_TOL_MM: a straight plunge, pecked in
      pitch_mm steps (0 = one plunge to full depth). Every peck but the last
      ends with a feed-rate lift back out of the hole to clear chips -- the
      pecks share one XY, so ToolPath.iter_moves would otherwise bridge them
      without retracting.
    - Otherwise: a helix from Z=0 down to -depth_mm, dropping at most
      pitch_mm per revolution (0 = the whole depth in one revolution),
      followed by one flat full circle at the bottom to clean up the floor's
      last ramp, then a move back to the centre so the retract doesn't drag
      along the wall. climb=True runs counter-clockwise (climb milling on an
      inside wall with a clockwise spindle), False clockwise.
    """
    if r_path < _PLUNGE_RADIUS_TOL_MM:
        n = max(1, int(np.ceil(depth_mm / pitch_mm - 1e-9))) if pitch_mm > 0 else 1
        zs = [-depth_mm * k / n for k in range(n + 1)]
        segs = []
        for k in range(n):
            seg = [(cx, cy, zs[k]), (cx, cy, zs[k + 1])]
            if k < n - 1:
                seg.append((cx, cy, 0.0))
            segs.append(seg)
        return segs

    n_rev = max(1, int(np.ceil(depth_mm / pitch_mm - 1e-9))) if pitch_mm > 0 else 1
    # Chord ≤ 5° of arc and ≤ 0.2 mm, whichever is finer.
    per_rev = max(72, int(np.ceil(2.0 * np.pi * r_path / 0.2)))
    sign = 1.0 if climb else -1.0
    t = np.linspace(0.0, n_rev + 1.0, int((n_rev + 1) * per_rev) + 1)
    theta = sign * 2.0 * np.pi * t
    z = -depth_mm * np.minimum(t / n_rev, 1.0)
    pts = [(cx + r_path * np.cos(a), cy + r_path * np.sin(a), zz)
           for a, zz in zip(theta, z)]
    pts.append((cx, cy, -depth_mm))
    return [[(float(x), float(y), float(zz)) for x, y, zz in pts]]


def compute_registration_holes_toolpath(
    tool, width_mm, height_mm, shape,
    pattern="Vertical axis",
    origin="Bottom-Left",
    diag_dir="+45°",
    hole_diameter_mm=6.0,
    hole_depth_mm=15.0,
    stepdown_mm=1.0,
    inset_mm=10.0,
    spacing_mm=None,
    cut_direction="Climb",
    max_depth=5.0,
    prior_accessible=None,
    previous_result=None,
    previous_alpha=None,
    progress=None,
):
    """Bore registration-pin holes for flipping the stock over.

    Holes are placed by registration_hole_centers and bored by
    bore_hole_segments, nearest-first from the G-code origin. Like
    compute_cutout_toolpath this is a flat, straight-wall cut that ignores
    the depth map and any Process Region gating -- the holes belong to the
    stock, not the artwork -- and hole_depth_mm is an absolute depth,
    independent of max_depth (typically through the stock and into the
    spoilboard, so the pins stay put while the stock is turned over).

    Parameters
    ----------
    tool : Tool
        Expected profile "End Mill" (enforced by the UI).
    width_mm, height_mm : float
        Physical dimensions of the work area (stock including margin) in mm.
    shape : (H, W)
        Pixel shape of the canvas's depth map, for the simulated
        result/accessible/alpha arrays.
    pattern, diag_dir, inset_mm, spacing_mm :
        See registration_hole_centers.
    origin : str
        G-code origin convention; also anchors the Diagonal flip line.
    hole_diameter_mm : float
        Finished hole diameter. Must be at least the tool diameter.
    hole_depth_mm : float
        Absolute hole depth in mm.
    stepdown_mm : float
        Helix pitch (Z drop per revolution), or peck depth for a hole the
        tool's own size. 0 = single pass.
    cut_direction : "Either" | "Climb" | "Conventional"
        Helix winding: Climb and Either run counter-clockwise, Conventional
        clockwise.
    max_depth : float
        Canvas's shared max cut depth in mm, for normalising the simulated
        arrays. A hole deeper than this is marked as cut through in alpha,
        using the same convention as Cutout.
    prior_accessible, previous_result, previous_alpha : 2-D arrays or None
        The previous step's cumulative surfaces, unioned into this step's.
    progress : callable(pct, message)

    Returns
    -------
    ToolPath
    """
    if progress is None:
        progress = lambda pct, msg: None
    r_path = (hole_diameter_mm - tool.diameter_mm) / 2.0
    if r_path < -1e-9:
        raise ValueError(
            f"Hole diameter ({hole_diameter_mm:g} mm) is smaller than the tool "
            f"diameter ({tool.diameter_mm:g} mm).")
    r_path = max(r_path, 0.0)

    H, W = shape
    px_per_mm_x = W / width_mm
    px_per_mm_y = H / height_mm
    W_mm = (W - 1) / px_per_mm_x
    H_mm = (H - 1) / px_per_mm_y
    _ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
           "Top-Left":    0,       "Top-Right":    W_mm,
           "Center":      W_mm / 2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm / 2}.get(origin, 0)

    progress(0, "Placing registration holes…")
    centers = registration_hole_centers(pattern, W_mm, H_mm, origin, diag_dir,
                                        inset_mm, spacing_mm)
    gcode_centers = [(x - _ox, y - _oy) for x, y in centers]

    remaining = list(gcode_centers)
    ordered = []
    cur = (0.0, 0.0)
    while remaining:
        i = min(range(len(remaining)),
                key=lambda k: np.hypot(remaining[k][0] - cur[0], remaining[k][1] - cur[1]))
        cur = remaining.pop(i)
        ordered.append(cur)

    climb = cut_direction != "Conventional"
    segments = []
    for k, (cx, cy) in enumerate(ordered):
        progress(10 + 40 * k // max(len(ordered), 1),
                 f"Boring hole {k + 1}/{len(ordered)}…")
        segments.extend(bore_hole_segments(cx, cy, r_path, hole_depth_mm,
                                           stepdown_mm, climb))

    progress(50, "Simulating tool-swept removal surface…")
    px_per_mm = (px_per_mm_x * px_per_mm_y) ** 0.5
    kernel, footprint = _build_kernel(tool, px_per_mm, max_depth)
    removed = simulate_toolpath_result(
        ToolPath(segments=segments), (H, W), px_per_mm_x, px_per_mm_y,
        _ox, _oy, max_depth, kernel, footprint)
    result = np.maximum(previous_result if previous_result is not None else 0.0,
                        removed).astype(np.float32)
    accessible = np.maximum(prior_accessible if prior_accessible is not None else 0.0,
                            removed).astype(np.float32)
    if hole_depth_mm > max_depth:
        alpha = np.minimum(previous_alpha if previous_alpha is not None else 1.0,
                           np.where(removed >= 1.0 - 1e-3, 0.0, 1.0)).astype(np.float32)
    else:
        alpha = previous_alpha.copy() if previous_alpha is not None else None

    axis_point, axis_dir = registration_flip_axis(pattern, W_mm, H_mm, origin, diag_dir)
    metadata = {
        'algorithm':        'Registration Holes',
        'profile':          tool.profile,
        'radius_mm':        tool.diameter_mm / 2,
        'max_depth':        max_depth,
        'pattern':          pattern,
        'hole_centers':     gcode_centers,
        'hole_diameter_mm': hole_diameter_mm,
        'hole_depth_mm':    hole_depth_mm,
        'stepdown_mm':      stepdown_mm,
        'cut_method':       'plunge' if r_path < _PLUNGE_RADIUS_TOL_MM else 'helix',
        'flip_axis':        ((axis_point[0] - _ox, axis_point[1] - _oy), axis_dir),
        'cut_direction':    cut_direction,
        'width_mm':         width_mm,
        'height_mm':        height_mm,
    }
    progress(100, "Registration holes tool path complete!")
    return ToolPath(segments=segments, result=result, accessible=accessible,
                    alpha=alpha, metadata=metadata)
