import numpy as np
from tool import Tool
from toolpath import ToolPath


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
            t     = np.dot(diff, seg) / (seg_len * seg_len)
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


def compute_raster_toolpath(depth_map, n_grey_levels, tool,
                            step_mm, stepdown_mm, max_depth,
                            width_cm, height_cm, raster_dir,
                            origin="Bottom-Left",
                            feed_rate=800.0,
                            work_surface_tol=None,
                            previous_result=None,
                            raster_offset_pct=0.0,
                            progress=None):
    """Compute a raster toolpath by scanning the depth map with the given tool.

    Parameters
    ----------
    depth_map         : 2-D float32 [0, 1]  — 1.0 = max cut depth (intaglio)
    n_grey_levels     : number of grey levels in source image (e.g. 256)
    tool              : Tool — encapsulates profile, diameter_mm, vbit_angle_deg
    step_mm           : lateral step-over in mm
    stepdown_mm       : max depth increment per z-pass in mm
    max_depth         : total maximum cut depth in mm
    width_cm          : physical image width in cm
    height_cm         : physical image height in cm
    raster_dir        : "X" | "Y" | "Both"
    raster_offset_pct : percentage [0, 100) to shift raster lines relative to the
                        step-over distance; e.g. 50 shifts every line by half a step,
                        useful when combining with a previous raster step at 0%.
    origin            : G-code origin convention ("Bottom-Left", "Top-Left", …)
    feed_rate         : cutting feed rate in mm/min
    work_surface_tol  : threshold (normalised [0,1]) applied to the residual
                        depth (depth_map − previous_result).  0.0 (default)
                        visits every pixel; higher values skip pixels where
                        the prior work is already shallow.
    previous_result   : optional 2-D float32 [0,1] cumulative result from a
                        prior step (i.e. ToolPath.result), used as the floor
                        for the output result so material already removed is
                        not un-cut.  When None (default) the floor is 0.
    progress          : callable(pct, subtask) — thread-safe progress reporter

    Returns
    -------
    ToolPath
    """
    from scipy.ndimage import grey_erosion, grey_dilation


    H, W = depth_map.shape

    if work_surface_tol is None:
        work_surface_tol = 0.0
    residual = (np.clip(depth_map - previous_result, 0.0, 1.0)
                if previous_result is not None else depth_map)

    px_per_mm_x = W / (width_cm  * 10.0)
    px_per_mm_y = H / (height_cm * 10.0)
    px_per_mm   = (px_per_mm_x + px_per_mm_y) / 2.0   # isotropic approx

    W_mm = (W - 1) / px_per_mm_x
    H_mm = (H - 1) / px_per_mm_y
    _ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
           "Top-Left":    0,       "Top-Right":    W_mm,
           "Center":      W_mm/2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm/2}.get(origin, 0)

    # ── Stage 1 · Build tool kernel (0–10 %) ─────────────────────────────
    progress(0, "Building tool geometry kernel…")
    kernel_raster, footprint_raster = _build_kernel(tool, px_per_mm, max_depth)
    ksize = kernel_raster.shape[0]
    progress(10, f"Kernel ready  ({ksize}×{ksize} px, profile: {tool.profile}).")

    # ── Stage 2 · Compute achievable depth surface (10–35 %) ─────────────
    progress(10, "Computing achievable depth surface (morphological opening)…")
    structure = -kernel_raster.astype(np.float64)

    # Max depth the tool can reach when centred at each pixel (erosion).
    # Used internally as per-pixel depth targets during toolpath planning.
    center_depth_contour_raster = grey_erosion(depth_map.astype(np.float64),
                              footprint=footprint_raster,
                              structure=structure).astype(np.float32)
    center_depth_contour_raster = np.clip(center_depth_contour_raster, 0.0, 1.0)

    # Dilation of the erosion gives the true achievable depth at each material
    # point: the tool can carve (x,y) while centred at an offset position.
    # Erosion followed by dilation = morphological opening.
    accessible_raster = grey_dilation(center_depth_contour_raster.astype(np.float64),
                               footprint=footprint_raster,
                               structure=structure).astype(np.float32)
    accessible_raster = np.clip(accessible_raster, 0.0, 1.0)
    progress(35, "Achievable depth surface computed.")

    # ── Stage 3 · Plan depth passes (35–40 %) ─────────────────────
    progress(35, "Planning depth passes…")
    n_passes    = max(1, int(np.ceil(max_depth / stepdown_mm)))
    pass_depths = [min((i + 1) * stepdown_mm, max_depth)
                   for i in range(n_passes)]
    step_px        = max(1, int(round(step_mm * px_per_mm)))
    raster_start_px = int(round(raster_offset_pct / 100.0 * step_px)) % step_px

    resid_tol_mm = max_depth / (n_grey_levels - 1)
    resid_tol    = 1.0 / (n_grey_levels - 1)   # normalised [0,1]

    directions = []
    if raster_dir in ("X", "Both"):
        directions.append("X")
    if raster_dir in ("Y", "Both"):
        directions.append("Y")

    progress(40, f"{n_passes} depth pass{'es' if n_passes != 1 else ''}  "
                 f"({stepdown_mm:.2f} mm each), "
                 f"step-over {step_px} px = {step_mm:.2f} mm, "
                 f"residual tol {resid_tol_mm*1000:.1f} µm.")

    # ── Stage 4 · Adaptive multi-pass raster (40–80 %) ────────────

    if work_surface_tol > 0.0:
        from scipy.ndimage import binary_dilation
        prior_mask = binary_dilation(
                         residual >= work_surface_tol,
                         structure=footprint_raster,
                     ).astype(np.float32)
        progress(40, f"Threshold {work_surface_tol*max_depth*1000:.0f} µm: "
                     f"{int(np.sum(prior_mask))}/{H*W} pixels in mask.")
    else:
        prior_mask = None

    all_segments = []  # list of (segment_type, [(x,y,z),...]) for ToolPath
    achieved_raster    = np.zeros_like(center_depth_contour_raster, dtype=np.float32)
    tool_pos    = np.array([0.0, 0.0])
    total_segs  = 0
    n_links_inserted = 0
    n_links          = 0

    link_threshold_mm = 2.0 * step_mm

    def bresenham_pixels(r0, c0, r1, c1):
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

    def geodesic_link_pixels(r0, c0, r1, c1, z_norm_val):
        """Dijkstra minimum-3D-arc-length path from (r0,c0) to (r1,c1).

        Prefers routes that avoid sharp depth edges over the straight
        Bresenham line, producing a smoother z-profile for terrain links.
        Closes over center_depth_contour_raster, H, W, px_per_mm_x/y, max_depth.
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
            z_cur = min(float(center_depth_contour_raster[r, c]), z_norm_val)
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if not (rlo <= nr <= rhi and clo <= nc <= chi):
                    continue
                z_nb = min(float(center_depth_contour_raster[nr, nc]), z_norm_val)
                dxy2 = (dr / px_per_mm_y) ** 2 + (dc / px_per_mm_x) ** 2
                dz   = (z_nb - z_cur) * max_depth
                nd   = d + (dxy2 + dz * dz) ** 0.5
                if nd < dist.get((nr, nc), float('inf')):
                    dist[(nr, nc)] = nd
                    prev[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (nd, nr, nc))

        if (r1, c1) not in prev and (r0, c0) != (r1, c1):
            return list(bresenham_pixels(r0, c0, r1, c1))

        path, cur = [], (r1, c1)
        while cur != (r0, c0):
            path.append(cur)
            cur = prev[cur]
        path.append((r0, c0))
        path.reverse()
        return path

    for pass_idx, z_depth_mm in enumerate(pass_depths):
        z_norm = z_depth_mm / max_depth

        # ── Compute per-pixel remaining depth at this z-level ─────
        remaining_depth = (np.minimum(center_depth_contour_raster, z_norm)
                           - achieved_raster).astype(np.float32)
        if prior_mask is not None:
            remaining_depth *= prior_mask

        # ── Per direction: find which raster lines need revisiting ─
        for direction in directions:

            # All raster line indices for this direction
            if direction == "X":
                all_line_ids = np.arange(raster_start_px, H, step_px)
            else:
                all_line_ids = np.arange(raster_start_px, W, step_px)

            n_lines = len(all_line_ids)

            # 1-D boolean: does this raster line have any pixel whose
            # remaining depth exceeds the tolerance?
            line_needs_cut = np.zeros(n_lines, dtype=bool)
            for li, idx in enumerate(all_line_ids):
                if direction == "X":
                    line_slice = remaining_depth[idx, :]
                else:
                    line_slice = remaining_depth[:, idx]
                line_needs_cut[li] = np.any(line_slice > resid_tol)

            n_active = int(np.sum(line_needs_cut))
            progress(40 + 40 * pass_idx / n_passes,
                     f"Pass {pass_idx+1}/{n_passes}, dir={direction} "
                     f"at z={z_depth_mm:.3f} mm: "
                     f"{n_active}/{n_lines} raster lines need cutting.")

            if n_active == 0:
                continue

            # ── Collect all spans across all active raster lines ──
            all_spans = []
            for li, idx in enumerate(all_line_ids):
                if not line_needs_cut[li]:
                    continue
                if direction == "X":
                    line_resid = remaining_depth[idx, :]
                else:
                    line_resid = remaining_depth[:, idx]
                if pass_idx == 0 and prior_mask is None:
                    # First pass with no prior-step gate: one continuous span
                    # from first to last non-zero pixel so the tool doesn't
                    # retract through zero-depth regions.
                    nonzero = np.where(line_resid > resid_tol)[0]
                    all_spans.append((idx, int(nonzero[0]), int(nonzero[-1])))
                else:
                    for s, e in mask_spans(line_resid > resid_tol):
                        all_spans.append((idx, s, e))

            n_spans = len(all_spans)
            progress(40 + 40 * pass_idx / n_passes,
                     f"Pass {pass_idx+1}/{n_passes}, dir={direction}: "
                     f"{n_spans} span(s) to cut across "
                     f"{n_active} raster line(s).")

            # ── Greedy nearest-endpoint TSP over all spans ─────────
            def span_endpoints_mm(idx, s, e):
                """Return (x_s,y_s), (x_e,y_e) for a span in G-code coords."""
                if direction == "X":
                    y = (H - 1 - idx) / px_per_mm_y - _oy
                    return ((s / px_per_mm_x - _ox, y),
                            (e / px_per_mm_x - _ox, y))
                else:
                    x = idx / px_per_mm_x - _ox
                    return ((x, (H - 1 - s) / px_per_mm_y - _oy),
                            (x, (H - 1 - e) / px_per_mm_y - _oy))

            unvisited = list(range(n_spans))

            while unvisited:
                best_i    = 0
                best_rev  = False
                best_dist = np.inf

                for ui, si in enumerate(unvisited):
                    idx, s, e = all_spans[si]
                    (x_s, y_s), (x_e, y_e) = span_endpoints_mm(idx, s, e)
                    d_s = np.hypot(x_s - tool_pos[0], y_s - tool_pos[1])
                    d_e = np.hypot(x_e - tool_pos[0], y_e - tool_pos[1])
                    if d_s <= d_e:
                        d, rev = d_s, False
                    else:
                        d, rev = d_e, True
                    if d < best_dist:
                        best_i, best_rev, best_dist = ui, rev, d

                si = unvisited.pop(best_i)
                idx, s, e = all_spans[si]
                px_range = (range(s, e + 1) if not best_rev
                            else range(e, s - 1, -1))

                points = []
                for px in px_range:
                    row, col = ((idx, px) if direction == "X"
                                else (px, idx))
                    target = min(center_depth_contour_raster[row, col], z_norm)
                    if target > achieved_raster[row, col]:
                        achieved_raster[row, col] = target
                    z_mm = achieved_raster[row, col] * max_depth
                    x_mm = col / px_per_mm_x - _ox
                    y_mm = (H - 1 - row) / px_per_mm_y - _oy   # flip: pixel rows go down, G-code Y goes up
                    points.append((x_mm, y_mm, -z_mm))

                if points:
                    all_segments.append(points)
                    total_segs += 1
                    tool_pos = np.array([points[-1][0], points[-1][1]])

                    # ── Inline terrain link to next span ──────────
                    if unvisited:
                        nxt_best_i    = 0
                        nxt_best_rev  = False
                        nxt_best_dist = np.inf
                        for ui, si2 in enumerate(unvisited):
                            idx2, s2, e2 = all_spans[si2]
                            (xs2, ys2), (xe2, ye2) = span_endpoints_mm(
                                idx2, s2, e2)
                            d_s2 = np.hypot(xs2 - tool_pos[0],
                                            ys2 - tool_pos[1])
                            d_e2 = np.hypot(xe2 - tool_pos[0],
                                            ye2 - tool_pos[1])
                            if d_s2 <= d_e2:
                                d2, rev2 = d_s2, False
                            else:
                                d2, rev2 = d_e2, True
                            if d2 < nxt_best_dist:
                                nxt_best_i, nxt_best_rev, nxt_best_dist \
                                    = ui, rev2, d2

                        if nxt_best_dist < link_threshold_mm:
                            si2      = unvisited[nxt_best_i]
                            idx2, s2, e2 = all_spans[si2]
                            (xs2, ys2), (xe2, ye2) = span_endpoints_mm(
                                idx2, s2, e2)
                            if not nxt_best_rev:
                                x_entry = xs2
                                y_entry = ys2
                            else:
                                x_entry = xe2
                                y_entry = ye2

                            x_exit_lnk = tool_pos[0]
                            y_exit_lnk = tool_pos[1]
                            c0 = int(round((x_exit_lnk + _ox) * px_per_mm_x))
                            r0 = int(round((H - 1) - (y_exit_lnk + _oy) * px_per_mm_y))
                            c1 = int(round((x_entry    + _ox) * px_per_mm_x))
                            r1 = int(round((H - 1) - (y_entry    + _oy) * px_per_mm_y))

                            link_points = []
                            for row_l, col_l in geodesic_link_pixels(
                                    r0, c0, r1, c1, z_norm):
                                z_link = min(center_depth_contour_raster[row_l, col_l],
                                             z_norm)
                                if z_link > achieved_raster[row_l, col_l]:
                                    achieved_raster[row_l, col_l] = z_link
                                x_lmm = col_l / px_per_mm_x - _ox
                                y_lmm = (H - 1 - row_l) / px_per_mm_y - _oy   # flip Y
                                link_points.append(
                                    (x_lmm, y_lmm, -z_link * max_depth))

                            if len(link_points) >= 2:
                                all_segments.append(link_points)
                                n_links_inserted += 1

                n_links += 1   # count every potential inter-span gap

                if total_segs % max(1, n_spans // 10) == 0:
                    done = n_spans - len(unvisited)
                    pct  = 40 + 40 * (pass_idx + done / n_spans) / n_passes
                    progress(pct,
                             f"Pass {pass_idx+1}/{n_passes}, "
                             f"dir={direction}: "
                             f"{done}/{n_spans} spans cut, "
                             f"{total_segs} segments total…")

        # remaining_depth is recomputed fresh at the top of each pass

    progress(80, f"Raster complete — {total_segs} cut segments, "
                 f"{n_links_inserted} terrain links across all passes.")

    # ── Stage 6 · Simplify, simulate result, create ToolPath (88–100 %) ─────
    progress(80, "Simplifying tool path…")
    rdp_tol = max(
        1.0 / px_per_mm_x,
        1.0 / px_per_mm_y,
        max_depth / (n_grey_levels - 1),
    )
    simplified = [simplify_line(pts, rdp_tol) for pts in all_segments]

    progress(88, "Simulating tool-swept removal surface…")
    material_removed_raster = simulate_toolpath_result(
        ToolPath(segments=simplified),
        (H, W), px_per_mm_x, px_per_mm_y, _ox, _oy,
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
            'max_depth': max_depth,
            'width_cm': width_cm,
            'height_cm': height_cm,
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
                            width_cm, height_cm,
                            origin="Bottom-Left",
                            feed_rate=800.0,
                            threshold_um=0.0,
                            min_feature_width_mm=0.0,
                            max_feature_width_mm=0.0,
                            stepdown_mm=0.0, final_stepdown_mm=0.0,
                            skeleton_mode="medial_axis",
                            prior_accessible=None, previous_result=None,
                            progress=None):
    """Compute a V-carving toolpath for fine details not reachable by raster.

    Parameters
    ----------
    depth_map             : 2-D float32 [0, 1] — 1.0 = max cut depth
    n_grey_levels         : number of grey levels (for RDP tolerance)
    tool                  : Tool — encapsulates profile, diameter_mm, vbit_angle_deg
    max_depth             : total max cut depth in mm (physical scale)
    width_cm              : physical image width in cm
    height_cm             : physical image height in cm
    origin                : G-code origin convention ("Bottom-Left", "Top-Left", …)
    feed_rate             : cutting feed rate in mm/min
    threshold_um          : minimum fine-detail depth in µm; regions shallower
                            than this are skipped
    min_feature_width_mm  : features narrower than this (mm) are skipped
    stepdown_mm           : max depth increment per z-pass in mm; 0 = single pass
    final_stepdown_mm     : depth increment for the finishing pass
                            (0 = same as stepdown_mm)
    skeleton_mode         : "medial_axis" | "voronoi" | "inscribed"
    prior_accessible      : optional 2-D float32 [0,1] geometrically accessible
                            surface of the prior tool (ToolPath.accessible).
                            Pixels where depth_map − prior_accessible > threshold
                            are treated as fine detail to V-carve.  None (default)
                            treats the full depth_map as fine detail.
    previous_result       : optional 2-D float32 [0,1] cumulative result from a
                            prior step (i.e. ToolPath.result), used as the ceiling
                            when computing the output result.  None defaults to depth_map.
    progress              : callable(pct, message) — thread-safe progress reporter

    Returns
    -------
    ToolPath
    """
    from scipy.ndimage import grey_erosion, distance_transform_edt, label

    _alg_name = ('V-Carving (Voronoi)'   if skeleton_mode == 'voronoi'
                 else 'V-Carving (Inscribed)' if skeleton_mode == 'inscribed'
                 else 'V-Carving')

    H, W = depth_map.shape

    px_per_mm_x = W / (width_cm  * 10.0)
    px_per_mm_y = H / (height_cm * 10.0)
    px_per_mm   = (px_per_mm_x + px_per_mm_y) / 2.0

    W_mm = (W - 1) / px_per_mm_x
    H_mm = (H - 1) / px_per_mm_y
    _ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
           "Top-Left":    0,       "Top-Right":    W_mm,
           "Center":      W_mm/2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm/2}.get(origin, 0)

    # ── Stage 1 · Unaccessible-depth mask (0–10 %) ────────────────
    progress(0, "Computing unaccessible-depth mask…")
    if prior_accessible is None:
        unaccessible = depth_map.copy()
    else:
        unaccessible = np.clip(depth_map - prior_accessible, 0.0, 1.0)

    threshold_norm = (threshold_um / 1000.0) / max_depth if max_depth else 0.0

    unaccessible_filtered, mask_vcarve = compute_vcarve_mask(
        unaccessible, threshold_norm, min_feature_width_mm, px_per_mm,
        max_feature_width_mm,
    )

    if not np.any(mask_vcarve):
        progress(100, "No fine-detail regions found — nothing to V-carve.")
        return ToolPath(
            segments=[],
            result=(previous_result.copy() if previous_result is not None
                    else np.zeros_like(depth_map, dtype=np.float32)),
            metadata={'algorithm': _alg_name,
                      'vbit_angle_deg': tool.vbit_angle_deg,
                      'threshold_um': threshold_um,
                      'max_depth': max_depth}
        )

    progress(10, f"Fine-detail mask: {np.count_nonzero(mask_vcarve)} pixels.")

    # ── Stage 2a · Build tool kernel (10–20 %) ────────────────────
    progress(10, "Building tool geometry kernel…")
    if tool.profile == "Ball" and tool.diameter_mm <= 0.0:
        raise ValueError("Ball profile requires a positive tool diameter.")
    kernel_vcarve, footprint_vcarve = _build_kernel(tool, px_per_mm, max_depth)
    ksize = kernel_vcarve.shape[0]
    progress(20, f"Kernel ready  ({ksize}×{ksize} px, profile: {tool.profile}).")

    # ── Stage 2b · Tool-centre depth limit (20–40 %) ─────────────
    progress(20, "Computing tool-centre depth limit (morphological erosion)…")
    center_depth_contour_vcarve = grey_erosion(
        depth_map.astype(np.float64),
        footprint=footprint_vcarve,
        structure=-kernel_vcarve.astype(np.float64),
    ).astype(np.float32)
    center_depth_contour_vcarve = np.clip(center_depth_contour_vcarve, 0.0, 1.0)
    progress(40, "Tool-centre depth limit computed.")

    # ── Stage 3 · Medial axis (40–60 %) ───────────────────────────
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
        raw_contours = _find_contours(unaccessible_filtered, threshold_norm)
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
            skeleton = np.zeros_like(mask_vcarve, dtype=bool)
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
                      'threshold_um': threshold_um,
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
    vcarve_link_mm  = 2.0 / min(px_per_mm_x, px_per_mm_y)
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
            x_s = sc / px_per_mm_x - _ox
            y_s = (H - 1 - sr) / px_per_mm_y - _oy
            x_e = ec / px_per_mm_x - _ox
            y_e = (H - 1 - er) / px_per_mm_y - _oy
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
        tool_pos[0] = ec / px_per_mm_x - _ox
        tool_pos[1] = (H - 1 - er) / px_per_mm_y - _oy

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
                x = c / px_per_mm_x - _ox
                y = (H - 1 - r) / px_per_mm_y - _oy
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

    all_segments = []

    if stepdown_mm <= 0.0:
        def _depth_single(r, c):
            return -_depth_norm[r, c] * max_depth
        all_segments = _build_pass_segments(ordered_branches, _depth_single)
    else:
        final_norm = final_stepdown_mm / max_depth
        sd_norm    = stepdown_mm / max_depth

        skel_rows, skel_cols = np.where(skeleton)
        if len(skel_rows):
            max_roughable_norm = float(
                (_depth_norm[skel_rows, skel_cols] - final_norm)
                .clip(min=0).max()
            )
        else:
            max_roughable_norm = 0.0

        if max_roughable_norm > 0 and sd_norm > 0:
            n_rough = max(1, int(np.ceil(max_roughable_norm / sd_norm)))
            pass_ceilings = [
                min((i + 1) * sd_norm, max_roughable_norm)
                for i in range(n_rough)
            ]
        else:
            pass_ceilings = []

        prev_ceiling = 0.0
        for ceiling_norm in pass_ceilings:
            _pc = prev_ceiling
            _cn = ceiling_norm
            _fn = final_norm

            def _depth_rough(r, c, _pc=_pc, _cn=_cn, _fn=_fn):
                roughing_target = _depth_norm[r, c] - _fn
                if roughing_target <= _pc:
                    return None
                return -min(roughing_target, _cn) * max_depth

            all_segments.extend(
                _build_pass_segments(ordered_branches, _depth_rough)
            )
            prev_ceiling = ceiling_norm

        if final_stepdown_mm > 0.0:
            def _depth_final(r, c):
                return -_depth_norm[r, c] * max_depth
            all_segments.extend(
                _build_pass_segments(ordered_branches, _depth_final)
            )

    progress(90, f"Generated {len(all_segments)} segments across all passes.")

    # ── Stage 6 · Simplify, simulate result, Simplify and assemble ToolPath (90–100 %) ───────
    progress(90, "Simplifying tool path (RDP)…")
    rdp_tol = max(
        1.0 / px_per_mm_x,
        1.0 / px_per_mm_y,
        max_depth / (n_grey_levels - 1),
    )
    simplified = [simplify_line(pts, rdp_tol) for pts in all_segments]

    progress(93, "Simulating tool-swept removal surface…")
    vcarve_result = simulate_toolpath_result(
        ToolPath(segments=simplified),
        (H, W), px_per_mm_x, px_per_mm_y, _ox, _oy,
        max_depth, kernel_vcarve, footprint_vcarve,
    )

    progress(97, "Creating ToolPath object…")
    toolpath = ToolPath(
        segments=simplified,
        result=np.maximum(
            previous_result if previous_result is not None else 0.0,
            vcarve_result).astype(np.float32),
        accessible=None,
        metadata={
            'algorithm':       _alg_name,
            'skeleton_mode':   skeleton_mode,
            'profile':         tool.profile,
            'vbit_angle_deg':  tool.vbit_angle_deg,
            'min_feature_width_mm':  min_feature_width_mm,
            'threshold_um':          threshold_um,
            'stepdown_mm':           stepdown_mm,
            'final_stepdown_mm':     final_stepdown_mm,
            'max_depth':       max_depth,
            'width_cm':        width_cm,
            'height_cm':       height_cm,
            'n_branches':      n_branches,
            'rdp_tolerance':   rdp_tol,
        }
    )
    progress(100, "V-carve tool path computation complete!")
    return toolpath


def compute_contour_toolpath(
    depth_map, n_grey_levels,
    tool, stepdown_mm, max_depth,
    width_cm, height_cm,
    origin="Bottom-Left",
    feed_rate=800.0,
    prior_accessible=None,
    previous_result=None,
    progress=None,
    adaptive_levels=False,
):
    """Compute a waterline (contour) tool path from a normalised depth map.

    The achievable depth surface is sliced at evenly-spaced Z-levels separated
    by *stepdown_mm*.  At each level the iso-depth contour lines are extracted
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
    stepdown_mm : float
        Z increment between contour levels in mm.  Also controls the number of
        passes in adaptive mode: n_passes = ceil(max_depth / stepdown_mm).
    adaptive_levels : bool
        When True, place the n_passes contour levels at equal-area (CDF-based)
        depth intervals rather than uniform Z spacing.  This concentrates passes
        where the depth map has the most pixels, avoiding redundant cuts between
        discrete grey-level transitions.
    max_depth : float
        Total maximum cut depth in mm.
    width_cm, height_cm : float
        Physical dimensions of the work area in cm.
    origin : str
        G-code origin convention ("Bottom-Left", "Top-Left", etc.).
    feed_rate : float
        Cutting feed rate in mm/min.
    prior_accessible : 2-D float32 array, values in [0, 1], or None
        Reserved for future use (analogous to vcarve's *prior_accessible*).
        Currently ignored; the accessible surface is always computed from
        *depth_map*.
    previous_result : 2-D float32 array, values in [0, 1], or None
        Cumulative result from a prior step (i.e. ToolPath.result); used as the
        ceiling for the output result.  When None, *depth_map* is used.
    progress : callable(pct, message)
        Thread-safe progress reporter.

    Returns
    -------
    ToolPath
    """
    import math
    from scipy.ndimage import grey_erosion, grey_dilation
    from skimage.measure import find_contours

    H, W = depth_map.shape

    px_per_mm_x = W / (width_cm  * 10.0)
    px_per_mm_y = H / (height_cm * 10.0)
    px_per_mm   = (px_per_mm_x + px_per_mm_y) / 2.0

    W_mm = (W - 1) / px_per_mm_x
    H_mm = (H - 1) / px_per_mm_y
    _ox = {"Bottom-Left": 0,       "Bottom-Right": W_mm,
           "Top-Left":    0,       "Top-Right":    W_mm,
           "Center":      W_mm / 2}.get(origin, 0)
    _oy = {"Bottom-Left": 0,       "Bottom-Right": 0,
           "Top-Left":    H_mm,    "Top-Right":    H_mm,
           "Center":      H_mm / 2}.get(origin, 0)

    # ── Stage 1 · Build tool kernel (0–10 %) ──────────────────────────────
    progress(0, "Building tool geometry kernel…")
    kernel_contour, footprint_contour = _build_kernel(tool, px_per_mm, max_depth)
    ksize = kernel_contour.shape[0]
    progress(10, f"Kernel ready  ({ksize}×{ksize} px, profile: {tool.profile}).")

    # ── Stage 2 · Compute achievable depth surface (10–35 %) ──────────────
    progress(10, "Computing achievable depth surface (morphological opening)…")
    structure = -kernel_contour.astype(np.float64)

    # Max depth the tool can reach when centred at each pixel (erosion).
    # Used for contour extraction and adaptive level planning.
    center_depth_contour = grey_erosion(depth_map.astype(np.float64),
                              footprint=footprint_contour,
                              structure=structure).astype(np.float32)
    center_depth_contour = np.clip(center_depth_contour, 0.0, 1.0)

    # Dilation of the erosion gives the true achievable depth at each material
    # point: the tool can carve (x,y) while centred at an offset position.
    # Erosion followed by dilation = morphological opening.
    accessible_contour = grey_dilation(center_depth_contour.astype(np.float64),
                                       footprint=footprint_contour,
                                       structure=structure).astype(np.float32)
    accessible_contour = np.clip(accessible_contour, 0.0, 1.0)
    progress(35, "Achievable depth surface computed.")

    # ── Stage 3 · Plan Z-levels (35–40 %) ─────────────────────────────────
    progress(35, "Planning contour levels…")
    n_passes = max(1, math.ceil(max_depth / stepdown_mm))
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
            z_levels = [min(i * stepdown_mm / max_depth, 1.0)
                        for i in range(1, n_passes + 1)]
        progress(40, f"{len(z_levels)} contour level{'s' if len(z_levels) != 1 else ''}  "
                     f"(adaptive, target stepdown {stepdown_mm:.2f} mm).")
    else:
        z_levels = [min(i * stepdown_mm / max_depth, 1.0)
                    for i in range(1, n_passes + 1)]
        progress(40, f"{n_passes} contour level{'s' if n_passes != 1 else ''}  "
                     f"({stepdown_mm:.2f} mm each).")

    # ── Stage 4 · Extract contours and build segments (40–80 %) ───────────
    rdp_tol = max(1.0 / px_per_mm_x,
                  1.0 / px_per_mm_y,
                  max_depth / (n_grey_levels - 1))

    # Half a grey-level step in normalised units.  Subtracting this from the
    # find_contours threshold keeps flat-region pixels firmly "inside" even
    # when z_norm lands at or just above their exact value (e.g. from CDF
    # quantile rounding).  Without it the contour jumps to the inner wall of
    # a flat step and the tool cuts air above the neighbouring deeper region.
    _contour_eps = 0.5 / max(n_grey_levels - 1, 1)

    all_segments = []
    current_pos  = np.array([0.0, 0.0])

    n_levels = len(z_levels)
    for pass_idx, z_norm in enumerate(z_levels):
        progress(40 + 40 * pass_idx // n_levels,
                 f"Extracting contours at Z = {-z_norm * max_depth:.3f} mm…")
        z_mm = -z_norm * max_depth

        raw_contours = find_contours(center_depth_contour, max(z_norm - _contour_eps, 0.0))
        contours     = [c for c in raw_contours if len(c) >= 3]
        if not contours:
            continue

        # Greedy nearest-neighbour ordering across contours at this Z-level.
        # For closed contours any rotation of the loop is valid, so we sample
        # candidate start points; for open contours only the two endpoints are
        # considered (flip=True covers the other end).
        remaining = contours[:]
        while remaining:
            best_idx  = 0
            best_flip = False
            best_rot  = 0
            best_dist = float('inf')

            for i, c in enumerate(remaining):
                n         = len(c)
                is_closed = np.linalg.norm(c[0] - c[-1]) < 1.5
                rot_cands = range(0, n, max(1, n // 20)) if is_closed else [0]
                for flip in (False, True):
                    orient = c[::-1] if flip else c
                    for j in rot_cands:
                        row0, col0 = orient[j]
                        x0 = col0 / px_per_mm_x - _ox
                        y0 = (H - 1 - row0) / px_per_mm_y - _oy
                        d  = (x0 - current_pos[0])**2 + (y0 - current_pos[1])**2
                        if d < best_dist:
                            best_dist = d
                            best_idx  = i
                            best_flip = flip
                            best_rot  = j

            c_chosen  = remaining.pop(best_idx)
            is_closed = np.linalg.norm(c_chosen[0] - c_chosen[-1]) < 1.5
            if best_flip:
                c_chosen = c_chosen[::-1]
            if best_rot > 0:
                c_chosen = np.roll(c_chosen, -best_rot, axis=0)
            if is_closed:
                c_chosen = np.vstack([c_chosen, c_chosen[0:1]])

            points = [(col / px_per_mm_x - _ox,
                       (H - 1 - row) / px_per_mm_y - _oy,
                       z_mm)
                      for row, col in c_chosen]
            points = simplify_line(points, rdp_tol)
            if len(points) < 2:
                continue

            all_segments.append(points)
            last = points[-1]
            current_pos[0] = last[0]
            current_pos[1] = last[1]

    progress(80, f"Contour segments built ({len(all_segments)} segments).")

    # ── Stage 5 · Simulate result (80–90 %) ───────────────────────────────
    progress(80, "Simulating tool-swept removal surface…")
    material_removed = simulate_toolpath_result(
        ToolPath(segments=all_segments),
        (H, W), px_per_mm_x, px_per_mm_y, _ox, _oy,
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
            'stepdown_mm':     stepdown_mm,
            'adaptive_levels': adaptive_levels,
            'n_passes':        len(z_levels),
            'width_cm':        width_cm,
            'height_cm':       height_cm,
            'rdp_tolerance':   rdp_tol,
        }
    )
    progress(100, "Contour tool path computation complete!")
    return tp
