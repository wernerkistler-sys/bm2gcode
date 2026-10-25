"""Undercut toolpaths: waterline roughing and finishing under overhangs.

A cutter whose head is wider than its neck (Lollipop, T-Slot, Dovetail) can
reach sideways under an overhang that a vertical plunge never reaches. This
module plans such cuts from the undercut gaps of the step's STL sources
(layers.Gaps) at uniform depth levels. At each level it finds where the
whole tool fits between the floor below and the ceiling above, cuts rings
that start in open sky and work toward the undercut wall, ends with the
waterline along the wall, and joins each chain of rings to open sky with
lead paths at the level's depth, so that every plunge and retract happens
where nothing is overhead. See README.md's "Undercut" section.

Depths are normalised as in algorithms.py: 0 = stock top, 1 = max_depth,
larger = deeper.
"""
import math

import numpy as np

from tool import Tool, UNDERCUT_PROFILES, undercut_geometry_error
from toolpath import ToolPath

SKY  = -2.0e6   # ceiling of a column with nothing overhead
NECK = 1.0e6    # tool top over the neck: the neck runs up without end
MAX_BANDS = 8   # underside-height bands a footprint is split into at most
TOP_EPS = 1e-6  # a gap ceiling this close to the top surface opens to the sky


def equivalent_2p5d_tool(tool):
    """The 2.5D tool with the same underside: what a vertical plunge sees."""
    if tool.profile == "Lollipop":
        return Tool("Ball", tool.diameter_mm)
    return Tool("End Mill", tool.diameter_mm)


def _head_profile(tool, r):
    """(lo, hi) in mm of the head at radii *r* (0 <= r <= R): the heights
    of its underside and top above the tip, ignoring the neck."""
    R = tool.diameter_mm / 2.0
    if tool.profile == "Lollipop":
        half = np.sqrt(np.maximum(0.0, R * R - r * r))
        return R - half, R + half
    if tool.profile == "T-Slot":
        return np.zeros_like(r), np.full_like(r, tool.head_height_mm)
    if tool.profile == "Dovetail":
        tan_a = math.tan(math.radians(tool.dovetail_angle_deg))
        return (np.zeros_like(r),
                np.clip((R - r) * tan_a, 0.0, tool.head_height_mm))
    raise ValueError(f"Not an undercut profile: {tool.profile!r}")


def build_undercut_kernels(tool, px_per_mm, max_depth, allowance_mm=0.0):
    """Return (lo, hi, footprint) for an undercut tool, normalised by
    max_depth and centred on the tool axis.

    The tool fills depths [d - hi, d - lo] in each column of its footprint
    when its tip is at depth d: lo is the underside's height above the tip,
    hi the top's. Over the neck hi is NECK. A positive *allowance_mm* grows
    the tool by a sphere of that radius (the exact Minkowski sum), so the
    real tool stays that far from the surface while the tip reference is
    unchanged. Because the tool is rotationally symmetric, the point of it
    nearest any point lies in that point's meridian plane, so the sum is
    taken on the (r, height) profile: lo'(r) = min lo(r') - sqrt(a^2 -
    (r - r')^2) over |r - r'| <= a, and hi' likewise with max and +.
    """
    R = tool.diameter_mm / 2.0
    r_n = tool.neck_diameter_mm / 2.0
    a = max(allowance_mm, 0.0)
    k_rad = max(1, int(np.ceil((R + a) * px_per_mm)))
    ky, kx = np.mgrid[-k_rad:k_rad + 1, -k_rad:k_rad + 1]
    r_px = np.sqrt(kx ** 2 + ky ** 2)
    footprint = r_px <= k_rad
    # Pixels between R + a and the footprint's edge take the rim's values,
    # so the discretised tool is never smaller than the real one.
    r_eff = np.minimum(r_px / px_per_mm, R + a)
    radii, inv = np.unique(r_eff, return_inverse=True)

    if a > 0.0:
        rp = np.linspace(0.0, R, max(400, int(np.ceil(R / (a / 100.0))) + 1))
        lo_p, hi_p = _head_profile(tool, rp)
        dr = radii[:, None] - rp[None, :]
        reach = np.abs(dr) <= a
        cap = np.sqrt(np.maximum(0.0, a * a - dr * dr))
        lo_r = np.min(np.where(reach, lo_p[None, :] - cap, np.inf), axis=1)
        hi_r = np.max(np.where(reach, hi_p[None, :] + cap, -np.inf), axis=1)
    else:
        lo_r, hi_r = _head_profile(tool, radii)

    lo = (lo_r / max_depth)[inv].reshape(r_px.shape)
    hi = (hi_r / max_depth)[inv].reshape(r_px.shape)
    # Half a pixel wider than the (grown) neck, so the discretised neck is
    # never thinner than the real one.
    hi[r_px / px_per_mm <= r_n + a + 0.5 / px_per_mm] = NECK
    lo = np.where(footprint, lo, 0.0)
    hi = np.where(footprint, hi, 0.0)
    return lo.astype(np.float64), hi.astype(np.float64), footprint


def _shank_limit(depth_map, tool, px_per_mm, max_depth, allowance_mm):
    """Deepest tip depth at which a shank wider than the neck stays above
    all material: the shank starts neck_length_mm above the tip, and its
    columns must be clear from there up. None if there is no wider shank."""
    if tool.shank_diameter_mm <= tool.neck_diameter_mm:
        return None
    from grey_morphology import grey_erosion
    r_s = tool.shank_diameter_mm / 2.0 + allowance_mm
    k = max(1, int(np.ceil(r_s * px_per_mm)))
    ky, kx = np.mgrid[-k:k + 1, -k:k + 1]
    fp = np.sqrt(kx ** 2 + ky ** 2) <= r_s * px_per_mm + 0.5
    top = grey_erosion(depth_map.astype(np.float64), footprint=fp)
    return top + (tool.neck_length_mm - allowance_mm) / max_depth


def _lo_bands(lo, footprint, step):
    """Split *footprint* into bands whose underside heights lo span at most
    *step*: [(lo_mid, band footprint)], lo_mid the middle of the band's lo
    range. A flat underside (T-Slot, Dovetail) is one band."""
    vals = lo[footprint]
    lo_min = float(vals.min())
    n = max(1, int(np.ceil((float(vals.max()) - lo_min) / step)))
    idx = np.minimum(((lo - lo_min) / step).astype(np.int64), n - 1)
    bands = []
    for j in range(n):
        band = footprint & (idx == j)
        if band.any():
            v = lo[band]
            bands.append((0.5 * (float(v.min()) + float(v.max())), band))
    return bands


def _floor_ceil_at(gaps, gap_pix, top, d):
    """Per-column floor and ceiling bounding the air at depth d (the
    tool's underside in that column): the deepest gap starting above d
    (the solid containing d, or the air just above it, lies below that
    gap's ceiling). Columns with no such gap have the top surface as floor
    and open sky above. Returns (F, C, active) with active marking the
    columns that use a gap."""
    F = top.astype(np.float64).copy()
    C = np.full(top.shape, SKY)
    active = np.zeros(top.shape, bool)
    idx = np.flatnonzero(gaps.ceil < d)
    if idx.size:
        pix = gap_pix[idx]
        last = idx[np.r_[pix[1:] != pix[:-1], True]]
        F.flat[gap_pix[last]] = gaps.floor[last]
        C.flat[gap_pix[last]] = gaps.ceil[last]
        active.flat[gap_pix[last]] = True
    return F, C, active


def _open_to_sky(depth_map, gaps):
    """(depth_map, gaps) with every gap that starts at its column's top
    surface (within TOP_EPS) merged into the open sky above it: no solid
    lies between, so the column's top drops to the gap's floor and the gap
    is dropped -- repeatedly, down a column. Such gaps are where the
    composite's top is not solid: a partly covered outline pixel, whose top
    blends the source with what lies beneath, or a grazed rim of zero
    thickness. Merged, the plunge-from-above surface, which reads the depth
    map, agrees with the gaps there."""
    from layers import Gaps
    pix = gaps.pixel_index()
    if pix.size == 0:
        return depth_map, gaps
    top = depth_map.astype(np.float64).ravel()
    rank = np.arange(pix.size) - gaps.offsets[pix]
    open_col = np.ones(top.size, bool)          # still open down from the top
    merged = np.zeros(pix.size, bool)
    for k in range(int(rank.max()) + 1):
        i = np.flatnonzero(rank == k)
        i = i[open_col[pix[i]]]
        hit = np.abs(gaps.ceil[i] - top[pix[i]]) <= TOP_EPS
        top[pix[i[hit]]] = gaps.floor[i[hit]]
        merged[i[hit]] = True
        open_col[pix[i[~hit]]] = False
        if not hit.any():
            break
    if not merged.any():
        return depth_map, gaps
    keep = ~merged
    return (top.reshape(depth_map.shape).astype(depth_map.dtype),
            Gaps.from_lists(gaps.shape, pix[keep], gaps.ceil[keep], gaps.floor[keep]))


def _gather_gaps(gaps, src_pix, new_shape):
    """Gaps for a new grid whose pixel i takes column src_pix[i] of *gaps*
    (src_pix < 0: no gaps)."""
    from layers import Gaps
    src_pix = np.asarray(src_pix, np.int64)
    valid = src_pix >= 0
    src = np.where(valid, src_pix, 0)
    start = gaps.offsets[src]
    n = np.where(valid, gaps.offsets[src + 1] - start, 0)
    total = int(n.sum())
    if total == 0:
        return Gaps.empty(new_shape)
    new_pix = np.repeat(np.arange(src_pix.size), n)
    k = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    gi = start[new_pix] + k
    return Gaps.from_lists(new_shape, new_pix, gaps.ceil[gi], gaps.floor[gi])


def _resample_gaps(gaps, new_shape):
    """Nearest-column resample of *gaps* onto *new_shape*, with the corner-
    aligned grid scipy.ndimage.zoom uses."""
    H, W = gaps.shape
    H2, W2 = new_shape
    rr = np.rint(np.arange(H2) * (H - 1) / max(H2 - 1, 1)).astype(np.int64)
    cc = np.rint(np.arange(W2) * (W - 1) / max(W2 - 1, 1)).astype(np.int64)
    src = (rr[:, None] * W + cc[None, :]).ravel()
    return _gather_gaps(gaps, src, new_shape)


def _windows(mask, pad, shape, max_windows=64):
    """Disjoint boxes (r0, r1, c0, c1) covering mask's connected components,
    each grown by *pad* and clipped to *shape*. Overlapping boxes are merged,
    so every component lies wholly inside exactly one box; one box around
    everything when there are more than max_windows components."""
    from scipy.ndimage import find_objects, label
    H, W = shape
    lab, n = label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    if n > max_windows:
        rows, cols = np.nonzero(mask)
        found = [(rows.min(), rows.max() + 1, cols.min(), cols.max() + 1)]
    else:
        found = [(s[0].start, s[0].stop, s[1].start, s[1].stop)
                 for s in find_objects(lab)]
    boxes = [(max(0, r0 - pad), min(H, r1 + pad), max(0, c0 - pad), min(W, c1 + pad))
             for r0, r1, c0, c1 in found]
    merged = True
    while merged:
        merged = False
        out = []
        for b in boxes:
            for i, o in enumerate(out):
                if b[0] < o[1] and o[0] < b[1] and b[2] < o[3] and o[2] < b[3]:
                    out[i] = (min(b[0], o[0]), max(b[1], o[1]),
                              min(b[2], o[2]), max(b[3], o[3]))
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def _sample(field, pts):
    from scipy.ndimage import map_coordinates
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return map_coordinates(field, [pts[:, 0], pts[:, 1]], order=1, mode='nearest')


def _segment_free(field, p, q, thr):
    """True if *field* (bilinear) stays >= thr along the straight line p->q,
    sampled at most half a pixel apart."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    n = max(2, int(np.ceil(np.hypot(*(q - p)) / 0.5)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return bool(np.all(_sample(field, p + (q - p) * t) >= thr))


def _bridge_gaps(pts, in_u, closed, max_len):
    """in_u with every stretch outside the zone that lies between two points
    inside it and spans at most max_len of contour arc length marked inside.
    For a closed contour pts holds each point once and wraps around."""
    n = len(pts)
    if in_u.all() or not in_u.any():
        return in_u
    nxt = np.roll(pts, -1, axis=0) if closed else pts[1:]
    seg = np.hypot(*(nxt - pts[:len(nxt)]).T)
    t = int(np.argmax(in_u)) if closed else 0
    order = (np.arange(n) + t) % n
    u = in_u[order]
    cum = np.r_[0.0, np.cumsum(seg[order] if closed else seg)]
    edges = np.diff(np.r_[0, (~u).astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if a == 0 or (b == n and not closed):
            continue
        if cum[b] - cum[a - 1] <= max_len:
            u[a:b] = True
    out = np.empty_like(in_u)
    out[order] = u
    return out


def _split_runs(pts, in_u, closed, max_gap):
    """Split a contour into runs of points inside the undercut zone, each
    extended by one neighbouring point outside it (which lies in clear sky)
    where there is one. Stretches outside the zone of at most max_gap arc
    length between two runs are kept in the run: the contour is free
    throughout, and following it is cheaper than an escape and re-entry.
    Returns [(points, is_closed)]."""
    if closed:
        pts, in_u = pts[:-1], in_u[:-1]
    in_u = _bridge_gaps(pts, in_u, closed, max_gap)
    if closed:
        if in_u.all():
            return [(pts, True)]
        k = int(np.argmin(in_u))            # first point outside the zone
        pts, in_u = np.roll(pts, -k, axis=0), np.roll(in_u, -k)
        pts = np.vstack([pts, pts[:1]])
        in_u = np.append(in_u, in_u[0])
    n = len(pts)
    edges = np.diff(np.r_[0, in_u.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = []
    for i0, i1 in zip(starts, stops):       # in_u[i0:i1] all True
        a, b = max(i0 - 1, 0), min(i1 + 1, n)
        if b - a >= 2:
            runs.append((pts[a:b], False))
    return runs


class _Level:
    """Free-space data of one depth level, on a crop of the canvas."""

    def __init__(self, d, r0, c0, g, g_sky, eps):
        from scipy.ndimage import binary_erosion
        from skimage.graph import MCP_Geometric
        self.d, self.r0, self.c0 = d, r0, c0
        self.g, self.g_sky, self.eps = g, g_sky, eps
        self.free = g >= -eps
        sky = g_sky >= -eps
        square = np.ones((3, 3), bool)
        domain = binary_erosion(self.free, structure=square)
        starts = binary_erosion(sky, structure=square) & domain
        self.starts = starts
        self.reachable = starts.any()
        self.cum = None
        if self.reachable:
            costs = np.where(domain, 1.0, np.inf)
            self.mcp = MCP_Geometric(costs, fully_connected=True)
            # Every path into the sky set enters through its frontier, so
            # only those pixels need to be seeds; the rest are sky already.
            frontier = starts & ~binary_erosion(starts, structure=square)
            cum, _ = self.mcp.find_costs(list(zip(*np.nonzero(frontier))))
            self.cum = np.where(starts, 0.0, cum)
            from scipy.ndimage import minimum_filter
            # Same reach as connect()'s search, so the estimate never calls
            # a point unreachable that connect() could join.
            self.cum_near = minimum_filter(self.cum, size=5, mode='constant',
                                           cval=np.inf)

    def connect(self, q):
        """Escape route from contour point q to open sky, as (cost_px,
        pixel path from a sky pixel to the pixel joined to q), or None."""
        if not self.reachable:
            return None
        H, W = self.g.shape
        r, c = int(round(q[0])), int(round(q[1]))
        best = None
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W) or not np.isfinite(self.cum[rr, cc]):
                    continue
                cost = self.cum[rr, cc] + math.hypot(rr - q[0], cc - q[1])
                if best is not None and cost >= best[0]:
                    continue
                if _segment_free(self.g, q, (rr, cc), -2.0 * self.eps - 1e-9):
                    best = (cost, (rr, cc))
        if best is None:
            return None
        if self.starts[best[1]]:
            return best[0], np.asarray([best[1]], np.float64)
        path = np.asarray(self.mcp.traceback(best[1]), np.float64)
        return best[0], path

    def seam_order(self, pts):
        """Indices of a closed ring's points, cheapest estimated escape
        first."""
        H, W = self.g.shape
        rr = np.clip(np.rint(pts[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.rint(pts[:, 1]).astype(int), 0, W - 1)
        est = self.cum_near[rr, cc] if self.reachable else np.full(len(pts), np.inf)
        return np.argsort(est, kind='stable'), est


def compute_undercut_toolpath(
        depth_map, gaps, tool, z_increment_mm, stepover_pct, max_depth,
        width_mm, height_mm, origin="Bottom-Left", feed_rate=800.0,
        previous_result=None, z_safe=5.0, feed_rapid=3000.0,
        cut_direction="Climb", extra_offset_mm=0.0, n_grey_levels=65536,
        progress=None, erosion_cache=None, erosion_cache_key=None):
    """Plan undercut roughing and finishing at uniform depth levels.

    Parameters
    ----------
    depth_map      : 2-D float32 [0, 1] composite depth map (1.0 = max_depth)
    gaps           : layers.Gaps on the same grid -- the air below the top
                     surface that only a sideways move can reach
    tool           : Tool with an undercut profile (tool.UNDERCUT_PROFILES)
    z_increment_mm : spacing between depth levels
    stepover_pct   : spacing between rings, % of the head diameter
    cut_direction  : "Either" | "Climb" | "Conventional"; Climb keeps the
                     uncut material on the right of travel, as in Contour
    extra_offset_mm: stock to leave on every surface (Minkowski allowance)
    previous_result: the prior step's ToolPath.result; passed through as
                     this step's result (see README.md)

    At each level d the tool fits at a pixel if its underside clears the
    floor below and its top (and neck) clears the ceiling above, in each
    column both taken from the gap the tool's underside is in there
    (_floor_ceil_at, per band of underside height), or if it can reach d by a
    vertical plunge from above (clear sky). g, the larger of the two
    clearances, is >= 0 exactly where the tool fits; its zero contour is the
    waterline and its distance transform gives the roughing rings. Only
    ring stretches outside clear sky are cut; every chain of them starts and
    ends with an escape path to clear sky at the level's depth.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt, label
    from grey_morphology import grey_dilation, grey_erosion
    from skimage.measure import find_contours
    from algorithms import _build_kernel, _retract_breakeven_mm, simplify_line

    if progress is None:
        progress = lambda pct, msg: None
    if tool.profile not in UNDERCUT_PROFILES:
        raise ValueError(f"Undercut needs a Lollipop, T-Slot or Dovetail tool, "
                         f"not {tool.profile!r}.")
    err = undercut_geometry_error(tool)
    if err:
        raise ValueError(err)
    if z_increment_mm <= 0.0:
        raise ValueError("Step-down must be greater than 0.")
    if stepover_pct <= 0.0:
        raise ValueError("Step-over must be greater than 0.")
    warnings = []

    def warn(msg, pct):
        warnings.append(msg)
        progress(pct, "Warning: " + msg)

    depth_map, gaps = _open_to_sky(depth_map, gaps)

    # ── Square pixels ─────────────────────────────────────────────────────
    H, W = depth_map.shape
    in_shape = (H, W)
    px_per_mm_x = W / width_mm
    px_per_mm_y = H / height_mm
    if px_per_mm_x != px_per_mm_y:
        from scipy.ndimage import zoom as _zoom
        target = max(px_per_mm_x, px_per_mm_y)
        zy, zx = target / px_per_mm_y, target / px_per_mm_x
        progress(0, "Resampling to square pixels…")
        depth_map = _zoom(depth_map, (zy, zx), order=1).astype(np.float32)
        if previous_result is not None:
            previous_result = _zoom(previous_result, (zy, zx), order=1).astype(np.float32)
        gaps = _resample_gaps(gaps, depth_map.shape)
        H, W = depth_map.shape
    px_per_mm = W / width_mm
    W_mm = (W - 1) / px_per_mm
    H_mm = (H - 1) / px_per_mm
    ox = {"Bottom-Left": 0, "Bottom-Right": W_mm, "Top-Left": 0,
          "Top-Right": W_mm, "Center": W_mm / 2}.get(origin, 0)
    oy = {"Bottom-Left": 0, "Bottom-Right": 0, "Top-Left": H_mm,
          "Top-Right": H_mm, "Center": H_mm / 2}.get(origin, 0)

    eps = 0.5 / max(n_grey_levels - 1, 1)
    depth64 = depth_map.astype(np.float64)

    # ── Kernels and the plunge-from-above surface ─────────────────────────
    progress(0, "Building tool geometry…")
    lo, hi, fp = build_undercut_kernels(tool, px_per_mm, max_depth, extra_offset_mm)
    k_rad = fp.shape[0] // 2
    real_kernel, real_fp = _build_kernel(equivalent_2p5d_tool(tool), px_per_mm,
                                         max_depth)
    cache_key = (erosion_cache_key, 'undercut', extra_offset_mm)
    cached = erosion_cache.get(cache_key) if erosion_cache is not None else None
    if cached is not None and cached[0].shape == depth_map.shape:
        center25, accessible = cached
    else:
        progress(2, "Computing plunge-accessible depth…")
        center25 = grey_erosion(depth64, footprint=fp, structure=-lo)
        center_real = grey_erosion(depth64, footprint=real_fp,
                                   structure=-real_kernel.astype(np.float64))
        accessible = np.clip(grey_dilation(
            np.clip(center_real, 0.0, 1.0), footprint=real_fp,
            structure=-real_kernel.astype(np.float64)), 0.0, 1.0).astype(np.float32)
        if erosion_cache is not None:
            erosion_cache[cache_key] = (center25, accessible)
    shank = _shank_limit(depth_map, tool, px_per_mm, max_depth, extra_offset_mm)
    sky_limit = center25 if shank is None else np.minimum(center25, shank)
    # A Lollipop's underside rises up to R above its tip, so a column at the
    # head's rim can lie in a gap above the one the tip is in. Each band of
    # the footprint picks its gaps at its own underside depth. A wrong pick
    # only rejects positions next to solid thinner than half a band, so
    # bands are a pixel's worth of depth thick, but at most MAX_BANDS.
    lo_span = float(lo[fp].max() - lo[fp].min())
    bands = _lo_bands(lo, fp, max(1.0 / (px_per_mm * max_depth), lo_span / MAX_BANDS))
    band_lo = np.array([b[0] for b in bands])

    gap_pix = gaps.pixel_index()
    if gap_pix.size == 0:
        warn("The step's sources have no undercuts.", 100)

    # ── Levels ────────────────────────────────────────────────────────────
    n_levels = max(1, math.ceil(max_depth / z_increment_mm))
    levels = [min(i * z_increment_mm / max_depth, 1.0) for i in range(1, n_levels + 1)]
    step_px = max(stepover_pct / 100.0 * tool.diameter_mm * px_per_mm, 1.0)
    max_rings = int(np.ceil(tool.diameter_mm * px_per_mm / step_px)) + 2
    reverse = (cut_direction == "Climb")
    thr_link = -2.0 * eps - 1e-9

    chains = []          # dict(level, pts (global row, col), rep pixel)
    u_union = np.zeros((H, W), bool)
    n_unreachable = 0

    for li, d in enumerate(levels):
        progress(5 + 85 * li / len(levels),
                 f"Level {li + 1}/{len(levels)} ({d * max_depth:.2f} mm)…")
        b_min, b_max = d - band_lo.max(), d - band_lo.min()
        if gap_pix.size == 0 or not np.any(gaps.ceil < b_max):
            continue
        # Bands pick the same gaps wherever no ceiling lies between their
        # underside depths; then one pass over the whole footprint does.
        if np.any((gaps.ceil >= b_min) & (gaps.ceil < b_max)):
            picks = [_floor_ceil_at(gaps, gap_pix, depth_map, d - lo_mid) + (band,)
                     for lo_mid, band in bands]
        else:
            picks = [_floor_ceil_at(gaps, gap_pix, depth_map, b_max) + (fp,)]
        active = np.logical_or.reduce([p[2] for p in picks])
        g_sky = sky_limit - d
        g = g_sky.copy()
        for r0, r1, c0, c1 in _windows(active, k_rad + 1, (H, W)):
            R0, R1 = max(0, r0 - k_rad), min(H, r1 + k_rad)
            C0, C1 = max(0, c0 - k_rad), min(W, c1 + k_rad)
            center_k = ceil_k = None
            for F, C, _, band in picks:
                e = grey_erosion(F[R0:R1, C0:C1], footprint=band, structure=-lo)
                c = grey_dilation(C[R0:R1, C0:C1], footprint=band, structure=hi)
                center_k = e if center_k is None else np.minimum(center_k, e)
                ceil_k = c if ceil_k is None else np.maximum(ceil_k, c)
            g_gap = np.minimum(center_k - d, d - ceil_k)
            if shank is not None:
                g_gap = np.minimum(g_gap, shank[R0:R1, C0:C1] - d)
            ir0, ic0 = r0 - R0, c0 - C0
            inner = g_gap[ir0:ir0 + (r1 - r0), ic0:ic0 + (c1 - c0)]
            g[r0:r1, c0:c1] = np.maximum(inner, g_sky[r0:r1, c0:c1])

        undercut = (g >= -eps) & (g_sky < -eps)
        if not undercut.any():
            continue
        u_union |= undercut
        # A straight link between rings at this depth replaces escape +
        # retract + plunge + re-entry; take it up to the length whose feed
        # time equals the retract it saves, and never less than a tool width.
        link_max_px = max(tool.diameter_mm, _retract_breakeven_mm(
            feed_rate, feed_rapid, z_safe, d * max_depth)) * px_per_mm
        pad = 2 * k_rad + int(np.ceil(max_rings * step_px)) + 8
        for r0, r1, c0, c1 in _windows(undercut, pad, (H, W)):
            lev = _Level(d, r0, c0, g[r0:r1, c0:c1], g_sky[r0:r1, c0:c1], eps)
            u_crop = undercut[r0:r1, c0:c1]
            if not u_crop.any():
                continue

            # Rings, outermost (nearest open sky) first, waterline last.
            edt = distance_transform_edt(lev.free)
            ring_runs = []
            for j in range(max_rings, -1, -1):
                raw = (find_contours(lev.g, -eps) if j == 0
                       else find_contours(edt, j * step_px))
                runs_j = []
                for c in raw:
                    if len(c) < 2:
                        continue
                    if reverse:
                        c = c[::-1]
                    closed = np.allclose(c[0], c[-1])
                    # Undercut only where the gap gives more room than a
                    # plunge: on a waterline the sky alone forms, g and
                    # g_sky are equal and sit at the threshold, where a test
                    # of g_sky alone flips from point to point.
                    gs = _sample(lev.g_sky, c)
                    in_u = (gs < -eps) & (_sample(lev.g, c) - gs > eps)
                    if not in_u.any():
                        continue
                    runs_j.extend(_split_runs(c, in_u, closed, link_max_px))
                if runs_j:
                    ring_runs.append(runs_j)
            if not ring_runs:
                continue

            # Chains: greedy nearest within each ring, linked straight when
            # the link stays free and short, else escaped to open sky.
            cur = None          # current chain's points (crop coords)
            cur_pos = None

            def close_chain():
                nonlocal cur
                if cur is None:
                    return
                out = lev.connect(cur[-1])
                # With no escape from the chain's end, retrace the chain
                # itself back to the open sky it started from.
                tail = out[1][::-1] if out is not None else cur[-2::-1]
                pts = np.vstack([cur, tail])
                rep = cur[len(cur) // 2]
                chains.append({'level': li,
                               'pts': pts + (r0, c0),
                               'rep': (int(round(rep[0])) + r0,
                                       int(round(rep[1])) + c0)})
                cur = None

            for runs_j in ring_runs:
                remaining = list(runs_j)
                while remaining:
                    # Pick the nearest run (any point of a closed ring).
                    if cur_pos is None:
                        pick, seam = 0, None
                    else:
                        best = None
                        for i, (pts, closed) in enumerate(remaining):
                            cand = pts if closed else pts[:1]
                            dd = np.hypot(cand[:, 0] - cur_pos[0], cand[:, 1] - cur_pos[1])
                            k = int(np.argmin(dd))
                            if best is None or dd[k] < best[0]:
                                best = (dd[k], i, k if closed else None)
                        _, pick, seam = best
                    pts, closed = remaining.pop(pick)

                    linked = False
                    if cur is not None:
                        if closed:
                            dd = np.hypot(pts[:, 0] - cur_pos[0], pts[:, 1] - cur_pos[1])
                            for k in np.argsort(dd)[:12]:
                                if dd[k] > link_max_px:
                                    break
                                if _segment_free(lev.g, cur_pos, pts[k], thr_link):
                                    seam, linked = int(k), True
                                    break
                        elif (np.hypot(*(pts[0] - cur_pos)) <= link_max_px
                              and _segment_free(lev.g, cur_pos, pts[0], thr_link)):
                            linked = True
                    if not linked:
                        close_chain()
                        if closed:
                            order, est = lev.seam_order(pts)
                            seam = None
                            for k in order[:8]:
                                if not np.isfinite(est[k]):
                                    break
                                if lev.connect(pts[k]) is not None:
                                    seam = int(k)
                                    break
                            if seam is None:
                                n_unreachable += 1
                                continue
                        route = lev.connect(pts[seam] if closed else pts[0])
                        if route is None:
                            n_unreachable += 1
                            continue
                    if closed:
                        ring = np.vstack([np.roll(pts, -seam, axis=0),
                                          np.roll(pts, -seam, axis=0)[:1]])
                    else:
                        ring = pts
                    if linked:
                        cur = np.vstack([cur, ring])
                    else:
                        cur = np.vstack([route[1], ring])
                    cur_pos = cur[-1]
            close_chain()

    if n_unreachable:
        warn(f"{n_unreachable} undercut stretch{'es' if n_unreachable != 1 else ''} "
             f"cannot be reached sideways from open sky at its level and "
             f"{'were' if n_unreachable != 1 else 'was'} skipped.", 90)
    if gap_pix.size and not chains:
        warn("No undercut is reachable with this tool (check the neck "
             "diameter, neck length and shank diameter).", 90)
    if previous_result is None:
        warn("No previous step: Undercut assumes the stock above the overhangs "
             "was already removed by earlier 2.5D steps.", 90)

    # ── Order: undercut regions nearest-first, levels top-down within each ─
    progress(90, "Ordering…")
    if chains:
        lab, _ = label(u_union, structure=np.ones((3, 3), bool))
        for ch in chains:
            ch['region'] = int(lab[ch['rep']])
        by_region = {}
        for i, ch in enumerate(chains):
            by_region.setdefault(ch['region'], []).append(i)
        ordered, pos = [], None
        pending = {r: sorted(ix, key=lambda i: (chains[i]['level'], i))
                   for r, ix in by_region.items()}
        while pending:
            if pos is None:
                region = min(pending)
            else:
                region = min(pending, key=lambda r: np.hypot(
                    *(chains[pending[r][0]]['pts'][0] - pos)))
            for i in pending.pop(region):
                ordered.append(chains[i])
            pos = ordered[-1]['pts'][-1]
        chains = ordered

    rdp_tol = 0.5 / px_per_mm
    segments = []
    for ch in chains:
        rc = ch['pts']
        z = -levels[ch['level']] * max_depth
        xs = rc[:, 1] / px_per_mm - ox
        ys = (H - 1 - rc[:, 0]) / px_per_mm - oy
        pts = [(float(x), float(y), z) for x, y in zip(xs, ys)]
        segments.append(simplify_line(pts, rdp_tol))

    # Material the head sweeps: within its radius of every position where
    # the tool fits under an overhang, on the caller's grid.
    reach_px = max(tool.diameter_mm / 2.0 - extra_offset_mm, 0.0) * px_per_mm
    if u_union.any() and reach_px >= 1.0:
        n = int(reach_px)
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        reach = binary_dilation(u_union, structure=yy ** 2 + xx ** 2 <= reach_px ** 2)
    else:
        reach = u_union.copy()
    if reach.shape != in_shape:
        rows = np.minimum((np.arange(in_shape[0]) + 0.5) * H / in_shape[0], H - 1)
        cols = np.minimum((np.arange(in_shape[1]) + 0.5) * W / in_shape[1], W - 1)
        reach = reach[np.ix_(rows.astype(np.int64), cols.astype(np.int64))]

    result = (previous_result.astype(np.float32).copy() if previous_result is not None
              else np.zeros((H, W), np.float32))
    n_cut_levels = len({ch['level'] for ch in chains})
    progress(100, f"Undercut: {len(segments)} chain{'s' if len(segments) != 1 else ''} "
                  f"on {n_cut_levels} level{'s' if n_cut_levels != 1 else ''}.")
    return ToolPath(
        segments=segments, result=result, accessible=accessible,
        undercut_reach=reach,
        metadata={
            'algorithm':       'Undercut',
            'profile':         tool.profile,
            'max_depth':       max_depth,
            'radius_mm':       tool.diameter_mm / 2,
            'neck_diameter_mm': tool.neck_diameter_mm,
            'z_increment_mm':  z_increment_mm,
            'stepover_pct':    stepover_pct,
            'extra_offset_mm': extra_offset_mm,
            'n_passes':        n_cut_levels,
            'n_unreachable':   n_unreachable,
            'warnings':        warnings,
            'width_mm':        width_mm,
            'height_mm':       height_mm,
            'rdp_tolerance':   rdp_tol,
        })
