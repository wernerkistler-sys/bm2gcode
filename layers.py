"""Pure logic for the depth-map layer stack: converting loaded images to
grayscale, resampling a layer onto the shared canvas grid, and compositing
the visible layers into the single flattened depth map that the rest of the
app (``_get_depth_map`` and everything downstream of it) consumes.

Compositing happens in *depth space* (0 = surface, 1 = full depth), not raw
grayscale, so blend modes are unambiguous regardless of the project's
white/black Z=0-colour polarity. Callers are responsible for converting to
and from raw grayscale (see ``to_depth``/``from_depth``).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import affine_transform


@dataclass
class Layer:
    name: str
    raw: np.ndarray                 # float32 [0, 1], native resolution — same
                                     # contract as the app's old current_image
    source_kind: str                # "image" | "stl"
    path: Optional[str] = None
    stl_params: Optional[dict] = None   # {'width_px','height_px','rot_x','rot_y','rot_z'}
    layer_id: str = ""               # stable id, assigned by the app for Treeview iids
    alpha: Optional[np.ndarray] = None  # float32 [0,1], same shape as raw, or None
                                         # (fully opaque / no alpha channel in source)
    offset_x_mm: float = 0.0         # +right, relative to canvas center
    offset_y_mm: float = 0.0         # +up, relative to canvas center
    rotation_deg: float = 0.0        # counter-clockwise
    scale_pct: float = 100.0
    blend_mode: str = "Replace"      # see constants.BLEND_MODES
    opacity_pct: float = 100.0
    min_depth_pct: float = 0.0       # depth (% of max_cut_depth) that this
                                      # layer's white pixels are mapped to
    max_depth_pct: float = 100.0     # depth (% of max_cut_depth) that this
                                      # layer's black pixels are mapped to;
                                      # max_depth_pct < min_depth_pct flips
                                      # the layer's polarity
    visible: bool = True
    mirror: str = "None"             # see constants.LAYER_MIRRORS; reflects
                                      # the placed layer across a stock flip
                                      # axis (see _mirror_axis_mm)
    # STL layers only: the mesh's stl_mesh.Dexels on the same grid as raw,
    # built on demand for Undercut steps and never serialised. dexels_key
    # records the (path, stl_params) it was built from, so a re-rasterised
    # layer is detected as stale.
    dexels: Optional[object] = field(default=None, repr=False, compare=False)
    dexels_key: Optional[tuple] = field(default=None, repr=False, compare=False)

    def is_identity_placement(self) -> bool:
        return (self.offset_x_mm == 0.0 and self.offset_y_mm == 0.0
                and self.rotation_deg == 0.0 and self.scale_pct == 100.0
                and self.mirror == "None")


def to_grayscale(raw: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert any loaded image array to (gray, alpha).

    gray  : 2-D float32 [0, 1] grayscale array, straight (not pre-composited
            onto any background) luminance.
    alpha : 2-D float32 [0, 1] array (1 = opaque) if the source had an alpha
            channel, else None. Preserved as genuine per-pixel coverage for
            compositing, rather than flattened onto white and discarded.
    """
    img = raw.astype(np.float32)

    # Normalise to [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = np.clip(img[..., 3], 0.0, 1.0)
        img = img[..., :3]

    # Convert RGB to grayscale (ITU-R BT.601 luminance weights)
    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    return np.clip(img, 0.0, 1.0), alpha


def resize_raw(raw: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """Resample a native-resolution array (grayscale or alpha) to a new pixel
    shape via a uniform, center-anchored stretch -- used when the user picks
    a canvas resolution different from a newly loaded base image's own
    native size. Unlike resample_layer()'s placement transform, every output
    pixel here is a legitimate resample of real content (there's no "off
    canvas" region), so edge handling uses mode='nearest' rather than a
    zero-fill cval.
    """
    H0, W0 = raw.shape
    H1, W1 = new_shape
    if (H0, W0) == (H1, W1):
        return raw
    sy, sx = H0 / H1, W0 / W1
    matrix = np.array([[sy, 0.0], [0.0, sx]])
    offset = np.array([(sy - 1.0) / 2.0, (sx - 1.0) / 2.0])
    return affine_transform(
        raw, matrix, offset=offset, output_shape=(H1, W1),
        order=1, mode="nearest", prefilter=False,
    ).astype(np.float32)


def to_depth(raw: np.ndarray, surface: str, invert: bool) -> np.ndarray:
    """Raw grayscale [0,1] -> depth space [0,1] (0 = surface, 1 = full depth).

    Mirrors the polarity convention already used by
    GCodeGeneratorApp._get_depth_map(): "white" means white pixels are the
    uncut surface, so depth = 1 - raw; "black" means depth = raw. The
    `invert` flag additionally flips the result; composite() always passes
    False, since a layer's own polarity is instead expressed by its
    min_depth_pct/max_depth_pct ordering (see Layer).
    """
    depth = (1.0 - raw) if surface == "white" else raw
    return (1.0 - depth) if invert else depth


def from_depth(depth: np.ndarray, surface: str) -> np.ndarray:
    """Inverse of to_depth() with invert=False — converts a depth-space
    composite back to raw grayscale in the project's own polarity, so it can
    be stored where the rest of the app expects a raw grayscale array."""
    return (1.0 - depth) if surface == "white" else depth


def _mirror_axis_mm(mirror: str, W_mm: float, H_mm: float, origin: str):
    """(point, unit direction) of a Mirror option's reflection axis, in stock
    mm (bottom-left corner = 0,0, y up) -- the same flip axes as the
    Registration Holes step (algorithms.registration_flip_axis): the stock's
    vertical/horizontal centre line, or the 45° line through the G-code
    origin. None for "None"."""
    if mirror == "Vertical axis":
        return (W_mm / 2.0, 0.0), (0.0, 1.0)
    if mirror == "Horizontal axis":
        return (0.0, H_mm / 2.0), (1.0, 0.0)
    if mirror in ("Diagonal (+45°)", "Diagonal (−45°)"):
        point = {"Bottom-Left": (0.0, 0.0),  "Bottom-Right": (W_mm, 0.0),
                 "Top-Left":    (0.0, H_mm), "Top-Right":    (W_mm, H_mm)
                 }.get(origin, (W_mm / 2.0, H_mm / 2.0))
        s = 1.0 / np.sqrt(2.0)
        return point, ((s, s) if mirror == "Diagonal (+45°)" else (s, -s))
    return None


def _mirror_params(mirror: str, canvas_shape: Tuple[int, int],
                   px_per_mm_x: float, px_per_mm_y: float, origin: str):
    """(R, t) of the canvas-pixel reflection q -> R @ q + t, q = [row, col],
    for a Mirror option, or None for "None".

    The reflection is built in stock mm -- where the flip axes are defined
    -- and conjugated into pixel space, so a 45° axis stays a true 45° line
    even when pixels aren't square. Stock mm spans pixel centres
    (x = col / px_per_mm_x, y = (H - 1 - row) / px_per_mm_y), the same
    convention the toolpath algorithms use, so a mirrored layer lines up
    with toolpaths flipped across the same axis.
    """
    H, W = canvas_shape
    axis = _mirror_axis_mm(mirror, (W - 1) / px_per_mm_x, (H - 1) / px_per_mm_y, origin)
    if axis is None:
        return None
    (px, py), (dx, dy) = axis
    A = np.array([[2 * dx * dx - 1, 2 * dx * dy],
                  [2 * dx * dy,     2 * dy * dy - 1]])
    P = np.array([px, py])
    # [x, y] = S @ [row, col] + b
    S = np.array([[0.0, 1.0 / px_per_mm_x], [-1.0 / px_per_mm_y, 0.0]])
    b = np.array([0.0, (H - 1) / px_per_mm_y])
    S_inv = np.linalg.inv(S)
    # Rounded so an exact flip stays exact: a stray 1e-16 term would put the
    # canvas's edge pixels a hair outside the layer, where affine_transform
    # samples its zero fill instead of the edge.
    return (np.round(S_inv @ A @ S, 12),
            np.round(S_inv @ (P + A @ (b - P) - b), 9))


def _affine_params(layer: Layer, canvas_shape: Tuple[int, int],
                    px_per_mm_x: float, px_per_mm_y: float,
                    origin: str = "Bottom-Left"):
    """(matrix, offset) for scipy.ndimage.affine_transform, which maps
    OUTPUT (canvas) pixel coords to INPUT (layer) pixel coords:

        [r_l, c_l] = matrix @ [r_c, c_c] + offset

    Derivation: canvas and layer each have their own pixel-space center;
    a layer point is scaled by scale_pct, rotated CCW by rotation_deg around
    its own center, then translated by (offset_x_mm, offset_y_mm) — stored in
    mm so placement survives a change of canvas resolution — before landing
    on the canvas relative to the canvas center. Row axis points down, so Y
    (physical, +up) is negated against the row axis throughout.

    scale_pct is canvas-size-relative, not pixel-for-pixel: at 100%, the
    layer's own aspect ratio is inscribed in the canvas rectangle
    (aspect-preserving "contain"), independent of the layer's native pixel
    count or the canvas's pixel density -- so a source's on-stock physical
    size only depends on scale_pct and the canvas's physical Width/Height,
    never on canvas resolution. fit_ratio (canvas px per layer-native px at
    scale_pct=100, along whichever axis the layer's aspect ratio binds)
    converts that into the pixel-space factor the matrix below needs.

    mirror, if set, reflects the placed layer across a stock flip axis (see
    _mirror_params) -- applied after placement, so an offset layer's offset
    is mirrored along with its content. *origin* (the G-code origin) anchors
    the diagonal axes.
    """
    import math

    H, W = canvas_shape
    H_l, W_l = layer.raw.shape
    fit_ratio = min(W / W_l, H / H_l)
    s = fit_ratio * max(layer.scale_pct, 1e-6) / 100.0
    theta = math.radians(layer.rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    cy_l, cx_l = (H_l - 1) / 2.0, (W_l - 1) / 2.0
    cy_c, cx_c = (H - 1) / 2.0, (W - 1) / 2.0
    ox = layer.offset_x_mm * px_per_mm_x
    oy = layer.offset_y_mm * px_per_mm_y

    matrix = np.array([[ cos_t / s,  sin_t / s],
                        [-sin_t / s,  cos_t / s]])

    k1 = cx_c + ox
    k2 = cy_c - oy
    offset = np.array([
        cy_l - (sin_t / s) * k1 - (cos_t / s) * k2,
        cx_l - (cos_t / s) * k1 + (sin_t / s) * k2,
    ])
    mirror = _mirror_params(layer.mirror, canvas_shape, px_per_mm_x, px_per_mm_y, origin)
    if mirror is not None:
        R, t = mirror
        matrix, offset = matrix @ R, matrix @ t + offset
    return matrix, offset


def resample_layer(layer: Layer, canvas_shape: Tuple[int, int],
                    px_per_mm_x: float, px_per_mm_y: float,
                    origin: str = "Bottom-Left"
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (resampled_raw, coverage), both float32 and canvas_shape.

    coverage is 1.0 where the canvas pixel falls inside the layer's own
    footprint AND the layer's own alpha channel (if any) is opaque there,
    0.0 where either is absent (fractional near an anti-aliased edge).
    """
    if layer.is_identity_placement() and layer.raw.shape == canvas_shape:
        coverage = (layer.alpha if layer.alpha is not None
                    else np.ones(canvas_shape, dtype=np.float32))
        return layer.raw, coverage

    matrix, offset = _affine_params(layer, canvas_shape, px_per_mm_x, px_per_mm_y, origin)
    resampled = affine_transform(
        layer.raw, matrix, offset=offset, output_shape=canvas_shape,
        order=1, cval=0.0, prefilter=False,
    ).astype(np.float32)
    coverage = affine_transform(
        np.ones(layer.raw.shape, dtype=np.float32), matrix, offset=offset,
        output_shape=canvas_shape, order=1, cval=0.0, prefilter=False,
    ).astype(np.float32)
    if layer.alpha is not None:
        resampled_alpha = affine_transform(
            layer.alpha, matrix, offset=offset, output_shape=canvas_shape,
            order=1, cval=0.0, prefilter=False,
        ).astype(np.float32)
        coverage = coverage * resampled_alpha
    return resampled, coverage


def _blend_op(mode: str, base: np.ndarray, layer_depth: np.ndarray) -> np.ndarray:
    if mode == "Replace":
        return layer_depth
    if mode == "Add":
        return base - layer_depth
    if mode == "Subtract":
        return base + layer_depth
    if mode == "Merge low":
        return np.maximum(base, layer_depth)
    if mode == "Merge high":
        return np.minimum(base, layer_depth)
    raise ValueError(f"Unknown blend mode: {mode!r}")


def composite(layers: List[Layer], canvas_shape: Tuple[int, int],
              px_per_mm_x: float, px_per_mm_y: float,
              surface: str, background_depth: float = 0.0,
              visibility_override: Optional[List[bool]] = None,
              origin: str = "Bottom-Left") -> np.ndarray:
    """Composite the visible layers bottom-to-top into a single float32 raw
    grayscale array (canvas_shape), in the project's own polarity.

    Pure function: canvas geometry is caller-supplied, not derived from
    layers[0], so it can be unit-tested without any app.py state.

    background_depth (canonical [0,1], 0 = no cut) fills any canvas pixel no
    layer covers. The bottom-most layer (index 0) is always treated as
    "Replace" regardless of its own stored blend_mode -- there's nothing
    beneath it but the background, so Add/Merge high there would otherwise
    corrupt the result; this is enforced structurally here rather than by
    mutating the layer's stored data, since layers can be reordered.

    visibility_override, if given, is a list index-aligned with *layers*
    (visibility_override[i] gates layers[i]) and takes precedence over that
    layer's own .visible -- lets a caller (e.g. a specific carving step's
    own stored per-source visibility) gate the composite without mutating
    any Layer's shared .visible state. Missing/short lists fall back to
    the layer's own .visible for the uncovered tail.

    origin is the G-code origin, which anchors the diagonal Mirror axes.
    """
    canvas_depth = np.full(canvas_shape, background_depth, dtype=np.float32)
    for i, layer in enumerate(layers):
        if visibility_override is not None and i < len(visibility_override):
            visible = visibility_override[i]
        else:
            visible = layer.visible
        if not visible or layer.opacity_pct <= 0.0:
            continue
        resampled, coverage = resample_layer(layer, canvas_shape, px_per_mm_x, px_per_mm_y,
                                             origin)
        depth = to_depth(resampled, surface, False)
        min_frac, max_frac = layer.min_depth_pct / 100.0, layer.max_depth_pct / 100.0
        layer_depth = np.clip(min_frac + (max_frac - min_frac) * depth, 0.0, 1.0)
        alpha = coverage * (layer.opacity_pct / 100.0)
        mode = "Replace" if i == 0 else layer.blend_mode
        blended = np.clip(_blend_op(mode, canvas_depth, layer_depth), 0.0, 1.0)
        canvas_depth = canvas_depth * (1.0 - alpha) + blended * alpha
    return from_depth(canvas_depth, surface).astype(np.float32)


def compute_n_grey_levels(layers: List[Layer], composite_raw: np.ndarray) -> int:
    """Strict generalisation of the app's existing single-image heuristic,
    now run on the composite; forced to 65536 (STL-derived, effectively
    continuous) if any layer in the stack is STL-sourced."""
    if any(layer.source_kind == "stl" for layer in layers):
        return 65536
    if composite_raw is None or composite_raw.size == 0:
        return 256
    levels = int(round((float(composite_raw.max()) - float(composite_raw.min())) * 255)) + 1
    return max(levels, 1)


# ── Undercut gaps ──────────────────────────────────────────────────────────

@dataclass
class Gaps:
    """Air gaps below the top surface of each canvas column, in canonical
    depth (0 = surface, 1 = full depth, larger = deeper).

    Column i (row-major) holds gaps offsets[i]:offsets[i+1], shallowest
    first: air from ceil (the underside of the solid above) down to floor
    (the top of the solid below). Columns without gaps are solid from their
    top surface down.
    """
    shape: tuple
    offsets: np.ndarray       # int64, H*W + 1
    ceil: np.ndarray          # float64
    floor: np.ndarray         # float64

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets).reshape(self.shape)

    @classmethod
    def empty(cls, shape) -> "Gaps":
        H, W = shape
        return cls((H, W), np.zeros(H * W + 1, np.int64),
                   np.zeros(0, np.float64), np.zeros(0, np.float64))

    @classmethod
    def from_lists(cls, shape, pix, ceil, floor) -> "Gaps":
        """Build from flat per-gap arrays (pixel index, ceil, floor) in any
        order; gaps of one column are sorted shallowest first."""
        H, W = shape
        pix = np.asarray(pix, np.int64)
        ceil = np.asarray(ceil, np.float64)
        floor = np.asarray(floor, np.float64)
        order = np.lexsort((ceil, pix))
        pix, ceil, floor = pix[order], ceil[order], floor[order]
        offsets = np.zeros(H * W + 1, np.int64)
        np.cumsum(np.bincount(pix, minlength=H * W), out=offsets[1:])
        return cls((H, W), offsets, ceil, floor)

    def pixel_index(self) -> np.ndarray:
        """Pixel index of every gap, aligned with ceil/floor."""
        return np.repeat(np.arange(self.offsets.size - 1), np.diff(self.offsets))


def overhang_masks(gaps: Gaps, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """(closed, open_bottom) pixel masks of *gaps*.

    closed: pixels with a gap bounded below by solid above full depth --
    air enclosed by the source, reachable only by an undercut. open_bottom:
    pixels whose gaps all reach full depth (the stock's bottom face), such
    as under an object floating above it -- reachable from the back once the
    stock is turned over.
    """
    H, W = gaps.shape
    pix = gaps.pixel_index()
    is_closed = gaps.floor < 1.0 - tol
    closed = np.bincount(pix[is_closed], minlength=H * W).reshape(H, W) > 0
    any_gap = gaps.counts > 0
    return closed, any_gap & ~closed


def pad_gaps(gaps: Gaps, pad_y: int, pad_x: int) -> Gaps:
    """*gaps* moved onto a grid padded by pad_y rows and pad_x columns on
    every side (the stock margin), with no gaps in the padding."""
    if pad_y <= 0 and pad_x <= 0:
        return gaps
    H, W = gaps.shape
    pix = gaps.pixel_index()
    rows, cols = pix // W, pix % W
    H2, W2 = H + 2 * pad_y, W + 2 * pad_x
    return Gaps.from_lists((H2, W2), (rows + pad_y) * W2 + cols + pad_x,
                           gaps.ceil, gaps.floor)


def _layer_depth_affine(layer: Layer, surface: str):
    """(a, b) with layer depth = a * z + b for the layer's mesh z, following
    the same chain composite() applies to the layer's raw heightmap:
    normalise by the mesh's z bounds (normalize_and_orient; the top
    surface's range for dexels without them), to_depth, then the min/max
    depth % mapping. None if that range is flat."""
    if layer.dexels.z_bounds is not None:
        z_min, z_max = layer.dexels.z_bounds
    else:
        tops = layer.dexels.top
        if not np.isfinite(tops).any():
            return None
        z_min, z_max = float(np.nanmin(tops)), float(np.nanmax(tops))
    if z_max - z_min < 1e-12:
        return None
    # raw = (z - z_min) / range;  depth = 1 - raw (white) or raw (black)
    a, b = 1.0 / (z_max - z_min), -z_min / (z_max - z_min)
    if surface == "white":
        a, b = -a, 1.0 - b
    min_frac, max_frac = layer.min_depth_pct / 100.0, layer.max_depth_pct / 100.0
    return (max_frac - min_frac) * a, min_frac + (max_frac - min_frac) * b


def composite_gaps(layers: List[Layer], canvas_shape: Tuple[int, int],
                   px_per_mm_x: float, px_per_mm_y: float, surface: str,
                   composite_depth: np.ndarray,
                   visibility_override: Optional[List[bool]] = None,
                   origin: str = "Bottom-Left",
                   tol: float = 1e-4,
                   background_depth: float = 0.0) -> Tuple[Gaps, List[str]]:
    """The undercut gaps of the visible STL layers on the canvas grid.

    composite_depth is composite()'s result in canonical depth (to_depth of
    its raw output), and background_depth the one it was built with. A
    layer's gaps are adopted at a canvas pixel only where that layer
    produces the composite's top surface there, so the composite stays
    authoritative for the top surface whatever the blend modes; where
    several layers qualify, the topmost wins. At full coverage (and full
    opacity) that is where its own placed depth matches composite_depth
    within *tol*. At partial coverage c, as along the layer's outline when
    it is resampled, it is where composite_depth is c times its depth plus
    1 - c times the composite of the layers beneath it (the background
    alone for the bottom layer): the layer replacing what is beneath over
    part of the pixel. What is beneath is taken as solid from its top down,
    so there the layer's air (above its top and in its gaps) only counts
    between composite_depth and that top.

    Gaps are read from the nearest layer column (placement, rotation, scale
    and Mirror as in resample_layer; none outside the layer's raster),
    clipped at full depth, and dropped when they start within *tol* of it
    or below it -- a bottom face resting on full depth maps to it only up
    to rounding, and must not leave a zero-height gap there.

    Layers without dexels are ignored. Returns (gaps, warnings).
    """
    H, W = canvas_shape
    warnings = []
    owner = np.full(H * W, -1, np.int64)        # source column per canvas pixel
    owner_layer = np.full(H * W, -1, np.int64)
    owner_cap = np.full(H * W, np.inf)          # top of the solid beneath, partial coverage
    flat_depth = composite_depth.ravel()
    affines = {}
    for i, layer in enumerate(layers):
        if visibility_override is not None and i < len(visibility_override):
            visible = visibility_override[i]
        else:
            visible = layer.visible
        if (not visible or layer.opacity_pct <= 0.0 or layer.source_kind != "stl"
                or layer.dexels is None):
            continue
        if layer.dexels.shape != layer.raw.shape:
            warnings.append(f"{layer.name}: undercut data is out of date; ignored.")
            continue
        ab = _layer_depth_affine(layer, surface)
        if ab is None:
            continue
        if ab[0] >= 0.0:
            warnings.append(f"{layer.name}: inverted depth polarity -- its "
                            f"undercuts are ignored.")
            continue
        affines[i] = ab
        if layer.opacity_pct < 100.0:
            continue

        resampled, coverage = resample_layer(layer, canvas_shape, px_per_mm_x,
                                             px_per_mm_y, origin)
        min_frac, max_frac = layer.min_depth_pct / 100.0, layer.max_depth_pct / 100.0
        depth = np.clip(min_frac + (max_frac - min_frac)
                        * to_depth(resampled, surface, False), 0.0, 1.0)
        wins = (coverage >= 0.999) & (np.abs(depth - composite_depth) <= tol)
        partial = (coverage > tol) & (coverage < 1.0) & ~wins
        below = None
        if partial.any():
            below_vis = (list(visibility_override[:i]) if visibility_override is not None
                         else None)
            below = to_depth(composite(layers[:i], canvas_shape, px_per_mm_x, px_per_mm_y,
                                       surface, background_depth, below_vis, origin),
                             surface, False)
            partial &= (np.abs(coverage * depth + (1.0 - coverage) * below
                               - composite_depth) <= tol)
        wins = (wins | partial).ravel()
        if not wins.any():
            continue

        H_l, W_l = layer.raw.shape
        if layer.is_identity_placement() and (H_l, W_l) == (H, W):
            src = np.arange(H * W)
            ok = np.ones(H * W, bool)
        else:
            matrix, offset = _affine_params(layer, canvas_shape, px_per_mm_x,
                                            px_per_mm_y, origin)
            rr, cc = np.mgrid[0:H, 0:W]
            pts = np.stack([rr.ravel(), cc.ravel()]).astype(np.float64)
            src_rc = matrix @ pts + offset[:, None]
            r_l = np.rint(src_rc[0]).astype(np.int64)
            c_l = np.rint(src_rc[1]).astype(np.int64)
            ok = (r_l >= 0) & (r_l < H_l) & (c_l >= 0) & (c_l < W_l)
            src = np.where(ok, r_l * W_l + c_l, 0)
        partial = partial.ravel()
        sel = wins & (ok | partial)
        owner[sel] = np.where(ok[sel], src[sel], -1)
        owner_layer[sel] = i
        owner_cap[sel] = np.inf
        if below is not None:
            p = sel & partial
            owner_cap[p] = below.ravel()[p]

    pix_out, ceil_out, floor_out = [], [], []
    for i, (a, b) in affines.items():
        mine = owner_layer == i
        dex = layers[i].dexels

        # Full coverage: the column's gaps below its top.
        canvas_pix = np.flatnonzero(mine & np.isinf(owner_cap))
        src = owner[canvas_pix]
        start = dex.offsets[src]
        n_hits = dex.offsets[src + 1] - start
        n_gaps = n_hits // 2            # every exit hit opens a gap
        total = int(n_gaps.sum())
        if total:
            col = np.repeat(np.arange(canvas_pix.size), n_gaps)
            m = np.arange(total) - np.repeat(np.cumsum(n_gaps) - n_gaps, n_gaps)
            col_start = start[col]
            exit_idx = col_start + 2 * m + 1
            enter_pos = 2 * m + 2
            has_floor = enter_pos < n_hits[col]
            ceil = a * dex.z[exit_idx] + b
            floor = np.full(total, np.inf)
            floor[has_floor] = a * dex.z[col_start[has_floor] + enter_pos[has_floor]] + b
            keep = ceil < 1.0 - tol
            pix_out.append(canvas_pix[col[keep]])
            ceil_out.append(np.maximum(ceil[keep], 0.0))
            floor_out.append(np.minimum(floor[keep], 1.0))

        # Partial coverage: the column's air -- above its top (all of it for
        # a column without hits) and in its gaps -- between the composite's
        # top and the solid beneath.
        canvas_pix = np.flatnonzero(mine & np.isfinite(owner_cap))
        if canvas_pix.size == 0:
            continue
        src = owner[canvas_pix]
        has_src = src >= 0
        start = np.where(has_src, dex.offsets[np.maximum(src, 0)], 0)
        n_hits = np.where(has_src, dex.offsets[np.maximum(src, 0) + 1] - start, 0)
        n_air = n_hits // 2 + 1
        col = np.repeat(np.arange(canvas_pix.size), n_air)
        m = np.arange(col.size) - np.repeat(np.cumsum(n_air) - n_air, n_air)
        col_start, col_hits = start[col], n_hits[col]
        ceil = np.full(col.size, -np.inf)
        above = m > 0
        ceil[above] = a * dex.z[col_start[above] + 2 * m[above] - 1] + b
        floor = np.full(col.size, np.inf)
        has_floor = 2 * m < col_hits
        floor[has_floor] = a * dex.z[col_start[has_floor] + 2 * m[has_floor]] + b
        ceil = np.maximum(ceil, flat_depth[canvas_pix[col]])
        floor = np.minimum(np.minimum(floor, owner_cap[canvas_pix[col]]), 1.0)
        keep = (floor > ceil) & (ceil < 1.0 - tol)
        pix_out.append(canvas_pix[col[keep]])
        ceil_out.append(np.maximum(ceil[keep], 0.0))
        floor_out.append(floor[keep])

    if not pix_out:
        return Gaps.empty((H, W)), warnings
    return (Gaps.from_lists((H, W), np.concatenate(pix_out),
                            np.concatenate(ceil_out), np.concatenate(floor_out)),
            warnings)
