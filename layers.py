"""Pure logic for the depth-map layer stack: converting loaded images to
grayscale, resampling a layer onto the shared canvas grid, and compositing
the visible layers into the single flattened depth map that the rest of the
app (``_get_depth_map`` and everything downstream of it) consumes.

Compositing happens in *depth space* (0 = surface, 1 = full depth), not raw
grayscale, so blend modes are unambiguous regardless of the project's
white/black Z=0-colour polarity. Callers are responsible for converting to
and from raw grayscale (see ``to_depth``/``from_depth``).
"""

from dataclasses import dataclass
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

    def is_identity_placement(self) -> bool:
        return (self.offset_x_mm == 0.0 and self.offset_y_mm == 0.0
                and self.rotation_deg == 0.0 and self.scale_pct == 100.0)


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


def _affine_params(layer: Layer, canvas_shape: Tuple[int, int],
                    px_per_mm_x: float, px_per_mm_y: float):
    """(matrix, offset) for scipy.ndimage.affine_transform, which maps
    OUTPUT (canvas) pixel coords to INPUT (layer) pixel coords:

        [r_l, c_l] = matrix @ [r_c, c_c] + offset

    Derivation: canvas and layer each have their own pixel-space center;
    a layer point is scaled by scale_pct, rotated CCW by rotation_deg around
    its own center, then translated by (offset_x_mm, offset_y_mm) — stored in
    mm so placement survives a change of canvas resolution — before landing
    on the canvas relative to the canvas center. Row axis points down, so Y
    (physical, +up) is negated against the row axis throughout.
    """
    import math

    H, W = canvas_shape
    H_l, W_l = layer.raw.shape
    s = max(layer.scale_pct, 1e-6) / 100.0
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
    return matrix, offset


def resample_layer(layer: Layer, canvas_shape: Tuple[int, int],
                    px_per_mm_x: float, px_per_mm_y: float
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

    matrix, offset = _affine_params(layer, canvas_shape, px_per_mm_x, px_per_mm_y)
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
        return base + layer_depth
    if mode == "Subtract":
        return base - layer_depth
    if mode == "Max":
        return np.maximum(base, layer_depth)
    if mode == "Min":
        return np.minimum(base, layer_depth)
    raise ValueError(f"Unknown blend mode: {mode!r}")


def composite(layers: List[Layer], canvas_shape: Tuple[int, int],
              px_per_mm_x: float, px_per_mm_y: float,
              surface: str, background_depth: float = 0.0) -> np.ndarray:
    """Composite the visible layers bottom-to-top into a single float32 raw
    grayscale array (canvas_shape), in the project's own polarity.

    Pure function: canvas geometry is caller-supplied, not derived from
    layers[0], so it can be unit-tested without any app.py state.

    background_depth (canonical [0,1], 0 = no cut) fills any canvas pixel no
    layer covers. The bottom-most layer (index 0) is always treated as
    "Replace" regardless of its own stored blend_mode -- there's nothing
    beneath it but the background, so Subtract/Min there would otherwise
    corrupt the result; this is enforced structurally here rather than by
    mutating the layer's stored data, since layers can be reordered.
    """
    canvas_depth = np.full(canvas_shape, background_depth, dtype=np.float32)
    for i, layer in enumerate(layers):
        if not layer.visible or layer.opacity_pct <= 0.0:
            continue
        resampled, coverage = resample_layer(layer, canvas_shape, px_per_mm_x, px_per_mm_y)
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
