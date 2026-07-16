"""Plain assert-based tests for the layer-compositing helpers (layers.py).

Run with: python tests/test_layers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from layers import (Layer, to_depth, from_depth, to_grayscale, resize_raw,
                    resample_layer, composite, compute_n_grey_levels)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


def make_layer(raw, **kw):
    kw.setdefault('name', 'test')
    kw.setdefault('source_kind', 'image')
    kw.setdefault('path', None)
    return Layer(raw=raw, **kw)


# --------------------------------------------------------------------------
# to_depth / from_depth polarity round-trip
# --------------------------------------------------------------------------
print("to_depth / from_depth")

raw = np.array([[0.0, 0.25, 1.0]], dtype=np.float32)
check("white polarity: depth is 1-raw",
      np.allclose(to_depth(raw, "white", False), 1.0 - raw))
check("black polarity: depth is raw",
      np.allclose(to_depth(raw, "black", False), raw))
check("invert flips depth regardless of polarity",
      np.allclose(to_depth(raw, "white", True), raw))
for surface in ("white", "black"):
    depth = to_depth(raw, surface, False)
    check(f"from_depth inverts to_depth exactly ({surface})",
          np.array_equal(from_depth(depth, surface), raw))


# --------------------------------------------------------------------------
# composite: single default layer is a bit-exact no-op
# --------------------------------------------------------------------------
print("composite: identity / no-op")

img = np.array([[0.0, 0.5, 1.0], [0.2, 0.8, 0.3]], dtype=np.float32)
# composite() itself only needs to reproduce the input to float32 precision --
# a "white" polarity round-trips raw through 1-(1-x), which is not guaranteed
# bit-exact for every float32 value. True bit-exactness for the common
# single-layer case is the job of app.py's _recompute_composite() outer
# shortcut (which bypasses composite() entirely), not composite() itself.
for surface in ("white", "black"):
    layer = make_layer(img.copy())
    out = composite([layer], img.shape, 1.0, 1.0, surface)
    check(f"single default layer reproduces input to float precision ({surface})",
          np.allclose(out, img, atol=1e-6))
    check("composite() returns float32", out.dtype == np.float32)
check("black polarity (raw==depth, no round-trip) is bit-exact",
      np.array_equal(composite([make_layer(img.copy())], img.shape, 1.0, 1.0, "black"), img))


# --------------------------------------------------------------------------
# blend modes
# --------------------------------------------------------------------------
print("blend modes")

# Black polarity makes raw == depth directly (to_depth/from_depth are the
# identity for surface="black", invert=False), so expected values can be
# hand-computed and compared bit-exactly without any polarity conversion.
SURFACE = "black"

base_add = make_layer(np.array([[0.3, 0.9]], dtype=np.float32))
top_add  = make_layer(np.array([[0.4, 0.4]], dtype=np.float32), blend_mode="Add")
out = composite([base_add, top_add], (1, 2), 1.0, 1.0, SURFACE)
check("Add sums depths and clips at full depth",
      np.allclose(out, [[0.7, 1.0]]))

base_sub = make_layer(np.array([[0.9, 0.3]], dtype=np.float32))
top_sub  = make_layer(np.array([[0.4, 0.5]], dtype=np.float32), blend_mode="Subtract")
out = composite([base_sub, top_sub], (1, 2), 1.0, 1.0, SURFACE)
check("Subtract subtracts depths and clips at zero",
      np.allclose(out, [[0.5, 0.0]]))

base_mm = make_layer(np.array([[0.5, 0.5]], dtype=np.float32))
top_max = make_layer(np.array([[0.3, 0.7]], dtype=np.float32), blend_mode="Max")
out = composite([base_mm, top_max], (1, 2), 1.0, 1.0, SURFACE)
check("Max keeps the deeper of the two", np.allclose(out, [[0.5, 0.7]]))

top_min = make_layer(np.array([[0.3, 0.7]], dtype=np.float32), blend_mode="Min")
out = composite([base_mm, top_min], (1, 2), 1.0, 1.0, SURFACE)
check("Min keeps the shallower of the two", np.allclose(out, [[0.3, 0.5]]))

base_normal = make_layer(np.array([[1.0, 1.0]], dtype=np.float32))
top_normal  = make_layer(np.array([[0.9, 0.1]], dtype=np.float32), blend_mode="Replace")
out = composite([base_normal, top_normal], (1, 2), 1.0, 1.0, SURFACE)
check("Replace fully replaces the base wherever it has coverage",
      np.array_equal(out, top_normal.raw))


# --------------------------------------------------------------------------
# invisible layer contributes nothing, even for Subtract/Min against a
# zero-depth base -- this is the failure mode a missing "Replace" default
# blend mode would otherwise hit for a lone visible layer.
# --------------------------------------------------------------------------
print("visibility")

zero_base = make_layer(np.ones((2, 2), dtype=np.float32))   # white polarity -> depth 0 everywhere
hidden_sub = make_layer(np.zeros((2, 2), dtype=np.float32), blend_mode="Subtract", visible=False)
out = composite([zero_base, hidden_sub], (2, 2), 1.0, 1.0, "white")
check("an invisible layer (even Subtract) leaves the composite unchanged",
      np.array_equal(out, zero_base.raw))

hidden_min = make_layer(np.zeros((2, 2), dtype=np.float32), blend_mode="Min", visible=False)
out = composite([zero_base, hidden_min], (2, 2), 1.0, 1.0, "white")
check("an invisible Min layer leaves the composite unchanged",
      np.array_equal(out, zero_base.raw))


# --------------------------------------------------------------------------
# opacity / coverage act as a multiplicative alpha
# --------------------------------------------------------------------------
print("opacity")

base = make_layer(np.zeros((1, 2), dtype=np.float32))          # depth 1.0 (black polarity)
overlay = make_layer(np.ones((1, 2), dtype=np.float32), opacity_pct=50.0)  # depth 1.0
out = composite([base, overlay], (1, 2), 1.0, 1.0, "black")
check("50% opacity halves the contribution", np.allclose(out[0], [0.5, 0.5]))


# --------------------------------------------------------------------------
# resample_layer: fast path and off-canvas transform
# --------------------------------------------------------------------------
print("resample_layer")

identity_layer = make_layer(img.copy())
resampled, coverage = resample_layer(identity_layer, img.shape, 1.0, 1.0)
check("identity-placement fast path returns the same array object",
      resampled is identity_layer.raw)
check("identity-placement fast path has full coverage", np.all(coverage == 1.0))

marker = np.zeros((10, 10), dtype=np.float32)
marker[3, 4] = 1.0   # row 3, col 4
marker_layer = make_layer(marker, offset_x_mm=2.0, offset_y_mm=-1.0)
resampled, coverage = resample_layer(marker_layer, (10, 10), 1.0, 1.0)
# +x offset shifts content toward larger columns; +y offset in the layer's own
# convention is "up" (physical), i.e. toward smaller row indices; here offset_y
# is negative ("down"), so the marker should move toward larger row indices too.
ry, rx = np.unravel_index(np.argmax(resampled), resampled.shape)
check("translated marker lands at the expected column (+offset_x -> larger col)",
      rx == 4 + 2)
check("translated marker lands at the expected row (-offset_y -> larger row)",
      ry == 3 + 1)

far_layer = make_layer(np.ones((4, 4), dtype=np.float32), offset_x_mm=10_000.0)
_, coverage = resample_layer(far_layer, (4, 4), 1.0, 1.0)
check("a layer transformed far off-canvas has zero coverage", np.all(coverage == 0.0))
out = composite([make_layer(np.full((4, 4), 0.5, dtype=np.float32)), far_layer],
                (4, 4), 1.0, 1.0, "white")
check("an off-canvas layer leaves the composite below it unchanged",
      np.allclose(out, 0.5))


# --------------------------------------------------------------------------
# compute_n_grey_levels
# --------------------------------------------------------------------------
print("compute_n_grey_levels")

img8 = np.array([[0.0, 1.0]], dtype=np.float32)
check("image-only stack uses the (max-min)*255+1 heuristic",
      compute_n_grey_levels([make_layer(img8, source_kind="image")], img8) == 256)
check("any STL layer forces 65536",
      compute_n_grey_levels(
          [make_layer(img8, source_kind="image"), make_layer(img8, source_kind="stl")],
          img8) == 65536)


# --------------------------------------------------------------------------
# to_grayscale: alpha preserved as genuine coverage, not composited onto white
# --------------------------------------------------------------------------
print("to_grayscale")

rgba = np.zeros((1, 1, 4), dtype=np.float32)
rgba[0, 0] = [0.0, 0.0, 0.0, 0.0]   # black ink, fully transparent
gray, alpha = to_grayscale(rgba)
check("RGBA with alpha=0 is a straight (non-white-composited) conversion",
      gray[0, 0] < 0.01)
check("alpha channel is preserved (not None) for RGBA input", alpha is not None)
check("alpha value matches the source alpha", np.isclose(alpha[0, 0], 0.0))

rgb = np.zeros((1, 1, 3), dtype=np.float32)
_, alpha2 = to_grayscale(rgb)
check("RGB (no alpha channel) input returns alpha=None", alpha2 is None)


# --------------------------------------------------------------------------
# resize_raw
# --------------------------------------------------------------------------
print("resize_raw")

small = (np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0)
big = resize_raw(small, (8, 10))
check("resize_raw produces the exact requested output shape (upsize)", big.shape == (8, 10))
smaller = resize_raw(small, (2, 2))
check("resize_raw produces the exact requested output shape (downsize)", smaller.shape == (2, 2))
same = resize_raw(small, (4, 4))
check("resize_raw with an unchanged shape returns the content unmodified",
      np.array_equal(same, small))


# --------------------------------------------------------------------------
# min_depth_pct / max_depth_pct
# --------------------------------------------------------------------------
print("min_depth_pct / max_depth_pct")

layer_full = make_layer(np.array([[0.6, 0.8]], dtype=np.float32))
layer_half = make_layer(np.array([[0.6, 0.8]], dtype=np.float32), max_depth_pct=50.0)
out_full = composite([layer_full], (1, 2), 1.0, 1.0, "black")
out_half = composite([layer_half], (1, 2), 1.0, 1.0, "black")
check("halving max_depth_pct halves the composited depth contribution",
      np.allclose(out_half, out_full * 0.5))

layer_swapped = make_layer(np.array([[0.0, 1.0]], dtype=np.float32),
                           min_depth_pct=100.0, max_depth_pct=0.0)
out_swapped = composite([layer_swapped], (1, 2), 1.0, 1.0, "black")
check("max_depth_pct < min_depth_pct flips the layer's polarity",
      np.allclose(out_swapped, [[1.0, 0.0]]))


# --------------------------------------------------------------------------
# base layer (index 0) is always forced to Replace, regardless of its own
# stored blend_mode -- the direct regression test for the original
# Subtract/Min-against-an-empty-canvas footgun, now fixed structurally.
# --------------------------------------------------------------------------
print("base layer forced to Replace")

bg = 0.4
base_vals = np.array([[0.9, 0.9]], dtype=np.float32)
out_subtract_at_base = composite([make_layer(base_vals.copy(), blend_mode="Subtract")],
                                 (1, 2), 1.0, 1.0, "black", background_depth=bg)
out_min_at_base = composite([make_layer(base_vals.copy(), blend_mode="Min")],
                            (1, 2), 1.0, 1.0, "black", background_depth=bg)
out_replace_at_base = composite([make_layer(base_vals.copy(), blend_mode="Replace")],
                                (1, 2), 1.0, 1.0, "black", background_depth=bg)
check("a stored Subtract blend_mode at index 0 is structurally ignored (forced to Replace)",
      np.array_equal(out_subtract_at_base, out_replace_at_base))
check("a stored Min blend_mode at index 0 is structurally ignored (forced to Replace)",
      np.array_equal(out_min_at_base, out_replace_at_base))
check("...and does not actually subtract from the background",
      not np.allclose(out_subtract_at_base, np.clip(bg - base_vals, 0.0, 1.0)))


# --------------------------------------------------------------------------
# background_depth fills canvas area no layer covers
# --------------------------------------------------------------------------
print("background_depth")

offcanvas_layer = make_layer(np.full((2, 2), 0.9, dtype=np.float32), offset_x_mm=1000.0)
out = composite([offcanvas_layer], (2, 2), 1.0, 1.0, "black", background_depth=0.3)
check("background_depth fills canvas area no layer covers",
      np.allclose(out, 0.3))


# --------------------------------------------------------------------------
# alpha-driven coverage: the case Add/Max can already fake via zero-valued
# content but Replace and Min structurally cannot.
# --------------------------------------------------------------------------
print("alpha-driven coverage")

canvas_shape = (3, 3)
base_val = 0.5
base_layer = make_layer(np.full(canvas_shape, base_val, dtype=np.float32))  # index 0

alpha_mask = np.zeros(canvas_shape, dtype=np.float32)
alpha_mask[1, 1] = 1.0   # opaque only at the center pixel

replace_layer = make_layer(np.full(canvas_shape, 0.9, dtype=np.float32),
                           alpha=alpha_mask.copy(), blend_mode="Replace")
out = composite([base_layer, replace_layer], canvas_shape, 1.0, 1.0, "black")
expected = np.full(canvas_shape, base_val, dtype=np.float32)
expected[1, 1] = 0.9
check("Replace with alpha coverage only overrides the opaque pixel (identity-placement path)",
      np.allclose(out, expected))

min_layer = make_layer(np.zeros(canvas_shape, dtype=np.float32),
                       alpha=alpha_mask.copy(), blend_mode="Min")
out2 = composite([base_layer, min_layer], canvas_shape, 1.0, 1.0, "black")
expected2 = np.full(canvas_shape, base_val, dtype=np.float32)
expected2[1, 1] = 0.0
check("Min with alpha coverage only affects the opaque pixel, not the rest (identity-placement path)",
      np.allclose(out2, expected2))

# General (non-identity) affine path: an offset placement must still fold
# the layer's own alpha channel into the returned coverage.
alpha_partial = np.ones((4, 4), dtype=np.float32)
alpha_partial[0, 0] = 0.0
offset_layer = make_layer(np.full((4, 4), 0.9, dtype=np.float32),
                          alpha=alpha_partial, offset_x_mm=1.0)
resampled, coverage = resample_layer(offset_layer, (4, 4), 1.0, 1.0)
check("a non-identity (offset) placement still incorporates the layer's own alpha into coverage",
      coverage.min() < 1.0)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
