"""Plain assert-based tests for the layer-compositing helpers (layers.py).

Run with: python tests/test_layers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from layers import (Layer, to_depth, from_depth, to_grayscale, resize_raw,
                    resample_layer, composite, compute_n_grey_levels,
                    _mirror_axis_mm, Gaps, overhang_masks)
from algorithms import registration_flip_axis

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

base_sub = make_layer(np.array([[0.3, 0.9]], dtype=np.float32))
top_sub  = make_layer(np.array([[0.4, 0.4]], dtype=np.float32), blend_mode="Subtract")
out = composite([base_sub, top_sub], (1, 2), 1.0, 1.0, SURFACE)
check("Subtract sums depths (deepens) and clips at full depth",
      np.allclose(out, [[0.7, 1.0]]))

base_add = make_layer(np.array([[0.9, 0.3]], dtype=np.float32))
top_add  = make_layer(np.array([[0.4, 0.5]], dtype=np.float32), blend_mode="Add")
out = composite([base_add, top_add], (1, 2), 1.0, 1.0, SURFACE)
check("Add subtracts depths (raises the surface) and clips at zero",
      np.allclose(out, [[0.5, 0.0]]))

base_mm = make_layer(np.array([[0.5, 0.5]], dtype=np.float32))
top_mlow = make_layer(np.array([[0.3, 0.7]], dtype=np.float32), blend_mode="Merge low")
out = composite([base_mm, top_mlow], (1, 2), 1.0, 1.0, SURFACE)
check("Merge low keeps the deeper of the two", np.allclose(out, [[0.5, 0.7]]))

top_mhigh = make_layer(np.array([[0.3, 0.7]], dtype=np.float32), blend_mode="Merge high")
out = composite([base_mm, top_mhigh], (1, 2), 1.0, 1.0, SURFACE)
check("Merge high keeps the shallower of the two", np.allclose(out, [[0.3, 0.5]]))

base_normal = make_layer(np.array([[1.0, 1.0]], dtype=np.float32))
top_normal  = make_layer(np.array([[0.9, 0.1]], dtype=np.float32), blend_mode="Replace")
out = composite([base_normal, top_normal], (1, 2), 1.0, 1.0, SURFACE)
check("Replace fully replaces the base wherever it has coverage",
      np.array_equal(out, top_normal.raw))


# --------------------------------------------------------------------------
# invisible layer contributes nothing, even for Add/Merge high against a
# zero-depth base -- this is the failure mode a missing "Replace" default
# blend mode would otherwise hit for a lone visible layer.
# --------------------------------------------------------------------------
print("visibility")

zero_base = make_layer(np.ones((2, 2), dtype=np.float32))   # white polarity -> depth 0 everywhere
hidden_add = make_layer(np.zeros((2, 2), dtype=np.float32), blend_mode="Add", visible=False)
out = composite([zero_base, hidden_add], (2, 2), 1.0, 1.0, "white")
check("an invisible layer (even Add) leaves the composite unchanged",
      np.array_equal(out, zero_base.raw))

hidden_mhigh = make_layer(np.zeros((2, 2), dtype=np.float32), blend_mode="Merge high", visible=False)
out = composite([zero_base, hidden_mhigh], (2, 2), 1.0, 1.0, "white")
check("an invisible Merge high layer leaves the composite unchanged",
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
# Add/Merge high-against-an-empty-canvas footgun, now fixed structurally.
# --------------------------------------------------------------------------
print("base layer forced to Replace")

bg = 0.4
base_vals = np.array([[0.9, 0.9]], dtype=np.float32)
out_add_at_base = composite([make_layer(base_vals.copy(), blend_mode="Add")],
                            (1, 2), 1.0, 1.0, "black", background_depth=bg)
out_mhigh_at_base = composite([make_layer(base_vals.copy(), blend_mode="Merge high")],
                              (1, 2), 1.0, 1.0, "black", background_depth=bg)
out_replace_at_base = composite([make_layer(base_vals.copy(), blend_mode="Replace")],
                                (1, 2), 1.0, 1.0, "black", background_depth=bg)
check("a stored Add blend_mode at index 0 is structurally ignored (forced to Replace)",
      np.array_equal(out_add_at_base, out_replace_at_base))
check("a stored Merge high blend_mode at index 0 is structurally ignored (forced to Replace)",
      np.array_equal(out_mhigh_at_base, out_replace_at_base))
check("...and does not actually apply the Add math (bg - base_vals) against the background",
      not np.allclose(out_add_at_base, np.clip(bg - base_vals, 0.0, 1.0)))


# --------------------------------------------------------------------------
# background_depth fills canvas area no layer covers
# --------------------------------------------------------------------------
print("background_depth")

offcanvas_layer = make_layer(np.full((2, 2), 0.9, dtype=np.float32), offset_x_mm=1000.0)
out = composite([offcanvas_layer], (2, 2), 1.0, 1.0, "black", background_depth=0.3)
check("background_depth fills canvas area no layer covers",
      np.allclose(out, 0.3))


# --------------------------------------------------------------------------
# alpha-driven coverage: the case Subtract/Merge low can already fake via
# zero-valued content but Replace and Merge high structurally cannot.
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

mhigh_layer = make_layer(np.zeros(canvas_shape, dtype=np.float32),
                         alpha=alpha_mask.copy(), blend_mode="Merge high")
out2 = composite([base_layer, mhigh_layer], canvas_shape, 1.0, 1.0, "black")
expected2 = np.full(canvas_shape, base_val, dtype=np.float32)
expected2[1, 1] = 0.0
check("Merge high with alpha coverage only affects the opaque pixel, not the rest (identity-placement path)",
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


# --------------------------------------------------------------------------
# Mirror: reflection across the stock flip axes, applied after placement
# --------------------------------------------------------------------------
print("mirror")

rng = np.random.default_rng(0)
sq = rng.random((9, 9)).astype(np.float32)
for mirror, origin, expect in [
        ("Vertical axis",   "Bottom-Left", np.fliplr(sq)),
        ("Horizontal axis", "Bottom-Left", np.flipud(sq)),
        ("Diagonal (+45°)", "Bottom-Left", np.rot90(sq, 2).T),   # y = x, rows point down
        ("Diagonal (+45°)", "Top-Right",   np.rot90(sq, 2).T),
        ("Diagonal (+45°)", "Center",      np.rot90(sq, 2).T),
        ("Diagonal (−45°)", "Top-Left",    sq.T),
        ("Diagonal (−45°)", "Bottom-Right", sq.T),
        ("Diagonal (−45°)", "Center",      sq.T)]:
    out, cov = resample_layer(make_layer(sq.copy(), mirror=mirror), sq.shape, 1.0, 1.0, origin)
    check(f"{mirror} through {origin} on a square canvas",
          np.allclose(out, expect, atol=1e-5) and np.allclose(cov, 1.0, atol=1e-5))

marker = np.zeros((10, 10), dtype=np.float32)
marker[4:6, 4:6] = 1.0
off = make_layer(marker.copy(), offset_x_mm=3.0, mirror="Vertical axis")
out, _ = resample_layer(off, (10, 10), 1.0, 1.0)
cols = np.nonzero(out.max(axis=0) > 0.5)[0]
check("a mirrored layer's offset is mirrored too (+3 mm lands at -3 mm)",
      list(cols) == [1, 2])

wide = np.zeros((5, 9), dtype=np.float32)   # H=5, W=9, 1 px/mm: 8 x 4 mm stock
wide[4, 2] = 1.0                             # stock (x=2, y=0)
out, _ = resample_layer(make_layer(wide.copy(), mirror="Diagonal (+45°)"),
                        wide.shape, 1.0, 1.0, "Bottom-Left")
check("Diagonal (+45°) from Bottom-Left on a non-square canvas maps (x, y) -> (y, x)",
      out[4 - 2, 0] > 0.99 and out.sum() < 1.01)

for mirror, pattern, dd in [("Vertical axis", "Vertical axis", "+45°"),
                            ("Horizontal axis", "Horizontal axis", "+45°"),
                            ("Diagonal (+45°)", "Diagonal", "+45°"),
                            ("Diagonal (−45°)", "Diagonal", "−45°")]:
    origins = (["Bottom-Left", "Top-Right", "Center"] if mirror == "Diagonal (+45°)" else
               ["Top-Left", "Bottom-Right", "Center"] if mirror == "Diagonal (−45°)" else
               ["Bottom-Left", "Center"])
    for origin in origins:
        pm, dm = _mirror_axis_mm(mirror, 120.0, 80.0, origin)
        pr, dr = registration_flip_axis(pattern, 120.0, 80.0, origin, dd)
        same_dir = abs(abs(dm[0] * dr[0] + dm[1] * dr[1]) - 1.0) < 1e-9
        on_line = abs((pr[0] - pm[0]) * dm[1] - (pr[1] - pm[1]) * dm[0]) < 1e-9
        check(f"{mirror} through {origin} uses the Registration Holes flip axis",
              same_dir and on_line)


# --------------------------------------------------------------------------
# overhang_masks: gaps enclosed by the source vs. open down to full depth
# --------------------------------------------------------------------------
print("overhang_masks")

# Pixel 0: closed gap; 1: open-bottom gap; 2: both; 3: no gap.
g = Gaps.from_lists((1, 4), [0, 1, 2, 2], [0.2, 0.3, 0.1, 0.7], [0.6, 1.0, 0.5, 1.0])
closed, open_bottom = overhang_masks(g)
check("a gap with a floor above full depth is closed", closed[0, 0] and not open_bottom[0, 0])
check("a gap down to full depth is open-bottom", open_bottom[0, 1] and not closed[0, 1])
check("a pixel with both kinds counts as closed", closed[0, 2] and not open_bottom[0, 2])
check("a pixel without gaps is neither", not closed[0, 3] and not open_bottom[0, 3])
check("empty gaps give empty masks",
      not any(m.any() for m in overhang_masks(Gaps.empty((3, 3)))))


# --------------------------------------------------------------------------
# STL placement by the mesh's mm extent
# --------------------------------------------------------------------------
print("STL placement by mesh extent")

from layers import place_stl_layer, placement_key, raw_fills_canvas, native_aspect


def box_mesh(w, h, d):
    v = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
                  [0, 0, d], [w, 0, d], [w, h, d], [0, h, d]], float)
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in f], np.float32)


def stl_layer(raw_shape, mesh_w, mesh_h, **kw):
    hl, wl = raw_shape
    return make_layer(np.zeros(raw_shape, np.float32), source_kind='stl', path='box.stl',
                      stl_params={'width_px': wl, 'height_px': hl,
                                  'mesh_w_mm': mesh_w, 'mesh_h_mm': mesh_h}, **kw)


# A 100 x 3.3 mm box on a 100 x 50 mm canvas at 10 px/mm: its preview
# raster's rounded pixel count (33, 7, 6 or 2 rows) must not change where
# the mesh lands on the canvas grid.
thin = box_mesh(100.0, 3.3, 2.0)
footprints = []
for raw_shape in [(33, 1000), (7, 200), (6, 200), (2, 50)]:
    pr = place_stl_layer(stl_layer(raw_shape, 100.0, 3.3), thin, (500, 1000), 10.0, 10.0)
    cov = pr.alpha if pr.alpha is not None else np.ones((500, 1000))
    footprints.append(cov.astype(bool))
check("placed STL raster is independent of the preview raster's pixel count",
      all(np.array_equal(footprints[0], fp) for fp in footprints[1:]))
check("placed STL raster spans the mesh's width and height",
      footprints[0].any(0).sum() == 1000 and abs(footprints[0].any(1).sum() - 33) <= 1)

lk = stl_layer((7, 200), 100.0, 3.3)
key_a = placement_key(lk, (500, 1000), 10.0, 10.0, "Bottom-Left")
lk.raw = np.zeros((33, 1000), np.float32)
lk.stl_params = dict(lk.stl_params, width_px=1000, height_px=33)
check("placement key ignores the preview raster's size",
      key_a == placement_key(lk, (500, 1000), 10.0, 10.0, "Bottom-Left"))
check("native aspect of an STL layer is the mesh's", abs(native_aspect(lk) - 100.0 / 3.3) < 1e-12)

check("an image at the canvas's shape fills it 1:1",
      raw_fills_canvas(make_layer(np.zeros((40, 60), np.float32)), (40, 60)))
check("an STL raster whose mesh proportions differ from the grid's does not",
      not raw_fills_canvas(stl_layer((40, 60), 60.0, 40.5), (40, 60)))
check("an STL raster with the grid's proportions does",
      raw_fills_canvas(stl_layer((40, 60), 60.0, 40.0), (40, 60)))
check("a scaled source does not fill the canvas 1:1",
      not raw_fills_canvas(make_layer(np.zeros((40, 60), np.float32), scale_pct=90.0),
                           (40, 60)))


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
