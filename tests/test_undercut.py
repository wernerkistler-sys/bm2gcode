"""Plain assert-based tests for undercut planning: tool kernels (undercut.py),
mesh dexels (stl_mesh.py), gap compositing (layers.py) and the Undercut
toolpath itself.

Run with: python tests/test_undercut.py
"""
import math
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from layers import (Layer, Gaps, composite, composite_gaps, overhang_masks, pad_gaps,
                    to_depth)
from stl_mesh import (load_stl, generate_dexels, generate_heightmap, mesh_z_bounds,
                      normalize_and_orient)
from tool import Tool, undercut_geometry_error
from undercut import (build_undercut_kernels, compute_undercut_toolpath,
                      NECK, _floor_ceil_at, _head_profile, _resample_gaps)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


def lollipop(d=6.0, neck=3.0, **kw):
    return Tool("Lollipop", d, neck_diameter_mm=neck, **kw)


# --------------------------------------------------------------------------
# Inverted truncated cone standing on a plate: radius RB at the plate, RT at
# the top, so the flank overhangs the plate by RT - RB all around.
# --------------------------------------------------------------------------
SIZE_MM, PX = 60.0, 4.0
RB, RT, HC = 8.0, 12.0, 6.0          # max_depth = HC: plate top at full depth
N_PX = int(SIZE_MM * PX)
AXIS = ((N_PX - 1) / 2.0 / PX, (N_PX - 1) / 2.0 / PX)   # (x, y) mm, pixel-centred


def cone_fixture():
    """(depth_map, gaps) of the cone on the canvas grid, analytic."""
    rows, cols = np.mgrid[0:N_PX, 0:N_PX]
    x = cols / PX
    y = (N_PX - 1 - rows) / PX
    r = np.hypot(x - AXIS[0], y - AXIS[1])
    depth = np.where(r < RT, 0.0, 1.0).astype(np.float32)
    under = (r > RB) & (r < RT)
    pix = np.flatnonzero(under.ravel())
    ceil = 1.0 - (r.ravel()[pix] - RB) / (RT - RB)
    return depth, Gaps.from_lists((N_PX, N_PX), pix, ceil, np.ones(pix.size))


def cone_solid(xs, ys, zs):
    """True where (x, y, z mm; z = 0 at the cone top) is inside cone or plate."""
    r = np.hypot(xs - AXIS[0], ys - AXIS[1])
    zr = zs + HC                                   # height above the plate
    in_cone = (zr >= 0) & (zs <= 0) & (r <= RB + (RT - RB) * zr / HC)
    return (zr < 0) | in_cone


def tool_sample_points(tool, shrink):
    """Surface sample points of an undercut tool's head and neck, relative to
    the tip, pulled *shrink* mm inward (the check's tolerance)."""
    R = tool.diameter_mm / 2.0
    rn = tool.neck_diameter_mm / 2.0
    radii = np.linspace(0.0, R, 49)
    lo, hi = _head_profile(tool, radii)
    prof = []                                   # (r, h) outline points
    prof += [(r, l + shrink) for r, l in zip(radii, lo) if r <= R - shrink]
    prof += [(r, h - shrink) for r, h in zip(radii, hi) if rn <= r <= R - shrink]
    for h in np.linspace(lo[-1], hi[-1], 9):    # rim
        prof.append((R - shrink, h))
    top_neck = float(np.interp(rn, radii, hi))
    for h in np.linspace(top_neck, top_neck + 20.0, 41):
        prof.append((rn - shrink, h))
    pts = []
    for r, h in prof:
        if h > hi.max() + 25 or r < 0:
            continue
        for az in np.linspace(0, 2 * math.pi, 36, endpoint=False):
            pts.append((r * math.cos(az), r * math.sin(az), h))
    return np.array(pts)


def collisions(tp, tool, shrink, dz=0.0, solid=cone_solid):
    """Path points at which the sampled tool (its tip *dz* mm below the
    path's tip) intersects *solid*."""
    body = tool_sample_points(tool, shrink)
    hits = 0
    for seg in tp.segments:
        for x, y, z in seg:
            hits += int(solid(body[:, 0] + x, body[:, 1] + y,
                              body[:, 2] + z + dz).any())
    return hits


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def plunge_ok(x, y, z, tool, shrink):
    """A vertical plunge from above to (x, y, z) misses the cone (sampled)."""
    body = tool_sample_points(tool, shrink)
    for zz in np.linspace(z, 2.0, 30):
        if cone_solid(body[:, 0] + x, body[:, 1] + y, body[:, 2] + zz).any():
            return False
    return True


# --------------------------------------------------------------------------
# Tool geometry validation
# --------------------------------------------------------------------------
print("undercut tool validation")
check("valid lollipop", undercut_geometry_error(lollipop()) is None)
check("neck must be thinner than the head",
      undercut_geometry_error(lollipop(neck=6.0)) is not None)
check("wider shank needs a neck length",
      undercut_geometry_error(lollipop(shank_diameter_mm=6.0)) is not None)
check("T-Slot needs a head height",
      undercut_geometry_error(Tool("T-Slot", 10.0, neck_diameter_mm=4.0)) is not None)
check("Dovetail head must not be narrower than the neck at its top",
      undercut_geometry_error(Tool("Dovetail", 10.0, neck_diameter_mm=6.0,
                                   head_height_mm=5.0, dovetail_angle_deg=45.0))
      is not None)
check("non-undercut profiles are not validated",
      undercut_geometry_error(Tool("Ball", 6.0)) is None)


# --------------------------------------------------------------------------
# Kernels: lo/hi at known radii, neck sentinel, allowance
# --------------------------------------------------------------------------
print("undercut kernels")
md = 10.0
lo, hi, fp = build_undercut_kernels(lollipop(), 10.0, md)
c = lo.shape[0] // 2
check("lollipop tip: lo = 0", abs(lo[c, c]) < 1e-12)
check("lollipop over the neck: hi = NECK", hi[c, c] == NECK and hi[c, c + 15] == NECK)
r_mm = 2.0                                  # 20 px, between neck (1.5) and R (3)
check("lollipop lo/hi at r = 2 mm",
      abs(lo[c, c + 20] - (3 - math.sqrt(5)) / md) < 1e-9
      and abs(hi[c, c + 20] - (3 + math.sqrt(5)) / md) < 1e-9)
check("lollipop rim: lo = hi = R", abs(lo[c, c + 30] - 0.3) < 1e-9
      and abs(hi[c, c + 30] - 0.3) < 1e-9)

lo, hi, fp = build_undercut_kernels(Tool("T-Slot", 10.0, neck_diameter_mm=4.0,
                                         head_height_mm=2.0), 10.0, md)
c = lo.shape[0] // 2
check("T-Slot: flat underside, top at head height outside the neck",
      np.all(lo[fp] == 0.0) and abs(hi[c, c + 40] - 0.2) < 1e-12 and hi[c, c + 10] == NECK)

lo, hi, fp = build_undercut_kernels(Tool("Dovetail", 10.0, neck_diameter_mm=4.0,
                                         head_height_mm=2.0, dovetail_angle_deg=45.0),
                                    10.0, md)
c = lo.shape[0] // 2
check("Dovetail flank: hi = (R - r) tan a, capped at the head height",
      abs(hi[c, c + 40] - 0.1) < 1e-9 and abs(hi[c, c + 25] - 0.2) < 1e-9)

lo0, hi0, fp0 = build_undercut_kernels(lollipop(), 10.0, md)
lo1, hi1, fp1 = build_undercut_kernels(lollipop(), 10.0, md, allowance_mm=0.5)
c0, c1 = lo0.shape[0] // 2, lo1.shape[0] // 2
check("allowance grows the footprint by its radius",
      lo1.shape[0] == lo0.shape[0] + 10 and fp1[c1, c1 + 35] and fp0[c0, c0 + 30])
check("allowance lowers the underside by its radius at the tip",
      abs(lo1[c1, c1] - (-0.05)) < 1e-9)
# Lollipop grown by a: a sphere of R + a about the same centre (R above the
# tip), on a neck a wider.
check("allowance: underside and top follow the sphere of R + a",
      abs(lo1[c1, c1 + 25] - (3 - math.sqrt(3.5 ** 2 - 2.5 ** 2)) / md) < 0.003
      and abs(hi1[c1, c1 + 25] - (3 + math.sqrt(3.5 ** 2 - 2.5 ** 2)) / md) < 0.003)
check("allowance widens the neck by its radius",
      hi1[c1, c1 + 20] >= NECK and hi1[c1, c1 + 21] < NECK)


# --------------------------------------------------------------------------
# Floor / ceiling selection
# --------------------------------------------------------------------------
print("floor/ceiling selection")
g = Gaps.from_lists((1, 3), [1, 1, 2], [0.2, 0.6, 0.3], [0.4, 0.8, 1.0])
top = np.array([[0.0, 0.1, 0.1]])
F, C, act = _floor_ceil_at(g, g.pixel_index(), top, 0.5)
check("no gap: top surface floor, open sky", F[0, 0] == 0.0 and C[0, 0] < -1e5 and not act[0, 0])
check("depth in solid below gap 0 uses gap 0", F[0, 1] == 0.4 and C[0, 1] == 0.2)
F, C, act = _floor_ceil_at(g, g.pixel_index(), top, 0.7)
check("depth inside gap 1 uses gap 1", F[0, 1] == 0.8 and C[0, 1] == 0.6)
F, C, act = _floor_ceil_at(g, g.pixel_index(), top, 0.25)
check("depth above every gap ceiling: no gap", F[0, 2] == 0.1 and not act[0, 2])

g_top = Gaps.from_lists((1, 2), [0, 1], [0.3, 0.2], [0.9, 0.9])
F, C, act = _floor_ceil_at(g_top, g_top.pixel_index(), np.array([[0.3, 0.3]]), 0.5)
check("a gap starting at the top surface has open sky above",
      F[0, 0] == 0.9 and C[0, 0] < -1e5 and act[0, 0])
check("a gap starting above the top surface keeps its ceiling", C[0, 1] == 0.2)

rg = _resample_gaps(g, (1, 5))
check("nearest resample keeps each column's gaps",
      list(rg.counts[0]) == [0, 0, 2, 1, 1]
      and np.allclose(rg.ceil[rg.offsets[2]:rg.offsets[3]], [0.2, 0.6]))
pg = pad_gaps(g, 1, 2)
check("pad_gaps shifts gaps by the margin", pg.shape == (3, 7)
      and list(pg.counts[1]) == [0, 0, 0, 2, 1, 0, 0])


# --------------------------------------------------------------------------
# Dexels from a mesh: the inverted cone on a plate, as an ASCII STL
# --------------------------------------------------------------------------
print("dexels")


def _frustum_and_plate(n=96, plate=20.0, hp=2.0):
    """Closed triangles (outward winding) for the cone on a plate, in mm,
    plate top at z = hp: two closed shells touching at the cone's base."""

    def orient(tris, centre):
        tris = np.array(tris, np.float64)
        for k, t in enumerate(tris):
            nrm = np.cross(t[1] - t[0], t[2] - t[0])
            if np.dot(nrm, t.mean(axis=0) - centre) < 0:
                tris[k] = t[[0, 2, 1]]
        return tris

    def quads(*qs):
        out = []
        for a, b, c_, d_ in qs:
            out.extend([(a, b, c_), (a, c_, d_)])
        return out

    P = plate
    v = [(-P, -P, 0), (P, -P, 0), (P, P, 0), (-P, P, 0),
         (-P, -P, hp), (P, -P, hp), (P, P, hp), (-P, P, hp)]
    box = quads((v[0], v[3], v[2], v[1]), (v[4], v[5], v[6], v[7]),
                (v[0], v[1], v[5], v[4]), (v[1], v[2], v[6], v[5]),
                (v[2], v[3], v[7], v[6]), (v[3], v[0], v[4], v[7]))
    cone = []
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    for i in range(n):
        a0, a1 = ang[i], ang[(i + 1) % n]
        b0 = (RB * math.cos(a0), RB * math.sin(a0), hp)
        b1 = (RB * math.cos(a1), RB * math.sin(a1), hp)
        t0 = (RT * math.cos(a0), RT * math.sin(a0), hp + HC)
        t1 = (RT * math.cos(a1), RT * math.sin(a1), hp + HC)
        cone += quads((b0, b1, t1, t0))
        cone.append(((0, 0, hp + HC), t0, t1))
        cone.append(((0, 0, hp), b1, b0))
    return np.concatenate([orient(box, np.array([0, 0, hp / 2])),
                           orient(cone, np.array([0, 0, hp + HC / 2]))])


def _write_ascii_stl(path, tris):
    with open(path, 'w') as f:
        f.write("solid t\n")
        for t in tris:
            f.write(" facet normal 0 0 0\n  outer loop\n")
            for v in t:
                f.write(f"   vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
            f.write("  endloop\n endfacet\n")
        f.write("endsolid t\n")


with tempfile.TemporaryDirectory() as tmp:
    stl_path = os.path.join(tmp, "cone.stl")
    _write_ascii_stl(stl_path, _frustum_and_plate())
    tris = load_stl(stl_path)
n_mesh = 161                                   # 40 mm / 0.25 mm + 1
dex = generate_dexels(tris, n_mesh, n_mesh)
hm = generate_heightmap(tris, n_mesh, n_mesh)
counts = np.diff(dex.offsets).reshape(dex.shape)
check("Dexels.top is exactly the heightmap",
      np.array_equal(np.isnan(dex.top), np.isnan(np.flipud(hm)))
      and np.array_equal(dex.top[~np.isnan(dex.top)], np.flipud(hm)[~np.isnan(dex.top)]))
mm = np.linspace(-20, 20, n_mesh)
rows, cols = np.mgrid[0:n_mesh, 0:n_mesh]
xx, yy = mm[cols], mm[::-1][rows]
rr = np.hypot(xx, yy)
# Columns reduced to their top hit: a ray through the mesh's outline, where
# float rounding can drop one of two coincident crossings.
cnt_now = np.diff(dex.offsets).reshape(dex.shape)
on_silhouette = (np.abs(rr - RT) < 0.2) | (np.abs(xx) > 19.9) | (np.abs(yy) > 19.9)
check("inconsistent columns only where a ray grazes the outline",
      not np.any((cnt_now == 1) & ~on_silhouette))
under = (rr > RB + 0.3) & (rr < RT - 0.3)
check("columns under the overhang have 4 hits (top, flank, plate top, plate bottom)",
      np.all(counts[under] == 4))
check("columns through the cone core have 2 hits (cap/plate seam cancels)",
      np.all(counts[rr < RB - 0.3] == 2))
check("plate-only columns have 2 hits", np.all(counts[(rr > RT + 0.3) & (np.abs(xx) < 19.5)
                                                        & (np.abs(yy) < 19.5)] == 2))
i = int(np.flatnonzero((under & (np.abs(rr - 11.0) < 0.2)).ravel())[0])
zc = dex.z[dex.offsets[i]:dex.offsets[i + 1]]
flank = 2.0 + HC * (rr.ravel()[i] - RB) / (RT - RB)
check("flank hit at the analytic height", abs(zc[1] - flank) < 0.05 and abs(zc[2] - 2.0) < 1e-6)

open_tris = _frustum_and_plate()[2:]          # drop the plate bottom
dex_open = generate_dexels(open_tris, n_mesh, n_mesh)
cnt_open = np.diff(dex_open.offsets).reshape(dex_open.shape)
check("open-bottom columns end in a bottomless solid (odd count)",
      np.all(cnt_open[(rr > RT + 0.3) & (np.abs(xx) < 19.5) & (np.abs(yy) < 19.5)] % 2 == 1))
flipped = generate_dexels(_frustum_and_plate()[:, [0, 2, 1]], n_mesh, n_mesh)
check("an inside-out mesh gives the same dexels",
      np.array_equal(flipped.offsets, dex.offsets) and np.allclose(flipped.z, dex.z))


def _bicone(n=64, r=10.0, h=5.0):
    """Closed triangles (outward winding) of two cones base to base: a rim
    of radius r at z = 0, apexes at z = +-h. Rim vertices at 0, 90, 180 and
    270 degrees lie on the mesh's XY bounds."""
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    rim = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], axis=1)
    rim[np.abs(rim) < 1e-12] = 0.0
    up, dn = np.array([0.0, 0.0, h]), np.array([0.0, 0.0, -h])
    tris = []
    for i in range(n):
        p, q = rim[i], rim[(i + 1) % n]
        tris.append([p, q, up])
        tris.append([q, p, dn])
    return np.array(tris)


# Rays through the bounds' pixel centres graze the rim at one vertex: the
# upper and lower cone's hits cancel, yet the heightmap has a top there.
bic = _bicone()
dex_b = generate_dexels(bic, 81, 81)
nb = np.diff(dex_b.offsets)
first_b = np.full(nb.size, np.nan)
first_b[nb > 0] = dex_b.z[dex_b.offsets[:-1][nb > 0]]
cov_b = ~np.isnan(dex_b.top.ravel())
check("a grazed rim keeps the heightmap's top as the first crossing",
      np.array_equal(first_b[cov_b], dex_b.top.ravel()[cov_b]) and not np.any(nb[~cov_b]))
i_b = 40 * 81 + 80                                # (x, y) = (10, 0), the rim's tip
check("a grazed rim is a zero-thickness solid with air below",
      list(dex_b.z[dex_b.offsets[i_b]:dex_b.offsets[i_b + 1]]) == [0.0, 0.0]
      and dex_b.n_invalid == 0)


# --------------------------------------------------------------------------
# composite_gaps: the STL layer's gaps land where the composite top is its own
# --------------------------------------------------------------------------
print("composite_gaps")
raw, alpha = normalize_and_orient(hm, mesh_z_bounds(tris))
layer = Layer(name="cone", raw=raw, source_kind="stl", alpha=alpha)
layer.dexels = dex
shape = (n_mesh, n_mesh)
ppm = n_mesh / 40.0
comp = to_depth(composite([layer], shape, ppm, ppm, "white"), "white", False)
gp, warns = composite_gaps([layer], shape, ppm, ppm, "white", comp)
gc = gp.counts
check("one gap per column under the overhang", np.all(gc[under] == 1) and not warns)
check("no gaps outside the overhang", gc[~((rr > RB - 0.3) & (rr < RT + 0.3))].sum() == 0)
j = int(np.flatnonzero((under & (np.abs(rr - 11.0) < 0.2)).ravel())[0])
k = gp.offsets[j]
# depth = 1 - z / 8 over the mesh's z bounds: the plate bottom (z = 0) at
# full depth, the plate top (z = 2) at 0.75
from layers import _layer_depth_affine
a_, b_ = _layer_depth_affine(layer, "white")
flank_z = 2.0 + HC * (rr.ravel()[j] - RB) / (RT - RB)
check("gap ceiling = flank depth, floor = plate top, as the layer maps depth",
      abs(gp.ceil[k] - (a_ * flank_z + b_)) < 0.01
      and abs(gp.floor[k] - min(a_ * 2.0 + b_, 1.0)) < 1e-6)
check("top-surface depth of the gap columns matches the composite",
      np.all(np.abs(comp[under] - (a_ * 8.0 + b_)) < 1e-4))

shifted = Layer(name="cone", raw=raw, source_kind="stl", alpha=alpha,
                offset_x_mm=4.0, mirror="Vertical axis")
shifted.dexels = dex
comp_s = to_depth(composite([shifted], shape, ppm, ppm, "white"), "white", False)
gs, _ = composite_gaps([shifted], shape, ppm, ppm, "white", comp_s)
rr_s = np.hypot(xx + 4.0, yy)                   # offset +4 then mirrored to -4
under_s = (rr_s > RB + 0.4) & (rr_s < RT - 0.4)
check("placement and Mirror move the gaps with the layer",
      np.all(gs.counts[under_s] == 1)
      and gs.counts[~((rr_s > RB - 0.4) & (rr_s < RT + 0.4))].sum() == 0)

cover = Layer(name="block", raw=np.full(shape, 0.5, np.float32), source_kind="image",
              scale_pct=50.0)
comp_c = to_depth(composite([layer, cover], shape, ppm, ppm, "white"), "white", False)
gcv, _ = composite_gaps([layer, cover], shape, ppm, ppm, "white", comp_c)
covered = (np.abs(xx) < 9.5) & (np.abs(yy) < 9.5)
check("a layer covering the STL hides its gaps", gcv.counts[covered].sum() == 0
      and gcv.counts[under & ~((np.abs(xx) < 10.5) & (np.abs(yy) < 10.5))].min() == 1)

inv = Layer(name="cone", raw=raw, source_kind="stl", alpha=alpha,
            min_depth_pct=100.0, max_depth_pct=0.0)
inv.dexels = dex
comp_i = to_depth(composite([inv], shape, ppm, ppm, "white"), "white", False)
gi, warns_i = composite_gaps([inv], shape, ppm, ppm, "white", comp_i)
check("inverted polarity is ignored with a warning",
      gi.counts.sum() == 0 and len(warns_i) == 1)

# The plate top lies above full depth (the plate bottom, the mesh's lowest
# point, is at full depth), so the air under the flank is enclosed.
closed, open_bottom = overhang_masks(gp)
check("overhang over the plate is closed",
      np.all(closed[under]) and not open_bottom.any()
      and not closed[~((rr > RB - 0.3) & (rr < RT + 0.3))].any())
# With the layer's Max depth at 80 % the whole mesh floats above full depth:
# air from its bottom face down to the stock's bottom face is open-bottom.
shallow = Layer(name="cone", raw=raw, source_kind="stl", alpha=alpha, max_depth_pct=80.0)
shallow.dexels = dex
comp_sh = to_depth(composite([shallow], shape, ppm, ppm, "white"), "white", False)
gsh, _ = composite_gaps([shallow], shape, ppm, ppm, "white", comp_sh)
closed_sh, open_sh = overhang_masks(gsh)
plate_only = (rr > RT + 0.3) & (np.abs(xx) < 19.5) & (np.abs(yy) < 19.5)
check("a floating mesh: open-bottom under its bottom face, closed under the flank",
      np.all(open_sh[plate_only]) and np.all(closed_sh[under])
      and not (open_sh & closed_sh).any())


# Partial coverage: the bicone (a floating lens: its lower half overhangs
# open air) shifted by a fraction of a pixel, so its outline pixels are
# partly covered and blend with what lies beneath.
raw_b, alpha_b = normalize_and_orient(generate_heightmap(bic, 81, 81), mesh_z_bounds(bic))
lens = Layer(name="lens", raw=raw_b, source_kind="stl", alpha=alpha_b, offset_x_mm=0.1)
lens.dexels = dex_b
ppm_b = 80 / 20.0
_, cov_l = __import__("layers").resample_layer(lens, (81, 81), ppm_b, ppm_b)
edge_l = (cov_l > 1e-3) & (cov_l < 0.999)
for bg in (1.0, 0.0):
    comp_l = to_depth(composite([lens], (81, 81), ppm_b, ppm_b, "white", background_depth=bg),
                      "white", False)
    gl, _ = composite_gaps([lens], (81, 81), ppm_b, ppm_b, "white", comp_l,
                           background_depth=bg)
    pix_l = gl.pixel_index()
    on_edge = edge_l.ravel()[pix_l]
    if bg == 1.0:
        open_to_bed = np.zeros(81 * 81, bool)
        open_to_bed[pix_l[gl.floor >= 1.0]] = True
        check("partly covered outline over a full-depth background: air down to it",
              edge_l.sum() > 20 and np.all(open_to_bed.reshape(81, 81)[edge_l]))
        check("partly covered outline: gaps start at or below the composite top",
              np.all(gl.ceil[on_edge] >= comp_l.ravel()[pix_l[on_edge]] - 1e-9))
    else:
        check("partly covered outline over uncut stock: no air below its top",
              not on_edge.any() and gl.counts[~edge_l & (cov_l >= 0.999)].sum() > 0)


def _box(x0, x1, y0, y1, z0, z1):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    out = []
    for a, b, c_, d_ in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
        out += [(v[a], v[b], v[c_]), (v[a], v[c_], v[d_])]
    return np.array(out, np.float64)


# The outermost pixel centres lie on the mesh's XY bounds, exactly on the
# silhouette edges where the top and bottom faces meet: they see the top.
for n in (41, 57):
    box = _box(0, 10, 0, 10, 1.234, 9.876)
    hm_b = generate_heightmap(box, n, n)
    dex_b = generate_dexels(box, n, n)
    check(f"cube {n}x{n}: every pixel's height is the top face, edges included",
          np.allclose(hm_b, 9.876, rtol=0, atol=1e-9)
          and np.allclose(dex_b.top, 9.876, rtol=0, atol=1e-9))
    raw_b, _ = normalize_and_orient(hm_b, mesh_z_bounds(box))
    check(f"cube {n}x{n}: a flat top normalises to the top of the mesh's z bounds",
          np.all(raw_b == 1.0) and dex_b.z_bounds == (1.234, 9.876))

# A cube's bottom face maps to full depth only up to rounding (z 3..13 gives
# 1 - 2e-16): it rests on the stock's bottom face and leaves no gap there.
for z0, z1 in [(3.0, 13.0), (-12.7, 3.3), (1.234, 9.876), (-3.3, -0.7)]:
    box = _box(0, 10, 0, 10, z0, z1)
    raw_b, alpha_b = normalize_and_orient(generate_heightmap(box, 41, 41), mesh_z_bounds(box))
    for pct, floating in ((100.0, False), (80.0, True)):
        lb = Layer(name="cube", raw=raw_b, source_kind="stl", alpha=alpha_b,
                   max_depth_pct=pct)
        lb.dexels = generate_dexels(box, 41, 41)
        comp_b = to_depth(composite([lb], (41, 41), 4.1, 4.1, "white"), "white", False)
        gb, _ = composite_gaps([lb], (41, 41), 4.1, 4.1, "white", comp_b)
        closed_b, open_b = overhang_masks(gb)
        if floating:
            check(f"cube z {z0}..{z1} at 80 % Max depth floats (open-bottom)",
                  open_b.all() and not closed_b.any())
        else:
            check(f"cube z {z0}..{z1} on full depth has no gaps", gb.counts.sum() == 0)


# --------------------------------------------------------------------------
# Undercut toolpath on the analytic cone with a Lollipop
# --------------------------------------------------------------------------
print("undercut toolpath: inverted cone, Lollipop")
depth, gaps = cone_fixture()
tool = lollipop()                                 # R = 3, neck r = 1.5 -> reach 1.5
prev = depth.copy()                               # 2.5D roughing done
tp = compute_undercut_toolpath(depth, gaps, tool, 0.5, 30.0, HC,
                               SIZE_MM, SIZE_MM, previous_result=prev)
pts = np.array([p for s in tp.segments for p in s])
rho = np.hypot(pts[:, 0] - AXIS[0], pts[:, 1] - AXIS[1])
check("produces chains", len(tp.segments) > 0)
check("the head reaches under the lip (>= 1 mm)", rho.min() - 3.0 < RT - 1.0)
check("the neck axis stays outside the lip by the neck radius",
      rho.min() >= RT + 1.5 - 0.01)
check("no tool/part collision (sampled tool shrunk by 0.15 mm)",
      collisions(tp, tool, 0.15) == 0)
check("every chain starts and ends where a vertical plunge is clear",
      all(plunge_ok(*s[0], tool, 0.15) and plunge_ok(*s[-1], tool, 0.15)
          for s in tp.segments))
check("every chain stays at one depth", all(len({round(p[2], 9) for p in s}) == 1
                                            for s in tp.segments))
zs = [s[0][2] for s in tp.segments]
check("levels are cut top-down", all(a >= b - 1e-9 for a, b in zip(zs, zs[1:])))
check("the previous result is passed through unchanged", np.array_equal(tp.result, prev))
check("no warnings on a reachable undercut", tp.metadata['warnings'] == [])
_r_px = np.hypot(np.mgrid[0:N_PX, 0:N_PX][1] / PX - AXIS[0],
                 (N_PX - 1 - np.mgrid[0:N_PX, 0:N_PX][0]) / PX - AXIS[1])
reach = tp.undercut_reach
check("undercut_reach is on the depth map's grid",
      reach is not None and reach.shape == depth.shape and reach.dtype == bool)
check("undercut_reach covers the lip's outer band",
      np.all(reach[(_r_px > RT - 0.75) & (_r_px < RT - 0.25)]))
check("undercut_reach stays near the lip",
      not reach[_r_px < RB].any() and not reach[_r_px > RT + 6.5].any())

# Rings within a level: cut from open sky toward the wall (small step-over).
tp_r = compute_undercut_toolpath(depth, gaps, tool, 0.5, 8.0, HC,
                                 SIZE_MM, SIZE_MM, previous_result=prev)
check("small step-over: no tool/part collision", collisions(tp_r, tool, 0.15) == 0)
lvl = {}
for s in tp_r.segments:
    lvl.setdefault(round(s[0][2], 6), []).append(np.array(s))
ordered_ok, n_multi = True, 0
for z, segs in lvl.items():
    allp = np.vstack(segs)
    r_all = np.hypot(allp[:, 0] - AXIS[0], allp[:, 1] - AXIS[1])
    under_pts = r_all < 14.99                    # not reachable by a plunge
    if under_pts.sum() < 20:
        continue
    r_u = r_all[under_pts]
    half = len(r_u) // 2
    if np.ptp(r_u) > 0.8:
        n_multi += 1
        ordered_ok &= r_u[:half].mean() > r_u[half:].mean()
check("some level has more than one ring", n_multi > 0)
check("rings run from open sky inward, waterline last", ordered_ok)

# Climb: uncut material (the cone axis side) on the right of travel.
seg = max(tp.segments, key=len)
a = np.array(seg)
mid = len(a) // 2
tang = a[mid + 1, :2] - a[mid - 1, :2]
to_axis = np.array(AXIS) - a[mid, :2]
check("Climb keeps the material on the right", cross2(tang, to_axis) < 0)

tp_conv = compute_undercut_toolpath(depth, gaps, tool, 0.5, 30.0, HC, SIZE_MM, SIZE_MM,
                                    previous_result=prev, cut_direction="Conventional")
a = np.array(max(tp_conv.segments, key=len))
mid = len(a) // 2
tang = a[mid + 1, :2] - a[mid - 1, :2]
check("Conventional keeps the material on the left",
      cross2(tang, np.array(AXIS) - a[mid, :2]) > 0)


print("undercut toolpath: T-Slot and Dovetail")
for t_tool in (Tool("T-Slot", 6.0, neck_diameter_mm=3.0, head_height_mm=1.5),
               Tool("Dovetail", 7.0, neck_diameter_mm=3.0, head_height_mm=2.0,
                    dovetail_angle_deg=60.0)):
    tp_t = compute_undercut_toolpath(depth, gaps, t_tool, 0.5, 30.0, HC,
                                     SIZE_MM, SIZE_MM, previous_result=prev)
    p_t = np.array([p for s_ in tp_t.segments for p in s_])
    check(f"{t_tool.profile}: reaches under the lip",
          len(tp_t.segments) > 0
          and np.hypot(p_t[:, 0] - AXIS[0], p_t[:, 1] - AXIS[1]).min()
          - t_tool.diameter_mm / 2 < RT - 0.5)
    check(f"{t_tool.profile}: no tool/part collision", collisions(tp_t, t_tool, 0.15) == 0)
    check(f"{t_tool.profile}: chains start and end plunge-clear",
          all(plunge_ok(*s_[0], t_tool, 0.15) and plunge_ok(*s_[-1], t_tool, 0.15)
              for s_ in tp_t.segments))


print("undercut toolpath: overhang on one side only")
# Right half overhangs as before; the left half is a plain vertical wall at
# RB, whose waterline clear sky forms by itself.
_rows, _cols = np.mgrid[0:N_PX, 0:N_PX]
_x, _y = _cols / PX, (N_PX - 1 - _rows) / PX
_r = np.hypot(_x - AXIS[0], _y - AXIS[1])
_right = _x > AXIS[0]
depth_h = np.where((_r < RB) | ((_r < RT) & _right), 0.0, 1.0).astype(np.float32)
_pix = np.flatnonzero(((_r > RB) & (_r < RT) & _right).ravel())
gaps_h = Gaps.from_lists((N_PX, N_PX), _pix, 1.0 - (_r.ravel()[_pix] - RB) / (RT - RB),
                         np.ones(_pix.size))


def half_cone_solid(xs, ys, zs):
    """The cone on its right half, a cylinder of radius RB on its left; the
    face between them half a pixel right of the axis, where the pixel grid
    puts it."""
    zr = zs + HC
    in_cyl = (zr >= 0) & (zs <= 0) & (np.hypot(xs - AXIS[0], ys - AXIS[1]) <= RB)
    return np.where(xs > AXIS[0] + 0.5 / PX, cone_solid(xs, ys, zs), (zr < 0) | in_cyl)


tp_h = compute_undercut_toolpath(depth_h, gaps_h, tool, 0.5, 30.0, HC, SIZE_MM, SIZE_MM,
                                 previous_result=depth_h.copy())
lens_h = [np.hypot(*np.diff(np.array(s)[:, :2], axis=0).T).sum() for s in tp_h.segments]
check("one chain per level, not fragments along the plain wall",
      len(tp_h.segments) == len({s[0][2] for s in tp_h.segments}) and min(lens_h) > 20.0)
# The overhang side is the cone test's; check the plain-wall side here.
tp_h_left = SimpleNamespace(segments=[[p for p in s if p[0] < AXIS[0]] for s in tp_h.segments])
check("one-sided overhang: no tool/part collision along the plain wall",
      collisions(tp_h_left, tool, 0.15, solid=half_cone_solid) == 0)


print("undercut toolpath: two stacked grooves")
# A pulley of radius 15 mm, 16 mm tall, with two semicircular grooves of
# radius 3.5 mm about the rim at depths 4 and 12: a 1 mm middle flange at
# depth 7.5-8.5 between them. Cutting the upper groove's lower arm puts the
# Lollipop's tip below the middle flange, while its head stays in the upper
# groove.
RP, RG, HP = 15.0, 3.5, 16.0
depth_p = np.where(_r < RP, 0.0, 1.0).astype(np.float32)
_pix = np.flatnonzero(((_r > RP - RG) & (_r < RP)).ravel())
_h = np.sqrt(RG * RG - (RP - _r.ravel()[_pix]) ** 2)
gaps_p = Gaps.from_lists((N_PX, N_PX), np.r_[_pix, _pix],
                         np.r_[4.0 - _h, 12.0 - _h] / HP, np.r_[4.0 + _h, 12.0 + _h] / HP)


def pulley_solid(xs, ys, zs):
    rr_ = np.hypot(xs - AXIS[0], ys - AXIS[1])
    d_ = -zs
    return (((rr_ <= RP) & (d_ >= 0) & (np.hypot(RP - rr_, d_ - 4.0) >= RG)
             & (np.hypot(RP - rr_, d_ - 12.0) >= RG)) | (d_ > HP))


tool_p = Tool("Lollipop", 5.0, neck_diameter_mm=2.0)
tp_p = compute_undercut_toolpath(depth_p, gaps_p, tool_p, 0.5, 40.0, HP, SIZE_MM, SIZE_MM,
                                 previous_result=depth_p.copy())
levels_p = {round(-s[0][2], 6) for s in tp_p.segments}
check("the upper groove's lower arm is cut with the tip below the middle flange",
      {9.0, 9.5, 10.0} <= levels_p)
# 0.3 mm: the rim's vertical face falls between pixel centres.
check("stacked grooves: no tool/part collision",
      collisions(tp_p, tool_p, 0.3, solid=pulley_solid) == 0)


print("undercut toolpath: stock to leave")
tp_a = compute_undercut_toolpath(depth, gaps, tool, 0.5, 30.0, HC, SIZE_MM, SIZE_MM,
                                 previous_result=prev, extra_offset_mm=0.5)
pts_a = np.array([p for s in tp_a.segments for p in s])
# The tool grown by 0.5 mm: sphere of 3.5 about the same centre, so its tip
# is 0.5 mm below the real tip; neck 0.5 mm wider. Checked to within 0.2 mm
# (under one 0.25 mm pixel): the lip's sharp corner falls between pixel
# centres.
check("stock to leave keeps the tool 0.5 mm off (sampled, grown tool)",
      len(tp_a.segments) > 0
      and collisions(tp_a, Tool("Lollipop", 7.0, neck_diameter_mm=4.0), 0.2, dz=-0.5) == 0)
check("stock to leave reaches less far",
      np.hypot(pts_a[:, 0] - AXIS[0], pts_a[:, 1] - AXIS[1]).min() > rho.min())


print("undercut toolpath: shank wider than the neck")
tool_s = lollipop(shank_diameter_mm=8.0, neck_length_mm=2.0)
tp_s = compute_undercut_toolpath(depth, gaps, tool_s, 0.5, 30.0, HC, SIZE_MM, SIZE_MM,
                                 previous_result=prev)
check("a 2 mm neck cannot reach the undercut, which starts 3.5 mm down",
      tp_s.segments == [] and any("No undercut is reachable" in w
                                   for w in tp_s.metadata['warnings']))
tool_l = lollipop(shank_diameter_mm=8.0, neck_length_mm=4.5)
tp_l = compute_undercut_toolpath(depth, gaps, tool_l, 0.5, 30.0, HC, SIZE_MM, SIZE_MM,
                                 previous_result=prev)
zs_l = [p[2] for s in tp_l.segments for p in s]
check("a 4.5 mm neck keeps the wide shank above the cone top",
      len(zs_l) > 0 and min(zs_l) >= -4.5 - 1e-6 and max(zs_l) <= -3.5 + 1e-6)


print("undercut toolpath: unreachable and empty cases")
tool_fat = lollipop(d=6.0, neck=5.6)              # reach 0.2 mm < 1 px margin
tp_f = compute_undercut_toolpath(depth, gaps, tool_fat, 0.5, 30.0, HC, SIZE_MM, SIZE_MM,
                                 previous_result=prev)
check("a neck almost as wide as the head reaches (almost) nothing", collisions(tp_f, tool_fat, 0.15) == 0)
tp_e = compute_undercut_toolpath(depth, Gaps.empty(depth.shape), tool, 0.5, 30.0, HC,
                                 SIZE_MM, SIZE_MM, previous_result=prev)
check("no gaps: no chains, with a warning", tp_e.segments == []
      and any("no undercuts" in w for w in tp_e.metadata['warnings']))

print("undercut toolpath: non-square pixels")
depth_ns = depth[:, :-1]                          # 240 x 239 px over 59.75 x 60 mm
pix = gaps.pixel_index()
keep = pix % N_PX < N_PX - 1
r_, c_ = pix[keep] // N_PX, pix[keep] % N_PX
gaps_ns = Gaps.from_lists(depth_ns.shape, r_ * (N_PX - 1) + c_, gaps.ceil[keep],
                          gaps.floor[keep])
tp_ns = compute_undercut_toolpath(depth_ns, gaps_ns, tool, 1.0, 30.0, HC,
                                  SIZE_MM * (N_PX - 1) / N_PX, SIZE_MM,
                                  previous_result=depth_ns.copy())
check("non-square pixels still produce chains", len(tp_ns.segments) > 0)
check("non-square pixels: no tool/part collision", collisions(tp_ns, tool, 0.25) == 0)
check("non-square pixels: undercut_reach on the input grid",
      tp_ns.undercut_reach.shape == depth_ns.shape and tp_ns.undercut_reach.any())


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all ok")
