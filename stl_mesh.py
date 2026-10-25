import os
import re
from dataclasses import dataclass

import numpy as np


def load_stl(path):
    """Parse a binary or ASCII STL file.

    Returns an (N,3,3) float64 array: N triangles, 3 vertices each, xyz.
    Raises ValueError if the file has no triangles or can't be parsed.
    """
    triangles = _load_stl_native(path)
    if len(triangles) == 0:
        raise ValueError("STL file contains no triangles.")
    return triangles.astype(np.float64)


def _load_stl_native(path):
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        f.read(80)  # header
        count_bytes = f.read(4)
    if len(count_bytes) == 4:
        n = int.from_bytes(count_bytes, 'little')
        expected = 80 + 4 + n * 50
        if file_size == expected and n > 0:
            return _read_binary_stl(path, n)
    return _read_ascii_stl(path)


def _read_binary_stl(path, n):
    dtype = np.dtype([
        ('normal', np.float32, (3,)),
        ('v0',     np.float32, (3,)),
        ('v1',     np.float32, (3,)),
        ('v2',     np.float32, (3,)),
        ('attr',   np.uint16),
    ])
    with open(path, 'rb') as f:
        f.seek(84)
        records = np.frombuffer(f.read(n * 50), dtype=dtype)
    return np.stack([records['v0'], records['v1'], records['v2']], axis=1)


def _read_ascii_stl(path):
    with open(path, 'r', errors='replace') as f:
        content = f.read()
    matches = re.findall(r'vertex\s+([\S]+)\s+([\S]+)\s+([\S]+)', content)
    if not matches:
        raise ValueError("Cannot parse STL file: not valid binary or ASCII format.")
    verts = np.array([[float(x), float(y), float(z)] for x, y, z in matches],
                      dtype=np.float32)
    return verts.reshape(-1, 3, 3)


def mesh_xy_bounds(triangles):
    """Return (x_min, x_max, y_min, y_max) over all vertices."""
    verts = triangles.reshape(-1, 3)
    return (float(verts[:, 0].min()), float(verts[:, 0].max()),
            float(verts[:, 1].min()), float(verts[:, 1].max()))


def mesh_z_bounds(triangles):
    """Return (z_min, z_max) over all vertices."""
    verts = triangles.reshape(-1, 3)
    return float(verts[:, 2].min()), float(verts[:, 2].max())


# Exact (sin, cos) for quarter-turn angles -- avoids floating-point noise
# from calling sin()/cos() directly at these angles.
_SIN_COS = {0: (0.0, 1.0), 90: (1.0, 0.0), 180: (0.0, -1.0), 270: (-1.0, 0.0)}


def _axis_matrix(axis, degrees):
    s, c = _SIN_COS[degrees % 360]
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rotation_matrix(rot_x=0, rot_y=0, rot_z=0):
    """Combined rotation matrix for quarter-turn rotations applied around
    X, then Y, then Z, each about the original (fixed) axes. Each angle
    must be one of 0, 90, 180, 270.
    """
    return _axis_matrix('z', rot_z) @ _axis_matrix('y', rot_y) @ _axis_matrix('x', rot_x)


def apply_rotation(triangles, rot_x=0, rot_y=0, rot_z=0):
    """Rotate mesh vertices by quarter turns around X, Y, and/or Z.

    Every combination is a proper rotation (signed permutation matrix), so
    triangle winding needs no correction for the max-Z rasterization in
    generate_heightmap.
    """
    if rot_x % 360 == 0 and rot_y % 360 == 0 and rot_z % 360 == 0:
        return triangles
    R = _rotation_matrix(rot_x, rot_y, rot_z)
    return triangles @ R.T


def rotate_bounds(xy_bounds, z_bounds, rot_x=0, rot_y=0, rot_z=0):
    """Rotate an axis-aligned bounding box by the same quarter-turn
    rotation as apply_rotation, without needing the full triangle array.

    Exact (not approximate): because the rotation is a signed permutation,
    transforming the box's 8 corners and re-taking min/max reproduces the
    true rotated bounding box.
    """
    if rot_x % 360 == 0 and rot_y % 360 == 0 and rot_z % 360 == 0:
        return xy_bounds, z_bounds
    x_min, x_max, y_min, y_max = xy_bounds
    z_min, z_max = z_bounds
    corners = np.array([[x, y, z]
                         for x in (x_min, x_max)
                         for y in (y_min, y_max)
                         for z in (z_min, z_max)], dtype=np.float64)
    R = _rotation_matrix(rot_x, rot_y, rot_z)
    transformed = corners @ R.T
    new_min = transformed.min(axis=0)
    new_max = transformed.max(axis=0)
    return ((float(new_min[0]), float(new_max[0]),
             float(new_min[1]), float(new_max[1])),
            (float(new_min[2]), float(new_max[2])))


_EDGE_TOL = 1e-9   # barycentric slack for rays through triangle edges


def _cell_scale(lo, hi, n):
    """(scale, shift) mapping a mesh coordinate c to the pixel index
    (c - lo) * scale - shift: *n* pixel cells tile [lo, hi], each sampled at
    its centre, as every canvas pixel is a cell 1 / px_per_mm wide. A flat
    extent puts its one coordinate on pixel 0."""
    if hi > lo:
        return n / (hi - lo), 0.5
    return 1.0, 0.0


def generate_heightmap(triangles, width, height):
    """Rasterize triangles into a heightmap via per-triangle barycentric
    z-buffering (keeps the max Z at each pixel). The pixels are cells tiling
    the mesh's XY bounds, each sampled at its centre (_cell_scale).

    Returns a 2-D float64 (height, width) array of raw Z values; pixels
    covered by no triangle are NaN. Row 0 corresponds to the mesh's
    minimum Y (native rasterization convention) -- callers needing
    bm2gcode's row-0-is-max-Y convention must flip the result, see
    normalize_and_orient.
    """
    if width < 2 or height < 2:
        raise ValueError("Width and height must each be at least 2.")
    verts = triangles.reshape(-1, 3)
    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())

    sx, hx = _cell_scale(x_min, x_max, width)
    sy, hy = _cell_scale(y_min, y_max, height)

    heightmap = np.full((height, width), -np.inf)

    for tri in triangles:
        # Project the three vertices to pixel coordinates (float)
        px = (tri[:, 0] - x_min) * sx - hx
        py = (tri[:, 1] - y_min) * sy - hy

        ix0 = max(0,         int(np.floor(px.min())))
        ix1 = min(width - 1, int(np.ceil( px.max())))
        iy0 = max(0,         int(np.floor(py.min())))
        iy1 = min(height - 1,int(np.ceil( py.max())))
        if ix0 > ix1 or iy0 > iy1:
            continue

        denom = ((py[1] - py[2]) * (px[0] - px[2]) +
                 (px[2] - px[1]) * (py[0] - py[2]))
        if abs(denom) < 1e-10:
            continue

        iy, ix = np.mgrid[iy0:iy1 + 1, ix0:ix1 + 1]
        fx, fy = ix.astype(np.float64), iy.astype(np.float64)

        w0 = ((py[1] - py[2]) * (fx - px[2]) +
              (px[2] - px[1]) * (fy - py[2])) / denom
        w1 = ((py[2] - py[0]) * (fx - px[2]) +
              (px[0] - px[2]) * (fy - py[2])) / denom
        w2 = 1.0 - w0 - w1

        # A ray through an edge is taken by every triangle sharing it, so
        # rounding can't drop it from the upper of two faces meeting there,
        # as on a silhouette edge where a top and a bottom face meet.
        inside = (w0 >= -_EDGE_TOL) & (w1 >= -_EDGE_TOL) & (w2 >= -_EDGE_TOL)
        if not inside.any():
            continue

        z_vals = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        np.maximum.at(heightmap, (iy[inside], ix[inside]), z_vals[inside])

    # Pixels covered by no triangle stay as NaN; normalize_and_orient
    # decides how holes map into the final grayscale value.
    heightmap[np.isinf(heightmap)] = np.nan

    return heightmap


@dataclass
class Dexels:
    """Every surface crossing of each pixel's vertical ray through a mesh.

    Column i (row-major, row 0 = max Y, as normalize_and_orient orients a
    heightmap) holds z[offsets[i]:offsets[i+1]] in descending order. Walking
    down, the hits alternate entering and leaving the solid, so hits
    (2m, 2m+1) bound the m-th solid interval. An odd count means the last
    solid has no bottom: it runs down without end. Columns whose crossings
    are inconsistent (an open or self-intersecting mesh) keep only their top
    hit, which reads as solid from the top down -- the safe assumption.
    """
    shape: tuple
    offsets: np.ndarray       # int64, H*W + 1
    z: np.ndarray             # float64, descending within each column
    top: np.ndarray           # float64 (H, W): highest raw hit, NaN where none --
                              # exactly generate_heightmap's value
    n_invalid: int = 0        # columns reduced to their top hit
    z_bounds: tuple = None    # (z_min, z_max) of the mesh's vertices, the
                              # range normalize_and_orient maps to [0, 1]


def generate_dexels(triangles, width, height, merge_eps=None):
    """Rasterize triangles into Dexels on the same pixel grid as
    generate_heightmap (pixel cells tiling the mesh's XY bounds, sampled at
    their centres).

    A triangle facing up is where a downward ray enters the solid, one
    facing down is where it leaves; facing comes from the triangle's
    winding, flipped globally if most columns would otherwise start by
    leaving the solid (a mesh wound inside-out). Hits within *merge_eps* of
    each other are merged when they face the same way (a shared edge hit
    by both neighbours) and cancelled when they face opposite ways (the ray
    grazing a fold).
    """
    if width < 2 or height < 2:
        raise ValueError("Width and height must each be at least 2.")
    verts = triangles.reshape(-1, 3)
    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
    z_min, z_max = float(verts[:, 2].min()), float(verts[:, 2].max())
    if merge_eps is None:
        merge_eps = max(z_max - z_min, 1e-9) * 1e-7

    sx, hx = _cell_scale(x_min, x_max, width)
    sy, hy = _cell_scale(y_min, y_max, height)

    pix_parts, z_parts, sgn_parts = [], [], []
    for tri in triangles:
        px = (tri[:, 0] - x_min) * sx - hx
        py = (tri[:, 1] - y_min) * sy - hy

        ix0 = max(0,         int(np.floor(px.min())))
        ix1 = min(width - 1, int(np.ceil( px.max())))
        iy0 = max(0,         int(np.floor(py.min())))
        iy1 = min(height - 1, int(np.ceil(py.max())))
        if ix0 > ix1 or iy0 > iy1:
            continue

        denom = ((py[1] - py[2]) * (px[0] - px[2]) +
                 (px[2] - px[1]) * (py[0] - py[2]))
        if abs(denom) < 1e-10:
            continue

        iy, ix = np.mgrid[iy0:iy1 + 1, ix0:ix1 + 1]
        fx, fy = ix.astype(np.float64), iy.astype(np.float64)
        w0 = ((py[1] - py[2]) * (fx - px[2]) +
              (px[2] - px[1]) * (fy - py[2])) / denom
        w1 = ((py[2] - py[0]) * (fx - px[2]) +
              (px[0] - px[2]) * (fy - py[2])) / denom
        w2 = 1.0 - w0 - w1
        # A ray through an edge is taken by both neighbouring triangles (the
        # duplicate is merged below) rather than risk rounding dropping it
        # from both, as in generate_heightmap.
        inside = (w0 >= -_EDGE_TOL) & (w1 >= -_EDGE_TOL) & (w2 >= -_EDGE_TOL)
        if not inside.any():
            continue

        z_vals = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        # Row-flipped so row 0 is max Y, matching normalize_and_orient.
        pix = (height - 1 - iy[inside]).astype(np.int64) * width + ix[inside]
        pix_parts.append(pix)
        z_parts.append(z_vals[inside])
        # In pixel space (both axes scaled positively) denom has the sign of
        # the triangle's XY winding, i.e. of its normal's Z component.
        sgn_parts.append(np.full(pix.size, 1 if denom > 0 else -1, np.int8))

    n_px = width * height
    top = np.full(n_px, -np.inf)
    if not pix_parts:
        return Dexels((height, width), np.zeros(n_px + 1, np.int64),
                      np.zeros(0, np.float64), np.full((height, width), np.nan), 0,
                      (z_min, z_max))
    pix = np.concatenate(pix_parts)
    z = np.concatenate(z_parts)
    sgn = np.concatenate(sgn_parts)
    np.maximum.at(top, pix, z)
    top[np.isinf(top)] = np.nan

    order = np.lexsort((-z, pix))
    pix, z, sgn = pix[order], z[order], sgn[order]

    # Resolve each cluster of coincident hits (a ray through a shared edge
    # or vertex, a grazed fold, two shells touching) independently of their
    # order: hits facing one way collapse to one crossing, hits facing both
    # ways cancel to none.
    new_group = np.ones(pix.size, bool)
    new_group[1:] = (pix[1:] != pix[:-1]) | (z[:-1] - z[1:] > merge_eps)
    starts = np.flatnonzero(new_group)
    has_up = np.maximum.reduceat((sgn > 0).astype(np.int8), starts) > 0
    has_dn = np.maximum.reduceat((sgn < 0).astype(np.int8), starts) > 0
    keep = has_up != has_dn
    # A column's highest cluster cancelling (a ray grazing the silhouette at
    # a vertex or edge) still sets the heightmap's top there: it
    # becomes a solid of zero thickness, entered and left at that z, so the
    # column keeps that top and the air below it.
    g_pix = pix[starts]
    first_group = np.r_[True, g_pix[1:] != g_pix[:-1]]
    graze = first_group & has_up & has_dn
    graze_pix, graze_z = g_pix[graze], z[starts][graze]
    pix, z = g_pix[keep], z[starts][keep]
    sgn = np.where(has_up, 1, -1).astype(np.int8)[keep]

    counts = np.bincount(pix, minlength=n_px)
    offsets = np.zeros(n_px + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    rank = np.arange(pix.size) - offsets[pix]

    # Inside-out mesh: most columns would start by leaving the solid.
    first = rank == 0
    if np.count_nonzero(sgn[first] < 0) > np.count_nonzero(sgn[first] > 0):
        sgn = -sgn

    if graze_pix.size:
        pix = np.concatenate([pix, graze_pix, graze_pix])
        z = np.concatenate([z, graze_z, graze_z])
        sgn = np.concatenate([sgn, np.ones(graze_pix.size, np.int8),
                              -np.ones(graze_pix.size, np.int8)])
        order = np.lexsort((-sgn, -z, pix))     # entering before leaving
        pix, z, sgn = pix[order], z[order], sgn[order]
        counts = np.bincount(pix, minlength=n_px)
        offsets = np.zeros(n_px + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        rank = np.arange(pix.size) - offsets[pix]
        first = rank == 0

    expected = np.where(rank % 2 == 0, 1, -1)
    bad_hit = sgn != expected
    bad_col = np.zeros(n_px, bool)
    bad_col[pix[bad_hit]] = True
    n_invalid = int(np.count_nonzero(bad_col))
    if n_invalid:
        keep = ~bad_col[pix] | first
        pix, z = pix[keep], z[keep]
        counts = np.bincount(pix, minlength=n_px)
        offsets = np.zeros(n_px + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])

    return Dexels((height, width), offsets, z.astype(np.float64),
                  top.reshape(height, width), n_invalid, (z_min, z_max))


def normalize_and_orient(raw_heightmap, z_bounds=None):
    """Convert a raw rasterized heightmap (row 0 = min Y, per
    generate_heightmap) into bm2gcode's image convention: row 0 = max Y,
    float32, normalized to [0, 1].

    *z_bounds* (z_min, z_max) is the range mapped to [0, 1]: the mesh's
    bounding box (mesh_z_bounds of the rasterized triangles), so the mesh's
    lowest point lands on 0 -- Max cut depth, as the import dialog sizes it
    -- even where no ray sees it from above, as under a cube's flat top.
    None uses the heightmap's own range.

    Returns (gray, alpha): holes (NaN, pixels no triangle covered) are
    excluded via alpha=0.0 (gray is set to 0.0 there too, but is moot since
    zero alpha means the layer compositor ignores it entirely) rather than
    being silently treated as real geometry -- a hole is "no data", not "cut
    to the deepest point", so it should let whatever is beneath show through
    (e.g. under "Replace" blend mode) instead of gouging to max depth. alpha
    is None (not an all-ones array) when the mesh has no holes at all, so a
    fully-covered mesh doesn't pay for per-pixel alpha compositing it
    doesn't need.
    """
    oriented = np.flipud(raw_heightmap)
    has_holes = np.isnan(oriented).any()
    alpha = (~np.isnan(oriented)).astype(np.float32) if has_holes else None
    if z_bounds is None:
        z_min, z_max = np.nanmin(oriented), np.nanmax(oriented)
    else:
        z_min, z_max = z_bounds
    z_range = z_max - z_min
    if z_range < 1e-12:
        z_norm = np.zeros_like(oriented)
    else:
        z_norm = (oriented - z_min) / z_range
    z_norm = np.where(np.isnan(z_norm), 0.0, z_norm)
    return np.clip(z_norm, 0.0, 1.0).astype(np.float32), alpha


def physical_size_from_bounds(x_range, y_range, z_range):
    """Convert mesh-space ranges (assumed millimeters) into bm2gcode's
    physical-size fields: (width_mm, height_mm, max_depth_mm).
    """
    return x_range, y_range, z_range
