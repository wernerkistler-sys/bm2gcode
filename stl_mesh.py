import os
import re

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


def generate_heightmap(triangles, width, height):
    """Rasterize triangles into a heightmap via per-triangle barycentric
    z-buffering (keeps the max Z at each pixel).

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

    sx = (width  - 1) / (x_max - x_min) if x_max > x_min else 1.0
    sy = (height - 1) / (y_max - y_min) if y_max > y_min else 1.0

    heightmap = np.full((height, width), -np.inf)

    for tri in triangles:
        # Project the three vertices to pixel coordinates (float)
        px = (tri[:, 0] - x_min) * sx
        py = (tri[:, 1] - y_min) * sy

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

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z_vals = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        np.maximum.at(heightmap, (iy[inside], ix[inside]), z_vals[inside])

    # Pixels covered by no triangle stay as NaN; normalize_and_orient
    # decides how holes map into the final grayscale value.
    heightmap[np.isinf(heightmap)] = np.nan

    return heightmap


def normalize_and_orient(raw_heightmap):
    """Convert a raw rasterized heightmap (row 0 = min Y, per
    generate_heightmap) into bm2gcode's image convention: row 0 = max Y,
    float32, normalized to [0, 1], with holes (NaN) mapped to the deepest
    cut (0.0).
    """
    oriented = np.flipud(raw_heightmap)
    z_min = np.nanmin(oriented)
    z_max = np.nanmax(oriented)
    z_range = z_max - z_min
    if z_range < 1e-12:
        z_norm = np.zeros_like(oriented)
    else:
        z_norm = (oriented - z_min) / z_range
    z_norm = np.where(np.isnan(z_norm), 0.0, z_norm)
    return np.clip(z_norm, 0.0, 1.0).astype(np.float32)


def physical_size_from_bounds(x_range, y_range, z_range):
    """Convert mesh-space ranges (assumed millimeters) into bm2gcode's
    physical-size fields: (width_cm, height_cm, max_depth_mm).
    """
    return x_range / 10.0, y_range / 10.0, z_range
