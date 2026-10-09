"""Custom cutter profiles drawn as a chain of cubic Bezier segments.

A profile is the right half of the cutter's side silhouette, in mm: r is
the distance from the spindle axis, z the height above the lowest point of
the cutting edge. It is stored as a list of nodes, ordered from the axis
outward::

    {"p": [r, z], "in": [r, z], "out": [r, z]}

``p`` is the anchor; ``in``/``out`` are the absolute positions of the
handles controlling the segment arriving at / leaving the anchor. Segment
i runs from nodes[i].p (handle nodes[i].out) to nodes[i+1].p (handle
nodes[i+1].in). The first node's ``in`` and the last node's ``out`` are
unused.

The toolpath engine never sees the Bezier nodes. It uses the profile's
*radial height curve*: the lowest z the cutting edge reaches at each
radius (the lower envelope of the drawn curve), sampled on a uniform r
grid and shifted so its minimum is 0. A 3-axis cutter can only touch
material from above, so parts of the drawn curve hidden above a lower part
at the same radius (undercuts) do not change it. Nothing here touches Tk.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

CURVE_SAMPLES   = 257   # r grid points of a radial height curve
SEGMENT_SAMPLES = 64    # points evaluated per Bezier segment
_KAPPA          = 0.5522847498   # cubic handle length for a quarter circle

PRESETS = ("Point round-over", "Ball", "V-Bit", "Flat", "Bull nose")


# ── Node handling ─────────────────────────────────────────────────────────

def make_node(p, h_in=None, h_out=None):
    p = [float(p[0]), float(p[1])]
    return {"p": p,
            "in":  [float(v) for v in (h_in if h_in is not None else p)],
            "out": [float(v) for v in (h_out if h_out is not None else p)]}


def normalize_nodes(raw):
    """A clean copy of *raw* (as loaded from JSON), or None if it is not a
    usable profile: at least two nodes, each with numeric p/in/out pairs.
    Negative radii are clamped to 0 and the first anchor is pinned to the
    axis (r = 0)."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    nodes = []
    try:
        for n in raw:
            node = {}
            for key in ("p", "in", "out"):
                v = n.get(key, n["p"])
                r, z = float(v[0]), float(v[1])
                if not (math.isfinite(r) and math.isfinite(z)):
                    return None
                node[key] = [max(0.0, r), z]
            nodes.append(node)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None
    nodes[0]["p"][0] = 0.0
    return nodes


def sample_profile(nodes, per_segment=SEGMENT_SAMPLES):
    """Points (M, 2) along the drawn curve, from the axis outward."""
    t = np.linspace(0.0, 1.0, per_segment + 1)[:, None]
    parts = []
    for i in range(len(nodes) - 1):
        p0 = np.asarray(nodes[i]["p"], dtype=np.float64)
        p1 = np.asarray(nodes[i]["out"], dtype=np.float64)
        p2 = np.asarray(nodes[i + 1]["in"], dtype=np.float64)
        p3 = np.asarray(nodes[i + 1]["p"], dtype=np.float64)
        mt = 1.0 - t
        seg = (mt ** 3) * p0 + 3 * (mt ** 2) * t * p1 + 3 * mt * (t ** 2) * p2 + (t ** 3) * p3
        parts.append(seg if i == 0 else seg[1:])
    if not parts:
        return np.zeros((0, 2))
    pts = np.vstack(parts)
    pts[:, 0] = np.maximum(pts[:, 0], 0.0)
    return pts


def radial_height(nodes, n=CURVE_SAMPLES):
    """(r, h) arrays: the lower envelope of the drawn curve on a uniform
    grid from r = 0 to the largest radius, shifted so min(h) = 0. Returns
    None when the curve has no radial extent."""
    pts = sample_profile(nodes)
    if len(pts) < 2:
        return None
    R = float(pts[:, 0].max())
    if R <= 1e-6:
        return None
    r_grid = np.linspace(0.0, R, n)
    h = np.full(n, np.inf)
    ra, za = pts[:-1, 0], pts[:-1, 1]
    rb, zb = pts[1:, 0], pts[1:, 1]
    for r0, z0, r1, z1 in zip(ra, za, rb, zb):
        lo, hi = (r0, r1) if r0 <= r1 else (r1, r0)
        i0 = int(np.searchsorted(r_grid, lo, side="left"))
        i1 = int(np.searchsorted(r_grid, hi, side="right"))
        if i1 <= i0:
            continue
        rr = r_grid[i0:i1]
        if hi - lo < 1e-12:
            zz = np.full(rr.shape, min(z0, z1))
        else:
            zz = z0 + (rr - r0) * (z1 - z0) / (r1 - r0)
        np.minimum(h[i0:i1], zz, out=h[i0:i1])
    # Grid points in a gap (a jump the sampling skipped over) take their
    # neighbours' height.
    ok = np.isfinite(h)
    if not ok.any():
        return None
    if not ok.all():
        h = np.interp(r_grid, r_grid[ok], h[ok])
    return r_grid, h - h.min()


def profile_curve(nodes):
    """The radial height curve as a hashable tuple of (r, h) pairs, or None."""
    rh = radial_height(nodes)
    if rh is None:
        return None
    return tuple((round(float(r), 6), round(float(h), 6)) for r, h in zip(*rh))


def validate(nodes):
    """Human-readable problems with *nodes* (empty list = fine)."""
    issues = []
    pts = sample_profile(nodes)
    if len(pts) < 2 or pts[:, 0].max() <= 1e-6:
        return ["The profile has no width."]
    dr = np.diff(pts[:, 0])
    if (dr < -1e-6).any():
        issues.append("Undercut: the curve turns back toward the axis. "
                      "Only its lowest edge at each radius is used.")
    z_tip = pts[0, 1]
    if z_tip > pts[:, 1].min() + 1e-6:
        issues.append("The lowest point is off the axis; Z is measured from it.")
    return issues


# ── Curve helpers (operate on the tuple from profile_curve) ──────────────

def curve_arrays(curve):
    a = np.asarray(curve, dtype=np.float64)
    return a[:, 0], a[:, 1]


def curve_radius(curve):
    return float(curve[-1][0]) if curve else 0.0


def curve_key(curve):
    """Short stable digest of *curve*, for cache keys."""
    if not curve:
        return ""
    return hashlib.sha1(repr(curve).encode("ascii")).hexdigest()[:16]


def offset_curve(curve, offset_mm):
    """*curve* grown outward by *offset_mm*: a flat ring at the tip height,
    then the original curve shifted by the offset."""
    if offset_mm <= 0.0:
        return curve
    head = ((0.0, curve[0][1]),)
    return head + tuple((round(r + offset_mm, 6), h) for r, h in curve)


def inscribed_depth(curve, half_width_mm):
    """How deep the tip sinks into a groove of half-width *half_width_mm*
    (array or scalar): the lowest height the edge has at or beyond that
    radius. Beyond the cutter's radius it saturates at the outermost
    height."""
    r, h = curve_arrays(curve)
    floor = np.minimum.accumulate(h[::-1])[::-1]
    return np.interp(half_width_mm, r, floor)


def radius_at_depth(curve, depth_mm):
    """Smallest radius at which the edge is *depth_mm* above the tip, or
    the cutter's radius when it never gets that high."""
    r, h = curve_arrays(curve)
    hm = np.maximum.accumulate(h)
    if depth_mm >= hm[-1]:
        return float(r[-1])
    i = int(np.argmax(hm >= depth_mm))
    if i == 0:
        return 0.0
    h0, h1 = hm[i - 1], hm[i]
    t = 0.0 if h1 - h0 < 1e-12 else (depth_mm - h0) / (h1 - h0)
    return float(r[i - 1] + t * (r[i] - r[i - 1]))


# ── Presets ───────────────────────────────────────────────────────────────

def _line(p0, p1):
    """Two nodes joined by a straight segment (handles at thirds)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / 3.0
    return [make_node(p0, p0, p0 + d), make_node(p1, p1 - d, p1)]


def point_roundover(diameter, height, tip_angle_deg=0.0):
    """A pointed tip whose edge curves up and out to full diameter, leaving
    horizontal at the top -- a point-cutting round-over. tip_angle_deg is
    the included angle at the point (0 = edge leaves the tip vertically);
    with height = radius and a 0° tip it is a concave quarter circle."""
    R, H = diameter / 2.0, height
    half = math.radians(max(0.0, min(178.0, tip_angle_deg)) / 2.0)
    k = _KAPPA
    out = (k * H * math.sin(half) if half > 0 else 0.0, k * H * math.cos(half))
    return [make_node((0.0, 0.0), (0.0, 0.0), out),
            make_node((R, H), (R - k * R, H), (R, H))]


def ball(diameter):
    R = diameter / 2.0
    k = _KAPPA * R
    return [make_node((0.0, 0.0), (0.0, 0.0), (k, 0.0)),
            make_node((R, R), (R, R - k), (R, R))]


def vbit(diameter, angle_deg):
    R = diameter / 2.0
    half = math.radians(max(1.0, min(179.0, angle_deg)) / 2.0)
    return _line((0.0, 0.0), (R, R / math.tan(half)))


def flat(diameter):
    return _line((0.0, 0.0), (diameter / 2.0, 0.0))


def bull_nose(diameter, radius):
    """Flat bottom with a convex rounded corner (bull nose) of *radius*."""
    R = diameter / 2.0
    rc = max(0.0, min(radius, R))
    k = _KAPPA * rc
    flat_r = R - rc
    nodes = [make_node((0.0, 0.0), (0.0, 0.0), (flat_r / 3.0, 0.0))]
    if flat_r > 1e-6:
        nodes.append(make_node((flat_r, 0.0), (2.0 * flat_r / 3.0, 0.0),
                               (flat_r + k, 0.0)))
    else:
        nodes[0]["out"] = [k, 0.0]
    nodes.append(make_node((R, rc), (R, rc - k), (R, rc)))
    return nodes
