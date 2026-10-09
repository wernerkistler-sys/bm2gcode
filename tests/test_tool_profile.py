"""Plain assert-based tests for custom Bezier tool profiles
(tool_profile.py) and their use as a tool kernel (algorithms._build_kernel).

Run with: python tests/test_tool_profile.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import tool_profile as tp
from algorithms import _build_kernel, _planning_kernel
from tool import Tool

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


def custom_tool(nodes):
    curve = tp.profile_curve(nodes)
    return Tool("Custom", 2.0 * tp.curve_radius(curve), 90.0, curve)


# --------------------------------------------------------------------------
print("radial height curve")

r, h = tp.radial_height(tp.ball(6.0))
check("ball preset follows the sphere within 0.01 mm",
      np.abs(h - (3.0 - np.sqrt(9.0 - r ** 2))).max() < 0.01)
r, h = tp.radial_height(tp.vbit(10.0, 60.0))
check("V-bit preset is a straight cone",
      np.abs(h - r / math.tan(math.radians(30.0))).max() < 1e-6)
r, h = tp.radial_height(tp.point_roundover(12.0, 6.0))
check("point round-over (0° tip, height = radius) is a concave quarter circle",
      np.abs(h - np.sqrt(36.0 - (6.0 - r) ** 2)).max() < 0.01)
r, h = tp.radial_height(tp.flat(8.0))
check("flat preset has zero height everywhere", r[-1] == 4.0 and np.abs(h).max() < 1e-9)

# An S-shaped edge that turns back toward the axis (undercut): the
# envelope keeps only the lowest z at each radius and stays a function of r.
undercut = [tp.make_node((0, 0), (0, 0), (6, 0)),
            tp.make_node((2, 4), (6, 4), (-2, 4)),
            tp.make_node((4, 6), (2, 6), (4, 6))]
check("undercut is reported", any("Undercut" in s for s in tp.validate(undercut)))
r, h = tp.radial_height(undercut)
pts = tp.sample_profile(undercut)
check("undercut envelope never above the drawn curve at the same radius",
      all(np.interp(pr, r, h) <= pz + 1e-3 for pr, pz in pts[::8]))
check("clean profile has no issues", tp.validate(tp.ball(6.0)) == [])

lifted = [tp.make_node((0, 2), (0, 2), (1, 2)), tp.make_node((3, 0), (2, 0), (3, 0))]
r, h = tp.radial_height(lifted)
check("envelope is shifted so its lowest point is 0", abs(h.min()) < 1e-9 and h[0] > 1.9)

# --------------------------------------------------------------------------
print("curve helpers")

c = tp.profile_curve(tp.point_roundover(12.0, 6.0))
check("curve key is stable", tp.curve_key(c) == tp.curve_key(tp.profile_curve(
    tp.point_roundover(12.0, 6.0))))
check("curve key differs for different curves",
      tp.curve_key(c) != tp.curve_key(tp.profile_curve(tp.ball(12.0))))
check("radius_at_depth inverts the edge",
      abs(tp.radius_at_depth(c, 3.0) - (6.0 - math.sqrt(27.0))) < 0.02)
check("radius_at_depth saturates at the cutter radius", tp.radius_at_depth(c, 50.0) == 6.0)
check("inscribed depth beyond the radius saturates at the edge height",
      abs(float(tp.inscribed_depth(c, 20.0)) - 6.0) < 1e-6)
off = tp.offset_curve(c, 1.0)
check("offset curve adds a flat ring at the tip",
      off[0] == (0.0, 0.0) and off[1] == (1.0, 0.0) and tp.curve_radius(off) == 7.0)

check("normalize rejects garbage", tp.normalize_nodes([{"p": [0, 0]}]) is None
      and tp.normalize_nodes("x") is None
      and tp.normalize_nodes([{"p": [0, 0]}, {"p": ["a", 1]}]) is None)
n = tp.normalize_nodes([{"p": [0.5, 0]}, {"p": [3, 2]}])
check("normalize pins the first anchor to the axis and fills handles",
      n[0]["p"] == [0.0, 0.0] and n[1]["in"] == [3.0, 2.0])

# --------------------------------------------------------------------------
print("kernel")

for px_per_mm in (5.0, 10.0):
    k_ball, f_ball = _build_kernel(Tool("Ball", 6.0), px_per_mm, 5.0)
    k_cust, f_cust = _build_kernel(custom_tool(tp.ball(6.0)), px_per_mm, 5.0)
    # The Bezier quarter circle is ~0.003 mm off in r; near the rim, where
    # the sphere turns vertical, that is a large error in h, so compare
    # inside 95 % of the radius.
    kk = k_ball.shape[0] // 2
    yy, xx = np.mgrid[-kk:kk + 1, -kk:kk + 1]
    inside = f_ball & f_cust & (np.hypot(yy, xx) / px_per_mm <= 0.95 * 3.0)
    check(f"custom ball kernel matches Ball at {px_per_mm:g} px/mm",
          k_ball.shape == k_cust.shape and (f_ball == f_cust).all()
          and np.abs(k_ball[inside] - k_cust[inside]).max() < 0.01 / 5.0)

    k_v, f_v = _build_kernel(Tool("V-Bit", 6.0, 60.0), px_per_mm, 50.0)
    k_cv, f_cv = _build_kernel(custom_tool(tp.vbit(6.0, 60.0)), px_per_mm, 50.0)
    inside = f_v & f_cv
    check(f"custom V kernel matches V-Bit at {px_per_mm:g} px/mm",
          k_v.shape == k_cv.shape and np.abs(k_v[inside] - k_cv[inside]).max() < 1e-4)

tool = custom_tool(tp.point_roundover(6.0, 3.0))
k_real, f_real = _build_kernel(tool, 10.0, 5.0)
k_plan, f_plan = _planning_kernel(tool, 10.0, 5.0, 1.0, k_real, f_real)
check("planning kernel for Custom grows by the extra offset",
      k_plan.shape[0] == k_real.shape[0] + 20 and k_plan[k_plan.shape[0] // 2,
                                                         k_plan.shape[0] // 2 + 10] == 0.0)

try:
    _build_kernel(Tool("Custom", 6.0), 10.0, 5.0)
    check("Custom without a curve raises", False)
except ValueError:
    check("Custom without a curve raises", True)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All tests passed.")
