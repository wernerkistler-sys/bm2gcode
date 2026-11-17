TOOL_PROFILES     = ["Ball", "End Mill", "V-Bit", "Custom",
                     "Lollipop", "T-Slot", "Dovetail"]
WOUND_DIRECTIONS  = ["Coil (Outside-In)", "Coil (Inside-Out)",
                     "Spiral (Outside-In)", "Spiral (Inside-Out)"]
RASTER_DIRECTIONS = ["Horizontal", "Vertical", "Diagonal (45°)", "Diagonal (135°)"] \
                    + WOUND_DIRECTIONS
POCKET_ZIGZAG     = ["Zig-Zag (0°)", "Zig-Zag (45°)", "Zig-Zag (90°)", "Zig-Zag (135°)"]
POCKET_PATTERNS   = ["Onion Skin (Inside-Out)", "Onion Skin (Outside-In)"] + POCKET_ZIGZAG
CUT_DIRECTIONS    = ["Either", "Climb", "Conventional"]
# Undercut: what to do with undercuts open below (label -> undercut.py's
# open_undercuts mode).
UNDERCUT_OPEN_MODES = {"Cut": "cut",
                       "Skip, back uncut": "solid",
                       "Skip, back already cut": "air"}
ALGORITHMS        = ["Raster Scan", "Contour",
                     "V-Carving (Depth Map)",
                     "V-Carving (Sub-pixel)",
                     "V-Carving (Feature Width)",
                     "V-Carving (Hybrid)",
                     "V-Carving Clean-up",
                     "Cutout", "Pocket", "Level Clearing", "Registration Holes",
                     "Undercut"]

# Which compute_*_toolpath pipeline each Algorithm dropdown entry uses.
# V-Carving Clean-up displays under the V-Carving group but is really a
# Pocket operation (see algorithms.py's compute_pocket_toolpath) with its
# Boundary offset auto-derived from a preceding V-Carving (Hybrid) step's
# tool -- every dispatch on "is this a V-Carving step" / "is this a Pocket
# step" must key off this instead of string-matching the display name.
ALGORITHM_FAMILY  = {
    "Raster Scan":               "Raster Scan",
    "Contour":                   "Contour",
    "V-Carving (Depth Map)":     "V-Carving",
    "V-Carving (Sub-pixel)":     "V-Carving",
    "V-Carving (Feature Width)": "V-Carving",
    "V-Carving (Hybrid)":        "V-Carving",
    "V-Carving Clean-up":        "Pocket",
    "Cutout":                    "Cutout",
    "Pocket":                    "Pocket",
    "Level Clearing":            "Level Clearing",
    "Registration Holes":        "Registration Holes",
    "Undercut":                  "Undercut",
}

# Rough peak memory of one step's computation per canvas pixel, on top of
# what the app already holds (measured in docs/PERFORMANCE.md). Before a step
# is computed, the erosion cache frees this much if memory is short.
STEP_PEAK_BYTES_PER_PX = {"Undercut": 300}
STEP_PEAK_BYTES_PER_PX_DEFAULT = 100

# Registration Holes flip axes (see algorithms.py's registration_flip_axis).
# The Diagonal line always passes through the G-code origin; DIAG_DIRECTIONS
# picks its direction only for a Center origin, since a corner origin implies it.
REG_HOLE_PATTERNS = ["Vertical axis", "Horizontal axis", "Diagonal"]
DIAG_DIRECTIONS   = ["+45°", "−45°"]
# Per-layer Mirror options: the same flip axes, for back-side sources (see
# layers.py's _mirror_axis_mm). Diagonals are the 45° line through the G-code
# origin in the named direction.
LAYER_MIRRORS     = ["None", "Vertical axis", "Horizontal axis",
                     "Diagonal (+45°)", "Diagonal (−45°)"]
BLEND_MODES       = ["Replace", "Merge high", "Merge low", "Add", "Subtract"]

# Process Region modes (Raster Scan, V-Carving). GATING_DEPTH_MAP_CHANGE is
# offered only where the step's depth map differs from the prior step's (see
# app.py's _gating_mode_values).
GATING_DEPTH_MAP_CHANGE = "Depth map change"
GATING_MODES      = ["Full depth map",
                     GATING_DEPTH_MAP_CHANGE,
                     "Residual from prior step",
                     "Inaccessible by prior tool",
                     "Reachable residual",
                     "Reachable inaccessible"]

DEFAULT_PREAMBLE = (
    "G21         ; units = mm\n"
    "G90         ; absolute positioning\n"
    "G17         ; XY plane\n"
    "M3 S24000   ; spindle on (clockwise) at 24000 RPM\n"
    "G4 P30      ; dwell for 30 seconds to allow spindle to reach speed\n"
    "G0 Z{z_safe:.3f}   ; move to safe height\n"
)

DEFAULT_POSTAMBLE = (
    "G0 Z{z_safe:.3f}   ; move to safe height\n"
    "G0 X0.000 Y0.000   ; return to origin\n"
    "M5   ; spindle off\n"
    "M30  ; end of program"
)

DEFAULTS = {
    "width_mm":          "100.0",
    "height_mm":         "100.0",
    "min_px_per_mm":     "5.0",
    "max_megapixels":    "16.0",
    "max_depth":         "5.0",
    "margin_mm":         "0.0",
    "margin_depth_pct":  "0.0",
    "origin":            "Bottom-Left",
    "surface":           "white",
    # Global machine / G-code settings
    "z_safe":            "5.0",
    "feed_rapid":        "2000",
    "accel_mm_s2":       "300",
    "spindle_dwell_s":   "30",   # pause after each step's S word (s); 0 = no pause
    "spindle_max_rpm":   "24000",   # cap for recommended spindle speeds; 0 = no limit
    "spindle_min_rpm":   "6000",    # floor for recommended spindle speeds; 0 = no limit
    "feed_max":          "2000",    # cap for recommended feed rates (mm/min); 0 = no limit
    "preamble":          "",    # '' → use DEFAULT_PREAMBLE
    "postamble":         "",    # '' → use DEFAULT_POSTAMBLE
    # Per-layer defaults (used for every newly added layer)
    "layer_offset_x_mm":  "0.0",
    "layer_offset_y_mm":  "0.0",
    "layer_rotation_deg": "0.0",
    "layer_scale_pct":    "100.0",
    "layer_opacity_pct":  "100.0",
    "layer_blend_mode":   "Replace",  # see BLEND_MODES
    "layer_min_depth_pct": "0.0",
    "layer_max_depth_pct": "100.0",
    "layer_visible":      "1",
    "layer_mirror":       "None",     # see LAYER_MIRRORS
    # Background depth for canvas area no source covers (Canvas tab field)
    "canvas_background_depth_pct": "0.0",
    # Per-step defaults (prefix-free; used for every new step tab)
    "step_profile":      "End Mill",
    "step_diameter":     "6.0",
    "step_stepover":     "66",
    "step_raster_offset": "0",
    "step_stepdown":              "6.0",
    "step_raster_final_stepdown": "0.0",
    "step_extra_offset":          "0.0", # stock-to-leave allowance, lateral+vertical (mm); 0 = none
    "step_feed_rate":    "800",
    "step_plunge_rate":  "0",    # mm/min for vertical descents; 0 = use feed rate
    "step_spindle_rpm":  "0",    # S word emitted at the step's start; 0 = none
    "step_raster_dir":   "Vertical",
    "step_cut_direction": "Climb",
    "step_threshold":    "0",    # mm; 0 = visit all pixels (V-Carving/Cutout: 0 = auto-split at depth-map midpoint; Pocket: literal 0, a full-stock flattening pass -- check the field's Auto box to use the depth map's actual deepest point instead)
    # V-carving-specific defaults
    "step_algorithm":    "Raster Scan",  # see ALGORITHMS list
    "step_gating_mode": "Full depth map",  # initial default; overridden per-algorithm by _refresh_layout
    "step_vbit_angle":   "60",           # included angle in degrees
    "step_min_fw":       "0.0",          # min feature width for V-carve mask (mm)
    "step_max_fw":       "0.0",          # max feature width for V-carve mask (mm)
    "step_vcarve_stepdown":       "0.0", # max step-down per roughing pass (mm); 0 = single pass
    "step_vcarve_final_stepdown": "0.0", # final finishing pass depth (mm); 0 = none
    # Contour-specific defaults
    "step_contour_z_increment":   "0.2", # Z increment between contour levels (mm)
    "step_contour_adaptive":      "0",   # 0 = fixed levels, 1 = adaptive CDF levels
    "step_contour_lateral_tol":   "0.0", # lateral tolerance for redundancy pruning (mm); 0 = disabled
    # Cutout-specific defaults
    "step_cutout_depth":          "10.0", # absolute total cut-through depth (mm); typically > max_depth
    "step_cutout_stepdown":       "3.0",  # max step-down per roughing pass (mm); 0 = single pass
    "step_cutout_final_stepdown": "0.0",  # final finishing pass depth (mm); 0 = none
    "step_cutout_extra_offset":   "0.0",  # lateral-only outward push (mm); 0 = tool tangent to silhouette edge
    # Pocket-specific defaults (target depth reuses step_threshold, relabeled "Target depth")
    "step_pocket_boundary_offset": "0.0", # lateral-only outward push (mm); 0 = tool tangent to pocket wall
    # Registration-Holes-specific defaults (helix pitch / peck depth reuses step_stepdown)
    "step_reghole_pattern":   "Vertical axis",  # see REG_HOLE_PATTERNS
    "step_reghole_diag_dir":  "+45°",           # see DIAG_DIRECTIONS; Center origin only
    "step_reghole_diameter":  "6.0",   # finished hole diameter (mm); >= tool diameter
    "step_reghole_depth":     "15.0",  # absolute hole depth (mm); typically into the spoilboard
    "step_reghole_inset":     "10.0",  # hole centre distance from the stock edge (mm)
    "step_reghole_inset_auto":   "1",  # 1 = inset follows margin/2 (10 mm with no margin)
    "step_reghole_spacing":   "80.0",  # Diagonal L arm length (mm)
    "step_reghole_spacing_auto": "1",  # 1 = min(W, H) - 2 * inset
    # Undercut-tool geometry (Lollipop / T-Slot / Dovetail, see tool.UNDERCUT_FIELDS);
    # the Undercut ring spacing reuses step_stepover
    "step_neck_diameter":     "3.0",   # neck directly above the head (mm); < head diameter
    "step_neck_length":       "15.0",  # tip to where a wider shank starts (mm)
    "step_shank_diameter":    "0.0",   # shank above the neck (mm); <= neck = no wider shank
    "step_head_height":       "2.0",   # T-Slot disc / Dovetail head thickness (mm)
    "step_dovetail_angle":    "60",    # angle between the flank and the bottom face (deg)
    "step_undercut_z_increment": "1.0",  # Undercut depth spacing between levels (mm)
    "step_undercut_open": "Cut",         # key of UNDERCUT_OPEN_MODES
    # Depth band of Raster Scan, Contour and Undercut steps (% of max depth)
    "step_z_min_pct": "0",     # cut only below this; material above counts as removed
    "step_z_max_pct": "100",   # never cut below this
    # Path-ordering
    "step_tsp_2opt":              "0",   # 0 = greedy only, 1 = apply 2-opt post-pass
    "step_avoid_air":             "0",   # 1 = drop toolpath runs that would cut only air
    # Per-step G-code dress-ups (see dressup.py) — all opt-in, disabled by default
    "step_ramp_in_enabled":     "0",
    "step_ramp_angle_deg":      "30.0",
    "step_ramp_max_length_mm":  "15.0",
    "step_ramp_start_z_mm":     "0.0",
    "step_ramp_max_depth_mm":   "0.0",
    "step_backoff_enabled":     "0",
    "step_backoff_distance_mm": "0.5",
    "step_arc_rapids_enabled":  "0",
    "step_arc_radius_mm":       "3.0",
    "step_arc_feed_mm_min":     "1200.0",
    "step_tab_enabled":                 "0",
    "step_tab_count":                   "4",
    "step_tab_width_mm":                "5.0",
    "step_tab_height_mm":  "1.0",
    "step_tab_phase_pct":               "0.0",
    "step_tab_angle_deg":                "60.0",
}

DISPLAY_ORIGINAL    = "Original"
DISPLAY_STEP_RES    = "Step Residual"
DISPLAY_STEP_RESULT = "Step Result"
DISPLAY_STEP_3D     = "Step Path (3D)"
DISPLAY_STEP_ACCESSIBLE = "Step Accessible"
DISPLAY_STEP_UNACCESSIBLE = "Step Unaccessible"
DISPLAY_PROC_MASK   = "Processing Mask"
DISPLAY_DIFF_RES_UNCUT  = "Prev Residual − Unaccessible"
DISPLAY_DIFF_UNCUT_UNCUT = "Prev Unaccessible − Unaccessible"
DISPLAY_DIFF_DEPTH_MAP  = "Depth Map Change"
DISPLAY_MODES       = [DISPLAY_STEP_3D, DISPLAY_ORIGINAL, DISPLAY_STEP_ACCESSIBLE,
                       DISPLAY_STEP_RESULT, DISPLAY_STEP_RES,
                       DISPLAY_STEP_UNACCESSIBLE,
                       DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT,
                       DISPLAY_DIFF_DEPTH_MAP, DISPLAY_PROC_MASK]
DISPLAY_3D_MODES    = {DISPLAY_STEP_3D}
DISPLAY_STEP_MODES  = {DISPLAY_STEP_RES, DISPLAY_STEP_RESULT, DISPLAY_STEP_3D,
                       DISPLAY_STEP_ACCESSIBLE,
                       DISPLAY_STEP_UNACCESSIBLE, DISPLAY_PROC_MASK,
                       DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT,
                       DISPLAY_DIFF_DEPTH_MAP}
