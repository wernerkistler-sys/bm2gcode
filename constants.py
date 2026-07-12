TOOL_PROFILES     = ["Ball", "End Mill", "V-Bit"]
RASTER_DIRECTIONS = ["Horizontal", "Vertical"]
CUT_DIRECTIONS    = ["Either", "Climb", "Conventional"]
ALGORITHMS        = ["Raster Scan", "Contour",
                     "V-Carving (Depth Map)",
                     "V-Carving (Sub-pixel)",
                     "V-Carving (Feature Width)"]

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
    "width_cm":          "10.0",
    "height_cm":         "10.0",
    "max_depth":         "5.0",
    "margin_cm":         "0.0",
    "margin_depth_pct":  "0.0",
    "origin":            "Bottom-Left",
    "surface":           "white",
    # Global machine / G-code settings
    "z_safe":            "5.0",
    "feed_rapid":        "1500",
    "accel_mm_s2":       "300",
    "preamble":          "",    # '' → use DEFAULT_PREAMBLE
    "postamble":         "",    # '' → use DEFAULT_POSTAMBLE
    # Per-step defaults (prefix-free; used for every new step tab)
    "step_profile":      "End Mill",
    "step_diameter":     "6.0",
    "step_stepover":     "66",
    "step_raster_offset": "0",
    "step_stepdown":              "6.0",
    "step_raster_final_stepdown": "0.0",
    "step_feed_rate":    "800",
    "step_raster_dir":   "Vertical",
    "step_cut_direction": "Either",
    "step_threshold":    "0",    # mm; 0 = visit all pixels
    # V-carving-specific defaults
    "step_algorithm":    "Raster Scan",  # see ALGORITHMS list
    "step_gating_mode": "Full depth map",  # initial default; overridden per-algorithm by _refresh_layout
    "step_vbit_angle":   "60",           # included angle in degrees
    "step_min_fw":       "0.0",          # min feature width for V-carve mask (mm)
    "step_max_fw":       "0.0",          # max feature width for V-carve mask (mm)
    "step_vcarve_stepdown":       "0.0", # max step-down per roughing pass (mm); 0 = single pass
    "step_vcarve_final_stepdown": "0.0", # final finishing pass depth (mm); 0 = none
    # Contour-specific defaults
    "step_contour_stepdown":      "0.2", # Z increment between contour levels (mm)
    "step_contour_adaptive":      "0",   # 0 = fixed levels, 1 = adaptive CDF levels
    "step_contour_lateral_tol":   "0.0", # lateral tolerance for redundancy pruning (mm); 0 = disabled
    # Path-ordering
    "step_tsp_2opt":              "0",   # 0 = greedy only, 1 = apply 2-opt post-pass
}

DISPLAY_ORIGINAL    = "Original"
DISPLAY_STEP_RES    = "Step Residual"
DISPLAY_STEP_RESULT = "Step Result"
DISPLAY_STEP_3D     = "Step Path (3D)"
DISPLAY_STEP_UNACCESSIBLE = "Step Unaccessible"
DISPLAY_PROC_MASK   = "Processing Mask"
DISPLAY_DIFF_RES_UNCUT  = "Prev Residual − Unaccessible"
DISPLAY_DIFF_UNCUT_UNCUT = "Prev Unaccessible − Unaccessible"
DISPLAY_MODES       = [DISPLAY_ORIGINAL, DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
                       DISPLAY_STEP_3D, DISPLAY_STEP_UNACCESSIBLE,
                       DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT,
                       DISPLAY_PROC_MASK]
DISPLAY_3D_MODES    = {DISPLAY_STEP_3D}
DISPLAY_STEP_MODES  = {DISPLAY_STEP_RES, DISPLAY_STEP_RESULT, DISPLAY_STEP_3D,
                       DISPLAY_STEP_UNACCESSIBLE, DISPLAY_PROC_MASK,
                       DISPLAY_DIFF_RES_UNCUT, DISPLAY_DIFF_UNCUT_UNCUT}
