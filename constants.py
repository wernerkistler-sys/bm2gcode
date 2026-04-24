TOOL_PROFILES     = ["Ball", "End Mill", "V-Bit"]
RASTER_DIRECTIONS = ["X", "Y", "Both"]
ALGORITHMS        = ["Raster Scan", "V-Carving", "V-Carving (Voronoi)", "Contour"]

DEFAULTS = {
    "width_cm":          "10.0",
    "height_cm":         "10.0",
    "max_depth":         "5.0",
    "origin":            "Bottom-Left",
    "surface":           "white",
    # Per-step defaults (prefix-free; used for every new step tab)
    "step_profile":      "End Mill",
    "step_diameter":     "6.0",
    "step_stepover":     "50",
    "step_stepdown":     "6.0",
    "step_feed_rate":    "800",
    "step_raster_dir":   "X",
    "step_threshold":    "0",    # mm; 0 = visit all pixels
    # V-carving-specific defaults
    "step_algorithm":    "Raster Scan",  # "Raster Scan" | "V-Carving"
    "step_vbit_angle":   "90",           # included angle in degrees
    "step_min_fw":       "0.0",          # min feature width for V-carve mask (mm)
    "step_vcarve_stepdown":       "0.0", # max step-down per roughing pass (mm); 0 = single pass
    "step_vcarve_final_stepdown": "0.0", # final finishing pass depth (mm); 0 = none
    # Contour-specific defaults
    "step_contour_stepdown":      "1.0", # Z increment between contour levels (mm)
}

DISPLAY_ORIGINAL    = "Original"
DISPLAY_STEP_RES    = "Step Residual"
DISPLAY_STEP_RESULT = "Step Result"
DISPLAY_STEP_3D     = "Step Path (3D)"
DISPLAY_FINE_DETAIL = "Fine Detail"
DISPLAY_VCARVE_MASK = "V-carve Mask"
DISPLAY_MODES       = [DISPLAY_ORIGINAL, DISPLAY_STEP_RES, DISPLAY_STEP_RESULT,
                       DISPLAY_STEP_3D, DISPLAY_FINE_DETAIL, DISPLAY_VCARVE_MASK]
DISPLAY_3D_MODES    = {DISPLAY_STEP_3D}
DISPLAY_STEP_MODES  = {DISPLAY_STEP_RES, DISPLAY_STEP_RESULT, DISPLAY_STEP_3D,
                       DISPLAY_FINE_DETAIL, DISPLAY_VCARVE_MASK}
