from dataclasses import dataclass

# Cutters whose head is wider than their neck, so they can reach under an
# overhang from the side. Only the Undercut algorithm accepts them.
UNDERCUT_PROFILES = ("Lollipop", "T-Slot", "Dovetail")

# Extra geometry fields per undercut profile, as (Tool attribute, step-field
# name) pairs. Every undercut profile has a neck and an optional wider shank.
_NECK_FIELDS = (("neck_diameter_mm", "neck_diameter"),
                ("neck_length_mm", "neck_length"),
                ("shank_diameter_mm", "shank_diameter"))
UNDERCUT_FIELDS = {
    "Lollipop": _NECK_FIELDS,
    "T-Slot":   _NECK_FIELDS + (("head_height_mm", "head_height"),),
    "Dovetail": _NECK_FIELDS + (("head_height_mm", "head_height"),
                                ("dovetail_angle_deg", "dovetail_angle")),
}


@dataclass
class Tool:
    profile: str          # "Ball" | "End Mill" | "V-Bit" | "Custom" | one of UNDERCUT_PROFILES
    diameter_mm: float    # tool diameter in mm (undercut profiles: head diameter)
    vbit_angle_deg: float = 90.0  # included angle; only used when profile == "V-Bit"
    # Radial height curve ((r, h) pairs in mm, see tool_profile.profile_curve);
    # only used when profile == "Custom".
    profile_curve: tuple | None = None
    # Undercut profiles only (see UNDERCUT_FIELDS for which apply to which):
    neck_diameter_mm: float = 0.0   # narrowest part directly above the head
    neck_length_mm: float = 0.0     # tip to where a wider shank starts
    shank_diameter_mm: float = 0.0  # shank above the neck; <= neck = no wider shank
    head_height_mm: float = 0.0     # T-Slot disc / Dovetail head thickness
    dovetail_angle_deg: float = 60.0  # angle between the flank and the bottom face


def undercut_geometry_error(tool):
    """A message describing why *tool*'s undercut geometry is unusable, or
    None if it is valid (or the tool is not an undercut profile)."""
    if tool.profile not in UNDERCUT_PROFILES:
        return None
    from math import radians, tan
    if tool.diameter_mm <= 0.0:
        return "Head diameter must be greater than 0."
    if tool.neck_diameter_mm <= 0.0:
        return "Neck diameter must be greater than 0."
    if tool.neck_diameter_mm >= tool.diameter_mm:
        return "Neck diameter must be smaller than the head diameter."
    if tool.shank_diameter_mm > tool.neck_diameter_mm and tool.neck_length_mm <= 0.0:
        return "A shank wider than the neck needs a neck length greater than 0."
    if tool.profile in ("T-Slot", "Dovetail") and tool.head_height_mm <= 0.0:
        return "Head height must be greater than 0."
    if tool.profile == "Dovetail":
        if not 0.0 < tool.dovetail_angle_deg < 90.0:
            return "Dovetail angle must be between 0° and 90°."
        top_d = tool.diameter_mm - 2.0 * tool.head_height_mm / tan(
            radians(tool.dovetail_angle_deg))
        if top_d < tool.neck_diameter_mm:
            return ("Dovetail head is narrower than the neck at its top; "
                    "reduce the head height or neck diameter.")
    return None
