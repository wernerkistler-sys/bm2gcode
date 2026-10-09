from dataclasses import dataclass


@dataclass
class Tool:
    profile: str          # "Ball" | "End Mill" | "V-Bit" | "Custom"
    diameter_mm: float    # tool diameter in mm
    vbit_angle_deg: float = 90.0  # included angle; only used when profile == "V-Bit"
    # Radial height curve ((r, h) pairs in mm, see tool_profile.profile_curve);
    # only used when profile == "Custom".
    profile_curve: tuple | None = None
