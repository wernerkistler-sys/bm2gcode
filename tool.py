from dataclasses import dataclass


@dataclass
class Tool:
    profile: str          # "Ball" | "End Mill" | "V-Bit"
    diameter_mm: float    # tool diameter in mm
    vbit_angle_deg: float = 0.0  # included angle; only used when profile == "V-Bit"
