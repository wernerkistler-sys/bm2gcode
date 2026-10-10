"""User-wide library of cutting tools and per-material cutting data.

The library lives in one JSON file outside any project (see
``default_path``). Steps copy values out of it and remember which tool they
came from; projects never depend on the library being present. Nothing here
touches Tk -- the step-field mapping and sync check are plain functions so
they can be tested on their own.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass, field

from tool import Tool
import tool_profile

LIBRARY_VERSION = 1

# Status values returned by sync_state.
SYNC_IN_SYNC          = "in_sync"
SYNC_GEOMETRY_DIFFERS = "geometry_differs"
SYNC_MODIFIED         = "modified"
SYNC_MISSING          = "missing"
SYNC_NO_DATA          = "no_data"

# Every step-down field a library value is written to.
STEPDOWN_FIELDS = ('stepdown', 'vcarve_stepdown', 'cutout_stepdown')

# Cutter diameters (mm) a material's feed-per-tooth row is given for.
FZ_DIAMETERS_MM = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)

# Plunge rate recommended as a fraction of the recommended feed rate.
PLUNGE_FRACTION = 1.0 / 3.0

# Reference cutting data per material from the Stepcraft "Fräsparameter"
# sheet: cutting speed vc (m/min), feed per tooth fz (mm) for each of
# FZ_DIAMETERS_MM, and the slot depth per pass as a multiple of the cutter
# diameter (the sheet's conservative D-series values: 0.05 for non-ferrous
# metals, 0.5 for wood and plastics; steel, which the slot-depth table
# doesn't list, takes the metal value).
MATERIAL_PRESETS = {
    "Cast aluminium (>6 % Si)": (200, (0.010, 0.010, 0.010, 0.015, 0.015, 0.025, 0.030, 0.038, 0.050), 0.05),
    "Wrought aluminium":        (500, (0.010, 0.020, 0.025, 0.050, 0.050, 0.050, 0.064, 0.080, 0.100), 0.05),
    "Soft plastic":             (600, (0.025, 0.030, 0.035, 0.045, 0.065, 0.090, 0.100, 0.200, 0.300), 0.5),
    "Hard plastic":             (550, (0.015, 0.020, 0.025, 0.050, 0.060, 0.080, 0.089, 0.100, 0.150), 0.5),
    "Hardwood":                 (450, (0.020, 0.025, 0.030, 0.055, 0.065, 0.085, 0.095, 0.095, 0.155), 0.5),
    "Softwood":                 (500, (0.025, 0.030, 0.035, 0.060, 0.070, 0.090, 0.100, 0.110, 0.160), 0.5),
    "MDF":                      (450, (0.050, 0.070, 0.100, 0.150, 0.200, 0.300, 0.400, 0.500, 0.600), 0.5),
    "Brass, copper, bronze":    (365, (0.015, 0.020, 0.025, 0.025, 0.030, 0.050, 0.056, 0.065, 0.080), 0.05),
    "Steel":                    (75,  (0.010, 0.010, 0.012, 0.025, 0.030, 0.038, 0.045, 0.050, 0.080), 0.05),
}


@dataclass
class CuttingData:
    """Cutting data for one (tool, material) pair."""
    feed_rate: float          # mm/min
    stepdown_mm: float        # max step-down per pass
    plunge_rate: float = 0.0  # mm/min; 0 = use feed_rate
    spindle_rpm: float = 0.0  # 0 = don't emit S

    def to_dict(self):
        return {'feed_rate': self.feed_rate, 'stepdown_mm': self.stepdown_mm,
                'plunge_rate': self.plunge_rate, 'spindle_rpm': self.spindle_rpm}

    @classmethod
    def from_dict(cls, d):
        return cls(feed_rate=float(d['feed_rate']),
                   stepdown_mm=float(d['stepdown_mm']),
                   plunge_rate=float(d.get('plunge_rate', 0.0)),
                   spindle_rpm=float(d.get('spindle_rpm', 0.0)))


@dataclass
class ToolDef:
    """One physical cutter: its geometry plus cutting data per material."""
    id: str
    name: str
    profile: str                      # "Ball" | "End Mill" | "V-Bit" | "Custom"
    diameter_mm: float
    vbit_angle_deg: float = 90.0
    flute_length_mm: float = 0.0
    shank_diameter_mm: float = 0.0    # 0 = same as diameter_mm
    overall_length_mm: float = 0.0    # 0 = not specified
    flutes: int = 2                   # number of cutting edges (z)
    number: int | None = None         # tool number for a future M6 Tn
    notes: str = ""
    profile_points: list | None = None   # Bezier nodes of a Custom profile
    cutting: dict = field(default_factory=dict)   # material_id -> CuttingData

    def to_tool(self) -> Tool:
        if self.profile == "Custom" and self.profile_points:
            curve = tool_profile.profile_curve(self.profile_points)
            if curve is not None:
                return Tool(self.profile, 2.0 * tool_profile.curve_radius(curve),
                            self.vbit_angle_deg, curve)
        return Tool(self.profile, self.diameter_mm, self.vbit_angle_deg)

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'profile': self.profile,
            'diameter_mm': self.diameter_mm, 'vbit_angle_deg': self.vbit_angle_deg,
            'flute_length_mm': self.flute_length_mm,
            'shank_diameter_mm': self.shank_diameter_mm,
            'overall_length_mm': self.overall_length_mm, 'flutes': self.flutes,
            'number': self.number,
            'notes': self.notes, 'profile_points': self.profile_points,
            'cutting': {mid: cd.to_dict() for mid, cd in self.cutting.items()},
        }

    @classmethod
    def from_dict(cls, d):
        number = d.get('number')
        return cls(
            id=str(d['id']), name=str(d['name']), profile=str(d['profile']),
            diameter_mm=float(d['diameter_mm']),
            vbit_angle_deg=float(d.get('vbit_angle_deg', 90.0)),
            flute_length_mm=float(d.get('flute_length_mm', 0.0)),
            shank_diameter_mm=float(d.get('shank_diameter_mm', 0.0)),
            overall_length_mm=float(d.get('overall_length_mm', 0.0)),
            flutes=max(1, int(d.get('flutes', 2))),
            number=int(number) if number is not None else None,
            notes=str(d.get('notes', '')),
            profile_points=tool_profile.normalize_nodes(d.get('profile_points')),
            cutting={str(mid): CuttingData.from_dict(cd)
                     for mid, cd in (d.get('cutting') or {}).items()},
        )


@dataclass
class Material:
    """A workpiece material plus the reference data cutting recommendations
    are computed from (see ``recommend_cutting``)."""
    id: str
    name: str
    cutting_speed_m_min: float = 0.0        # vc; 0 = no reference data
    feed_per_tooth_mm: list | None = None   # fz per FZ_DIAMETERS_MM entry
    stepdown_factor: float = 0.0            # step-down per pass as a multiple of Ø

    def has_reference(self) -> bool:
        return (self.cutting_speed_m_min > 0 and self.feed_per_tooth_mm is not None
                and len(self.feed_per_tooth_mm) == len(FZ_DIAMETERS_MM)
                and all(fz > 0 for fz in self.feed_per_tooth_mm))

    def apply_preset(self, preset_name):
        """Copy the reference data of a MATERIAL_PRESETS entry."""
        vc, fz, factor = MATERIAL_PRESETS[preset_name]
        self.cutting_speed_m_min = float(vc)
        self.feed_per_tooth_mm = [float(x) for x in fz]
        self.stepdown_factor = float(factor)

    def to_dict(self):
        return {'id': self.id, 'name': self.name,
                'cutting_speed_m_min': self.cutting_speed_m_min,
                'feed_per_tooth_mm': self.feed_per_tooth_mm,
                'stepdown_factor': self.stepdown_factor}

    @classmethod
    def from_dict(cls, d):
        fz = d.get('feed_per_tooth_mm')
        return cls(id=str(d['id']), name=str(d['name']),
                   cutting_speed_m_min=float(d.get('cutting_speed_m_min', 0.0)),
                   feed_per_tooth_mm=[float(x) for x in fz] if fz else None,
                   stepdown_factor=float(d.get('stepdown_factor', 0.0)))


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ToolLibrary:
    tools: list = field(default_factory=list)       # [ToolDef]
    materials: list = field(default_factory=list)   # [Material]

    # ── Lookup ────────────────────────────────────────────────────────
    def find_tool(self, tool_id):
        for t in self.tools:
            if t.id == tool_id:
                return t
        return None

    def find_material(self, material_id):
        for m in self.materials:
            if m.id == material_id:
                return m
        return None

    def tools_for_profiles(self, profiles):
        return [t for t in self.tools if t.profile in profiles]

    def tools_using_material(self, material_id):
        return [t for t in self.tools if material_id in t.cutting]

    # ── Serialization ─────────────────────────────────────────────────
    def to_dict(self):
        return {'version': LIBRARY_VERSION,
                'materials': [m.to_dict() for m in self.materials],
                'tools': [t.to_dict() for t in self.tools]}

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise ValueError("tool library must be a JSON object")
        version = d.get('version', 1)
        if version > LIBRARY_VERSION:
            raise ValueError(f"unsupported tool library version {version}")
        return cls(tools=[ToolDef.from_dict(t) for t in d.get('tools', [])],
                   materials=[Material.from_dict(m) for m in d.get('materials', [])])

    @classmethod
    def load(cls, path):
        """Read a library file. Raises OSError/ValueError on a missing or
        damaged file."""
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(str(exc)) from exc
        try:
            return cls.from_dict(data)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed tool library: {exc}") from exc

    def save(self, path):
        """Write the library atomically: a temporary file next to *path*
        replaces it only once fully written."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def default_path():
        if sys.platform.startswith('win'):
            base = os.environ.get('APPDATA') or os.path.expanduser('~')
        else:
            base = (os.environ.get('XDG_CONFIG_HOME')
                    or os.path.join(os.path.expanduser('~'), '.config'))
        return os.path.join(base, 'bm2gcode', 'tool_library.json')

    @classmethod
    def load_or_seed(cls, path):
        """Return ``(library, warning)``.

        A missing file yields a seeded library, written to *path*. A
        damaged file is renamed to ``<path>.bak`` and replaced by a seeded
        library; *warning* then describes what happened, otherwise it is
        None.
        """
        if not os.path.exists(path):
            lib = cls.seeded()
            try:
                lib.save(path)
            except OSError:
                pass
            return lib, None
        try:
            return cls.load(path), None
        except (OSError, ValueError) as exc:
            backup = path + '.bak'
            try:
                os.replace(path, backup)
                moved = f"It was renamed to:\n{backup}\n\n"
            except OSError:
                moved = ""
            lib = cls.seeded()
            try:
                lib.save(path)
            except OSError:
                pass
            return lib, (f"The tool library could not be read ({exc}).\n\n"
                         f"{moved}A default tool library is used instead.")

    @classmethod
    def seeded(cls):
        """A small starter library: three tools, three materials."""
        softwood = Material('mat-softwood', 'Softwood')
        hardwood = Material('mat-hardwood', 'Hardwood')
        mdf      = Material('mat-mdf', 'MDF')
        for m in (softwood, hardwood, mdf):
            m.apply_preset(m.name)

        def data(rows):
            return {mid: CuttingData(feed, sd, plunge, rpm)
                    for mid, (feed, plunge, rpm, sd) in rows.items()}

        tools = [
            ToolDef('tool-6mm-end-mill', '6 mm End Mill', 'End Mill', 6.0,
                    flute_length_mm=22.0, shank_diameter_mm=6.0,
                    overall_length_mm=60.0, number=1,
                    cutting=data({softwood.id: (1500, 500, 18000, 3.0),
                                  hardwood.id: (1000, 300, 18000, 2.0),
                                  mdf.id:      (1800, 500, 18000, 3.0)})),
            ToolDef('tool-3mm-ball', '3 mm Ball Nose', 'Ball', 3.0,
                    flute_length_mm=12.0, shank_diameter_mm=6.0,
                    overall_length_mm=50.0, number=2,
                    cutting=data({softwood.id: (1200, 400, 20000, 1.5),
                                  hardwood.id: (800, 250, 20000, 1.0),
                                  mdf.id:      (1400, 400, 20000, 1.5)})),
            ToolDef('tool-60deg-vbit', '60° V-Bit', 'V-Bit', 12.0,
                    vbit_angle_deg=60.0, flute_length_mm=10.0,
                    shank_diameter_mm=6.0, overall_length_mm=45.0, number=3,
                    cutting=data({softwood.id: (1000, 400, 18000, 2.0),
                                  hardwood.id: (700, 250, 18000, 1.5),
                                  mdf.id:      (1200, 400, 18000, 2.0)})),
        ]
        return cls(tools=tools, materials=[softwood, hardwood, mdf])

    # ── Import ────────────────────────────────────────────────────────
    def merge(self, other, on_conflict):
        """Merge *other* into this library.

        Materials are matched by id; unknown ones are added. Tools are
        matched by id. For a tool whose id already exists,
        ``on_conflict(existing, incoming)`` returns ``"replace"``,
        ``"keep_both"`` (the incoming tool is added under a new id) or
        ``"skip"``. Returns the number of tools added or replaced.
        """
        for m in other.materials:
            if self.find_material(m.id) is None:
                self.materials.append(copy.deepcopy(m))
        changed = 0
        for incoming in other.tools:
            incoming = copy.deepcopy(incoming)
            existing = self.find_tool(incoming.id)
            if existing is None:
                self.tools.append(incoming)
                changed += 1
                continue
            choice = on_conflict(existing, incoming)
            if choice == "replace":
                self.tools[self.tools.index(existing)] = incoming
                changed += 1
            elif choice == "keep_both":
                incoming.id = new_id()
                incoming.name = f"{incoming.name} (imported)"
                self.tools.append(incoming)
                changed += 1
        return changed


# ── Step-field mapping ────────────────────────────────────────────────────

def format_value(x) -> str:
    """Format a number the way step fields display it: up to 4 decimals,
    trailing zeros dropped."""
    s = f"{float(x):.4f}".rstrip('0').rstrip('.')
    return s if s not in ('', '-0') else '0'


def cutting_for(tool, material_id):
    """The tool's cutting data for *material_id*, or None if it has none."""
    if tool is None or material_id is None:
        return None
    return tool.cutting.get(material_id)


def stepdown_field_for(family) -> str:
    """The step-down field the given algorithm family reads."""
    if family == "V-Carving":
        return 'vcarve_stepdown'
    if family == "Cutout":
        return 'cutout_stepdown'
    return 'stepdown'


def step_values_from(tool, cutting):
    """Step field values (as strings, keyed by step-field name) to write
    when applying *tool*. The V-bit angle is included only for V-bits,
    a deep copy of the Bezier nodes (key 'profile_nodes') only for Custom
    profiles, and cutting fields only when *cutting* is given."""
    values = {'profile': tool.profile, 'diameter': format_value(tool.diameter_mm)}
    if tool.profile == "V-Bit":
        values['vbit_angle'] = format_value(tool.vbit_angle_deg)
    if tool.profile == "Custom":
        values['profile_nodes'] = copy.deepcopy(tool.profile_points)
    if cutting is not None:
        values['feed_rate']   = format_value(cutting.feed_rate)
        values['plunge_rate'] = format_value(cutting.plunge_rate)
        values['spindle_rpm'] = format_value(cutting.spindle_rpm)
        for key in STEPDOWN_FIELDS:
            values[key] = format_value(cutting.stepdown_mm)
    return values


def _same(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _same_profile(nodes_a, nodes_b) -> bool:
    """True when two Bezier node lists give the same radial height curve."""
    if not nodes_a or not nodes_b:
        return not nodes_a and not nodes_b
    return tool_profile.profile_curve(nodes_a) == tool_profile.profile_curve(nodes_b)


def sync_state(step_values, tool, cutting, family) -> str:
    """Compare a linked step's values against its library tool.

    *step_values* maps step-field names to the step's current values
    (strings or numbers). *tool* is None when the linked tool is not in
    the library; *cutting* is None when the tool has no data for the
    project material. Geometry is checked before cutting data, and only
    the step-down field *family* actually reads is compared.
    """
    if tool is None:
        return SYNC_MISSING
    if step_values.get('profile') != tool.profile:
        return SYNC_GEOMETRY_DIFFERS
    if not _same(step_values.get('diameter'), tool.diameter_mm):
        return SYNC_GEOMETRY_DIFFERS
    if tool.profile == "V-Bit" and not _same(step_values.get('vbit_angle'),
                                             tool.vbit_angle_deg):
        return SYNC_GEOMETRY_DIFFERS
    if tool.profile == "Custom" and not _same_profile(step_values.get('profile_nodes'),
                                                      tool.profile_points):
        return SYNC_GEOMETRY_DIFFERS
    if cutting is None:
        return SYNC_NO_DATA
    expected = (('feed_rate',   cutting.feed_rate),
                ('plunge_rate', cutting.plunge_rate),
                ('spindle_rpm', cutting.spindle_rpm),
                (stepdown_field_for(family), cutting.stepdown_mm))
    for key, value in expected:
        if not _same(step_values.get(key), value):
            return SYNC_MODIFIED
    return SYNC_IN_SYNC


def update_cutting_from_step(tool, material_id, step_values, family):
    """A copy of *tool* whose cutting data for *material_id* is taken from
    the step. Geometry (profile, diameter, V angle, custom curve) is left
    unchanged.
    Raises ValueError if a step value is not a number."""
    updated = copy.deepcopy(tool)
    updated.cutting[material_id] = CuttingData(
        feed_rate=float(step_values['feed_rate']),
        stepdown_mm=float(step_values[stepdown_field_for(family)]),
        plunge_rate=float(step_values.get('plunge_rate') or 0.0),
        spindle_rpm=float(step_values.get('spindle_rpm') or 0.0))
    return updated


def tool_from_step(name, step_values, material_id, family):
    """A new library tool built from a step's values, with cutting data for
    *material_id* (if given). Raises ValueError on non-numeric values."""
    profile = step_values['profile']
    tool = ToolDef(
        id=new_id(), name=name, profile=profile,
        diameter_mm=float(step_values['diameter']),
        vbit_angle_deg=(float(step_values['vbit_angle'])
                        if profile == "V-Bit" else 90.0),
        profile_points=(copy.deepcopy(step_values.get('profile_nodes'))
                        if profile == "Custom" else None))
    if material_id is not None:
        tool = update_cutting_from_step(tool, material_id, step_values, family)
    return tool


# ── Cutting recommendations ───────────────────────────────────────────────

@dataclass
class Recommendation:
    """Recommended cutting data for one (tool, material) pair, with the
    intermediate values of the calculation."""
    diameter_mm: float
    flutes: int
    vc_m_min: float          # cutting speed after the adjustment
    fz_mm: float             # feed per tooth for the tool's diameter
    rpm_calc: float          # spindle speed reaching vc
    rpm: float               # spindle speed used, within the spindle's limits
    vc_eff_m_min: float      # cutting speed at the spindle speed used
    fz_eff_mm: float         # feed per tooth at the feed rate used
    feed_rate: float         # mm/min
    plunge_rate: float       # mm/min
    stepdown_mm: float
    notes: list = field(default_factory=list)      # how limits shaped the result
    warnings: list = field(default_factory=list)   # results that risk the cut

    def to_cutting(self) -> CuttingData:
        return CuttingData(feed_rate=self.feed_rate, stepdown_mm=self.stepdown_mm,
                           plunge_rate=self.plunge_rate, spindle_rpm=self.rpm)


def interpolate_fz(material, diameter_mm):
    """``(fz, note)``: the material's feed per tooth for *diameter_mm*,
    linearly interpolated between the FZ_DIAMETERS_MM entries. Outside
    their range the nearest end value is used and *note* says so;
    otherwise *note* is None."""
    ds, fz = FZ_DIAMETERS_MM, material.feed_per_tooth_mm
    if diameter_mm <= ds[0] or diameter_mm >= ds[-1]:
        i = 0 if diameter_mm <= ds[0] else -1
        note = None
        if not math.isclose(diameter_mm, ds[i]):
            note = (f"Ø {format_value(diameter_mm)} mm is outside the "
                    f"{format_value(ds[0])}–{format_value(ds[-1])} mm table; "
                    f"the feed per tooth for Ø {format_value(ds[i])} mm is used.")
        return float(fz[i]), note
    for k in range(1, len(ds)):
        if diameter_mm <= ds[k]:
            t = (diameter_mm - ds[k - 1]) / (ds[k] - ds[k - 1])
            return float(fz[k - 1] + t * (fz[k] - fz[k - 1])), None
    return float(fz[-1]), None


def _round_rpm(n, down=False):
    """*n* rounded to 100 RPM (to whole RPM below 100); *down* rounds down."""
    step = 100.0 if n >= 100 else 1.0
    return (math.floor(n / step) if down else round(n / step)) * step


def recommend_cutting(tool, material, max_rpm, vc_adjust_pct=0.0,
                      max_feed=0.0, min_rpm=0.0) -> Recommendation:
    """Recommended cutting data for *tool* in *material*.

    n = vc·1000 / (π·d), rounded to 100 RPM and kept within the spindle's
    [*min_rpm*, *max_rpm*] range (0 = no limit); feed = n · flutes · fz.
    When the feed exceeds *max_feed* (0 = no limit), the feed per tooth is
    kept and n is lowered instead: a lower cutting speed is harmless, while
    a thinner chip makes the edge rub rather than cut. Only when n would
    have to fall below *min_rpm* does the feed per tooth drop, which is
    reported in ``warnings``. Plunge = feed · PLUNGE_FRACTION; step-down =
    material's step-down factor · d, capped at the tool's flute length when
    one is set. *vc_adjust_pct* scales the material's cutting speed (e.g.
    -25 for roughing). Raises ValueError when the material has no reference
    data or the tool has no diameter.
    """
    if not material.has_reference():
        raise ValueError(f"Material \"{material.name}\" has no reference cutting data.")
    d = float(tool.diameter_mm)
    if d <= 0:
        raise ValueError("The tool has no diameter.")
    notes, warnings = [], []
    flutes = max(1, int(tool.flutes))
    vc = material.cutting_speed_m_min * (1.0 + vc_adjust_pct / 100.0)
    fz, fz_note = interpolate_fz(material, d)
    if fz_note:
        notes.append(fz_note)
    hi = float(max_rpm) if max_rpm and max_rpm > 0 else math.inf
    lo = min(float(min_rpm), hi) if min_rpm and min_rpm > 0 else 0.0

    def vc_at(n):
        return n * math.pi * d / 1000.0

    rpm_calc = vc * 1000.0 / (math.pi * d)
    rpm = _round_rpm(rpm_calc)
    if rpm > hi:
        rpm = hi
        notes.append(f"The calculated {rpm_calc:.0f} RPM exceeds the spindle's "
                     f"maximum; {format_value(hi)} RPM is used.")
    elif rpm < lo:
        rpm = lo
        notes.append(f"The calculated {rpm_calc:.0f} RPM is below the spindle's "
                     f"minimum; {format_value(lo)} RPM is used (vc "
                     f"{vc_at(rpm):.0f} m/min).")
    feed, fz_eff = rpm * flutes * fz, fz
    if max_feed and max_feed > 0 and feed > max_feed:
        rpm_fz = _round_rpm(max_feed / (flutes * fz), down=True)
        if rpm_fz >= max(lo, 1.0):
            rpm, feed = rpm_fz, rpm_fz * flutes * fz
            notes.append(f"The feed is limited to the machine maximum of "
                         f"{format_value(max_feed)} mm/min; the spindle speed is "
                         f"lowered to {rpm:.0f} RPM to keep the feed per tooth "
                         f"(vc {vc_at(rpm):.0f} m/min).")
        else:
            rpm = lo if lo > 0 else max(1.0, math.floor(max_feed / (flutes * fz)))
            feed = float(max_feed)
            fz_eff = feed / (rpm * flutes)
            warnings.append(f"Even at the spindle's minimum speed the feed would exceed "
                            f"the machine maximum of {format_value(max_feed)} mm/min, so "
                            f"the feed per tooth drops to {fz_eff:.4f} mm "
                            f"({(fz_eff / fz - 1.0) * 100.0:.0f} %). Thin chips make the "
                            f"edge rub and heat up; consider a cutter with fewer flutes.")
    feed = float(round(feed))
    if max_feed and max_feed > 0:
        feed = min(feed, float(max_feed))
    stepdown = material.stepdown_factor * d
    if tool.flute_length_mm > 0 and stepdown > tool.flute_length_mm:
        stepdown = tool.flute_length_mm
        notes.append("The step-down is limited to the flute length.")
    if tool.profile in ("V-Bit", "Custom"):
        notes.append("The calculation uses the full diameter; near the tip the "
                     "cutting diameter is smaller.")
    return Recommendation(
        diameter_mm=d, flutes=flutes, vc_m_min=vc, fz_mm=fz,
        rpm_calc=rpm_calc, rpm=float(rpm), vc_eff_m_min=vc_at(rpm), fz_eff_mm=fz_eff,
        feed_rate=feed, plunge_rate=float(round(feed * PLUNGE_FRACTION)),
        stepdown_mm=round(stepdown, 2), notes=notes, warnings=warnings)
