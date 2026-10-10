"""Plain assert-based tests for the tool library (tool_library.py).

Run with: python tests/test_tool_library.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tool_profile
from tool_library import (
    ToolLibrary, ToolDef, Material, CuttingData,
    cutting_for, step_values_from, stepdown_field_for, sync_state,
    update_cutting_from_step, tool_from_step, format_value,
    recommend_cutting, interpolate_fz, MATERIAL_PRESETS, PLUNGE_FRACTION,
    SYNC_IN_SYNC, SYNC_GEOMETRY_DIFFERS, SYNC_MODIFIED, SYNC_MISSING, SYNC_NO_DATA,
)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


TMP = tempfile.mkdtemp(prefix="bm2gcode_toollib_")


# --------------------------------------------------------------------------
# Save / load round-trip
# --------------------------------------------------------------------------
print("save / load")

lib = ToolLibrary.seeded()
path = os.path.join(TMP, "sub", "tool_library.json")
lib.save(path)
check("save creates missing directory", os.path.exists(path))
check("no temporary file left behind", not os.path.exists(path + ".tmp"))
reloaded = ToolLibrary.load(path)
check("round-trip preserves everything", reloaded.to_dict() == lib.to_dict())
check("round-trip keeps tool number", reloaded.tools[0].number == 1)
check("round-trip keeps shank diameter and overall length",
      reloaded.tools[0].shank_diameter_mm == 6.0
      and reloaded.tools[0].overall_length_mm == 60.0)
old = ToolDef.from_dict({'id': 'x', 'name': 'old', 'profile': 'End Mill',
                         'diameter_mm': 3.0})
check("missing shank diameter / overall length default to 0",
      old.shank_diameter_mm == 0.0 and old.overall_length_mm == 0.0)
check("seeded library has three tools and three materials",
      len(lib.tools) == 3 and len(lib.materials) == 3)
check("every seeded tool has data for every seeded material",
      all(set(t.cutting) == {m.id for m in lib.materials} for t in lib.tools))


# --------------------------------------------------------------------------
# load_or_seed
# --------------------------------------------------------------------------
print("load_or_seed")

missing = os.path.join(TMP, "missing", "tool_library.json")
lib2, warning = ToolLibrary.load_or_seed(missing)
check("missing file: seeded, no warning", warning is None and len(lib2.tools) == 3)
check("missing file: seeded library is written", os.path.exists(missing))

damaged = os.path.join(TMP, "damaged.json")
with open(damaged, "w", encoding="utf-8") as f:
    f.write("{ not json")
lib3, warning = ToolLibrary.load_or_seed(damaged)
check("damaged file: warning returned", warning is not None)
check("damaged file: renamed to .bak", os.path.exists(damaged + ".bak"))
with open(damaged + ".bak", encoding="utf-8") as f:
    check("damaged file: .bak keeps original content", f.read() == "{ not json")
check("damaged file: replaced by seeded library",
      ToolLibrary.load(damaged).to_dict() == ToolLibrary.seeded().to_dict())

malformed = os.path.join(TMP, "malformed.json")
with open(malformed, "w", encoding="utf-8") as f:
    f.write('{"version": 1, "tools": [{"name": "no id"}]}')
_, warning = ToolLibrary.load_or_seed(malformed)
check("malformed (valid JSON, missing keys) is treated as damaged", warning is not None)

future = os.path.join(TMP, "future.json")
with open(future, "w", encoding="utf-8") as f:
    f.write('{"version": 99, "tools": [], "materials": []}')
_, warning = ToolLibrary.load_or_seed(future)
check("newer library version is treated as damaged", warning is not None)


# --------------------------------------------------------------------------
# cutting_for
# --------------------------------------------------------------------------
print("cutting_for")

em = lib.find_tool('tool-6mm-end-mill')
check("returns the material's data",
      cutting_for(em, 'mat-hardwood').feed_rate == 1000)
check("no data for an unknown material (no fallback)",
      cutting_for(em, 'mat-oak') is None)
check("no material -> None", cutting_for(em, None) is None)
check("no tool -> None", cutting_for(None, 'mat-mdf') is None)


# --------------------------------------------------------------------------
# step_values_from
# --------------------------------------------------------------------------
print("step_values_from")

vals = step_values_from(em, cutting_for(em, 'mat-softwood'))
check("profile and diameter", vals['profile'] == "End Mill" and vals['diameter'] == "6")
check("all three step-down fields written",
      vals['stepdown'] == vals['vcarve_stepdown'] == vals['cutout_stepdown'] == "3")
check("feed / plunge / rpm",
      (vals['feed_rate'], vals['plunge_rate'], vals['spindle_rpm']) == ("1500", "500", "18000"))
check("V angle not written for a non-V-bit", 'vbit_angle' not in vals)
check("stepover is never written", 'stepover' not in vals)

vbit = lib.find_tool('tool-60deg-vbit')
check("V angle written for a V-bit",
      step_values_from(vbit, None).get('vbit_angle') == "60")
geo_only = step_values_from(em, None)
check("no cutting data -> geometry only",
      set(geo_only) == {'profile', 'diameter'})
check("format_value trims trailing zeros",
      format_value(2.50) == "2.5" and format_value(0.0) == "0" and format_value(1.23456) == "1.2346")


# --------------------------------------------------------------------------
# sync_state
# --------------------------------------------------------------------------
print("sync_state")

cut = cutting_for(em, 'mat-softwood')
base = step_values_from(em, cut)
check("fresh values are in sync",
      sync_state(base, em, cut, "Raster Scan") == SYNC_IN_SYNC)
check("float tolerance: '6.0000001' equals 6",
      sync_state(dict(base, diameter="6.0000001"), em, cut, "Raster Scan") == SYNC_IN_SYNC)
check("non-numeric field counts as different",
      sync_state(dict(base, feed_rate="abc"), em, cut, "Raster Scan") == SYNC_MODIFIED)
check("missing tool",
      sync_state(base, None, cut, "Raster Scan") == SYNC_MISSING)
check("diameter change -> geometry differs",
      sync_state(dict(base, diameter="5"), em, cut, "Raster Scan") == SYNC_GEOMETRY_DIFFERS)
check("profile change -> geometry differs",
      sync_state(dict(base, profile="Ball"), em, cut, "Cutout") == SYNC_GEOMETRY_DIFFERS)
check("geometry differs takes priority over modified",
      sync_state(dict(base, profile="Ball", feed_rate="1"), em, cut, "Raster Scan")
      == SYNC_GEOMETRY_DIFFERS)
check("feed change -> modified",
      sync_state(dict(base, feed_rate="900"), em, cut, "Raster Scan") == SYNC_MODIFIED)
check("rpm change -> modified",
      sync_state(dict(base, spindle_rpm="0"), em, cut, "Raster Scan") == SYNC_MODIFIED)
check("V angle ignored for a non-V-bit",
      sync_state(dict(base, vbit_angle="30"), em, cut, "Raster Scan") == SYNC_IN_SYNC)
vcut = cutting_for(vbit, 'mat-mdf')
vbase = step_values_from(vbit, vcut)
check("V angle compared for a V-bit",
      sync_state(dict(vbase, vbit_angle="90"), vbit, vcut, "V-Carving") == SYNC_GEOMETRY_DIFFERS)
check("no cutting data -> no_data",
      sync_state(base, em, None, "Raster Scan") == SYNC_NO_DATA)

# Only the active step-down field is compared.
check("stepdown_field_for families",
      (stepdown_field_for("Raster Scan"), stepdown_field_for("Contour"),
       stepdown_field_for("Pocket"), stepdown_field_for("Registration Holes"),
       stepdown_field_for("V-Carving"), stepdown_field_for("Cutout"))
      == ('stepdown', 'stepdown', 'stepdown', 'stepdown', 'vcarve_stepdown', 'cutout_stepdown'))
edited_vc = dict(base, vcarve_stepdown="0")
check("hidden V-Carving step-down ignored on Raster",
      sync_state(edited_vc, em, cut, "Raster Scan") == SYNC_IN_SYNC)
check("V-Carving step-down compared on V-Carving",
      sync_state(edited_vc, em, cut, "V-Carving") == SYNC_MODIFIED)
edited_co = dict(base, cutout_stepdown="1")
check("Cutout step-down compared on Cutout only",
      sync_state(edited_co, em, cut, "Pocket") == SYNC_IN_SYNC
      and sync_state(edited_co, em, cut, "Cutout") == SYNC_MODIFIED)


# --------------------------------------------------------------------------
# update_cutting_from_step / tool_from_step
# --------------------------------------------------------------------------
print("update_cutting_from_step")

step = dict(base, profile="Ball", diameter="3", feed_rate="1111",
            plunge_rate="222", spindle_rpm="16000", cutout_stepdown="1.25")
updated = update_cutting_from_step(em, 'mat-oak', step, "Cutout")
check("geometry unchanged",
      (updated.profile, updated.diameter_mm, updated.vbit_angle_deg)
      == (em.profile, em.diameter_mm, em.vbit_angle_deg))
check("cutting data taken from the step, active step-down field",
      updated.cutting['mat-oak'] == CuttingData(1111, 1.25, 222, 16000))
check("original tool untouched", 'mat-oak' not in em.cutting)
check("other materials kept", updated.cutting['mat-mdf'] == em.cutting['mat-mdf'])

new_tool = tool_from_step("My V", dict(vbase, vbit_angle="45"), 'mat-mdf', "V-Carving")
check("tool_from_step copies geometry",
      (new_tool.profile, new_tool.diameter_mm, new_tool.vbit_angle_deg) == ("V-Bit", 12.0, 45.0))
check("tool_from_step gets a fresh id", new_tool.id != vbit.id)
check("tool_from_step stores cutting data",
      new_tool.cutting['mat-mdf'].feed_rate == 1200)


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------
print("merge")

target = ToolLibrary.seeded()
incoming = ToolLibrary(
    tools=[ToolDef('tool-6mm-end-mill', '6 mm EM (shop)', 'End Mill', 6.0),
           ToolDef('tool-new', '1 mm EM', 'End Mill', 1.0)],
    materials=[Material('mat-mdf', 'MDF'), Material('mat-oak', 'Oak')])

n = target.merge(incoming, lambda existing, inc: "replace")
check("replace: count", n == 2)
check("replace: existing tool replaced",
      target.find_tool('tool-6mm-end-mill').name == '6 mm EM (shop)')
check("new tool added", target.find_tool('tool-new') is not None)
check("unknown material added, known one not duplicated",
      [m.id for m in target.materials].count('mat-mdf') == 1
      and target.find_material('mat-oak') is not None)

target = ToolLibrary.seeded()
target.merge(incoming, lambda existing, inc: "keep_both")
ems = [t for t in target.tools if t.profile == "End Mill" and t.diameter_mm == 6.0]
check("keep both: two tools, distinct ids",
      len(ems) == 2 and ems[0].id != ems[1].id)
check("keep both: original untouched",
      target.find_tool('tool-6mm-end-mill').name == '6 mm End Mill')

target = ToolLibrary.seeded()
n = target.merge(incoming, lambda existing, inc: "skip")
check("skip: only the new tool counted", n == 1)
check("merge copies, does not alias",
      target.find_tool('tool-new') is not incoming.tools[1])


# --------------------------------------------------------------------------
# Custom profiles
# --------------------------------------------------------------------------
print("custom profiles")

nodes = tool_profile.point_roundover(12.7, 6.35)
custom = ToolDef(id='tool-custom', name='Round-over', profile='Custom',
                 diameter_mm=12.7, profile_points=nodes)
custom.cutting['mat-mdf'] = CuttingData(feed_rate=900.0, stepdown_mm=1.0)
lib = ToolLibrary.seeded()
lib.tools.append(custom)
cpath = os.path.join(TMP, "custom", "tool_library.json")
lib.save(cpath)
loaded = ToolLibrary.load(cpath).find_tool('tool-custom')
check("custom nodes survive save / load", loaded.profile_points == nodes)
t = loaded.to_tool()
check("to_tool carries the sampled curve",
      t.profile == "Custom" and t.profile_curve == tool_profile.profile_curve(nodes)
      and abs(t.diameter_mm - 12.7) < 1e-6)

bad = custom.to_dict()
bad['profile_points'] = [{'p': 'oops'}]
check("malformed nodes load as None", ToolDef.from_dict(bad).profile_points is None)

values = step_values_from(custom, custom.cutting['mat-mdf'])
check("step values include a copy of the nodes",
      values['profile_nodes'] == nodes and values['profile_nodes'] is not nodes)
check("custom step in sync",
      sync_state(values, custom, custom.cutting['mat-mdf'], "Raster Scan") == SYNC_IN_SYNC)
edited = dict(values, profile_nodes=tool_profile.point_roundover(12.7, 5.0))
check("different curve is a geometry difference",
      sync_state(edited, custom, custom.cutting['mat-mdf'], "Raster Scan")
      == SYNC_GEOMETRY_DIFFERS)
check("missing curve is a geometry difference",
      sync_state(dict(values, profile_nodes=None), custom, None, "Raster Scan")
      == SYNC_GEOMETRY_DIFFERS)
new_tool = tool_from_step("From step", values, 'mat-mdf', "Raster Scan")
check("tool_from_step keeps the curve", new_tool.profile_points == nodes)
check("non-custom step values have no nodes",
      'profile_nodes' not in step_values_from(ToolLibrary.seeded().tools[0], None))


# --------------------------------------------------------------------------
# Cutting recommendations
# --------------------------------------------------------------------------
print("cutting recommendations")

def preset_material(name):
    m = Material("mat-" + name.lower().replace(' ', '-'), name)
    m.apply_preset(name)
    return m

seeded = ToolLibrary.seeded()
check("seeded materials have reference data",
      all(m.has_reference() for m in seeded.materials))
check("seeded tools have 2 flutes", all(t.flutes == 2 for t in seeded.tools))
check("all presets are complete",
      all(preset_material(n).has_reference() for n in MATERIAL_PRESETS))

# Flutes and material reference data survive a round-trip.
t = ToolDef('t-z', 'Z', 'End Mill', 6.0, flutes=3)
check("flutes round-trip", ToolDef.from_dict(t.to_dict()).flutes == 3)
alu = preset_material("Wrought aluminium")
alu2 = Material.from_dict(alu.to_dict())
check("material reference round-trip",
      alu2.cutting_speed_m_min == 500 and alu2.feed_per_tooth_mm == alu.feed_per_tooth_mm
      and alu2.stepdown_factor == 0.05)
old = ToolLibrary.from_dict({'materials': [{'id': 'm', 'name': 'Old'}],
                             'tools': [{'id': 't', 'name': 'T', 'profile': 'Ball',
                                        'diameter_mm': 3.0}]})
check("old library loads with defaults",
      old.tools[0].flutes == 2 and not old.materials[0].has_reference())

# The sheet's worked example: wrought aluminium, Ø 8, 2 flutes.
mill8 = ToolDef('t8', '8 mm', 'End Mill', 8.0)
rec = recommend_cutting(mill8, alu, 0)
check("example rpm", rec.rpm == 19900 and abs(rec.rpm_calc - 19894.4) < 0.1)
check("example feed", rec.feed_rate == round(19900 * 2 * 0.064))
check("plunge is a third of feed",
      rec.plunge_rate == round(rec.feed_rate * PLUNGE_FRACTION))
check("metal step-down", rec.stepdown_mm == 0.4)
check("no notes for an in-table end mill", rec.notes == [])

capped = recommend_cutting(mill8, alu, 18000)
check("rpm capped at spindle max", capped.rpm == 18000)
check("capped feed", capped.feed_rate == 2304)
check("cap noted", len(capped.notes) == 1)

check("more flutes, more feed",
      recommend_cutting(ToolDef('t', 't', 'End Mill', 8.0, flutes=3), alu, 0).feed_rate
      == round(19900 * 3 * 0.064))
rough = recommend_cutting(mill8, alu, 0, vc_adjust_pct=-25)
check("roughing lowers rpm", rough.rpm < rec.rpm and abs(rough.vc_m_min - 375) < 1e-9)

fz7, note = interpolate_fz(alu, 7.0)
check("fz interpolated between diameters", abs(fz7 - 0.057) < 1e-9 and note is None)
fz_small, note_small = interpolate_fz(alu, 0.5)
fz_big, note_big = interpolate_fz(alu, 16.0)
check("fz clamps below the table", fz_small == 0.010 and note_small)
check("fz clamps above the table", fz_big == 0.100 and note_big)
check("fz at table end has no note", interpolate_fz(alu, 12.0)[1] is None)

wood = preset_material("Softwood")
mill6 = ToolDef('t6', '6 mm', 'End Mill', 6.0)
check("wood step-down is half the diameter",
      recommend_cutting(mill6, wood, 0).stepdown_mm == 3.0)
short = ToolDef('t6s', '6 mm short', 'End Mill', 6.0, flute_length_mm=2.0)
rec_short = recommend_cutting(short, wood, 0)
check("step-down capped at flute length", rec_short.stepdown_mm == 2.0 and rec_short.notes)
check("v-bit gets a diameter note",
      recommend_cutting(seeded.tools[2], wood, 0).notes)
check("to_cutting carries the values",
      rec.to_cutting() == CuttingData(rec.feed_rate, rec.stepdown_mm,
                                      rec.plunge_rate, rec.rpm))

# Machine feed limit: fz is kept by lowering n, down to the spindle minimum.
mdf = preset_material("MDF")
mill6z3 = ToolDef('t6z3', '6 mm 3F', 'End Mill', 6.0, flutes=3)
unlimited = recommend_cutting(mill6z3, mdf, 24000)
check("feed limit off matches plain result",
      recommend_cutting(mill6z3, mdf, 24000, max_feed=0, min_rpm=0) == unlimited)
check("unlimited feed is high", unlimited.feed_rate == round(23900 * 3 * 0.3))
kept = recommend_cutting(mill6z3, mdf, 24000, max_feed=5000, min_rpm=0)
check("feed limit lowers rpm, keeps fz",
      kept.rpm == 5500 and kept.feed_rate == 4950 and kept.fz_eff_mm == 0.3)
check("feed-limited note, no warning", kept.notes and not kept.warnings)
check("effective vc follows rpm",
      abs(kept.vc_eff_m_min - 5500 * 3.141592653589793 * 6 / 1000) < 1e-9)
floor = recommend_cutting(mill6z3, mdf, 24000, max_feed=5000, min_rpm=6000)
check("min rpm floor reduces fz",
      floor.rpm == 6000 and floor.feed_rate == 5000
      and abs(floor.fz_eff_mm - 5000 / (6000 * 3)) < 1e-9)
check("reduced fz is a warning", len(floor.warnings) == 1)
check("plunge follows the limited feed",
      floor.plunge_rate == round(5000 * PLUNGE_FRACTION))
for d_ in (1.0, 3.0, 6.0, 12.0):
    for z_ in (1, 2, 4):
        r = recommend_cutting(ToolDef('x', 'x', 'End Mill', d_, flutes=z_), mdf, 24000,
                              max_feed=3000, min_rpm=8000)
        if r.feed_rate > 3000 or r.rpm < 8000 or r.rpm > 24000:
            check(f"limits hold for Ø{d_} z{z_}", False)
check("limits hold across diameters and flutes", True)

# Spindle minimum raises n when vc asks for less; fz is kept.
steel = preset_material("Steel")
mill12 = ToolDef('t12', '12 mm', 'End Mill', 12.0)
slow = recommend_cutting(mill12, steel, 24000, min_rpm=6000)
check("min rpm raises a slow spindle",
      slow.rpm == 6000 and slow.fz_eff_mm == 0.080 and slow.feed_rate == 6000 * 2 * 0.080)
check("raised vc reported", slow.vc_eff_m_min > slow.vc_m_min and slow.notes)

try:
    recommend_cutting(mill8, Material('m', 'Bare'), 0)
    check("no reference data raises", False)
except ValueError:
    check("no reference data raises", True)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All tests passed.")
