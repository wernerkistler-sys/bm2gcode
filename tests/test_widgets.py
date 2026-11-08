"""Plain assert-based tests for the dialogs in widgets.py.

Reads the source rather than building the dialogs, so it runs without a
display. A ``command=self._handler`` naming a method the class lacks raises
inside the dialog's ``__init__``, before ``wait_window``: the window stays
open, but its caller never sees the result.

Run with: python tests/test_widgets.py
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


def self_commands(cls):
    """Names of the self attributes passed as ``command=`` anywhere in *cls*."""
    names = set()
    for node in ast.walk(cls):
        if not isinstance(node, ast.keyword) or node.arg != 'command':
            continue
        value = node.value
        if (isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == 'self'):
            names.add(value.attr)
    return names


def self_attributes(cls):
    """Names of the methods and instance attributes *cls* defines."""
    names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(cls):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and node.value.id == 'self'):
            names.add(node.attr)
    return names


with open(os.path.join(ROOT, 'widgets.py'), encoding='utf-8') as f:
    tree = ast.parse(f.read())

print("button commands")
classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
check("found dialog classes", len(classes) > 5)
for cls in classes:
    missing = sorted(self_commands(cls) - self_attributes(cls))
    check(f"{cls.name}: every command exists{missing or ''}", not missing)


print("export tool groups")
try:
    sys.path.insert(0, ROOT)
    from widgets import export_tool_groups
except Exception as exc:          # no tkinter: widgets.py can't be imported
    print(f"  skipped ({exc})")
else:
    def _info(computed, profile, diam, tool_id="", tool_name="", key=""):
        return {'label': "", 'computed': computed, 'profile': profile,
                'diameter_mm': diam, 'profile_key': key,
                'tool_id': tool_id, 'tool_name': tool_name}
    _groups = export_tool_groups([
        _info(True,  "Ball", 6.0),
        _info(True,  "Flat", 3.175, "t1", "Endmill"),
        _info(False, "Ball", 6.0),
        _info(True,  "Ball", 6.0),
        _info(True,  "Flat", 3.175, "t1", "Endmill"),
        _info(True,  "Flat", 3.175, "t2", "Endmill"),
        _info(True,  "Custom", 4.0, key="a"),
        _info(True,  "Custom", 4.0, key="b"),
    ])
    check("one group per tool, uncomputed steps left out",
          [idx for _, idx in _groups] == [[0, 3], [1, 4], [5], [6], [7]])
    check("labels name the tool, repeats get a suffix",
          [lbl for lbl, _ in _groups] == [
              "Ball ∅6 mm", "Flat ∅3.175 mm (Endmill)",
              "Flat ∅3.175 mm (Endmill) #2", "Custom ∅4 mm",
              "Custom ∅4 mm #2"])
    check("no computed steps, no groups",
          export_tool_groups([_info(False, "Ball", 6.0)]) == [])

print("range entry")
try:
    import tkinter as tk
    _root = tk.Tk()
except Exception as exc:          # no display: nothing to build the widget on
    print(f"  skipped ({exc})")
else:
    sys.path.insert(0, ROOT)
    from widgets import LabeledRangeEntry
    _rng = LabeledRangeEntry(_root, "Z range", "%", "0", "100",
                             minval=0.0, maxval=100.0, increment=5.0)
    check("defaults", _rng.get() == ("0", "100"))
    _rng.set(25, 75.5)
    check("set/get", _rng.get() == ("25", "75.5"))
    _rng.set(-10, 250)
    check("values clamp to the bounds", _rng.get() == ("0", "100"))
    _rng.lo_entry.var.set("40")
    _rng.lo_entry._commit()
    check("a typed bound commits", _rng.get() == ("40", "100"))
    _root.destroy()

if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All tests passed.")
