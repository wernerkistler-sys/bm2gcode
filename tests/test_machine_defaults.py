"""Plain assert-based tests for the default machine settings
(machine_defaults.py).

Run with: python tests/test_machine_defaults.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import machine_defaults
from constants import DEFAULTS

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


TMP = tempfile.mkdtemp(prefix="bm2gcode_machine_")


print("built-in")
b = machine_defaults.builtin()
check("every key present", set(b) == set(machine_defaults.KEYS))
check("numbers are floats", all(isinstance(b[k], float)
                                for k in machine_defaults.NUMERIC_KEYS))
check("z_safe from DEFAULTS", b['z_safe'] == float(DEFAULTS['z_safe']))


print("missing file")
path = os.path.join(TMP, 'missing', 'machine_defaults.json')
d, warning = machine_defaults.load(path)
check("built-in settings", d == machine_defaults.builtin())
check("no warning", warning is None)
check("no file written", not os.path.exists(path))


print("save / load round-trip")
path = os.path.join(TMP, 'sub', 'machine_defaults.json')
s = machine_defaults.builtin()
s.update(z_safe=12.5, spindle_max_rpm=18000.0, preamble="G21\nM3 S10000",
         postamble="M5")
s['unrelated'] = 1
machine_defaults.save(path, s)
d, warning = machine_defaults.load(path)
check("no warning", warning is None)
check("values kept", d['z_safe'] == 12.5 and d['spindle_max_rpm'] == 18000.0
      and d['preamble'] == "G21\nM3 S10000" and d['postamble'] == "M5")
check("only machine keys saved", 'unrelated' not in json.load(open(path)))
check("no temp file left", not os.path.exists(path + '.tmp'))


print("partial file")
with open(path, 'w') as f:
    json.dump({'feed_max': 3000, 'extra': 'x'}, f)
d, warning = machine_defaults.load(path)
check("given value used", d['feed_max'] == 3000.0)
check("missing keys built-in", d['z_safe'] == machine_defaults.builtin()['z_safe'])
check("unknown key ignored", 'extra' not in d)


print("damaged file")
for name, content in (("bad json", "{not json"),
                      ("wrong type", json.dumps({'z_safe': "high"})),
                      ("not an object", json.dumps([1, 2]))):
    with open(path, 'w') as f:
        f.write(content)
    d, warning = machine_defaults.load(path)
    check(f"{name}: built-in settings", d == machine_defaults.builtin())
    check(f"{name}: warning", warning is not None)
    check(f"{name}: moved to .bak", os.path.exists(path + '.bak')
          and not os.path.exists(path))


if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All tests passed.")
