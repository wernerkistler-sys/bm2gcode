"""Plain assert-based tests for staleness.py: input fingerprints and which
steps' results and cuts are current.

Run with: python tests/test_staleness.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from staleness import StepState, fingerprint, step_currency
from tool import Tool
from dressup import TabSettings

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


# ── fingerprint ───────────────────────────────────────────────────────────
print("fingerprint")
base = {'algorithm': "Raster Scan", 'tool': Tool("Ball", 3.0), 'step_mm': 0.6,
        'feed_rate': 800.0, 'z_range': (0.0, 1.0), 'tab_settings': TabSettings(),
        'depth_inputs': {'sources': [{'index': 0, 'content': [1, None]}]}}
check("equal inputs give equal fingerprints",
      fingerprint(base) == fingerprint(dict(base)))
check("key order does not matter",
      fingerprint(base) == fingerprint(dict(reversed(list(base.items())))))
check("tuple and list are the same",
      fingerprint({**base, 'z_range': [0.0, 1.0]}) == fingerprint(base))
check("a changed stepover changes it",
      fingerprint({**base, 'step_mm': 0.7}) != fingerprint(base))
check("a changed feed rate changes it",
      fingerprint({**base, 'feed_rate': 900.0}) != fingerprint(base))
check("a changed tool field changes it",
      fingerprint({**base, 'tool': Tool("Ball", 4.0)}) != fingerprint(base))
check("a changed dataclass field changes it",
      fingerprint({**base, 'tab_settings': TabSettings(enabled=True)})
      != fingerprint(base))
check("a changed source token changes it",
      fingerprint({**base, 'depth_inputs': {'sources': [{'index': 0, 'content': [2, None]}]}})
      != fingerprint(base))
a = np.arange(1000, dtype=np.float32)
b = a.copy()
b[500] += 1.0
check("arrays compare by content, past numpy's repr truncation",
      fingerprint({'c': a}) == fingerprint({'c': a.copy()})
      and fingerprint({'c': a}) != fingerprint({'c': b}))
check("numpy scalars equal Python ones",
      fingerprint({'x': np.float64(0.5)}) == fingerprint({'x': 0.5}))


# ── step_currency ─────────────────────────────────────────────────────────
print("step_currency")


def computed(fp, reads_prior, serial, prior_serial, tp_fp=None):
    return StepState(fp, reads_prior, tp_fp or fp, serial, prior_serial, computed=True)


chain = [computed("a", False, 1, None),
         computed("b", True, 2, 1),
         computed("c", False, 3, 2)]
check("a freshly computed chain is current",
      all(c.result_current and c.cuts_current for c in step_currency(chain)))

edited = [computed("a2", False, 1, None, tp_fp="a")] + chain[1:]
cur = step_currency(edited)
check("an edited step is outdated", not cur[0].cuts_current and not cur[0].result_current)
check("a later step reading it has outdated cuts",
      not cur[1].cuts_current and not cur[1].result_current)
check("a later step not reading it keeps its cuts but not its result",
      cur[2].cuts_current and not cur[2].result_current)

recomputed = [computed("a2", False, 4, None)] + chain[1:]
cur = step_currency(recomputed)
check("after recomputing step 1, step 2 (reads prior) is outdated",
      cur[0].result_current and not cur[1].cuts_current)
recomputed = [computed("a2", False, 4, None), computed("b", True, 5, 4), chain[2]]
cur = step_currency(recomputed)
check("after recomputing steps 1-2, step 3 only has an outdated result",
      cur[1].cuts_current and cur[2].cuts_current and not cur[2].result_current)

cur = step_currency([StepState("a", False), computed("b", True, 2, None)])
check("a step not computed is outdated", not cur[0].cuts_current)
check("a step reading a missing prior step is outdated",
      not cur[1].cuts_current and not cur[1].result_current)
cur = step_currency([StepState("a", False), computed("b", False, 2, None)])
check("a step not reading a missing prior step keeps its cuts",
      cur[1].cuts_current and not cur[1].result_current)
cur = step_currency([computed("a", False, 1, None), StepState(None, False)])
check("a step whose fields don't parse is outdated",
      not cur[1].cuts_current)
cur = step_currency([computed(None, False, 1, None, tp_fp="a")])
check("an invalid fingerprint never matches", not cur[0].cuts_current)

# Removing step 2 of 3: the old step 3 now follows step 1, which it was
# not computed against.
cur = step_currency([chain[0], chain[2]])
check("a step whose previous step was removed has an outdated result",
      cur[1].cuts_current and not cur[1].result_current)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
