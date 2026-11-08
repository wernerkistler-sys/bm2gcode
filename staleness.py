"""Whether computed step toolpaths are still current (no Tk code).

A step's toolpath is computed from the step's own inputs (its parameters,
its depth map's inputs, the machine settings the algorithms read) and from
the previous step's toolpath, whose result/accessible/alpha every
algorithm folds into its own result map. The app fingerprints a step's own
inputs when it computes the step and stores the fingerprint, a serial for
the new toolpath and the serial of the previous toolpath it read on the
ToolPath (see toolpath.ToolPath). step_currency compares those against the
current state:

- A step's *result* is current when its own inputs are unchanged, it was
  computed against the previous step's present toolpath, and that one's
  result is current -- transitive through every earlier step, since
  results accumulate.
- A step's *cuts* are current when its own inputs are unchanged and, only
  if its cuts read the previous step at all (``reads_prior``), its
  previous step's result is current and was the one it read.
"""

import dataclasses
import hashlib
import json
from typing import List, NamedTuple, Optional

import numpy as np


def _canonical(obj):
    """*obj* as plain JSON-able data that equal inputs map to identically:
    dataclasses by type name and fields, numpy arrays by dtype, shape and
    a hash of their bytes, mappings with string keys."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {'__type__': type(obj).__name__,
                **{f.name: _canonical(getattr(obj, f.name))
                   for f in dataclasses.fields(obj)}}
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {'__array__': [arr.dtype.str, list(arr.shape),
                              hashlib.sha1(arr.tobytes()).hexdigest()]}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def fingerprint(obj) -> str:
    """A stable hex digest of *obj* (nested dicts/lists/tuples, dataclasses,
    numpy arrays and scalars): equal inputs give equal fingerprints."""
    text = json.dumps(_canonical(obj), sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class StepState(NamedTuple):
    """One step's current inputs and what its stored toolpath was computed
    from. The tp_* fields are None when the step has no toolpath."""
    fingerprint:     Optional[str]   # current own inputs; None if invalid
    reads_prior:     bool            # its cuts read the previous step's toolpath
    tp_fingerprint:  Optional[str] = None
    tp_serial:       Optional[int] = None
    tp_prior_serial: Optional[int] = None
    computed:        bool = False


class StepCurrency(NamedTuple):
    result_current: bool   # result map (and so the next step's input) is current
    cuts_current:   bool   # toolpath / G-code is current


def step_currency(states: List[StepState]) -> List[StepCurrency]:
    """Currency of every step of *states*, in step order (see the module
    docstring)."""
    out = []
    prev_serial = None
    prev_result_current = True
    for s in states:
        own = (s.computed and s.fingerprint is not None
               and s.tp_fingerprint == s.fingerprint)
        prior_ok = prev_result_current and s.tp_prior_serial == prev_serial
        result_current = own and prior_ok
        out.append(StepCurrency(result_current,
                                own and (prior_ok or not s.reads_prior)))
        prev_serial = s.tp_serial if s.computed else None
        prev_result_current = result_current
    return out
