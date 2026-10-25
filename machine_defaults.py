"""User-wide default machine settings (the Settings dialog's values).

The defaults live in one JSON file next to the tool library (see
``default_path``). They seed the settings of the startup project and of
every new project; a saved project keeps its own copy of the settings, so
opening it never depends on the defaults file. Nothing here touches Tk.
"""
from __future__ import annotations

import json
import os

from constants import DEFAULTS
from tool_library import ToolLibrary

NUMERIC_KEYS = ('z_safe', 'feed_rapid', 'accel_mm_s2', 'spindle_dwell_s',
                'spindle_max_rpm', 'spindle_min_rpm', 'feed_max')
# '' stands for the built-in DEFAULT_PREAMBLE / DEFAULT_POSTAMBLE.
TEXT_KEYS = ('preamble', 'postamble')
KEYS = NUMERIC_KEYS + TEXT_KEYS


def builtin():
    """The built-in machine settings from ``constants.DEFAULTS``."""
    d = {k: float(DEFAULTS[k]) for k in NUMERIC_KEYS}
    d.update({k: DEFAULTS[k] for k in TEXT_KEYS})
    return d


def default_path():
    return os.path.join(os.path.dirname(ToolLibrary.default_path()),
                        'machine_defaults.json')


def from_dict(data):
    """Machine settings from *data*: the built-in value for every key
    *data* lacks; unknown keys are ignored. Raises ValueError on a value
    of the wrong type."""
    if not isinstance(data, dict):
        raise ValueError("machine defaults must be a JSON object")
    d = builtin()
    for k in NUMERIC_KEYS:
        if k in data:
            v = data[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"'{k}' must be a number")
            d[k] = float(v)
    for k in TEXT_KEYS:
        if k in data:
            if not isinstance(data[k], str):
                raise ValueError(f"'{k}' must be a string")
            d[k] = data[k]
    return d


def save(path, settings):
    """Write the machine settings in *settings* atomically: a temporary
    file next to *path* replaces it only once fully written."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({k: settings[k] for k in KEYS}, f, indent=2)
    os.replace(tmp, path)


def load(path):
    """Return ``(settings, warning)``.

    A missing file yields the built-in settings. A damaged file is renamed
    to ``<path>.bak`` and the built-in settings are used; *warning* then
    describes what happened, otherwise it is None.
    """
    if not os.path.exists(path):
        return builtin(), None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(str(exc)) from exc
        return from_dict(data), None
    except (OSError, ValueError) as exc:
        backup = path + '.bak'
        try:
            os.replace(path, backup)
            moved = f"It was renamed to:\n{backup}\n\n"
        except OSError:
            moved = ""
        return builtin(), (f"The default machine settings could not be read "
                           f"({exc}).\n\n{moved}The built-in settings are used "
                           f"instead.")
