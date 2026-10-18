# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

bm2gcode converts a depth map (image or STL mesh) into CNC G-code for relief carving, via an adaptive multi-step workflow (roughing, finishing, cutout, pocketing, etc.). It's a Tkinter desktop app. `README.md` is the authoritative spec for every user-facing parameter and for how each algorithm's pipeline works internally (Raster Scan, Contour, V-Carving, Cutout, Pocket, Registration Holes, dress-ups) — read the relevant section there before touching an algorithm, since the pipeline stage order and the reasoning behind it (e.g. why climb/conventional locking restricts reordering differently per algorithm, or how Cutout's through-cut channel works) is documented in detail and easy to get subtly wrong by guessing from code alone.

## Commands

Run the app:
```
python bm2gcode.py
```

Run tests (plain assert-based scripts, not pytest — each prints `ok`/`FAIL` per case and exits nonzero on failure):
```
python tests/test_dressup.py
python tests/test_layers.py
python tests/test_algorithms.py
python tests/test_tool_library.py
python tests/test_grey_morphology.py
```
There is no test runner config, build step, or lint config in this repo.

Dependencies (no requirements.txt/pyproject.toml — install manually): `numpy`, `scipy`, `scikit-image`, `matplotlib`, `tkinter`.

## Architecture

Entry point `bm2gcode.py` creates the Tk root and `app.GCodeGeneratorApp`, forcing `LC_NUMERIC=C` so matplotlib's locale switch doesn't break Tk spinbox float parsing.

Module responsibilities:
- **`app.py`** — the Tk UI (`GCodeGeneratorApp`): canvas/source/step tabs, menus, view rendering. Orchestrates the other modules; holds no algorithmic logic of its own.
- **`widgets.py`** — reusable/standalone Tk dialogs and controls (`SettingsDialog`, `DressUpDialog`, `ExportToolpathDialog`, `ImportSTLDialog`, `ProgressDialog`, `LabeledEntry`, etc.).
- **`constants.py`** — static option lists (`ALGORITHMS`, `TOOL_PROFILES`, `BLEND_MODES`, ...), the `DEFAULTS` dict backing every parameter's initial value, and the default G-code preamble/postamble.
- **`layers.py`** — the `Layer` dataclass and depth-map compositing: resampling a source onto the canvas grid (`resample_layer`), per-layer Mirror across the stock flip axes, depth/grayscale polarity conversion (`to_depth`/`from_depth`), and blending multiple sources into one depth map (`composite`).
- **`stl_mesh.py`** — STL loading (binary and ASCII, no external mesh library) and rasterizing a mesh into a heightmap for use as a `Layer`.
- **`tool.py`** — the `Tool` dataclass (profile, diameter, V-bit angle).
- **`tool_library.py`** — the user-wide tool library (no Tk code): `ToolLibrary`/`ToolDef`/`Material`/`CuttingData`, JSON load/save at a per-user path (atomic save; a damaged file is moved to `.bak` and reseeded), import merging, and the pure step-field mapping used by the step tab — `step_values_from`, `sync_state` (compares only the step-down field the algorithm family reads), `update_cutting_from_step` (never changes geometry). Steps copy library values and store only a link (`library_tool_id`/`_name`/`_material_id`), so projects stay self-contained.
- **`algorithms.py`** — the six toolpath algorithms (`compute_raster_toolpath`, `compute_contour_toolpath`, `compute_vcarve_toolpath`, `compute_cutout_toolpath`, `compute_pocket_toolpath`, `compute_registration_holes_toolpath`) plus shared machinery: tool-accessibility morphological opening, multi-pass depth-slicing, path ordering (greedy + 2-opt), geodesic terrain linking, RDP simplification. This is the core of the project — see `README.md`'s "Technical Details" section for the pipeline stages before modifying.
- **`grey_morphology.py`** — `grey_erosion`/`grey_dilation`/`grey_opening` with `scipy.ndimage`'s results but memory bounded for large kernels (row-by-row decomposition; scipy's own offset table grows with r⁴). `algorithms.py` imports these instead of scipy's — use them for any new grey morphology.
- **`dressup.py`** — the four per-step G-code dress-ups: ramp-in, back-off-before-retract, arc-rounded rapid corners, and Cutout tabs (`RampSettings`/`BackoffSettings`/`ArcSettings`/`TabSettings` + `generate_ramp`/`compute_backoff_point`/`generate_corner_arc`/`apply_tabs`). Tabs alter cut geometry (consumed inside `algorithms.py`'s Cutout pipeline); the other three only alter G-code emission (consumed by `toolpath.py`).
- **`toolpath.py`** — `ToolPath`: holds computed segments/result map, emits G-code (`to_gcode`), applies the three emission-time dress-ups, and computes run-time/distance stats.

Data flow: sources (`layers.py`/`stl_mesh.py`) → composited depth map → an algorithm in `algorithms.py` (gated by a step's Process Region mode against the prior step's result map) → a `ToolPath` (`toolpath.py`) → optionally dressed up (`dressup.py`) → G-code.

## Code style

- Comments must describe only what the code currently does — never phrase them relative to a prior version (no "now does X", "changed to Y", "removed Z", "also handles..."). Write as if the code had always been this way.
