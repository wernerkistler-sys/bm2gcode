# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

bm2gcode converts a depth map (image or STL mesh) into CNC G-code for relief carving, via an adaptive multi-step workflow (roughing, finishing, cutout, pocketing, etc.), plus an Undercut step that cuts sideways under the overhangs of STL sources. It's a Tkinter desktop app. `README.md` is the authoritative spec for every user-facing parameter and for how each algorithm's pipeline works internally (Raster Scan, Contour, V-Carving, Cutout, Pocket, Registration Holes, Undercut, dress-ups) — read the relevant section there before touching an algorithm, since the pipeline stage order and the reasoning behind it (e.g. why climb/conventional locking restricts reordering differently per algorithm, or how Cutout's through-cut channel works) is documented in detail and easy to get subtly wrong by guessing from code alone.

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
python tests/test_machine_defaults.py
python tests/test_grey_morphology.py
python tests/test_undercut.py
```
There is no test runner config, build step, or lint config in this repo.

Dependencies (no requirements.txt/pyproject.toml — install manually): `numpy`, `scipy`, `scikit-image`, `matplotlib`, `tkinter`.

## Architecture

Entry point `bm2gcode.py` creates the Tk root and `app.GCodeGeneratorApp`, forcing `LC_NUMERIC=C` so matplotlib's locale switch doesn't break Tk spinbox float parsing.

Module responsibilities:
- **`app.py`** — the Tk UI (`GCodeGeneratorApp`): canvas/source/step tabs, menus, view rendering. Orchestrates the other modules; holds no algorithmic logic of its own.
- **`widgets.py`** — reusable/standalone Tk dialogs and controls (`SettingsDialog`, `DressUpDialog`, `ExportToolpathDialog`, `ImportSTLDialog`, `ProgressDialog`, `LabeledEntry`, etc.).
- **`constants.py`** — static option lists (`ALGORITHMS`, `TOOL_PROFILES`, `BLEND_MODES`, ...), the `DEFAULTS` dict backing every parameter's initial value, and the default G-code preamble/postamble.
- **`layers.py`** — the `Layer` dataclass and depth-map compositing: resampling a source onto the canvas grid (`resample_layer`) — or, for an STL source rasterized directly on the canvas grid at its placement before a step is computed (`place_stl_layer`, cached on the layer as a `PlacedRaster` while its `placement_key` matches), using that exact raster instead, per-layer Mirror across the stock flip axes, depth/grayscale polarity conversion (`to_depth`/`from_depth`), and blending multiple sources into one depth map (`composite`). Also the undercut `Gaps` (air below each canvas pixel's top surface) built from STL layers' dexels by `composite_gaps`, which adopts a layer's gaps only where that layer alone produces the composite's top surface. `overhang_masks` splits gap pixels into enclosed (solid below, above full depth) and open-bottom (air down to full depth) for the View → Show overhangs overlay.
- **`stl_mesh.py`** — STL loading (binary and ASCII, no external mesh library), rasterizing a mesh into a heightmap for use as a `Layer`, and into `Dexels` (every surface crossing of each pixel's vertical line through the mesh) for the Undercut step.
- **`tool.py`** — the `Tool` dataclass (profile, diameter, V-bit angle, a Custom profile's radial curve, and the neck/shank/head geometry of the undercut profiles Lollipop, T-Slot and Dovetail — `UNDERCUT_FIELDS` says which apply to which), plus `undercut_geometry_error`.
- **`tool_library.py`** — the user-wide tool library (no Tk code): `ToolLibrary`/`ToolDef`/`Material`/`CuttingData`, JSON load/save at a per-user path (atomic save; a damaged file is moved to `.bak` and reseeded), import merging, and the pure step-field mapping used by the step tab — `step_values_from`, `sync_state` (compares only the step-down field the algorithm family reads), `update_cutting_from_step` (never changes geometry). Steps copy library values and store only a link (`library_tool_id`/`_name`/`_material_id`), so projects stay self-contained.
- **`machine_defaults.py`** — the user-wide default machine settings (the Settings dialog's values; no Tk code): JSON load/save next to the tool library (atomic save; a damaged file is moved to `.bak` and the built-in values are used). They seed the startup project, every New Project, and any setting missing from an opened project; a project's own saved settings always win.
- **`algorithms.py`** — the six toolpath algorithms (`compute_raster_toolpath`, `compute_contour_toolpath`, `compute_vcarve_toolpath`, `compute_cutout_toolpath`, `compute_pocket_toolpath`, `compute_registration_holes_toolpath`) plus shared machinery: tool-accessibility morphological opening, multi-pass depth-slicing, path ordering (greedy + 2-opt), geodesic terrain linking, RDP simplification. This is the core of the project — see `README.md`'s "Technical Details" section for the pipeline stages before modifying.
- **`grey_morphology.py`** — `grey_erosion`/`grey_dilation`/`grey_opening` with `scipy.ndimage`'s results but memory bounded for large kernels (row-by-row decomposition; scipy's own offset table grows with r⁴). `algorithms.py` imports these instead of scipy's — use them for any new grey morphology.
- **`undercut.py`** — the Undercut algorithm (`compute_undercut_toolpath`): per depth level, the free set from the floor/ceiling of each pixel's gap grey-eroded/dilated with the tool's lower/upper profile (`build_undercut_kernels`), unioned with the plunge-accessible set (clear sky); rings from open sky to the waterline; lead paths to clear sky via `skimage.graph.MCP_Geometric`. Its `ToolPath.undercut_reach` (head radius around every position where the tool fits) colours reached overhangs in that overlay. Reuses helpers from `algorithms.py` and `grey_morphology.py`; read README.md's "Undercut Pipeline" before modifying.
- **`dressup.py`** — the four per-step G-code dress-ups: ramp-in, back-off-before-retract, arc-rounded rapid corners, and Cutout tabs (`RampSettings`/`BackoffSettings`/`ArcSettings`/`TabSettings` + `generate_ramp`/`compute_backoff_point`/`generate_corner_arc`/`apply_tabs`). Tabs alter cut geometry (consumed inside `algorithms.py`'s Cutout pipeline); the other three only alter G-code emission (consumed by `toolpath.py`).
- **`toolpath.py`** — `ToolPath`: holds computed segments/result map, emits G-code (`to_gcode`), applies the three emission-time dress-ups, and computes run-time/distance stats.

Data flow: sources (`layers.py`/`stl_mesh.py`) → composited depth map → an algorithm in `algorithms.py` (gated by a step's Process Region mode against the prior step's result map) → a `ToolPath` (`toolpath.py`) → optionally dressed up (`dressup.py`) → G-code. Undercut steps additionally take the composited gaps (`layers.composite_gaps`) into `undercut.py`; their segments start and end in clear sky, and ramp-in/back-off are never applied to them.

## Code style

- Comments must describe only what the code currently does — never phrase them relative to a prior version (no "now does X", "changed to Y", "removed Z", "also handles..."). Write as if the code had always been this way.

## Git

- Commit directly to `main`; never create a branch for a commit.
