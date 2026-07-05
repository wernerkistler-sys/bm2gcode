# bm2gcode

bm2gcode converts a depth map, represented by a grayscale bitmap image, into CNC G-code for relief carving. 

The central idea is an **adaptive multi-step workflow**. This allows multiple cutting steps to be defined using different algorithms and tools. Each step can be configured to process only the residual material left behind by previous steps, such as material that the previous step failed to remove, or regions that the previous tool geometry could not access. This makes it easy to combine strategies. For example, a large end mill could be used to rough out flat regions, followed by a ball bit to target curved surfaces and then a V-bit to cut fine details that the ball bit could not reach.

A set of visualization modes supports strategy planning. In addition to the 3D toolpath preview, available views include the simulated surface after each step, the material not removed by a step, regions inaccessible to the tool geometry, and a binary processing mask showing which areas the current gating configuration will cut. These views, together with the estimated machining time for each step, allow users to optimize the cutting strategy for a particular depth map.

![bm2gcode screenshot](docs/screenshot.png)

---

## Installation

**Requirements:** Python 3.8+, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `tkinter` (bundled with most Python distributions).

```bash
git clone https://github.com/wernerkistler-sys/bm2gcode.git
cd bm2gcode
pip install numpy scipy scikit-image matplotlib
python bm2gcode.py
```

---

## Quick Start

1. **File → Open Image** (Ctrl+O) — load a grayscale image (PNG, JPG, BMP, TIFF, WEBP, …).
2. Set **Physical Image Size** — enter the real-world dimensions and maximum cut depth.
3. Configure one or more **steps** — choose algorithm, tool, and region targeting for each.
4. **Compute → Compute All Steps** (F6) — generate toolpaths.
5. Use the **View** menu (hotkeys 1–7) to inspect the 3D path, simulated result, residual depth, processing mask, etc.
6. **File → Export Toolpath** — save a `.nc` G-code file for your CNC machine.

Save your work at any time with **File → Save Project** (Ctrl+S); reload it later with **File → Open Project** (Ctrl+Shift+O). Project files store all parameters and the path to the source image.

---

## Parameter Reference

### Physical Image Size

| Parameter | Description |
|---|---|
| Width / Height | Physical size of the carving area in cm. The lock icon maintains the image's aspect ratio; unlock it to set width and height independently. |
| Max cut depth | Maximum cut depth in mm. The pixel representing full depth (black when Z=0 colour = "white", white when Z=0 colour = "black") is carved to this depth. |
| Stock margin | Flat uncarved border around the image, in cm. |
| Margin depth | How deep to cut the stock margin, as a percentage of Max cut depth. |
| Z=0 colour | Which pixel colour represents the uncut surface. "White": white = surface (z = 0), black = deepest cut. "Black": the reverse. |
| G-code origin | Which corner — or the center — of the work area corresponds to X0 Y0 in the G-code output. |

---

### Steps

Each cutting step appears as a numbered tab. Use the **Add Step After** and **Remove this step** buttons at the bottom of each tab to manage the sequence. Each tab contains an **Operation** group and a **Tool** group.

---

#### Step / Operation

**Algorithm** selects the cutting strategy. The parameters displayed below the selector adapt to the chosen algorithm.

---

##### Step / Operation / Raster Scan

Sweeps the cutter in parallel lines across the material. 

| Parameter | Description |
|---|---|
| Step-over | Lateral spacing between scan lines, as a percentage of tool diameter. |
| Line offset | Shifts alternate scan lines by this percentage, so that the in-between lines can be cut in a separate step with a different setting. |
| Scan direction | Horizontal, Vertical, or Both. |

---

##### Step / Operation / Contour

Traces the depth map as a stack of iso-depth contours, like topographic lines on a map. 

| Parameter | Description |
|---|---|
| Step-down | Vertical distance between contour levels (mm). |
| Adaptive levels (CDF) | Distributes levels non-uniformly. Instead of equal Z spacing, levels are placed where many pixels share the same gray value ("plateaus"), so contour lines align with the flat areas of the depth map rather than spanning empty gray ranges. |
| Lateral tolerance | Discards contour segments that lie within this distance (mm) of an adjacent deeper-level contour, removing redundant passes over areas already covered. |

---

##### Step / Operation / V-Carving

Uses a V-bit or ball-nose bit whose effective cutting width varies with depth to follow the features of the depth map. Three variants are available:

- **V-Carving (Depth Map)** — the tool follows the pixel mask; cut depth at each point is given by the depth map directly.
- **V-Carving (Sub-pixel)** — the mask boundary is traced as a smooth contour rather than a pixel-grid staircase before the skeleton is extracted. This produces smoother centrelines for curved or flowing features.
- **V-Carving (Feature Width)** — cut depth at each skeleton point is determined by the width of the feature at that point (the diameter of the largest inscribed circle). Best for line art, lettering, and narrow grooves where the V-bit should naturally fill the groove width. Note: the cut depth derived from feature width may exceed the depth map value at that pixel, but is always clipped to Max cut depth.

| Parameter | Description |
|---|---|
| Min feature width | Skips features narrower than this (mm). Features too narrow for the tool to enter cleanly are excluded from the skeleton. |
| Max feature width | Limits the maximum carved feature width (mm). "Auto" derives this limit from the tool geometry and Max cut depth — the widest feature the tool can reach at full depth. |

---

##### Step / Operation / Process Region

The **Process Region** selector is the adaptive gating mechanism. It defines which areas of the depth map this step will process, based on what prior steps have already accomplished.

| Mode | What it processes |
|---|---|
| Full depth map | The entire depth map — no reference to prior steps. |
| Residual from prior step | Only regions where material remains that the depth map says should have been removed — areas where the prior step fell short of the target depth. |
| Inaccessible by prior tool | Only material that the prior tool's geometry could not remove (too narrow, wrong shape), regardless of the depth achieved. |
| Reachable residual | Intersection of "this tool can reach here" AND "the prior step left residual here." |
| Reachable inaccessible | Intersection of "this tool can reach here" AND "the prior tool could not reach here." |

Use the **Processing Mask** view (hotkey 8) and the **Prev Residual − Unaccessible** view (hotkey 6) to verify the gating is configured as intended before computing.

**Z-threshold** — a depth threshold in mm. In "Full depth map" mode: only process pixels deeper than this value. In all other modes: only process regions where the residual (or inaccessible depth) exceeds this value, skipping areas that deviate from the reference by less than this amount.

---

##### Step / Operation / Path Optimization

**Optimize path order (2-opt)** — reorders path segments using a 2-opt TSP solver to reduce total rapid travel distance between segments.

---

#### Step / Tool

| Parameter | Description |
|---|---|
| Profile | Cutter shape: Ball (hemispherical tip), End Mill (flat tip), V-Bit (angled V). |
| Diameter | Cutter diameter in mm. |
| V-bit angle | Included angle of the V-bit in degrees. Shown only when V-Bit is selected. |
| Feed rate | Cutting feed rate in mm/min. |

**Step-down controls (Raster Scan and V-Carving):**

| Parameter | Description |
|---|---|
| Max step-down | Maximum depth per roughing pass (mm). If the required depth exceeds this, multiple passes are generated automatically. |
| Final step-down | Depth of a separate finishing pass (mm). 0 means no finishing pass; the last roughing pass goes directly to full depth. |



---

### Menus

#### File Menu

| Item | Shortcut | Description |
|---|---|---|
| New Project | Ctrl+N | Discard the current steps and parameters and restart from default values. |
| Open Project | Ctrl+Shift+O | Reload a previously saved `.prj` file, restoring all parameters and step configuration. |
| Save Project | Ctrl+S | Save all current parameters and step configuration to a `.prj` file. |
| Open Image | Ctrl+O | Load a grayscale bitmap. |
| Export Toolpath | — | Export G-code for selected computed steps to a `.nc` file. |
| Settings | — | Configure machine parameters (see below). |
| Exit | Ctrl+Q | Quit the application. |

New Project, Open Project, and Exit all check for unsaved changes first: if the current project differs from what was last saved (or loaded), you're prompted to save, discard, or cancel before continuing.

**Settings dialog:**

| Parameter | Description |
|---|---|
| Safe height | Z height used for all rapid travel moves between cuts (mm). |
| Rapid feed | Feed rate for rapid moves, used only for time estimates (mm/min). |
| G-code Preamble | G-code inserted before the cutting program (spindle start, dwell, etc.). |
| G-code Postamble | G-code inserted after cutting (spindle stop, return to home, etc.). |

---

#### Compute Menu

| Item | Shortcut | Description |
|---|---|---|
| Compute Current Step | F5 | Compute the toolpath for the currently active step tab. |
| Compute All Steps | F6 | Compute all steps in sequence. Required when any step uses a gating mode that references a prior step's result. |

---

#### View Menu

All visualization modes show data for the currently active step. Cut moves and rapid moves can be toggled in any mode with **C** and **R**.

| Mode | Hotkey | Shows |
|---|---|---|
| Original | 1 | The source image, with stock margin if configured. |
| Step Path (3D) | 2 | 3D visualization of the computed toolpath. |
| Step Result | 3 | The simulated surface that remains after this step — the depth achieved, pixel by pixel. Requires toolpath to be computed.|
| Step Residual | 4 | Material that this step failed to remove. Positive (warm) values indicate regions where the target depth was not reached; cf. the "Residual from prior step" option for "Process region". Requires toolpath to be computed.|
| Step Unaccessible | 5 | Material that the current tool profile cannot remove. Positive (warm) values indicate regions where the target depth cannot be reached, regardless of strategy; cf. the "Inaccessible by prior tool" option for "Process region". Requires recomputation when tool parameters are changed.|
| Prev Residual − Unaccessible | 6 | Difference between the prior step's residual and the current tool's inaccessible material. Positive (warm) values indicate residual material from the previous step that can be removed by the present tool; cf. the "Reachable residual" option for "Process region". Requires recomputation when tool parameters are changed.|
| Prev Unaccessible − Unaccessible | 7 | Difference between the prior step's inaccessible material and the current tool's inaccessible material. Positive (warm) values indicate regions the current tool can reach that the prior tool could not; negative (cool) values indicate the current tool is more restricted than its predecessor; cf. the "Reachable inaccessible" option for "Process region". Requires recomputation when tool parameters are changed.|
| Processing Mask | 8 | Binary mask of which regions will be cut in this step, based on the Process Region setting. Live update when Z-threshold is changed.|

**Canvas navigation:**
- All modes: scroll to zoom, middle-click drag to pan, double-click to reset.
- 3D mode: drag to rotate, Shift+drag to pan, scroll to zoom.

---

## Examples

*(To be added — each example will include an image, recommended step configuration, and annotated parameter settings.)*

Planned examples:
- Simple single-pass raster carving (coin / medallion)
- Two-pass: rough ball-nose mill + V-bit detail finish
- Letter/logo carving with V-Carving (Feature Width)
- Deep relief with contour roughing + raster finish

---

## Technical Details

This section describes how each algorithm works internally — the pipeline stages that transform the depth map into a toolpath.

### Core Concepts

**Depth map.** The loaded image is converted to a float32 array where each pixel value represents the desired cut depth as a fraction of Max cut depth (0 = surface, 1 = full depth). The Z=0 colour setting controls polarity.

**Tool accessibility (morphological opening).** Before path planning, bm2gcode computes where the tool can physically reach. Grey-erosion of the depth map with a tool-shaped kernel gives the maximum depth achievable with the tool centred at each pixel. A subsequent grey-dilation recovers the reachable depth at each material point — because the tool can be centred nearby rather than directly above. Together these form a morphological opening and are the basis for all gating modes. The kernel shape depends on the tool profile:

- *Ball*: spherical cap offsets — `(R − √(R² − r²)) / max_depth`
- *V-Bit*: cone slope — `r / (tan(half-angle) × max_depth)`
- *End Mill*: zero inside the radius, infinity outside

**Accessibility caching.** The grey-erosion/grey-dilation pair above is the most expensive part of computing a step, and it depends only on the tool geometry, Max cut depth, and the work area's physical dimensions — not on step-down, threshold, gating mode, or any other per-step parameter. bm2gcode keeps a cache of these accessibility maps keyed by that combination, so recomputing a step after tweaking an unrelated parameter, or computing a later step with the same tool, reuses the previously computed result instead of rerunning the erosion/dilation. The cache is cleared whenever the source image or project changes.

**Result map.** After each step is computed, a float32 *result map* records how deep the material has been cut at every pixel. This is passed to the next step as the prior reference, enabling the Residual and Reachable gating modes.

---

### Raster Scan Pipeline

1. **Kernel construction** — Tool geometry is discretised into a depth-offset kernel and a boolean footprint disk.
2. **Accessibility map** — Grey-erosion → grey-dilation (morphological opening) yields the reachable depth at each material point.
3. **Pass planning** — Max step-down and Final step-down define an ordered sequence of depth ceilings (e.g. 0.5 mm → 1.0 mm → 1.5 mm → 1.8 mm (final)).
4. **Scan line extraction** — For each raster line at the step-over interval, contiguous spans of the gated work region are extracted by edge-detection on the binary mask. Each span becomes a full-depth segment.
5. **Depth-slicing (multi-pass)** — Full-depth segments are split into per-pass sub-segments. Roughing passes skip pixels already at target depth or shallower than the final step-down reserve. The final pass covers all remaining pixels. Segments are broken wherever the gate changes.
6. **Path ordering** — Greedy nearest-endpoint ordering, optionally refined by 2-opt TSP (see [2-opt TSP path reordering](#2-opt-tsp-path-reordering)).
7. **Terrain linking** — Short gaps between segments are bridged by a geodesic path along the roughing surface rather than a full retract (see [Geodesic terrain linking](#geodesic-terrain-linking)). Longer gaps use retract → rapid → plunge.
8. **Simplification** — Ramer-Douglas-Peucker reduces point count to within pixel resolution (see [Ramer-Douglas-Peucker simplification](#ramer-douglas-peucker-simplification)).
9. **Result simulation** — The toolpath is resampled at sub-pixel spacing and swept through a grey-dilation to compute the material state (result map) for subsequent steps.

---

### Contour Pipeline

1. **Kernel + accessibility map** — Identical to Raster (steps 1–2).
2. **Level planning** — Two modes:
   - *Uniform*: Z-levels at equal step-down intervals.
   - *Adaptive (CDF)*: Z-levels placed at quantile boundaries of the depth-map pixel distribution. This concentrates levels where many pixels share the same depth value ("plateaus"), so contour lines align with flat areas rather than spanning empty gray ranges.
3. **Contour extraction** — `skimage.find_contours` traces iso-depth curves at each Z-level on the accessibility map.
4. **Lateral pruning (optional)** — For each level, the distance transform to all committed deeper-level cuts is computed. Contour segments closer than the Lateral tolerance are discarded. Short pruned runs are restored if the retract/plunge overhead would cost more machine time than the cut saves.
5. **Per-level ordering** — Greedy nearest-neighbour with support for closed-contour rotation; optional 2-opt within each level.
6. **Cross-level assembly** — Optional 2-opt across adjacent levels, with a Z-jump penalty proportional to Max step-down. This reduces depth-change rapids between levels while discouraging jumps that exceed the step-down budget.
7. **Terrain linking** — Same geodesic approach as Raster for short inter-segment gaps.
8. **Simplification + result simulation** — Same as Raster (steps 8–9).

---

### V-Carving Pipeline

All three V-Carving variants share stages 1–2 and 4–8; they differ in how the skeleton is extracted (stage 3).

1. **Kernel + accessibility map** — Identical to Raster.
2. **V-carving mask** — Computes the region to carve:
   - Pixels shallower than Z-threshold are excluded.
   - Features narrower than Min feature width are removed by morphological opening.
   - Feature interiors wider than Max feature width (where the tool cannot reach the configured depth) are excluded using the distance transform.
3. **Skeleton extraction** — one of three methods:
   - *Depth Map*: medial axis of the mask (`skimage.morphology.medial_axis`). Cut depth at each skeleton point is taken directly from the depth map.
   - *Sub-pixel*: Voronoi diagram of the mask boundary contour. Ridges with a small opening angle — arising from closely-spaced same-side boundary points, not true centrelines — are discarded using the Attali–Montanvert criterion. Remaining ridges are discretised to 1-pixel width. Cut depth follows the depth map but the skeleton geometry is smoother than the pixel-grid medial axis.
   - *Feature Width*: same Voronoi skeleton. Cut depth at each point is derived from the inscribed-circle radius at that point (the radius of the widest circle fitting the mask there), converted to a physical depth via the tool geometry (V-bit cone formula or ball-nose formula). The resulting depth may exceed the depth map value at that pixel but is always clipped to Max cut depth.
4. **Branch decomposition** — The skeleton is split at junction pixels into polyline branches.
5. **Branch ordering** — Greedy nearest-endpoint ordering across all branches; optional 2-opt.
6. **Multi-pass depth slicing** — Same mechanism as Raster: branches are split into per-pass sub-segments when Max step-down > 0.
7. **Per-pass ordering** — Optional 2-opt.
8. **Simplification + result simulation** — Same as Raster.

---

### Supporting Algorithms

#### 2-opt TSP path reordering

Iteratively tests whether reversing a contiguous block of segments reduces total travel cost. The cost function accounts for whether a gap qualifies for terrain-linking (feed-rate cost only) or requires a retract + rapid + plunge (fixed overhead + distance). A configurable Z-jump penalty discourages large depth changes for cross-level reordering in Contour.

#### Ramer-Douglas-Peucker simplification

Reduces G-code move count while preserving path shape to within pixel resolution. Implemented iteratively (not recursively) to handle long paths without hitting Python's call-stack limit. Finds the point with the greatest perpendicular distance from the current chord; if it exceeds the tolerance, that point is kept and the chord is split there.

#### Geodesic terrain linking

Connects nearby segments with a path that follows the roughing surface, avoiding unnecessary retracts. Implemented as Dijkstra shortest-path on the pixel grid with 8-connected neighbours and 3-D arc-length as the edge cost (√(Δxy² + Δz²)). Falls back to a straight Bresenham line if no path is found within the search margin.

#### Bresenham line rasterisation

Classic integer-arithmetic line drawing, used as the geodesic-link fallback and for other pixel-grid operations.

#### Contour lateral pruning

For each contour level, a distance transform to committed deeper-level cuts identifies redundant segments. A breakeven calculation (retract/plunge overhead vs. time saved by skipping) determines whether short pruned runs should be restored before the contour is split into segments.

#### CDF adaptive level spacing

Extracts the empirical depth distribution of the accessibility map and places contour levels at quantile boundaries. This concentrates passes in depth ranges with many pixels and avoids passes in sparse gray ranges.

#### Medial axis / Voronoi skeleton

The medial axis (`skimage.morphology.medial_axis`) is the set of centres of all maximal inscribed circles in the mask — the standard definition of a shape's skeleton. The Voronoi variant builds the Voronoi diagram of the mask boundary contours and discards ridges with a small opening angle (those arise from closely-spaced same-side boundary points and are not genuine centrelines). The result is geometrically smoother for curved or flowing features.

#### G-code generation

Iterates over the ordered segments. Consecutive segments whose endpoints touch are joined as G1 feed moves. Gaps trigger a retract (G0 to safe height), a rapid to the next entry XY position, and a plunge (G1 to entry Z). The preamble and postamble are injected verbatim. A comment header records tool profile, work area dimensions, estimated run time, cut distance, and rapid distance.

---

## License

[GPL v3](LICENSE) — any distributed modifications must remain open source under the same license.
