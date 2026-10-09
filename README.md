# bm2gcode

bm2gcode converts a depth map into CNC G-code for relief carving. The depth map is composited on a **canvas** from one or more **sources** — grayscale images and/or 3D meshes (STL) — each independently positioned, scaled, rotated, and blended together.

The central idea is an **adaptive multi-step workflow**. This allows multiple cutting steps to be defined using different algorithms and tools. Each step can be configured to process only the residual material left behind by previous steps, such as material that the previous step failed to remove, or regions that the previous tool geometry could not access. This makes it easy to combine strategies. For example, a large end mill could be used to rough out flat regions, followed by a ball bit to target curved surfaces, then a V-bit to cut fine details that the ball bit could not reach, and finally a Cutout pass to free the finished part from the surrounding stock. Each step can also choose which sources on the canvas it sees, so different steps can carve different subsets of the same composition.

Generated toolpaths can be refined with optional **tool path dress-ups**: ramp-in eases the tool into the stock with a zig-zag descent instead of a straight plunge, back-off pulls the tool away from a just-cut wall before retracting to avoid rubbing marks, arc-rounding smooths a rapid move's sharp corners so the machine doesn't have to stop twice per rapid, and — for Cutout steps only — tabs leave short uncut bridges around the perimeter so the finished part stays attached to the surrounding stock. Each step has its own dress-up settings, configured via the **Dress-up…** button on that step's tab; a newly added step inherits the dress-up settings of the step it was added after.

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

1. **File → Open Source…** (Ctrl+O) — load a grayscale image or an STL mesh; the file's extension decides which. This creates the canvas's base source and, for an image, sizes the canvas to it; for an STL, an orientation dialog lets you populate the canvas's physical size from the mesh's bounding box.
2. Review the **Canvas** tab — physical width/height, precision, max cut depth, and other canvas-wide settings (see [Canvas](#canvas) below).
3. Optionally add more sources with **Add Source** (on the Canvas tab, or on any source's own tab) and arrange them with each source's Offset, Rotation, Scale, and Blend mode (see [Sources](#sources) below).
4. Configure one or more **steps** — choose algorithm, tool, region targeting, and which sources it should see for each.
5. **Compute → Compute All Steps** (F6) — generate toolpaths.
6. Use the **View** menu (hotkeys 1–8) to inspect the 3D path, simulated result, residual depth, processing mask, etc.
7. **File → Export Toolpath…** — save a `.nc` G-code file for your CNC machine.

Save your work at any time with **File → Save Project…** (Ctrl+S); reload it later with **File → Open Project…** (Ctrl+Shift+O). Project files store the canvas, every source's placement, all step configuration, and a path back to each source's original image or STL file (re-rasterized with the same settings on reload).

---

## Parameter Reference

### Canvas

The Canvas tab is the notebook's permanent first tab, above the source tabs. Its settings are canvas-wide — shared by every source and every step, not tied to any one of them.

| Parameter | Description |
|---|---|
| Width / Height | Physical size of the carving area, in mm. The **Lock** checkbox next to Height maintains the current aspect ratio when either field changes; uncheck it to set width and height independently. Changing Width/Height rescales every source's offset so the composed layout is preserved (Scale needs no rescaling — see Scale below), and warns first if the resulting pixel grid would be very large. |
| Min. px/mm | The floor on the canvas's working depth-map grid density. If no image source is present (an STL-only or empty canvas), this is the grid density, exactly. If one or more image sources are present, the canvas may automatically use a *finer* density — up to whatever the highest-resolution visible image actually offers, bounded by Max Megapixels — so imported detail isn't thrown away; it never goes coarser than this floor. See **Effective resolution** below for what was actually chosen. |
| Max Megapixels | Upper bound on the canvas's total pixel count (Width × Height at the chosen density), so a high-resolution image (or a small physical canvas) can't blow the working grid up indefinitely. Applies whether or not an image source is present — if Min. px/mm alone would already exceed it for the current Width/Height, you're asked to confirm before proceeding. |
| Effective resolution | Read-only: the pixel grid size and density the canvas actually settled on, between the Min. px/mm floor and the Max Megapixels ceiling. Recomputed when a source is added, when Width/Height/Min. px/mm/Max Megapixels change, via **Compute → Recompute Canvas Resolution**, or automatically before each Compute Step — not continuously while dragging a source's Scale/Offset. |
| Max cut depth | Maximum cut depth in mm. The pixel representing full depth (black when z=0 colour = "white", white when z=0 colour = "black") is carved to this depth. |
| Material | The workpiece material, chosen from the [Tool Library](#tool-library)'s materials. It decides which cutting data a step linked to a library tool receives. Choosing a different material lists the linked steps and asks which of them should receive that material's cutting data; nothing changes without confirmation. Saved with the project. |
| Stock margin | Flat uncarved border around the canvas, in mm. |
| Margin depth | How deep to cut the stock margin, as a percentage of Max cut depth. |
| Background depth | Depth, as a percentage of Max cut depth, given to any canvas pixel that no source covers. |
| z=0 colour | Which pixel colour represents the uncut surface. "white": white = surface (z = 0), black = deepest cut. "black": the reverse. |
| G-code origin | Which corner — or the center — of the work area corresponds to X0 Y0 in the G-code output. |

The **Add Source** button at the bottom of the Canvas tab adds a new source at the bottom of the stack (or establishes the base source if the canvas is empty) — see [Sources](#sources).

---

### Sources

Each source — an opened image or imported STL — gets its own tab in the notebook, labeled with its stack position and name (e.g. `1: photo.png`, `2: logo.stl`), in stack order from bottom to top. A source's tab holds its compositing properties:

| Parameter | Description |
|---|---|
| Offset X / Offset Y | Position relative to the canvas center, in mm. +X is right, +Y is up. |
| Rotation | Counter-clockwise rotation, in degrees. |
| Scale | Size as a percentage of the canvas's own physical size, aspect-preserving: at 100%, the source is inscribed in the canvas rectangle along whichever of its own axes binds ("contain"), independent of the source's native resolution or the canvas's pixel density. An STL source's default Scale reproduces the mesh's true physical size (see [Import STL](#import-stl)). |
| Mirror | Reflects the placed source across one of the stock flip axes used by the [Registration Holes](#step--operation--registration-holes) step, for setting up the back side of a two-sided job: **Vertical axis** or **Horizontal axis** (the stock's centre lines), or **Diagonal (+45°)** / **Diagonal (−45°)** (the 45° line through the G-code origin, in that direction). The reflection is applied after Offset/Rotation/Scale, so an offset source's position is mirrored along with its content. A back-view depth image (rendered from behind the object), or an STL imported rotated 180° about the matching axis, is already in flipped orientation and needs no mirror; Mirror is for back-side depth maps drawn in top-view coordinates. None by default. |
| Blend mode | How this source combines with everything beneath it — see below. The bottom-most source in the stack always reads "Replace (base source)" and can't be changed, since there's nothing under it to blend with. |
| Opacity | How strongly this source's blended result replaces what's beneath it, as a percentage; 0 hides it. |
| Min depth / Max depth | Remaps this source's black/white pixels into a depth range, each a percentage of the canvas's Max cut depth: white maps to Min depth and black to Max depth. Setting Max depth below Min depth flips the source's polarity. |
| Visible | Whether this source is included in the depth map for the *currently selected step* — see [Per-step source visibility](#per-step-source-visibility) below. |

**Blend modes**, applied in depth-space (0 = surface, 1 = full depth) against the composite of everything below:

| Mode | Result |
|---|---|
| Replace | This source's surface (depth value) outright. |
| Merge high | The higher of the two (whichever surface is shallower). |
| Merge low | The deeper of the two (whichever surface is deeper). |
| Add | Base depth − this source's depth (raises the surface). |
| Subtract | Base depth + this source's depth (deepens the surface). |

Each source tab also has **Duplicate** (clones the source and inserts the copy right after it), **Add Source** (inserts a newly opened source immediately after this one in the stack), and **Remove this source** (removes it — including the last remaining source, which leaves the canvas empty until a new one is added; canvas settings are unaffected).

---

#### Per-step source visibility

Each step has its own Visible/hidden choice for every source on the canvas: when a step tab is selected, each source's **Visible** checkbox shows and edits that step's own setting rather than a canvas-wide one, so different steps can carve different subsets of the same source stack — for example, one step carving a photo while a second step carves an overlaid logo, or a finishing step that excludes a source a roughing step included. A newly added step inherits the visibility of the step it was added after.

---

### Steps

Each cutting step appears as a numbered tab. Use the **Add Step After** and **Remove this step** buttons at the bottom of each tab to manage the sequence; **Add Step After** copies all settings — including dress-up — from the step it was added after. The **Dress-up…** button opens that step's tool path dress-up settings (see [Step / Dress-up](#step--dress-up) below). Each tab also contains an **Operation** group and a **Tool** group.

---

#### Step / Operation

**Algorithm** selects the cutting strategy. The parameters displayed below the selector adapt to the chosen algorithm.

| Algorithm | What it does | Typical use |
|---|---|---|
| [Raster Scan](#step--operation--raster-scan) | Sweeps the tool in parallel lines (or a rectangular spiral), following the depth map's relief along each line. | Roughing and finishing of general 3D reliefs. |
| [Contour](#step--operation--contour) | Cuts the relief as stacked iso-depth (waterline) contours, level by level. | Roughing terraced reliefs; finishing steep walls a raster leaves stepped. |
| [V-Carving (Depth Map)](#step--operation--v-carving) | Runs a V-bit or ball-nose along the medial axis of the carved region, with cut depth taken from the depth map. | Engraving where the depth map itself encodes the desired depth. |
| [V-Carving (Sub-pixel)](#step--operation--v-carving) | As Depth Map, but with a smoother centreline traced from the region's sub-pixel boundary. | Curved or flowing artwork where a pixel-grid centreline looks jagged. |
| [V-Carving (Feature Width)](#step--operation--v-carving) | Runs along the centreline, with cut depth set by the feature's local width so the V-bit's flanks touch both edges. | Classic V-carved lettering and line art. |
| [V-Carving (Hybrid)](#step--operation--v-carving) | Feature Width's centreline plus waterline contour rings that clear features too wide for the V-bit to fill in one stroke, down to Max cut depth. V-Bit only. | V-carved lettering and artwork with wide strokes or open areas, leaving sharp V-shaped walls and corners. |
| [V-Carving Clean-up](#step--operation--v-carving-clean-up) | Pocket-clears the flat floor that a preceding V-Carving (Hybrid) step's V-bit couldn't reach, without touching its sloped walls. | Flattening the bottom of wide V-carved areas with an end mill. |
| [Cutout](#step--operation--cutout) | Cuts straight down around the outer silhouette, through the full stock thickness, with optional tabs. | Freeing the finished part from the stock. |
| [Pocket](#step--operation--pocket) | Clears a flat-bottomed recess at a fixed Target depth with zig-zag, spiral, or onion-skin infill. | Flat recesses, or facing/flattening the whole stock. |
| [Registration Holes](#step--operation--registration-holes) | Bores pin holes symmetric about a flip axis, independent of the depth map. | Aligning the stock for two-sided milling. |

---

##### Step / Operation / Raster Scan

Sweeps the cutter in parallel lines across the material. 

| Parameter | Description |
|---|---|
| Step-over | Lateral spacing between scan lines, as a percentage of tool diameter. |
| Line offset | Shifts alternate scan lines by this percentage, so that the in-between lines can be cut in a separate step with a different setting. |
| Scan direction | Horizontal, Vertical, Diagonal (45°), or Diagonal (135°) scan in parallel lines at that angle. Spiral (Outside-In) or Spiral (Inside-Out) instead traces the work area as one continuous coil of shrinking (or growing) concentric rectangles, without lifting the tool between rows and columns — see [Spiral raster pattern](#spiral-raster-pattern). |
| Cut direction | Climb or Conventional forces every scan line to travel in one consistent direction relative to the spindle rotation, at the cost of some travel-time optimisation — see [Climb / conventional cut direction](#climb--conventional-cut-direction). Either leaves direction unconstrained, so path ordering is free to reverse segments for shorter travel. For a Spiral scan direction, Climb/Conventional instead pick which way the coil winds (clockwise or counter-clockwise) rather than reversing individual lines. |

---

##### Step / Operation / Contour

Traces the depth map as a stack of iso-depth contours, like topographic lines on a map. 

| Parameter | Description |
|---|---|
| Z increment | Vertical distance between contour levels (mm). |
| Adaptive levels (CDF) | Distributes levels non-uniformly. Instead of equal Z spacing, levels are placed where many pixels share the same gray value ("plateaus"), so contour lines align with the flat areas of the depth map rather than spanning empty gray ranges. |
| Lateral tolerance | Discards contour segments that lie within this distance (mm) of an adjacent deeper-level contour, removing redundant passes over areas already covered. |
| Cut direction | Climb or Conventional forces every contour to travel in one consistent direction relative to the spindle rotation, at the cost of some travel-time optimisation — see [Climb / conventional cut direction](#climb--conventional-cut-direction). Either leaves direction unconstrained, so path ordering is free to reverse segments for shorter travel. |

---

##### Step / Operation / V-Carving

Uses a V-bit or ball-nose bit whose effective cutting width varies with depth to follow the features of the depth map. Four variants are available:

- **V-Carving (Depth Map)** — the tool follows the pixel mask; cut depth at each point is given by the depth map directly.
- **V-Carving (Sub-pixel)** — the mask boundary is traced as a smooth contour rather than a pixel-grid staircase before the skeleton is extracted. This produces smoother centrelines for curved or flowing features.
- **V-Carving (Feature Width)** — cut depth at each skeleton point is determined by the width of the feature at that point (the diameter of the largest inscribed circle). Best for line art, lettering, and narrow grooves where the V-bit should naturally fill the groove width. Note: the cut depth derived from feature width may exceed the depth map value at that pixel, but is always clipped to Max cut depth.
- **V-Carving (Hybrid)** — combines the Feature Width centreline with waterline contour rings. Where a feature is narrow, the centreline alone carves it, exactly as in Feature Width. Where a feature is wider than the V-bit can fill from its centreline at Max cut depth, Feature Width would cut a single deep groove and leave material standing on either side; Hybrid instead also traces contour rings around the feature at every pass level down to Max cut depth, each ring offset from the feature edge by exactly as far as the V-bit's cone reaches at that depth. The result is a V-shaped wall along every edge — with sharp corners preserved by the centreline — around a flat floor at Max cut depth. That floor, inside the deepest ring, is wider than the V-bit can clear; follow up with a [V-Carving Clean-up](#step--operation--v-carving-clean-up) step to flatten it with an end mill. Hybrid requires a V-Bit tool.

| Parameter | Description |
|---|---|
| Min feature width | Skips features narrower than this (mm). Features too narrow for the tool to enter cleanly are excluded from the skeleton. |
| Max feature width | Limits the maximum carved feature width (mm). "Auto" derives this limit from the tool geometry and Max cut depth — the widest feature the tool can reach at full depth. |
| Cut direction | V-Carving (Hybrid) only. Climb or Conventional forces every contour ring to travel in one consistent direction relative to the spindle rotation — see [Climb / conventional cut direction](#climb--conventional-cut-direction). Either leaves direction unconstrained. The centreline strokes are unaffected, since a centreline has no inside or outside wall to keep on a particular side of the tool. |

For V-Carving (Hybrid), the Max step-down and Final step-down fields under [Step / Tool](#step--tool) also set the spacing of the contour rings: one ring level per pass, from the first step-down down to Max cut depth (a single ring at Max cut depth when Max step-down is 0). If the V-bit's diameter is too small for its cone to reach one step-down before the shank would take over, computing the step shows a warning — the deepest centreline cuts in wide areas then can't be a clean V-taper.

---

##### Step / Operation / Cutout

Traces the outer perimeter of the depth map's silhouette and cuts straight down through the full stock thickness, freeing the finished part from the surrounding waste material — unlike Raster Scan, Contour, and V-Carving, which all follow the depth map's own relief detail. Restricted to an End Mill tool, since a straight vertical wall only makes sense with a flat-bottomed bit. Interior holes in the silhouette (e.g. the counter of a letter "O") are always filled before tracing, so only the outer perimeter of each shape is cut; cutting interior holes is not supported yet.

| Parameter | Description |
|---|---|
| Cut depth | Absolute total cut depth, in mm — independent of, and typically deeper than, Max cut depth, since this must cut all the way through the stock rather than follow the depth map's relief detail. |
| Max step-down | Maximum depth per roughing pass (mm). If Cut depth exceeds this, multiple passes are generated automatically. |
| Final step-down | Depth of a separate finishing pass (mm). 0 means no finishing pass. |
| Boundary offset | Lateral-only outward push of the tool-centre path beyond the minimum tool-radius clearance, in mm. 0 = tool tangent to the traced silhouette edge. |
| Cut direction | Climb or Conventional forces the perimeter cut to travel in one consistent direction relative to the spindle rotation — see [Climb / conventional cut direction](#climb--conventional-cut-direction). Either leaves direction unconstrained. |

There's no Process Region selector for Cutout — **Split level** is always applied to the full depth map, never to a prior step's residual or accessible surface, so what any prior step accomplished has no bearing on which pixels get cut. Split level is the depth-map split point between "the object" and "the background", in mm. Since depth map polarity — whether the object is the deeply-cut region or the shallow/uncarved region — depends on how the source was authored, both split directions are tried automatically; see [Cutout Pipeline](#cutout-pipeline) below.

---

##### Step / Operation / Pocket

Clears a flat-bottomed recess: every position where the tool can be centred and still reach Target depth without its own footprint overhanging shallower material is treated as the pocket's interior, and that interior is cleared flat at Target depth using the same scan-line/spiral infill as Raster Scan — the tool centre naturally stays a full tool-radius clear of any boundary, the same way Contour's own iso-depth contours do. Unlike Raster Scan, Contour, and V-Carving, the pocket floor is flat rather than following the depth map's own relief detail, and unlike Cutout, only the interior is cleared — no wall-finishing pass is cut around the pocket boundary, and the tool-radius clearance leaves interior corners rounded rather than sharp, the way any pocketing operation with a finite tool radius does. When the depth map contains more than one disjoint region at or beyond Target depth, each region is cleared independently, with its own infill grid or spiral coil sized and centred on that region rather than on the stock as a whole.

| Parameter | Description |
|---|---|
| Target depth | Depth (mm) defining the pocket floor and which positions count as pocket interior — the tool must be able to centre there and still reach this depth without its footprint overhanging shallower material. Used literally, including 0 (the default): since every reachable position clears a depth of 0, a Target depth of 0 makes the whole reachable surface pocket interior, cleared flat at Z=0 — a full-stock flattening/facing pass. Check **Auto** to disable manual entry and instead track the depth map's own actual deepest point in mm (the peak of the composited depth map times Max cut depth, kept in sync as sources or Max cut depth change) rather than a hand-typed value — this clears only the deepest region(s), the same result the old "0 = auto, use Max cut depth" convention gave whenever some pixel actually reached full depth. |
| Step-over | Lateral spacing between infill lines, as a percentage of tool diameter. |
| Line offset | Shifts alternate infill lines by this percentage, so that the in-between lines can be cut in a separate step with a different setting. |
| Scan direction | Zig-Zag (0°), (45°), (90°), or (135°) scan each pocket region in parallel lines at that angle, same as Raster Scan's own Scan direction, but wherever a gap between two lines can be bridged by walking the region's own boundary rather than a geodesic search, it is — see [Pocket Pipeline](#pocket-pipeline). Spiral (Outside-In) or Spiral (Inside-Out) instead traces each pocket region as one continuous coil of shrinking (or growing) concentric rectangles, sized and centred on that region — see [Spiral raster pattern](#spiral-raster-pattern). Onion Skin (Outside-In) or Onion Skin (Inside-Out) traces each pocket region as one continuous chain of rings offset inward from that region's own boundary by Step-over, following the region's actual silhouette rather than a rectangular grid — see [Onion skin pocket pattern](#onion-skin-pocket-pattern). |
| Cut direction | Climb or Conventional forces every infill line to travel in one consistent direction relative to the spindle rotation, at the cost of some travel-time optimisation — see [Climb / conventional cut direction](#climb--conventional-cut-direction). Either leaves direction unconstrained. For a Spiral scan direction, Climb/Conventional instead pick which way each region's coil winds; for an Onion Skin scan direction, Climb/Conventional instead pick which way each ring winds. |
| Boundary offset | Lateral-only push of the tool-centre path away from the pocket wall, beyond the minimum tool-radius clearance, in mm. 0 = tool tangent to that wall; larger values retreat the tool centre further from the wall, leaving more material uncut near the boundary. Same "away from the boundary" direction as Cutout's own Boundary offset field — see [Step / Operation / Cutout](#step--operation--cutout) — but with no vertical component, since Pocket's cut depth is already fixed independently by Target depth. Shown here instead of the shared Extra offset field under [Step / Tool](#step--tool). |

There's no Process Region selector for Pocket, either — Target depth is always applied to the full depth map, never to a prior step's residual or accessible surface, so what any prior step accomplished has no bearing on which pixels count as pocket interior.

See also [V-Carving Clean-up](#step--operation--v-carving-clean-up), which reuses this same pipeline against a substituted reference surface instead of the raw depth map.

---

##### Step / Operation / V-Carving Clean-up

Runs the exact [Pocket](#step--operation--pocket) pipeline described above — same parameters, same flat-floor infill — but targets the flat interior that a preceding **V-Carving (Hybrid)** step's V-bit couldn't reach: Hybrid mode traces waterline contours down to Max cut depth, but the floor inside the deepest ring, where a feature is wider than the V-bit's cone can reach at that depth, is never cleared (see [V-Carving Pipeline](#v-carving-pipeline)). Clean-up flattens exactly that leftover floor, with any tool — not necessarily the V-bit used for V-Carving.

The one difference from Pocket: Stage 1's achievable-depth erosion (see [Pocket Pipeline](#pocket-pipeline)) runs against the nearest preceding V-Carving (Hybrid) step's own idealized-V-bit achievable-depth surface instead of the raw depth map — i.e. Clean-up's tool-kernel erosion is applied a *second* time, on top of the V-bit's own. This makes the resulting boundary exact for any clearing tool profile (End Mill, Ball, or V-bit, at any angle or diameter): the clearing tool retreats exactly as far as its own geometry requires to avoid gouging the V-bit's actual sloped wall, never more and never less — a plain scalar offset can't express this in general, since how far a Ball or V-bit clearing tool must retreat depends on depth the same way the original V-bit's own reach did. Computing a Clean-up step with no preceding V-Carving (Hybrid) step in the sequence raises a validation error.

**Target depth** and **Boundary offset** keep their ordinary Pocket meaning and are both freely editable — Boundary offset is a genuinely optional *extra* margin layered on top of the already-exact boundary above, same as it is for Pocket. Target depth's literal-0 default means a fresh Clean-up step with Target depth untouched flattens its *entire* reachable surface at Z=0, not just the V-bit's leftover floor — check **Auto** (see [Step / Operation / Pocket](#step--operation--pocket)) to clear only the deepest reachable point(s) instead, which for a typical Clean-up step is exactly that leftover floor.

The tool used for Clean-up (profile, diameter, feed rate, etc.) is independent of the source V-Carving step's tool.

---

##### Step / Operation / Registration Holes

Bores holes for registration pins, so the stock can be turned over and carved on its back side in exact alignment with the front. The holes are placed on the stock alone — the depth map, Process Region and prior steps play no part in where they go — and are symmetric about a chosen flip axis, so after flipping, the same pins fit the same holes. Hole depth is absolute and typically goes through the stock into the spoilboard: the pins then stay in the spoilboard while the stock is turned over. Restricted to an End Mill tool.

A hole the tool's own diameter is plunged straight down, pecked in Max step-down increments with a lift out of the hole after each peck to clear chips. A larger hole is bored as a helix: the tool spirals down along the hole wall, dropping at most Max step-down per revolution, finishes with one flat circle at full depth, then moves back to the hole centre before retracting. Max step-down is the shared field under [Step / Tool](#step--tool); there's no Final step-down.

| Parameter | Description |
|---|---|
| Flip axis | **Vertical axis**: two holes on the stock's vertical centre line, one near the bottom edge and one near the top. **Horizontal axis**: two holes on the horizontal centre line, near the left and right edges. **Diagonal**: an L of three holes, symmetric about the 45° line through the G-code origin — see below. |
| Diagonal direction | Which 45° line a Diagonal flip uses: +45° (rising to the right) or −45°. Shown only for a Center G-code origin; with a corner origin the direction follows from the corner (+45° through Bottom-Left/Top-Right, −45° through Top-Left/Bottom-Right). |
| Hole diameter | Finished hole diameter, in mm. Must be at least the tool diameter. |
| Hole depth | Absolute hole depth, in mm — independent of Max cut depth. |
| Edge inset | Distance from the stock edge (margin included) to each hole centre, in mm. **Auto** uses half the Stock margin — the middle of the margin strip, clear of the artwork — or 10 mm with no margin. |
| Spacing | Diagonal only: length of each arm of the L, from the corner hole to each arm hole, in mm. **Auto** uses the shorter stock side minus twice the Edge inset, so both arm holes sit the same inset from the edges as the corner hole. |
| Cut direction | Winding of a helical bore: Climb (and Either) run counter-clockwise, Conventional clockwise — for a clockwise spindle, counter-clockwise travel along an inside wall is climb milling. |

Hole placement is shown in the 2D views before computing, as dashed outlines with the flip axis drawn through them (toggle with **H** — see [View Menu](#view-menu)). The tab also shows a one-line reminder of how to set up the back side for the current flip axis and G-code origin. Holes aren't checked against the stock edges or the artwork; placement is left to the user.

**Two-sided milling workflow**

1. In the front-side project, add a Registration Holes step and run it with the rest of the front side. Its holes go into the spoilboard; set pins in them.
2. Flip the stock over the flip axis onto the pins. **Don't re-zero X/Y**: the pins locate the stock relative to the machine's unchanged zero. Re-zeroing Z is fine.
3. Carve the back side from a second project with the same Stock margin and G-code origin, and the same canvas Width/Height — swapped for a Diagonal flip on non-square stock. Give each back-side source whatever Mirror its depth map needs (see [Sources](#sources)).

Why this works: the back-side G-code needs its origin at the same physical point as the front side's. A flip about the stock's vertical or horizontal centre line maps the stock's footprint onto itself, so every G-code origin (any corner, or Center) lands back on the same point, and the back-side canvas has the same size. A 45° flip turns a W×H footprint into an H×W one, and only points on the flip line stay put — which is why the Diagonal flip line always passes through the G-code origin. The origin corner (or, for Center, the stock centre, which stays the centre of the transposed footprint) is then the same physical point on both sides, and the back-side canvas has Width and Height swapped.

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

Cutout's **Split level** and Pocket's **Target depth** are the same underlying field, relabeled, but no longer share one convention: Split level keeps the "0 = auto, split at the depth-map midpoint" behaviour (no Auto checkbox — it's shown only for the Pocket family), while Target depth treats 0 literally (a full-stock flattening pass at Z=0) and exposes its own **Auto** checkbox for the old "track the depth map's deepest point" behaviour instead; see their own sections above.

---

##### Step / Operation / Path Optimization

**Optimize path order (2-opt)** — reorders path segments using a 2-opt TSP solver to reduce total rapid travel distance between segments. Greyed out and has no effect when **Cut direction** is set to Climb or Conventional, since a locked direction cannot be reordered — see [Climb / conventional cut direction](#climb--conventional-cut-direction). For [Pocket](#step--operation--pocket), it's also greyed out whenever **Scan direction** is Zig-Zag or Onion Skin, regardless of Cut direction — both use their own dedicated ordering (region-sequential for Zig-Zag, fixed-orientation for Onion Skin — see [Pocket Pipeline](#pocket-pipeline)) that never consults 2-opt; only Spiral's ordering can fall back to it.

**Avoid cutting air** — removes the parts of every pass where the tool would only feed through air, because an earlier step has already removed the material there, or because the move stays at the stock surface (Z=0) where there's nothing to cut. Applies to every algorithm and to every pass, including the final pass and single-pass steps: a refining step's finishing pass over an area the previous step already cut to depth is skipped as well. An air run at the start or end of a segment is always trimmed; an air run in the middle of a segment is removed (splitting it in two) only if it's long enough that the extra retract and plunge take less time than feeding straight through — the same break-even rule as Contour's Lateral tolerance, using Safe height and Rapid feed from Settings. The step's result map is unchanged. Off by default. See [Air-cut pruning](#air-cut-pruning).

---

#### Step / Tool

| Parameter | Description |
|---|---|
| Library | The [Tool Library](#tool-library) tool this step uses, or (none). Lists only tools whose profile the step's algorithm accepts. Choosing a tool copies its profile, diameter, V-bit angle and — for the project's Material — feed rate, plunge rate, spindle speed and step-down into the step's fields, and links the step to the tool. The fields stay editable. A status next to the field shows when the step no longer matches the library: **● modified** (cutting data differs), **● geometry differs** (profile, diameter or V-bit angle differs — e.g. after switching to an algorithm that forces another profile), **● no data for *material*** (the tool has no cutting data for the project's Material; only the geometry was copied) or **● not in library** (the tool was deleted; the step keeps its values). The **⋯** button offers Re-apply from library, Update library tool from this step (writes the step's cutting data for the project's Material into the tool; never its geometry), Save as new library tool…, Unlink and Open Tool Library…. |
| Profile | Cutter shape: Ball (hemispherical tip), End Mill (flat tip), V-Bit (angled V). Cutout and Registration Holes restrict this to End Mill only. |
| Diameter | Cutter diameter in mm. |
| V-bit angle | Included angle of the V-bit in degrees. Shown only when V-Bit is selected. |
| Extra offset | Stock-to-leave allowance (mm), applied both laterally (x/y) and vertically (z). The step deliberately stops this far short of the target depth map, so a following step (typically the same tool with Extra offset = 0) can refine the result. 0 = none. Not shown for Cutout or Pocket, which each have their own separate Boundary offset field instead — see [Step / Operation / Cutout](#step--operation--cutout) and [Step / Operation / Pocket](#step--operation--pocket). |
| Feed rate | Cutting feed rate in mm/min. |
| Plunge rate | Feed rate in mm/min for vertical descents: plunges, and pecks inside a cut (e.g. Registration Holes' pecked holes). Ramps keep the cutting feed, since they move mostly sideways. 0 = use Feed rate. |
| Spindle speed | Spindle speed in RPM. When > 0, the step's G-code starts with an `S` word followed by the spin-up dwell from [Settings](#file-menu); consecutive exported steps at the same speed don't repeat it. 0 = no `S` word (the preamble's speed applies). |

**Step-down controls (Raster Scan and V-Carving):**

| Parameter | Description |
|---|---|
| Max step-down | Maximum depth per roughing pass (mm). If the required depth exceeds this, multiple passes are generated automatically. |
| Final step-down | Depth of a separate finishing pass (mm). 0 means no finishing pass; the last roughing pass goes directly to full depth. |

V-Carving has its own independent pair of fields with the same labels and
meaning — changing Raster Scan's Max/Final step-down has no effect on a
V-Carving step, and vice versa.

Contour also shows a **Max step-down** field here, but with a different
meaning: it bounds how many redundant levels lateral pruning may collapse
and the window (in Z-levels) that contour ordering and 2-opt are allowed
to mix contours across — see
[Contour lateral pruning](#contour-lateral-pruning). Contour's actual
Z-level spacing is the separate **Z increment** field under
[Step / Operation / Contour](#step--operation--contour).

Cutout shows its own independent Max/Final step-down pair too, alongside
its own Cut depth and Boundary offset fields — see
[Step / Operation / Cutout](#step--operation--cutout).

Registration Holes shows only **Max step-down**, as its helix pitch (Z drop
per revolution) or peck depth — see
[Step / Operation / Registration Holes](#step--operation--registration-holes).

---

#### Step / Dress-up

The **Dress-up…** button on each step's tab opens that step's tool path dress-up settings. Ramp-in, back-off, and arc-rounding are optional, off-by-default G-code refinements that only change how the tool travels between cuts — the cut geometry itself is unaffected; see [G-code generation](#g-code-generation) for how they fit into path emission. Tabs is the exception: available only for Cutout steps, it changes the cut geometry itself, applied earlier in the Cutout pipeline rather than at G-code emission time — see [Cutout Pipeline](#cutout-pipeline). Dress-up settings are per-step: a step added with **Add Step After** inherits the settings of the step it was added after, and different steps (e.g. a roughing pass vs. a finishing pass) can use different dress-up configurations.

**Ramp-in** — replaces a straight vertical plunge with a zig-zag ramp for the portion of the descent that lies inside the stock (below the surface, z < 0). Any travel above the surface remains a plain vertical plunge. The ramp retraces the upcoming cut path's own footprint — so it never cuts outside it — and never rises above the depth the final pass leaves at that position.

| Parameter | Description |
|---|---|
| Ramp angle | Target descent angle of the zig-zag, in degrees. |
| Max ramp length | How far ahead along the upcoming cut path the zig-zag is allowed to range, in mm. |

**Back off before retract** — before retracting to safe height, backs the tool away a short distance from the wall it just cut by retracing its own path backwards, to avoid rubbing or burning the surface while lifting. If the last move had too little lateral travel to cover the distance (e.g. a near-vertical raise), the back-off continues into the moves before it. It rises at 45° while backing off, but never drops below the surface the tool just cut — where the path came down more steeply than that, the back-off follows the path's own slope instead.

| Parameter | Description |
|---|---|
| Back-off distance | How far to back away before retracting, in mm. |

**Arc-round rapid corners** — rounds the two sharp 90° turns of a rapid move (retract → traverse → plunge) into tangent arcs, so the tool can carry speed through the corner instead of coming to a full stop twice. Both corners always share the same radius. Unlike the plain rapid it replaces, the arc is emitted as coordinated G1 motion (see [G-code generation](#g-code-generation)), so it needs its own literal feedrate rather than reusing the Rapid feed setting (which is time-estimate only and never appears in the G-code).

| Parameter | Description |
|---|---|
| Arc radius | Requested corner radius, in mm. Automatically reduced when there isn't enough room — e.g. a shallow cut, a short rapid, or a small safe height — and never exceeds the safe height (a global machine parameter, set under File → Settings), since a corner's lateral sweep is only safe above the stock surface. |
| Arc feed | Literal feedrate commanded on the arc's G1 moves, in mm/min. |

**Tabs** — leaves short, uncut bridges around the perimeter of a Cutout step's closed cutting loop, so the finished part stays attached to the surrounding stock. Unlike the three dress-ups above, tabs change the cut geometry itself rather than just how the tool travels between cuts — see [Cutout Pipeline](#cutout-pipeline). Available only for Cutout steps; greyed out (but still editable/persisted) for every other algorithm.

| Parameter | Description |
|---|---|
| Tab count | Number of tabs, evenly spaced around each closed loop's perimeter. A step whose silhouette has several disjoint shapes (e.g. separate letters) gives each shape its own full set of tabs. |
| Tab width | Each tab's footprint along the cut path, in mm. |
| Tab height | How much material each tab leaves uncut, in mm — within a tab, the tool stops this far short of full depth, however many passes that takes (see [Cutout Pipeline](#cutout-pipeline)). |
| Phase | Shifts every tab together, as a percentage of the spacing between two adjacent tabs. Useful for nudging a tab off an unwanted spot (e.g. a sharp corner). |
| Tab angle | Wall angle of the tab's taper, from horizontal, in degrees. 90° is a vertical, rectangular-walled tab (the tool steps straight up onto it); lower values taper the tab into a trapezoid instead, so the tool ramps onto and off it. |

Being a Z-only change, a tab doesn't alter the toolpath's outline, so it's otherwise invisible in the 2D preview. Each tab is instead marked there with a filled circle sized to its width, toggled together with cut moves and rapids — see [View Menu](#view-menu).

---

### Menus

#### File Menu

| Item | Shortcut | Description |
|---|---|---|
| New Project | Ctrl+N | Discard the current canvas, sources, steps, and parameters and restart from default values. |
| Open Project… | Ctrl+Shift+O | Reload a previously saved `.prj` file, restoring the canvas, every source's placement, and all step configuration. |
| Save Project… | Ctrl+S | Save the current canvas, sources, and step configuration to a `.prj` file. |
| Open Source… | Ctrl+O | Load a grayscale image or import an STL mesh as the canvas's base source, replacing the whole project (see below). |
| Export Toolpath… | — | Export G-code for selected computed steps to a `.nc` file. |
| Settings… | — | Configure machine parameters (see below). |
| Exit | Ctrl+Q | Quit the application. |

New Project, Open Project…, and Exit all check for unsaved changes first: if the current project differs from what was last saved (or loaded), you're prompted to save, discard, or cancel before continuing.

**Open Source…** opens a single file picker that accepts either an image (PNG, JPG, BMP, GIF, TIFF, WEBP) or an STL mesh; which one you pick decides what happens next. An image becomes the canvas's base source directly. An STL brings up an **Import STL** dialog first:

| Parameter | Description |
|---|---|
| Rotate around X / Y / Z axis | Rotates the mesh by 0°, 90°, 180°, or 270° around each axis before rasterizing, applied in X → Y → Z order. A 90°/270° rotation around X or Y turns a *side* of the mesh into the top-down view, so a side profile can become the depth map, not just the mesh's native top or bottom face. |
| Set width/height/max depth from mesh bounding box | Shown only when this STL is establishing the canvas's base source. When checked (the default), populates the Canvas tab's Width, Height, and Max cut depth from the rotated mesh's bounding box, assumed to be in millimeters — the common STL convention — and locks the aspect ratio. A live preview shows the resulting physical size as the rotation changes. |

The rasterization resolution itself isn't chosen in this dialog — it follows automatically from the canvas's effective resolution (see Min. px/mm / Max Megapixels on the Canvas tab) and the source's own Scale, which defaults to reproducing the mesh's true physical size.

Adding an STL as an additional source (via a source tab's or the Canvas tab's **Add Source** button, rather than as the very first/base source) shows the same dialog without the Physical Size section, since a non-base source must never resize the canvas.

**File → Save Project…** after an STL import stores the STL file's path together with its chosen rotation, instead of a bitmap reference. Reopening the project re-parses the STL and re-rasterizes the depth map with the same settings, so the result is reproduced exactly. If a source's file has since been moved or deleted, the project still loads with every other source and setting intact, and a warning explains that it must be re-added manually via **Open Source…** or a source tab's **Add Source** button.

**Settings dialog:**

| Parameter | Description |
|---|---|
| Safe height | Z height used for all rapid travel moves between cuts (mm). |
| Rapid feed | Feed rate for rapid moves, used only for time estimates (mm/min). |
| Max acceleration | Machine acceleration, used only for time estimates (mm/s²). |
| Spindle spin-up dwell | Pause in seconds emitted as `G4 P…` after each step's `S` word, so the spindle reaches its new speed before cutting. 0 = no pause. `G4 P` is in seconds on GRBL and LinuxCNC; some controllers read it as milliseconds — check yours. |
| G-code Preamble | G-code inserted before the cutting program (spindle start, dwell, etc.). |
| G-code Postamble | G-code inserted after cutting (spindle stop, return to home, etc.). |

Tool path dress-ups (ramp-in, back-off, arc-rounding) are configured per step instead, via the **Dress-up…** button on each step's tab — see [Step / Dress-up](#step--dress-up) above.

---

#### Compute Menu

| Item | Shortcut | Description |
|---|---|---|
| Compute Current Step | F5 | Compute the toolpath for the currently active step tab. |
| Compute All Steps | F6 | Compute all steps in sequence. Required when any step uses a gating mode that references a prior step's result. |

---

#### Tools Menu

| Item | Shortcut | Description |
|---|---|---|
| Tool Library… | Ctrl+T | Open the [Tool Library](#tool-library) editor. |

##### Tool Library

The tool library is a list of cutting tools shared by all projects. It lives in one file outside any project: `%APPDATA%m2gcode	ool_library.json` on Windows, `~/.config/bm2gcode/tool_library.json` elsewhere. On first run it is created with a 6 mm end mill, a 3 mm ball nose and a 60° V-bit, with cutting data for Softwood, Hardwood and MDF. If the file can't be read, it is renamed to `tool_library.json.bak`, a warning is shown, and the starter library is used instead.

Each tool has a name, tool number, profile, diameter, V-bit angle, flute length and notes, plus cutting data per material: feed rate, plunge rate, spindle speed and max step-down. Stepover isn't part of the library — it describes the finish a step wants, so it stays a step setting.

The editor's **Tools** tab lists the tools (New, Duplicate, Delete), shows the selected tool's geometry with a side-view preview, and its cutting data per material: pick a material, enter the values and press **Set**; **Remove** deletes the selected material's row. The **Materials** tab adds, renames and deletes materials; deleting one that tools have cutting data for asks first and removes that data too. **Import…** merges another library file (tools already present by id can be replaced, kept alongside, or skipped); **Export…** writes the library to a file. **OK** saves the library; **Cancel** discards the changes.

Projects stay self-contained: a step copies the library values into its own fields and only remembers which tool (and material) they came from, so a project opens with the same values on a machine whose library lacks that tool. Editing the library doesn't change any step by itself — linked steps show **● modified** until re-applied. Library edits don't mark the project as changed; linking, unlinking or re-applying does.

Each library value is written to all three step-down fields (Raster Scan / Pocket / Contour / Registration Holes, V-Carving, Cutout), so switching the step's algorithm later doesn't leave a stale value; the status only compares the field the current algorithm uses.

---

#### View Menu

All visualization modes show data for the currently active step. Cut moves, rapid moves, tab markers, and registration hole outlines can be toggled in any mode with **C**, **R**, **T**, and **H**. When ramp-in, back-off, or arc-rounding dress-ups are enabled for a step (see [Step / Dress-up](#step--dress-up), above), the 3D toolpath preview (mode 2) renders them too, toggled together with rapid moves. Tabs are different: they're baked into the cut geometry itself, so they always show up as real Z-height changes in the 3D preview regardless of any toggle. The **T** toggle instead controls a 2D-only marker — a filled circle at each tab, sized to its width — since a top-down view has no other way to show a Z-only feature. The **H** toggle (**Show registration holes**) draws every Registration Holes step's hole outlines and flip axis in the 2D modes, from the step's current settings, so the placement can be checked before computing — see [Step / Operation / Registration Holes](#step--operation--registration-holes).

**Show shading** (**S**) renders any of the 2D modes below as a hillshaded relief — simulated raking light and shadow over the mode's own colours — instead of a flat colour ramp, with vertical exaggeration auto-scaled from the max cut depth vs. the canvas footprint so shallow relief remains visible. It has no effect in Step Path (3D) mode.

| Mode | Hotkey | Shows |
|---|---|---|
| Original | 1 | The composited depth map for the current step — every source it can see, blended together — with stock margin if configured. |
| Step Path (3D) | 2 | 3D visualization of the computed toolpath. |
| Step Result | 3 | The simulated surface that remains after this step — the depth achieved, pixel by pixel. For a Cutout step whose Cut depth exceeds Max cut depth (or a Registration Holes step whose Hole depth does), pixels actually cut all the way through render as a distinct highlighted colour instead of the ordinary depth grayscale, since a through-cut clips to the same value as reaching full relief depth — see [Cutout Pipeline](#cutout-pipeline). Requires toolpath to be computed.|
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

**Depth map.** Each source is converted to a float32 grayscale array and resampled onto the canvas's pixel grid according to its Offset, Rotation, and Scale. The visible sources for a given step (see [Per-step source visibility](#per-step-source-visibility)) are then composited bottom-to-top — using each source's Blend mode, Opacity, and Min/Max depth remap — into a single depth map for that step, where each pixel value is the desired cut depth as a fraction of Max cut depth (0 = surface, 1 = full depth). The z=0 colour setting controls overall polarity; canvas pixels not covered by any source get the Background depth.

**Canvas resolution.** The canvas's pixel grid density is chosen automatically between the Min. px/mm floor and the Max Megapixels ceiling (Canvas tab): with no image source present it's exactly the floor; with one or more image sources present, it's the finest of their own native px/mm (each image's native pixel count relative to the physical footprint its Scale gives it), still bounded by the floor and ceiling. This is only re-evaluated at explicit points — adding a source, editing Width/Height/Min. px/mm/Max Megapixels, **Compute → Recompute Canvas Resolution**, or automatically just before a Compute Step runs — never live while dragging a source's own Scale/Offset. An STL source's rasterization resolution follows the same settled density.

**Tool accessibility (morphological opening).** Before path planning, bm2gcode computes where the tool can physically reach. Grey-erosion of the depth map with a tool-shaped kernel gives the maximum depth achievable with the tool centred at each pixel. A subsequent grey-dilation recovers the reachable depth at each material point — because the tool can be centred nearby rather than directly above. Together these form a morphological opening and are the basis for all gating modes. The kernel shape depends on the tool profile:

- *Ball*: spherical cap offsets — `(R − √(R² − r²)) / max_depth`
- *V-Bit*: cone slope — `r / (tan(half-angle) × max_depth)`
- *End Mill*: zero inside the radius, infinity outside

**Accessibility caching.** The grey-erosion/grey-dilation pair above is the most expensive part of computing a step, and it depends only on the tool geometry, Max cut depth, and the work area's physical dimensions — not on step-down, threshold, gating mode, or any other per-step parameter. bm2gcode keeps a cache of these accessibility maps keyed by that combination, so recomputing a step after tweaking an unrelated parameter, or computing a later step with the same tool, reuses the previously computed result instead of rerunning the erosion/dilation. The cache is cleared whenever a source, the canvas, or the project changes.

**Result map.** After each step is computed, a float32 *result map* records how deep the material has been cut at every pixel. This is passed to the next step as the prior reference, enabling the Residual and Reachable gating modes.

---

### Raster Scan Pipeline

1. **Kernel construction** — Tool geometry is discretised into a depth-offset kernel and a boolean footprint disk.
2. **Accessibility map** — Grey-erosion → grey-dilation (morphological opening) yields the reachable depth at each material point.
3. **Pass planning** — Max step-down and Final step-down define an ordered sequence of depth ceilings (e.g. 0.5 mm → 1.0 mm → 1.5 mm → 1.8 mm (final)).
4. **Scan line / spiral extraction** —
   - *Horizontal/Vertical/Diagonal*: for each raster line at the step-over interval (perpendicular to the scan angle), contiguous spans of the gated work region are extracted by edge-detection on the binary mask. Each span becomes a full-depth segment. Diagonal (45°/135°) lines are sampled the same way along the tilted line — pixel positions where row+col (45°) or row-col (135°) is constant — at roughly √2 px point spacing instead of Horizontal/Vertical's 1 px. If Cut direction is Climb or Conventional, spans are reversed as needed so every line travels the same physical direction.
   - *Spiral*: the work area is divided into a grid of rows and columns, stepped over independently on each axis, then traced as one continuous perimeter walk — clockwise or counter-clockwise depending on Cut direction — and split into straight legs (see [Spiral raster pattern](#spiral-raster-pattern)). Each leg's contiguous spans of the gated work region are then extracted the same way as a raster line.

   See [Climb / conventional cut direction](#climb--conventional-cut-direction) for what Climb/Conventional/Either mean for each scan direction.
5. **Depth-slicing (multi-pass)** — Full-depth segments are split into per-pass sub-segments. Roughing passes skip pixels already at target depth or shallower than the final step-down reserve. The final pass covers all remaining pixels. Segments are broken wherever the gate changes. If Avoid cutting air is on, each pass's air-cutting runs are then removed (see [Air-cut pruning](#air-cut-pruning)), before ordering, so ordering and linking handle the resulting fragments like any other gated segments.
6. **Path ordering** — Greedy nearest-endpoint ordering, optionally refined by 2-opt TSP (see [2-opt TSP path reordering](#2-opt-tsp-path-reordering)). A locked Cut direction instead uses a line-adjacency-preserving oscillating sweep for Horizontal/Vertical (see [Climb / conventional cut direction](#climb--conventional-cut-direction)); 2-opt is skipped in that case. For Spiral, ordering is skipped entirely whenever the pass is still a single unbroken coil — locked or not, since there is no travel left to optimise — and falls back to ordinary greedy/2-opt reordering only once depth-slicing or masking has fragmented it into disconnected pieces.
7. **Terrain linking** — Short gaps between segments are bridged by a geodesic path along the roughing surface rather than a full retract (see [Geodesic terrain linking](#geodesic-terrain-linking)). Longer gaps use retract → rapid → plunge.
8. **Simplification** — Ramer-Douglas-Peucker reduces point count to within pixel resolution (see [Ramer-Douglas-Peucker simplification](#ramer-douglas-peucker-simplification)).
9. **Result simulation** — The toolpath is resampled at sub-pixel spacing and swept through a grey-dilation to compute the material state (result map) for subsequent steps.

---

### Contour Pipeline

1. **Kernel + accessibility map** — Identical to Raster (steps 1–2).
2. **Level planning** — Two modes:
   - *Uniform*: Z-levels at equal Z-increment intervals.
   - *Adaptive (CDF)*: Z-levels placed at quantile boundaries of the depth-map pixel distribution. This concentrates levels where many pixels share the same depth value ("plateaus"), so contour lines align with flat areas rather than spanning empty gray ranges.
3. **Contour extraction** — `skimage.find_contours` traces iso-depth curves at each Z-level on the accessibility map. If Cut direction is Climb or Conventional, every contour is reversed (or not) by a single global rule applied at this stage (see [Climb / conventional cut direction](#climb--conventional-cut-direction)).
4. **Lateral pruning (optional)** — For each level, the distance transform to all committed deeper-level cuts is computed. Contour segments closer than the Lateral tolerance are discarded. Short pruned runs are restored if the retract/plunge overhead would cost more machine time than the cut saves. If Avoid cutting air is on, each level's contours are then also pruned where the tool would cut only air (see [Air-cut pruning](#air-cut-pruning)), with the same short-run restore.
5. **Ordering** — Contours are grouped into non-overlapping windows of adjacent Z-levels sized so no window spans more than Max step-down of Z travel, then greedy nearest-neighbour ordering runs across every contour in the window at once — not just one Z-level — with support for closed-contour rotation. Because a contour's climb/conventional correctness comes entirely from its own orientation (set in step 3), not from visiting order, a locked Cut direction still gets this full cross-window freedom; it only disables reversal (flip) and the optional 2-opt refinement pass, since 2-opt's block-reversal move would undo the forced direction — see [Climb / conventional cut direction](#climb--conventional-cut-direction). When the nearest candidate is a genuine rapid rather than a short hop, the greedy step prefers whichever candidate is closest to the window's shallowest level, so forced long jumps still trend top-to-bottom instead of landing at an arbitrary depth. 2-opt, when it runs, applies a Z-jump penalty proportional to Max step-down across the same window.
6. **Terrain linking** — Same geodesic approach as Raster for short inter-segment gaps.
7. **Simplification + result simulation** — Same as Raster (steps 8–9).

---

### V-Carving Pipeline

All four V-Carving variants share stages 1–2 and 4–8; they differ in how the skeleton is extracted (stage 3), and Hybrid adds waterline contour rings to the path (stage 5).

1. **Kernel + accessibility map** — Identical to Raster.
2. **V-carving mask** — Computes the region to carve:
   - Pixels shallower than Z-threshold are excluded. A Z-threshold of 0 is treated as "not set" and auto-splits at the midpoint of the depth map's own value range instead — otherwise a literal 0 would keep the flat background (always at depth 0) in the mask alongside the artwork, turning the skeleton (stage 3) into the medial axis of the whole canvas rectangle rather than the artwork's own shape.
   - Features narrower than Min feature width are removed by morphological opening.
   - Feature interiors wider than Max feature width (where the tool cannot reach the configured depth) are excluded using the distance transform.
3. **Skeleton extraction** — one of three methods (Hybrid uses Feature Width's):
   - *Depth Map*: medial axis of the mask (`skimage.morphology.medial_axis`). Cut depth at each skeleton point is taken directly from the depth map.
   - *Sub-pixel*: Voronoi diagram of the mask boundary contour. Ridges with a small opening angle — arising from closely-spaced same-side boundary points, not true centrelines — are discarded using the Attali–Montanvert criterion. Remaining ridges are discretised to 1-pixel width. Cut depth follows the depth map but the skeleton geometry is smoother than the pixel-grid medial axis.
   - *Feature Width*: same Voronoi skeleton. Cut depth at each point is derived from the inscribed-circle radius at that point (the radius of the widest circle fitting the mask there), converted to a physical depth via the tool geometry (V-bit cone formula or ball-nose formula). The resulting depth may exceed the depth map value at that pixel but is always clipped to Max cut depth.
4. **Branch decomposition** — The skeleton is split at junction pixels into polyline branches.
5. **Branch ordering** — Greedy nearest-endpoint ordering across all branches; optional 2-opt.

   *Hybrid only — waterline contour rings*: the skeleton is kept whole (unpruned) and 2-opt is applied to it on its own; contour rings are then appended after it. The rings are extracted from the achievable-depth surface of an idealized V-bit — same angle as the real tool, but just large enough that its cone tapers to a point exactly at Max cut depth (`compute_vcarve_synth_achievable_depth`) — rather than the real tool's surface, so the rings follow where a V-bit's flank can actually reach at each depth regardless of the real tool's diameter. Ring levels are the pass ceilings from Max step-down / Final step-down down to Max cut depth. The rings are extracted on the unmasked surface and only then clipped to the V-carving mask from stage 2, so clipping doesn't invent spurious geometry along the mask edge. They are ordered with Contour's windowed ordering (see [Contour Pipeline](#contour-pipeline) step 5), with Cut direction applied to each ring's orientation. The combined skeleton + rings path then continues through the remaining stages. The idealized V-bit's achievable-depth surface is also what a later [V-Carving Clean-up](#step--operation--v-carving-clean-up) step erodes against (see [Pocket Pipeline](#pocket-pipeline)).
6. **Multi-pass depth slicing** — Same mechanism as Raster: branches are split into per-pass sub-segments when Max step-down > 0. If Avoid cutting air is on, air-cutting runs are then removed from each pass (see [Air-cut pruning](#air-cut-pruning)).
7. **Per-pass ordering** — Optional 2-opt.
8. **Simplification + result simulation** — Same as Raster.

---

### Cutout Pipeline

Cutout doesn't share Raster/Contour/V-Carving's kernel + accessibility (morphological opening) stage — since it's a straight-wall cut, only the tool's lateral radius matters, not a depth-shaping kernel — and it ignores Process Region gating entirely, always working from its own Split level against the current depth map regardless of what any prior step accomplished.

1. **Silhouette mask** — Splits the depth map into "object" and "background" at Split level. Because depth map polarity — whether the object is the deeply-cut region or the shallow/uncarved region left standing proud of a deeply-recessed background — depends on how the source was authored, both split directions (`<=` and `>=` the threshold) are tried; whichever produces a workable mask (neither empty nor spanning the entire canvas) is used. A Split level of 0 is treated as "not set" and auto-splits at the midpoint of the depth map's own value range instead. Interior holes are then filled — keeping only the outer perimeter of each shape — and the result is dilated outward by the tool radius plus Boundary offset, so the traced boundary already accounts for the tool's own width.
2. **Contour extraction** — `skimage.find_contours` traces the outer boundary of the dilated mask. If Cut direction is Climb or Conventional, every contour is reversed (or not) by the same single global rule Contour uses (see [Climb / conventional cut direction](#climb--conventional-cut-direction)).
3. **Tabs (optional) + depth-slicing (multi-pass)** — If the Tabs dress-up is enabled (see [Step / Dress-up](#step--dress-up)), each traced loop's Z is first raised to Tab height for a Tab width-wide, optionally tapered (Tab angle) stretch around each of Tab count evenly-spaced points, before depth-slicing runs. Every other point of every traced loop targets the same full-depth Z (Cut depth) — unlike the other algorithms' per-pixel varying depth — so Max step-down / Final step-down normally just produce a sequence of identical loops at progressively deeper ceilings; with tabs, a roughing pass instead stops short wherever a tab's raised Z has already been reached, so a tab is respected by every pass it would otherwise be cut through by, not just the final one — the same per-point "stop once the target depth is reached" mechanism as the [Raster Scan pipeline](#raster-scan-pipeline)'s depth-slicing step, which handles per-pixel varying depth there. If Avoid cutting air is on, air-cutting runs are then removed from each pass (see [Air-cut pruning](#air-cut-pruning)) — including anywhere a previous Cutout step already cut all the way through — turning a partly-pruned closed loop into an open segment.
4. **Ordering** — Greedy nearest-endpoint ordering per pass, with support for closed-loop rotation; optional 2-opt. No terrain linking: unlike Raster/Contour, there's no surface underneath to route a gap-bridging move along, since this is a flat through-cut, so gaps become a plain retract → rapid → plunge instead.
5. **Simplification + result simulation** — Same as Raster.

Because Cut depth is typically deeper than Max cut depth, a genuine through-cut clips to the same depth-map value (1.0) as ordinary full-depth relief carving elsewhere in the project — the two would otherwise be indistinguishable in the Step Result view. To keep them distinguishable, the result carries a second, separate channel recording exactly which pixels were actually cut all the way through (deeper than Max cut depth — a shallower Cutout pass is just an ordinary partial cut, not a hole, and needs no special marking); the Step Result view renders those pixels fully transparent, showing a distinct highlight colour through instead of the ordinary depth grayscale. A tab only counts as "cut all the way through" once its own raised Z has actually reached Max cut depth or deeper — a tab shallower than that renders as an ordinary partial-depth grey in the Step Result view, not a hole.

---

### Pocket Pipeline

Pocket shares Raster's kernel + accessibility (morphological opening) stage, but replaces Raster's per-pixel varying depth with a single flat floor at Target depth.

1. **Kernel + accessibility map** — Same `_build_kernel`/grey-erosion/grey-dilation morphological opening as Raster, but without Raster's own Extra offset kernel inflation: Pocket's Boundary offset (step 2 below) is a lateral-only push applied after this stage, not a depth-shaped kernel change, since Pocket's cut depth is already fixed independently by Target depth.
2. **Target-Z interior mask** — Every position whose tool-centre erosion (the same surface Contour extracts its iso-depth contours from — never the dilated/opened accessible surface, which would let the tool centre travel right up to a boundary only reachable via a nearby centre position) is at least Target depth (normalised, with the same half-grey-level epsilon Contour uses so flat regions land firmly inside the mask rather than teetering on a quantisation boundary) is pocket interior. If Boundary offset is set, this mask is then eroded by a disk of that radius, retreating the tool-centre path further from the pocket wall — the same "away from the boundary" push Cutout applies to its own silhouette mask (see [Cutout Pipeline](#cutout-pipeline) step 1), just implemented as an erosion here rather than a dilation since Pocket's mask already represents the safe interior rather than the material being cut around — before Process Region gating (if active) is combined with (not replacing) it, using Z-threshold as the residual tolerance exactly as Raster does.
3. **Per-region infill generation** — The interior mask is split into connected regions (`scipy.ndimage.label`). Each region gets its own scan-line grid, spiral coil, or onion-skin ring chain, sized and centred on (or, for onion skin, offset inward from) that region's own bounding box/boundary rather than the full stock, so a Spiral or Onion Skin pattern forms a coil or ring chain concentric with the pocket rather than fragments centred on the canvas. Gating uses each region's own label mask, so two disjoint regions with overlapping bounding boxes never double-cut each other's pixels. An Onion Skin ring chain that erodes down to a narrower neck than expected and splits into multiple sub-islands (a dumbbell-shaped region) has each sub-island chained independently — see [Onion skin pocket pattern](#onion-skin-pocket-pattern). For a Zig-Zag scan direction, each region's own un-eroded boundary contour (one ring, plus one ring per interior hole/island) is also extracted here via `skimage.find_contours`, for step 6's contour linking to use. See [Climb / conventional cut direction](#climb--conventional-cut-direction) for what Climb/Conventional/Either mean for each scan direction, applied independently within each region.
4. **Depth-slicing (multi-pass)** — Since every pocket point targets the same flat floor Z, this slices a constant depth rather than Raster's per-pixel terrain, the same mechanism Cutout uses for its own constant Cut depth. If Avoid cutting air is on, air-cutting runs are then removed from each pass (see [Air-cut pruning](#air-cut-pruning)); every piece of a split segment keeps its region, so Zig-Zag and Onion Skin still order one region at a time.
5. **Path ordering** — For Spiral and Onion Skin, same greedy/2-opt/locked-direction machinery as Raster (Spiral), or fixed-orientation region grouping (Onion Skin — see [Onion skin pocket pattern](#onion-skin-pocket-pattern)), operating across all regions' segments together. For Zig-Zag, one region's own sweep always completes before the next region's begins — never interleaved — and each region enters from whichever of its own two extreme lines (e.g. top-left vs. bottom-right corner for a 45° sweep) is nearer the incoming tool position, the same choice Raster's own locked-direction sweep already makes for a single scan area, just now scored per region instead of across every region's segments pooled together; see [Zig-Zag per-region ordering](#zig-zag-per-region-ordering).
6. **Terrain/contour linking** — For a Zig-Zag scan direction, a short gap between two segments is bridged by walking the region's own boundary contour from step 3 whenever both gap endpoints lie on it (within a couple of pixels) — since every Zig-Zag segment already starts and ends on that same boundary, this reuses the exact ring rather than approximating it, and stays correct across a concave notch that splits one scan line into several segments, since walking either direction along a region's own boundary can never leave the region. Every other gap — any non-Zig-Zag scan direction, or a Zig-Zag gap whose two endpoints don't resolve to one shared ring (e.g. it crosses between two different regions, or between one region's outer boundary and an interior hole/island) — falls back to a geodesic path (see [Geodesic terrain linking](#geodesic-terrain-linking)), the same mechanism Raster uses, but following the flat floor abstraction from step 2 rather than the depth map's own relief: within the pocket interior this is constant at the current pass's ceiling, so a short gap is bridged by a level cutting move instead of a retract; a gap that must cross outside the pocket interior rises only to the surface, never dipping to match relief the pocket floor isn't actually cutting to. Longer gaps use retract → rapid → plunge.
7. **Simplification + result simulation** — Same as Raster.

Pocket only clears the interior — it does not cut a wall-finishing pass around the pocket boundary. Every scan direction's own infill lines still stop at the interior mask's pixel-grid boundary, the same stair-stepped edge Raster leaves at any masked boundary — Zig-Zag's contour linking (step 6) only smooths the connecting move between lines, not each line's own endpoint. Onion Skin is the exception: its rings are extracted via sub-pixel marching squares (`skimage.find_contours`) directly, so its outermost ring traces a smoother boundary than the per-pixel stair-step.

V-Carving Clean-up runs this identical pipeline with one substitution: step 1's achievable-depth erosion runs against `compute_vcarve_synth_achievable_depth`'s output — the nearest preceding V-Carving (Hybrid) step's own idealized-V-bit achievable-depth surface (see [V-Carving Pipeline](#v-carving-pipeline)) — instead of the raw depth map, so the clearing tool's kernel is eroded against the V-bit's own achievable-depth surface rather than the flat depth map. Target depth still defaults to 0, but that now means literal Z=0 (see [Step / Operation / Pocket](#step--operation--pocket)) — check its **Auto** box to track the depth map's own deepest point instead (same Auto behaviour as ordinary Pocket; it reads the raw depth map, not this step's own V-bit-eroded reference surface), mirroring what "0 = auto" used to give by default; see [Step / Operation / V-Carving Clean-up](#step--operation--v-carving-clean-up).

---

### Registration Holes Pipeline

Like Cutout, Registration Holes is a flat, straight-wall End Mill cut with no kernel/accessibility stage and no Process Region gating; unlike Cutout, it doesn't read the depth map at all.

1. **Hole placement** — Hole centres are computed in stock coordinates (the stock's bottom-left corner, margin included, is 0,0), spanning pixel centres like every other algorithm, so a flip axis through the stock centre is exactly the depth map's own centre line. **Vertical axis**/**Horizontal axis** put two holes on the centre line, Edge inset from the two stock edges it crosses. **Diagonal** builds an L: the flip line is the 45° line through the G-code origin (for Center, the line through the stock centre in the chosen direction); the corner hole sits on that line, Edge inset from both edges at the line's origin end (for Center, at the far end of the line's centred square of side min(W, H)); the two arm holes sit Spacing from the corner hole along the two stock edge directions pointing into the stock. Reflecting across the flip line fixes the corner hole and swaps the arm holes, so every pattern maps onto itself. The centres are then converted to G-code coordinates for the selected origin.
2. **Boring** — Holes are visited nearest-first from the origin. Each is a single segment for a helix — Z=0 down to Hole depth along a circle of radius (Hole diameter − tool diameter)/2, at an equal Z drop per revolution no larger than Max step-down, then one flat revolution at full depth and a move to the centre — or, when Hole diameter is within 0.02 mm of the tool diameter, a peck per Max step-down. The pecks share one XY position, and [G-code generation](#g-code-generation) only retracts between segments whose XY differs, so each peck but the last carries its own feed-rate lift back to Z=0.
3. **Result simulation** — Same as Raster, unioned with the previous step's result. A hole deeper than Max cut depth is marked as cut all the way through in the same separate channel Cutout uses (see [Cutout Pipeline](#cutout-pipeline)), so it shows as a hole in the Step Result view.

The **Processing Mask** view shows the finished holes, placed exactly as this pipeline places them.

---

### Supporting Algorithms

#### Air-cut pruning

Used by **Avoid cutting air** in every algorithm. A tool centred at c with its tip at depth z removes material at c+d down to z − k(d), where k is the tool's depth-offset kernel. Where the previous step's result map (the cumulative removed depth, zero when there is no previous step) is r, the tool cuts nothing iff z ≤ r(c+d) + k(d) at every d of the footprint. The minimum of that bound over the footprint — the deepest air-only tip depth at each centre — is a grey erosion of r by the tool kernel, the same operation that builds the achievable-depth surface from the depth map. It uses the real tool kernel, not one widened by Extra offset, since this is about where the tool physically touches material. For Cutout, pixels a previous Cutout step cut all the way through are treated as having no material at any depth.

Each pass segment is sampled every half pixel along its length. A sample cuts material only if it's deeper than that bound by more than one grey level (so pixel rounding doesn't count as cutting). The result is a run of cutting and air samples along each segment:

- Air runs at a segment's start or end are always trimmed, since that adds no retract.
- An air run in the middle is removed only if it's at least as long as feed rate × 2 × (Safe height + depth) / Rapid feed, the distance the tool could feed in the time an extra retract and plunge take. Shorter air runs are kept. A closed loop is first rotated to start in its longest air run, so removing that run doesn't split the loop.
- Each cutting run is extended by one sample into the air on both sides. Its ends then lie within half a pixel of the true air/material boundary, on the air side, so no material is left behind.

Kept pieces keep the segment's original vertices plus the two boundary points, and keep their direction of travel, so Climb/Conventional stays valid. Only earlier steps count as removed material: within one step, depth-slicing already drops points that don't go deeper than the previous pass at the same position.

#### 2-opt TSP path reordering

Iteratively tests whether reversing a contiguous block of segments reduces total travel cost. The cost function accounts for whether a gap qualifies for terrain-linking (feed-rate cost only) or requires a retract + rapid + plunge (fixed overhead + distance). A configurable Z-jump penalty discourages large depth changes for cross-level reordering in Contour.

#### Climb / conventional cut direction

With a clockwise spindle (`M3`, the default G-code preamble, editable under File → Settings), a cut keeps uncut material on the tool's right for climb milling and on its left for conventional milling. Raster and Contour each apply this as a single, cheap, upfront reversal rule rather than analysing each segment's surrounding geometry:

- **Raster (Horizontal/Vertical)**: horizontal scan lines default to left→right and vertical lines to top→bottom. Climb keeps horizontal lines as extracted and reverses vertical lines; Conventional does the opposite.
- **Raster (Diagonal)**: 45° lines default to run from their top-right end toward their bottom-left end; 135° lines default to run from their top-left end toward their bottom-right end. Climb reverses both (bottom-left→top-right for 45°, bottom-right→top-left for 135°); Conventional keeps both as extracted.
- **Raster (Spiral)**: there is no per-segment reversal — Phase 1 always generates a single continuous, correctly-wound coil directly, so Climb/Conventional instead pick which way that coil winds. For Outside-In, Climb winds clockwise and Conventional winds counter-clockwise. Inside-Out is the exact point-reversal of that same path (see [Spiral raster pattern](#spiral-raster-pattern)) — and because reversing a wound path preserves its climb/conventional character while flipping its travel direction, the winding sense that produces a given label flips too: Inside-Out Climb winds counter-clockwise and Conventional winds clockwise. This matches the same real-world convention as milling a pocket's boundary from the inside versus the outside.
- **Contour**: `skimage.find_contours` always keeps higher-valued pixels on a fixed side of the direction it returns a contour in, regardless of whether that contour encloses a peak or a valley, open or closed. Climb keeps every contour as extracted; Conventional reverses all of them.

Once a direction is locked, reversal is off the table everywhere, but how much freedom ordering keeps differs between the algorithms — it depends on whether correctness is a property of one segment alone or of the visiting sequence:

- **Raster (Horizontal/Vertical/Diagonal)**: correctness depends on visiting order, not just each segment's own direction — a segment cut out of turn could put a still-uncut line's material on the wrong side even though its own travel direction is correct. So consecutive cuts are restricted to the current raster line or the immediately next one in the active sweep direction (grouped by y for Horizontal, x for Vertical, x-y for Diagonal 45°, x+y for Diagonal 135°), and a line can never be visited out of turn. Depth-slicing for multi-pass roughing can fragment a line into several short segments, leaving some unvisited when a greedy jump favours a closer line further along; when the sweep reaches one edge of the map with segments still unvisited elsewhere, both the sweep direction and each newly-visited segment's travel direction flip together and it resumes from that same edge, oscillating back and forth until nothing is left. Reversing progression and per-segment travel together preserves which side carries uncut material, so every line-to-line transition stays climb/conventional-correct; 2-opt is skipped, since its block-reversal move would undo the forced direction.
- **Raster (Spiral)**: the generated leg order already is the correct visiting order — a locked direction always leaves it untouched rather than attempting a raster-style oscillating sweep, which assumes independent parallel lines and doesn't generalise to a spiral's alternating row/column legs. Reordering is only ever a travel-time concern for Spiral (see [Spiral raster pattern](#spiral-raster-pattern)), never a correctness one, since climb/conventional correctness is baked into the winding direction chosen at generation time.
- **Contour**: correctness comes entirely from each contour's own fixed orientation (set in the single global reversal rule above), not from visiting order, so locking direction doesn't need to restrict which contour is visited next or from which Z-level — the greedy step still draws candidates from the whole max_step_down window, same as when unlocked. It only removes the flip option (a closed contour can still start at any point along itself, just never reversed) and skips 2-opt, since its block-reversal move would undo the forced direction.

This trades some travel-time optimisation for a mechanically consistent cut.

#### Spiral raster pattern

Outside-In generation works in two stages. First, the work area's row and column step-over grid lines — each independently rounded to fit the work area exactly, so non-square depth maps are fully covered on both axes — are traced in pure array-index space using the classic "spiral matrix" traversal (top row, right column, bottom row, left column, shrinking inward one ring at a time), clockwise or counter-clockwise depending on Cut direction (see [Climb / conventional cut direction](#climb--conventional-cut-direction)). Second, that sequence of adjacent grid points is run-length-encoded into maximal straight legs, each a fixed row or column with a start and end index; this automatically absorbs the degenerate case of a ring shrinking to a single row or column wide, with no special-casing needed. Inside-Out is generated by reversing the entire Outside-In leg sequence — both the leg order and each leg's own point order — rather than being built independently from the centre outward; reversing a wound path preserves its climb/conventional character while flipping its travel direction, which is why the winding sense that produces a given Climb/Conventional label inverts between Outside-In and Inside-Out.

Because consecutive legs always share their corner point exactly, Path ordering (see the Raster Scan pipeline above) can detect whether a pass is still a single unbroken chain and, if so, leave it untouched — reordering an already-optimal path can only be a no-op at best. This also sidesteps a failure mode of naive greedy nearest-endpoint search: its search anchor is the tool's assumed start position, which can coincide exactly with a *non-initial* vertex of the spiral (for example, the G-code origin corner corresponding to the boundary between two rings rather than the coil's true start). Without the unbroken-chain check, that exact-distance tie makes greedy search pick the wrong leg first and chain through a run of further ties, closing each ring into a standalone rectangle linked to the next by a diagonal jump instead of preserving the spiral. Once depth-slicing (multi-pass roughing) or region gating (a masked refining step) has actually fragmented a pass into disconnected pieces, this check no longer holds and ordering falls back to ordinary greedy/2-opt reordering among the fragments, exactly as for any other pattern — there is no longer a single correct order to preserve at that point, so travel-time optimisation is the right goal again.

#### Onion skin pocket pattern

Outside-In generation works ring by ring. The region's own mask is repeatedly shrunk by Step-over via binary erosion; each shrunk mask's boundary is extracted with `skimage.find_contours` (the same sub-pixel marching-squares technique Contour uses for its iso-depth contours), giving one ring per erosion. The rings are stitched into one continuous chain rather than a series of separately-retracted loops. Each ring is traversed in full, entering and leaving at the same point, its *seam*, and the move from one ring's seam to the next ring's seam is cut as an ordinary in-material move, not a rapid. That move is an S-shaped link wherever one fits: two circular arcs (a biarc), the first leaving the finished ring along its own direction of travel, the second arriving tangentially on the next ring, each with a radius of at least one Step-over, so the tool changes ring without a sharp turn. With rings one Step-over apart, arcs of that radius put the next ring's seam about 1.7 Step-overs further along than the previous one, so the seams spiral forward from ring to ring. Both rings are still cut in full, so the S-link leaves no sliver at the pocket wall; it only replaces the move through the band between them, which is cleared anyway. Where no S-link fits (a ring too small for arcs of that radius, as with the last few rings at a pocket's centre, or an outer ring and a hole ring whose directions don't line up), a straight link is used instead.

The seams of all rings in a chain are chosen together, as a shortest path through a layered graph (one layer per ring, one node per candidate seam spaced about a quarter Step-over apart along the ring) minimising the chain's total link cost: an S-link costs its length, and a straight link its length plus a penalty of 8 Step-overs, so a straight link is only used where no S-link fits. A ring one erosion step further in is exactly the set of points whose Step-over-radius disk fits inside the previous ring's area, so from every point of it a link of about one Step-over leads back out, and that link lies inside the pocket. Choosing the seams globally lets the seam line settle on a straight flank of the region, where consecutive rings are about one Step-over apart, instead of drifting into an acute tip, a concave cusp, or a protrusion that the next ring no longer reaches, where the gap is much wider. Every candidate link, S-shaped or straight, is also checked against the region's own mask (sampled along its length at the same 0.5 iso-level `find_contours` traces), and only links that stay inside the region are allowed, so a link can never cut material outside a concave pocket boundary. If two consecutive rings have no inside link at all, which can only happen between an outer ring and a hole ring of the same erosion step, the chain is split there, and Stage 6 ordering and linking joins the two pieces like any other pair of chains. Climb/Conventional is decided once per ring, immediately after extraction (reversing every ring's point order for Conventional, the same single reversal rule Contour applies to its own iso-depth contours), before any stitching happens — so neither seam selection nor the choice of Outside-In vs. Inside-Out ever changes a ring's own winding, only which of its points the chain enters and leaves on. Inside-Out's reversal also reverses each S-link, which is then still tangent to both rings. Inside-Out is generated by reversing the entire Outside-In chain — both the ring order and each ring's own point order — the same way Spiral's Inside-Out is the exact reversal of its Outside-In leg sequence (see [Spiral raster pattern](#spiral-raster-pattern) above).

Erosion continues until the mask empties out or, for a region much smaller than one Step-over, after just its own single outermost ring. If an erosion step disconnects the region into two or more pieces (a dumbbell-shaped region whose narrow neck has been fully eroded away), the chain built so far is closed off there and each new piece is chained independently from scratch, rather than attempting to bridge across the former neck — a bridge would have to re-cross material the tool already cleared when the outer rings were cut. A region with a hole (an annulus) needs no special handling: `find_contours` already returns both the outer and inner boundary of a single erosion step as separate rings, which are visited in nearest-ring order (whichever remaining ring comes closest to the ring just cut) and stitched with the same seam selection as consecutive rings.

#### Ramer-Douglas-Peucker simplification

Reduces G-code move count while preserving path shape to within the source's resolution. Implemented iteratively (not recursively) to handle long paths without hitting Python's call-stack limit. Finds the point with the greatest deviation from the current chord; if it exceeds the tolerance, that point is kept and the chord is split there.

Raster Scan, Contour, V-Carving and Pocket use independent lateral and vertical tolerances: one pixel's width (1 / px per mm) in X/Y, and one grey level's depth (Max depth / (grey levels − 1)) in Z. Z is scaled by the ratio of the two before distances are measured, so every dropped point lies within an ellipsoid of those two semi-axes around the simplified path. A coarse canvas therefore can't let the path dip a whole pixel-width below its target depth. The tradeoff is more points on curved surfaces, especially for STL sources, which count as 65536 grey levels. Cutout, whose Z comes from pass depths and tabs rather than the depth map, uses the single one-pixel tolerance in all three axes.

#### Geodesic terrain linking

Connects nearby segments with a path that follows the roughing surface, avoiding unnecessary retracts. Implemented as Dijkstra shortest-path on the pixel grid with 8-connected neighbours and 3-D arc-length as the edge cost (√(Δxy² + Δz²)). Falls back to a straight Bresenham line if no path is found within the search margin.

#### Zig-Zag per-region ordering

Within one depth pass, Zig-Zag orders each pocket region's own segments independently and never interleaves two regions' segments in the visiting order: at each step, every not-yet-visited region's own best sweep is computed against the current tool position, and whichever region's resulting entry point is nearest overall is visited next, with the tool position advancing to that region's own last point before the next region is considered. Both this outer region-to-region sequencing and each region's own starting line are unaffected by Cut direction, including Either — a region's entry is always one of its own two extreme lines (e.g. the top-left vs. bottom-right corner's diagonal for a 45° sweep), whichever is nearer, never an arbitrary line partway through it, and (since best entry is recomputed against wherever the tool ends up after the previous region) a given region doesn't always start from the same corner from one pass to the next.

What differs by Cut direction is how a region's *own* lines are swept once entered. Climb and Conventional reuse Raster's own locked-direction oscillating sweep (see [Climb / conventional cut direction](#climb--conventional-cut-direction)), which keeps every line travelling the same physical direction so uncut material stays on one consistent side between lines — the same correctness guarantee it already provides for a single scan area, at the same travel-time cost: consecutive lines can't enter from whichever end is nearer, since that would put material on inconsistent sides across the transition. Either has no such constraint, so each line is instead free to enter from whichever of its own two ends is nearer the tool's position after the previous line — letting the fill alternate side to side (a true zig-zag) instead of returning to the same starting side every line, which is what makes a near-continuous single path through a whole region possible in the first place.

#### Zig-Zag contour linking

Pocket's Zig-Zag scan direction (see [Pocket Pipeline](#pocket-pipeline) step 6) bridges a short gap between two segments by walking a sub-arc of the pocket region's own boundary contour instead of the geodesic search above, whenever that's possible: each region's un-eroded boundary ring(s) — one outer ring, plus one per interior hole/island — are extracted once via `skimage.find_contours` during infill generation (step 3), the same sub-pixel marching-squares technique Onion Skin uses for its own rings. Since every Zig-Zag segment already starts and ends on its region's boundary by construction, bridging a gap is a matter of finding both endpoints on that same ring (snapping each to its nearest ring vertex, within a couple of pixels) and extracting the shorter-arc-length sub-path between them, walking whichever of the ring's two directions is shorter. Every point on that sub-path is itself one of the ring's own vertices, so the resulting link can never leave the region regardless of how concave its boundary is — the same walk correctly threads a scan line's own concave notch (where the notch splits one raster line into two or more segments) without needing any special-casing for that case, since going around the notch along the ring is exactly what "walk the boundary" already means there. If the two endpoints don't resolve to the same ring — the gap crosses between two different pocket regions, or between one region's outer boundary and an interior hole/island — this falls back to the geodesic search above unchanged.

#### Bresenham line rasterisation

Classic integer-arithmetic line drawing, used as the geodesic-link fallback and for other pixel-grid operations.

#### Contour lateral pruning

For each contour level, a distance transform to committed deeper-level cuts identifies redundant segments. A breakeven calculation (retract/plunge overhead vs. time saved by skipping) determines whether short pruned runs should be restored before the contour is split into segments.

#### CDF adaptive level spacing

Extracts the empirical depth distribution of the accessibility map and places contour levels at quantile boundaries. This concentrates passes in depth ranges with many pixels and avoids passes in sparse gray ranges.

#### Medial axis / Voronoi skeleton

The medial axis (`skimage.morphology.medial_axis`) is the set of centres of all maximal inscribed circles in the mask — the standard definition of a shape's skeleton. The Voronoi variant builds the Voronoi diagram of the mask boundary contours and discards ridges with a small opening angle (those arise from closely-spaced same-side boundary points and are not genuine centrelines). The result is geometrically smoother for curved or flowing features.

#### G-code generation

Iterates over the ordered segments. Consecutive segments whose endpoints touch are joined as G1 feed moves. Gaps trigger a retract (G0 to safe height), a rapid to the next entry XY position, and a plunge (G1 to entry Z). Three optional per-step dress-ups (see [Step / Dress-up](#step--dress-up), off by default) modify this base behaviour without changing the underlying cut geometry:

- **Ramp-in** replaces the plunge below the stock surface (z < 0) with a zig-zag that retraces the upcoming cut path's own footprint; any travel above the surface stays a plain vertical plunge.
- **Back off before retract** inserts a short move away from the wall just cut, immediately before each retract. It retraces the just-cut path's XY footprint backwards (across as many prior moves as needed to cover the distance), emitted as G1 X/Y/Z moves whose Z rises at 45° but is never lower than the path's own Z at that XY — so every waypoint sits directly above a position the tool already occupied.
- **Arc-round rapid corners** replaces the retract → traverse → plunge rapid's two sharp 90° corners with tangent arcs sized to the available clearance (never exceeding the safe height), so both corners always share one radius. The arc's waypoints are emitted as G1 moves at its own configured feed (rather than G0), since G0 doesn't guarantee coordinated linear interpolation between waypoints on every controller — G1 is what makes the polyline actually trace the intended arc instead of a per-axis independent rapid.

The fourth per-step dress-up, **Tabs**, isn't part of this stage at all — it modifies the cut geometry itself, earlier in the Cutout pipeline (see [Cutout Pipeline](#cutout-pipeline)), so by the time G-code generation runs, a tab is already just an ordinary Z change baked into the segments like any other.

The preamble and postamble are injected verbatim. A comment header records tool profile, work area dimensions, estimated run time, cut distance, and rapid distance. Vertical descents (plunges, and same-XY descending moves inside a cut) use the step's Plunge rate when it is > 0. When a step's Spindle speed is > 0 and differs from the last speed emitted in the file, its body starts with `S<rpm>` and, if the spin-up dwell is > 0, `G4 P<seconds>`; the run-time estimate includes both the plunge rate and the dwell.

---

## License

[GPL v3](LICENSE) — any distributed modifications must remain open source under the same license.
