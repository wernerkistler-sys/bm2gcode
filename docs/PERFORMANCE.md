# Performance and memory analysis at high resolution

Analysis of where bm2gcode spends memory and time at a high but realistic
canvas resolution of **20 px/mm**, done on 2026-11-02 against commit `076a2b0`.
Recommendations 1–5 were followed up on 2026-11-04; see
[Follow-up: path storage and `ToolPath.resample`](#follow-up-path-storage-and-toolpathresample),
[Follow-up: terrain links](#follow-up-terrain-links),
[Follow-up: Pocket onion skin](#follow-up-pocket-onion-skin),
[Follow-up: morphology in float32](#follow-up-morphology-in-float32) and
[Follow-up: Undercut memory](#follow-up-undercut-memory).

## Summary

- At 20 px/mm, most of the time goes to **per-point Python code**, not to the
  big arrays: one Python tuple per pixel along every path, plus loops over
  those points (terrain links, depth slicing, resampling).
- **Grey morphology** with a non-flat structure (Ball, V-Bit, Custom) takes
  over at higher resolutions or with larger tools. Its cost grows with
  stock area × (px/mm)⁴ × tool diameter².
- **Memory** during a computation runs at roughly 70–115 bytes per canvas
  pixel. The worst spikes came from the removal simulation
  (`ToolPath.resample`, vectorised since; see the follow-up) and from the
  Undercut step, which the follow-up measured at ~600 B/px.

## Scale

| Stock | Resolution | Canvas | One float32 map | One float64 map |
|---|---|---|---|---|
| 200×150 mm | 20 px/mm | 4000×3000 = 12 Mpx | 48 MB | 96 MB |
| 400×300 mm | 20 px/mm | 8000×6000 = 48 Mpx | 192 MB | 384 MB |

Memory scales linearly with pixel count, so the 12 Mpx peaks below should be
multiplied by about 4 for a 400×300 mm stock.

## Method

- Machine: Windows 11 ARM64, 12 cores, Python 3.13. **numba was not
  installed**, so non-flat morphology used the scipy row-decomposition path
  in `grey_morphology.py`.
- Input: a synthetic relief (sums of sines, with a raised central plateau for
  the region-based algorithms), built by the scripts in [`bench/`](../bench).
- Timing: wall clock per progress-callback stage, plus `cProfile` sorted by
  self time.
- Memory: `tracemalloc` (which tracks numpy allocations), peak per
  progress-callback stage. Measured in a separate run, because tracemalloc
  slows Python-heavy code several times over.
- Undercut was **not benchmarked** in this first run; its figures are
  estimated from the code. The follow-up measured it with
  `bench/bench_undercut.py`, and the estimate turned out about 3× too low.

Reproduce from the repo root:

```
python bench/bench_raster.py 200 150 20 3 Ball 2.0        # timing per stage
python bench/bench_raster.py 200 150 20 3 Ball 2.0 mem    # memory per stage
python bench/bench_algos.py 200 150 20 pocket             # contour | vcarve | pocket | cutout
python bench/bench_algos.py 200 150 20 pocket mem
python bench/bench_undercut.py 20                         # umbilic torus, Lollipop 6 mm
python bench/bench_undercut.py 20 mem
```

`bench_raster.py` arguments: width mm, height mm, px/mm, tool diameter,
profile, step-down mm. Step-over is 20 % of the tool diameter.
`bench_algos.py` uses fixed tools: Ball 3 mm (Contour), V-Bit 6 mm 60°
(V-carve), End Mill 6 mm (Pocket onion skin, Cutout).

## Results: memory

Peak traced memory per computation at 12 Mpx (200×150 mm, 20 px/mm):

| Step / stage | Peak | Main cause |
|---|---|---|
| Raster: morphology (erosion + dilation) | 782 MB (~65 B/px) | `grey_morphology.py` works in float64: input copy, padded copy and output are each 8 B/px, and the 12 parallel tiles each allocate a full float64 temporary per footprint row (`_row_erosion`). **Fixed since** (see the follow-up on float32). |
| Raster: removal simulation | **1383 MB** | `ToolPath.resample` builds a Python list with one small numpy array per sample, every 0.5 px. At 20 px/mm that is 40 samples per mm of path, about 6M arrays for this job. The `achieved` map and the dilation are float64. **Fixed since** (see the follow-up): the samples are computed with numpy and painted in chunks. |
| Raster: path storage | ~140 B per point | `_line_points` creates one `(x, y, z)` tuple per pixel on each line, and `make_multipass` copies everything as 4-tuples (`_insert_ceiling_crossings`). Grows with path length × px/mm. Phase 1 added ~145 MB here, depth slicing another ~165 MB. |
| V-carve | 1117 MB | Morphology plus the result map. |
| Pocket | 1220 MB | Morphology plus the result map. |
| Contour | 862 MB | Morphology plus the result map. |
| Undercut (**estimated**) | ~190 B/px, so ~9 GB at 48 Mpx | Up to `MAX_BANDS = 8` `_GapPicker` objects, each holding `F` and `C` as float64 plus `active` (17 B/px each), on top of `depth64`, `center25`, `sky_limit`, `shank`, `g` and `g_sky`. Dexels and Gaps add an int64 `offsets` array (8 B/px each). The non-placed path of `layers.composite_gaps` adds ~80 B/px of temporaries (`mgrid`, float64 coordinates, int64 source indices). | **Measured since:** ~600 B/px on the umbilic torus, mostly duplicated per-picker copies; fixed to ~270 B/px (see the follow-up).

Memory the app keeps between computations:

- Every computed step holds `result` and `accessible` at full size
  (8 B/px, i.e. 384 MB per step at 48 Mpx); Cutout adds `alpha`.
- `GCodeGeneratorApp._erosion_cache` holds two float32 maps per tool
  signature (another 384 MB each at 48 Mpx). It has no size limit and is only
  emptied when invalidated.

## Results: time

### Raster across resolutions

3 mm Ball, 200×150 mm, 3 passes, Horizontal:

| Resolution | Canvas | Total | Morphology |
|---|---|---|---|
| 10 px/mm | 3 Mpx | 16 s | – |
| 20 px/mm | 12 Mpx | 26 s | ~4 s opening + ~2 s simulation dilation |
| 40 px/mm | 48 Mpx | 194 s | 103 s opening + ~50 s simulation dilation |

From 10 to 20 px/mm the total barely grows, because per-point Python work
dominates. From 20 to 40 px/mm the morphology grows ~25× (theory: 16×, i.e.
4× pixels × 4× footprint; the rest is likely cache and memory bandwidth).

### Raster profile at 20 px/mm

cProfile self time (the profiler inflates Python-heavy code; total 41 s
profiled versus 26 s plain):

| Function | Self time | Notes |
|---|---|---|
| `algorithms._geodesic_link` | 12.1 s | Pure-Python Dijkstra (dict + heapq) for each of 3661 terrain links. The search box per link grows with px/mm². |
| scipy `min_or_max_filter` | 4.9 s | The morphology. |
| `ToolPath.resample` | 4.5 s | Python loop over every 0.5 px sample. |
| `make_multipass` + `rough_target` + `_insert_ceiling_crossings` | ~5 s | Per-point Python. |
| `_order_free_raster_pass` | 2.3 s | Greedy nearest-endpoint ordering. |
| `simplify_line` | 1.5 s | RDP per segment. |

### Grey morphology versus tool size

One `grey_erosion` on a 2000×2000 array at 20 px/mm:

| Tool | Radius | Footprint pixels | Time | Per Mpx |
|---|---|---|---|---|
| Ball 1 mm | 10 px | 317 | 0.20 s | 0.05 s |
| End Mill 1 mm | 10 px | 317 | 0.08 s | 0.02 s |
| Ball 3 mm | 30 px | 2821 | 1.13 s | 0.28 s |
| End Mill 3 mm | 30 px | 2821 | 0.23 s | 0.06 s |
| Ball 6 mm | 60 px | 11289 | 5.63 s | 1.41 s |
| End Mill 6 mm | 60 px | 11289 | 0.50 s | 0.13 s |

Non-flat structures (Ball, V-Bit, Custom) cost about pixel count × footprint
pixels. Flat structures (End Mill) use `minimum_filter1d` per footprint row,
so they only grow linearly with the radius. A 6 mm Ball at 48 Mpx needs about
70 s per operation, and Raster runs three (erosion, dilation, simulation
dilation).

### Other algorithms at 12 Mpx

| Algorithm | Total | Where the time goes |
|---|---|---|
| Contour (Ball 3 mm, 0.5 mm levels) | 11 s | ~5 s morphology, ~3 s `_contour_graph`/ordering, ~1 s `label`. |
| V-carve (V-Bit 6 mm 60°) | 40 s | 28 s morphology (non-flat cone, ~58 px radius), 6 s `skimage.medial_axis`. |
| Pocket (End Mill 6 mm, onion skin) | 51 s | `_onion_skin_chains` calls `binary_erosion(disk(step_px))` once per ring: 97 full morphology calls, 23 s of 1-D filtering, plus ~15 s in `numpy.zeros` because scipy allocates a new output array for every footprint row. |
| Cutout (End Mill 6 mm) | 8 s | About half is morphology. |

## Follow-up: path storage and `ToolPath.resample`

Done on 2026-11-04 on Linux (AMD Ryzen 5 7600, 12 threads, Python 3.13,
numba installed), with the same Raster job:
`bench_raster.py 200 150 20 3 Ball 2.0`.

The original recommendation 1 combined two changes. Measured separately,
they differ a lot in both cost and payoff.

**The final path is small.** After RDP, the job's `ToolPath.segments` holds
7,586 segments with 68,359 points, about 10 MB as tuples. Storing them as
arrays saves almost nothing. `ToolPath.resample` has one caller,
`simulate_toolpath_result`, and runs on this simplified path. It could be
fixed without changing how paths are stored.

**The dense data is the intermediate.** Phase 1 builds one point per pixel:
251 lines × 4,000 points, about 1M points per pass, kept as tuples through
depth slicing, air pruning, ordering and terrain linking until RDP. Without
the profiler, its per-point Python costs about 1.5 s (`_line_points` ~0.4 s,
`make_multipass` ~1 s), not ~10 s. The cProfile figures above inflate it,
and most of their ~10 s was `resample`. Its memory is the ~310 MB already
listed under "Raster: path storage": a 536 MB peak during slicing, with
~460 MB held until RDP.

### Vectorised `resample` (implemented)

`ToolPath.iter_resampled` computes each segment's cumulative XY arc length
and places its samples with `np.arange` and `np.searchsorted`, in chunks of
at most 2¹⁸ samples. A single long Spiral or Coil segment therefore stays
bounded too. `simulate_toolpath_result` paints each chunk as it comes, and
`resample` concatenates them. Before is commit `7f0554b`, after is the
vectorised version:

| | Before | After |
|---|---|---|
| Raster total | 16.0 s | **12.7 s** |
| Simulation stage | 4.5 s | **1.2 s** |
| Simulation stage peak (tracemalloc) | 1176 MB | **860 MB** |
| Of which above the 566 MB held before the stage | 610 MB | **294 MB** |

The remaining 294 MB is the float64 `achieved` map (96 MB) and the float64
temporaries of the dilation, which is recommendation 4. The test suite
passed before and after.

The samples are the same as before, except where the old loop's running sum
(`pos += spacing`) drifted across an exact edge boundary. About 30 % of the
samples sit exactly on a 0.5 px rounding tie, so noise of 10⁻¹³ mm decides
their pixel either way. On this job the removal map differs at 19.5 % of
pixels: 4.6 µm on average and 15 µm at most, below one grey level (19.6 µm).
It comes out deeper about as often as shallower (12.1 % against 7.4 % of
pixels), so there is no bias. A sample that lands exactly on a plunge's
vertex now takes the plunge's bottom.

### Paths as arrays throughout (not recommended)

Converting `ToolPath.segments` to `(N, 3)` arrays touches every algorithm's
point builders, `_insert_ceiling_crossings`/`make_multipass`,
`prune_air_segments`, the ordering functions (`list(reversed(seg))`), the
terrain-link anchors, `dressup.apply_tabs`, `iter_moves`/`to_gcode` (slower
over numpy scalars unless converted back with `.tolist()`), `app.py`
(`if not pts:` fails on arrays) and the tests that compare `tp.segments` with
lists. All of this only affects the ~1.5 s and ~310 MB of the dense
intermediate, which have a cheaper fix:

**Simplify before depth slicing.** V-Carving already RDP-simplifies its
full-depth segments before `make_multipass` (`algorithms.py`, "full_depth_segs
= [simplify_line(...)"), and `make_multipass` inserts exact ceiling crossings
for this case. Ordering and terrain linking only read segment endpoints,
which RDP keeps. A one-line trial of this for Raster cut the memory held
during ordering and linking from ~460 MB to ~165 MB and the slicing peak from
536 MB to 291 MB. Time stayed about the same (15.9 s against 16.2 s). The
output moved by up to 83 µm, about four grey levels, probably because the
path was then simplified twice. Doing this properly means skipping the final
RDP for cut segments and updating the README's Raster pipeline. Pocket's
raster fill uses the same helpers and would benefit too.

## Follow-up: terrain links

Same machine and Raster job as the previous follow-up.

`_geodesic_link` costs each 8-connected move by its 3-D length,
√(Δxy² + Δz²), so links go around walls and follow level ground. The
suggested `skimage.graph.MCP_Geometric` can't express that.
It costs a move as its length × the mean of the two pixels' values, so it
never sees the depth *difference* between them. A slope-based value per
pixel would charge a link running level along a slope as if it climbed, and
would smear a vertical step over two pixels. The links would come out
different, not just faster.

Measured on 300 random links of up to 24 px on the benchmark relief:

| Search | Cost | Time per link | Same pixels as the heapq version |
|---|---|---|---|
| Python `heapq` Dijkstra (before) | exact 3-D length | 2.9 ms | — |
| `scipy.sparse.csgraph.dijkstra` | exact 3-D length | **0.18 ms** | 300/300 |
| `skimage.graph.MCP_Flexible`, `travel_cost` overridden | exact 3-D length | 0.68 ms | 259/300 (equal length, other ties) |

`_geodesic_link` now builds the window's 8-connected graph with numpy, with
edge costs computed the same way as before, and solves it with
`csgraph.dijkstra`. The window, surface, return value and Bresenham fallback
are unchanged, and so are its callers in Raster, Pocket and Contour.
`csgraph` computes the whole window instead of stopping at the target,
which costs little because the windows are small (around 40×40 px).

| | Before (`5181cf3`) | After |
|---|---|---|
| Raster total | 12.9 s | **8.2 s** |
| Depth slicing + ordering + terrain links (3,661) | 7.5 s | **3.0 s** |

The job's toolpath is identical before and after (all 7,586 segments
compared). A new test compares against the heapq version on relief and on
flat floors with steps, with ceilings, finishing allowance and canvas edges:
120/120 links are identical. Exactly tied shortest paths may still be broken
differently in rare cases, with the same length. The Pocket and Contour
benchmarks hardly link (0 and 7 links), so their times don't change.

## Follow-up: Pocket onion skin

Same machine as the previous follow-ups, with
`bench_algos.py 200 150 20 pocket` (End Mill 6 mm, Step-over 2.4 mm = 48 px,
Onion Skin (Outside-In), one region of 8.1 Mpx).

A profile of the onion skin phase before the change: 19.8 s, of which 16.5 s
was 23 `binary_erosion(disk(step_px))` calls on the whole region.
`find_contours`, `label`, ring ordering and stitching took the other ~3 s.

`_onion_skin_chains` now computes one `distance_transform_edt` of the region
and takes level k as the pixels whose squared distance to the outside
exceeds (k · step_px)². Within a branch after a fork, it intersects that
with the branch's mask. Labelling, the stray-fragment filter, forks,
`find_contours` and stitching are unchanged.

- **Level 1 is identical** to one erosion by `disk(step_px)`, which keeps
  exactly the pixels with squared distance > step_px².
- **Deeper levels differ**, and come out more accurate. Eroding by a
  *digital* disk again and again adds up its polygonal error. On a disk of
  radius 420 px, the eroded rings drift up to 12 px (a whole step-over of
  12 px) from the ideal radii 420 − 12k over 35 levels. The distance-transform
  rings stay within 1 px. With a 48 px step-over the drift is 1.7 px against
  1 px. Rectangles come out identical.
- **Links stay inside the region.** A pixel of level k+1 lies more than
  (k+1) · step_px from the outside, so its whole `disk(step_px)` lies inside
  level k (triangle inequality). The guarantee that every ring has a short
  link back to the previous one, inside the region, holds as before.

| | Before (`e9ba8f5`) | After |
|---|---|---|
| Pocket total | 24.6 s | **10.0 s** |
| Onion skin phase | 18.4 s | **3.7 s** |
| Onion skin phase peak (tracemalloc) | 851 MB | **747 MB** |
| Cut length | 27,776 mm | 27,741 mm |
| Pocket floor pixels at target depth | 100 % | 100 % |

All tests passed after the change, including the existing onion-skin
regressions (dumbbell forks, stray fragments, links inside the region,
S-links). A new test checks that the rings on a 300 px disk lie on the ideal
radii R − k · step: 96.7 % of the chain's points are within 1.5 px, against
83.0 % with repeated erosions. The remaining ~3 s are per-level
`find_contours` and `label` on the whole region crop; cropping each level to
its shrinking bounding box would reduce them further.

## Follow-up: morphology in float32

Same machine as the previous follow-ups. Raster:
`bench_raster.py 200 150 20 3 Ball 2.0`, once with numba's compiled kernel
and once with `grey_morphology.USE_NUMBA = False` (the scipy row path, which
the original Windows runs used). Pocket: `bench_algos.py 200 150 20 pocket`
(End Mill 6 mm, a flat footprint).

`grey_morphology.py` already keeps the input's dtype. The float64 came from
the callers, which converted their float32 maps to float64 before every
erosion and dilation and converted the result back. The changes:

- **The callers in `algorithms.py` pass float32 maps:** the achievable-depth
  erosion and dilation of Raster, Pocket, Contour and V-carve, V-carve's
  feature-width opening, `compute_air_free_depth`, and the `achieved` map of
  the removal simulation. The structures stay float64.
- **The scipy row path reuses one buffer per tile.** `_row_erosion` writes
  into it with `output=`, where scipy used to allocate and zero a new array
  for every footprint row.
- **The compiled kernel pads in the inputs' dtype** and rounds into an output
  of that dtype, instead of padding in float64.

**Results.** Both scipy and the compiled kernel subtract the structure in
double precision and round only when storing, and rounding is monotone. A
float32 input therefore gives exactly the float32 rounding of the float64
result, which is what the callers kept anyway. A new test checks this for
both paths. On 100×75 mm versions of the benchmark jobs (Raster on both
paths, a Raster refining step with Avoid air cutting, Pocket, Contour and
V-carve), every toolpath and every accessible map is identical. So are
Pocket's and V-carve's result maps. The Raster and Contour result maps differ
by at most 0.0003 µm, one float32 rounding step, because the simulation now
rounds each painted sample depth to float32 before the dilation. All tests
passed.

Stage time / stage peak (tracemalloc, including what is already held):

| Job | Stage | Before (`ecf2178`) | After |
|---|---|---|---|
| Raster, numba | achievable depth | 1.6 s / 488 MB | **1.3 s / 244 MB** |
| | removal simulation | 1.2 s / 863 MB | 1.2 s / **667 MB** |
| | total | 8.2 s | 7.9 s |
| Raster, scipy path | achievable depth | 3.9 s / 768 MB | 4.2 s / **301 MB** |
| | removal simulation | 2.4 s / 1145 MB | 2.7 s / **725 MB** |
| | total | 11.7 s | 12.5 s |
| Pocket (flat) | achievable depth | 3.3 s / 839 MB | **1.3 s / 336 MB** |
| | removal simulation | 1.7 s / 1150 MB | **0.8 s / 695 MB** |
| | total | 9.5 s | **6.7 s** |

The highest stage peak of each job drops by 23–40 %. The memory each stage
adds on top of what is already held drops by more than half (Raster
simulation with numba: 295 MB → 99 MB). Flat footprints get faster because
the zeroed temporaries per row are gone. The one cost is that scipy's
non-flat erosion (Ball, V-Bit and Custom without numba) runs about 8 %
slower on float32 input when using all 12 threads: one 12 Mpx Ball erosion
took 2.10 s against 1.94 s. Single-threaded it isn't slower, which points to
memory bandwidth or conversion in scipy's inner loop rather than to the
buffer. The first run after the change also compiles the kernel's float32
version once (about 1 s); numba caches it.

The Undercut step (`undercut.py`) is covered by the next follow-up.

## Follow-up: Undercut memory

Same machine as the previous follow-ups, with the new
`bench/bench_undercut.py`. It runs the Undercut step on
`umbilic-torus-3-1.stl` (118 × 124 × 50 mm) with a 6 mm Lollipop (3 mm neck),
2 mm levels and 20 % step-over. That gives 8 bands, and 71 % of the pixels
have gaps.

**The estimate was too low.** Measured at 10 px/mm, the step peaked at
606 B/px, not ~190 B/px, and already held 340 B/px before its first level. A
tracemalloc snapshot showed that most of it was the 8 band pickers
(`_GapPicker`), each with its own copies:

| Per picker (× 8) | B/px, all 8 |
|---|---|
| `self.top`, a float64 copy of the depth map, read only by `_reset` | 64 |
| `F`, float64 | 64 |
| `C`, float64 | 64 |
| `ceil_sorted`, a copy of every gap ceiling in sorted order | 47 |
| `active` | 8 |

During each level, `banded_grey_erosion`/`banded_grey_dilation` stack and
pad all bands' F (or C) windows on top of this.

Of the two alternatives in recommendation 5, sharing F/C between the band
pickers would save the most but is complex. The pickers sit at different
depths, so it needs a shared base plus per-band overrides, and the banded
morphology still needs a dense array per band and window. Keeping F/C only
within the active windows doesn't help a model like this, whose windows
cover nearly the whole canvas. Two simpler steps were done instead:

1. **Shared read-only data.** All pickers share `top` and `ceil_sorted`,
   which the caller already computes. The output is bit-identical.
2. **float32.** F and C, and the depth map, `center25`, the shank limit,
   `sky_limit`, `g` and `g_sky` are float32. The banded morphology then pads
   in float32 too (see the previous follow-up). STL sources count as 65,536
   grey levels, so the free-space tests compare against
   eps = 7.6 · 10⁻⁶. Float32's rounding near full depth is below 10⁻⁷.

| | Before (`30b2426`) | Step 1 | Steps 1 + 2 |
|---|---|---|---|
| 10 px/mm (1.46 Mpx): highest stage peak | 884 MB (606 B/px) | 723 MB (496 B/px) | **416 MB (285 B/px)** |
| 10 px/mm: held before the first level | 496 MB | – | **218 MB** |
| 10 px/mm: time | 6.5 s | – | **5.2 s** |
| 20 px/mm (5.83 Mpx): highest stage peak | 3429 MB (588 B/px) | – | **1556 MB (267 B/px)** |
| 20 px/mm: held before the first level | 1883 MB | – | **770 MB** |
| 20 px/mm: time | 71.3 s | – | **55.4 s** |

At 48 Mpx that is about 13 GB instead of about 28 GB.

Step 2 isn't bit-identical. At both resolutions, every chain has the same
number of points as before, and the points move by at most 0.001 µm. The
result map, the reach overlay (`undercut_reach`) and the warnings are
identical. The accessible map differs by one float32 rounding step, because
the 2.5-D erosion's result is rounded to float32 before the dilation. All
tests passed. `test_undercut.py`'s picker checks now compare against the
float32 rounding of the gaps' float64 values.

The remaining ~270 B/px are mostly the 8 pickers' float32 F/C (64 B/px),
the Gaps arrays (~40 B/px on this model), and the banded morphology's
per-level stack. Sharing F/C between band pickers would address the first,
for Lollipop tools only (T-Slot and Dovetail have one band).

## Recommendations

Ordered by expected payoff.

1. **Simulation and path storage.** Vectorising `ToolPath.resample` is done
   (see the follow-up). For the dense Raster intermediate, simplify the
   full-depth segments before `make_multipass`, as V-Carving does, instead of
   converting every path to `(N, 3)` arrays. Keep `ToolPath.segments` as
   lists of tuples.
2. **Terrain links.** Done with `scipy.sparse.csgraph.dijkstra`, not
   `MCP_Geometric`, whose per-pixel cost can't express 3-D arc length (see
   the follow-up).
3. **Pocket onion skin.** Done: one `distance_transform_edt` per region,
   thresholded at k · step (see the follow-up). Remaining option: crop each
   level to its shrinking bounding box before `find_contours` and `label`.
4. **Morphology memory.** Done: the algorithms pass float32 maps, and the
   scipy row path reuses one `output=` buffer per tile (see the follow-up).
   Results are the float32 rounding of the float64 ones, as before.
5. **Undercut memory.** Done: shared read-only picker data and float32
   (see the follow-up). Remaining option: share F/C between the band
   pickers of a Lollipop.
6. **Non-flat morphology speed:** numba's compiled kernel in
   `grey_morphology.py` is the intended fix. Whether numba wheels exist for
   Windows ARM64 with Python 3.13 was not checked. Without it, Ball and V-Bit
   tools at high resolution are inherently px/mm⁴.
7. **App memory:** put a size limit on `_erosion_cache` (LRU), so it doesn't
   grow with every tool tried.
