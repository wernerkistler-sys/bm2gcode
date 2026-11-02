# Performance and memory analysis at high resolution

Analysis of where bm2gcode spends memory and time at a high but realistic
canvas resolution of **20 px/mm**, done on 2026-11-02 against commit `076a2b0`.

## Summary

- At 20 px/mm, most of the time goes to **per-point Python code**, not to the
  big arrays: one Python tuple per pixel along every path, plus loops over
  those points (terrain links, depth slicing, resampling).
- **Grey morphology** with a non-flat structure (Ball, V-Bit, Custom) takes
  over at higher resolutions or with larger tools. Its cost grows with
  stock area × (px/mm)⁴ × tool diameter².
- **Memory** during a computation runs at roughly 70–115 bytes per canvas
  pixel. The worst spikes come from the removal simulation
  (`ToolPath.resample`) and, by estimate, from the Undercut step.

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
- Undercut was **not benchmarked**; its figures are estimated from the code.

Reproduce from the repo root:

```
python bench/bench_raster.py 200 150 20 3 Ball 2.0        # timing per stage
python bench/bench_raster.py 200 150 20 3 Ball 2.0 mem    # memory per stage
python bench/bench_algos.py 200 150 20 pocket             # contour | vcarve | pocket | cutout
python bench/bench_algos.py 200 150 20 pocket mem
```

`bench_raster.py` arguments: width mm, height mm, px/mm, tool diameter,
profile, step-down mm. Step-over is 20 % of the tool diameter.
`bench_algos.py` uses fixed tools: Ball 3 mm (Contour), V-Bit 6 mm 60°
(V-carve), End Mill 6 mm (Pocket onion skin, Cutout).

## Results: memory

Peak traced memory per computation at 12 Mpx (200×150 mm, 20 px/mm):

| Step / stage | Peak | Main cause |
|---|---|---|
| Raster: morphology (erosion + dilation) | 782 MB (~65 B/px) | `grey_morphology.py` works in float64: input copy, padded copy and output are each 8 B/px, and the 12 parallel tiles each allocate a full float64 temporary per footprint row (`_row_erosion`). |
| Raster: removal simulation | **1383 MB** | `ToolPath.resample` builds a Python list with one small numpy array per sample, every 0.5 px. At 20 px/mm that is 40 samples per mm of path, about 6M arrays for this job. The `achieved` map and the dilation are float64. |
| Raster: path storage | ~140 B per point | `_line_points` creates one `(x, y, z)` tuple per pixel on each line, and `make_multipass` copies everything as 4-tuples (`_insert_ceiling_crossings`). Grows with path length × px/mm. Phase 1 added ~145 MB here, depth slicing another ~165 MB. |
| V-carve | 1117 MB | Morphology plus the result map. |
| Pocket | 1220 MB | Morphology plus the result map. |
| Contour | 862 MB | Morphology plus the result map. |
| Undercut (**estimated**) | ~190 B/px, so ~9 GB at 48 Mpx | Up to `MAX_BANDS = 8` `_GapPicker` objects, each holding `F` and `C` as float64 plus `active` (17 B/px each), on top of `depth64`, `center25`, `sky_limit`, `shank`, `g` and `g_sky`. Dexels and Gaps add an int64 `offsets` array (8 B/px each). The non-placed path of `layers.composite_gaps` adds ~80 B/px of temporaries (`mgrid`, float64 coordinates, int64 source indices). |

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

## Recommendations

Ordered by expected payoff.

1. **Store paths as `(N, 3)` numpy arrays and vectorise `ToolPath.resample`**
   (cumulative arc length plus `np.interp`). Removes the 1.4 GB simulation
   spike and ~10 s of per-point Python in `_line_points`, `make_multipass` and
   `resample`. This is the largest change: segments are lists of tuples
   throughout `algorithms.py`, `toolpath.py` and `dressup.py`.
2. **Replace `_geodesic_link`** with `skimage.graph.MCP_Geometric` on the link
   window (already used in `undercut.py`), or a compiled routine. That moves
   the biggest single cost in Raster into C.
3. **Pocket onion skin:** compute one `distance_transform_edt` per region and
   threshold it at k·step, instead of repeated `binary_erosion` calls. The
   cost no longer depends on radius or ring count. The rings come out
   slightly different from discrete repeated erosions, so tests need
   checking.
4. **Morphology memory:** work in float32 instead of float64 and pass a
   preallocated `output=` to the per-row scipy calls in `grey_morphology.py`.
   That roughly halves the morphology peak and removes the allocation churn.
   First check that float32 is precise enough for the module's
   "identical to scipy" contract and for the algorithms' tolerances.
5. **Undercut memory:** share the `F`/`C` arrays between band pickers, or keep
   them in float32 and only within the active windows.
6. **Non-flat morphology speed:** numba's compiled kernel in
   `grey_morphology.py` is the intended fix. Whether numba wheels exist for
   Windows ARM64 with Python 3.13 was not checked. Without it, Ball and V-Bit
   tools at high resolution are inherently px/mm⁴.
7. **App memory:** put a size limit on `_erosion_cache` (LRU), so it doesn't
   grow with every tool tried.
