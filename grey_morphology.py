"""Memory-bounded 2-D grey morphology with scipy.ndimage's semantics.

scipy.ndimage.grey_erosion/grey_dilation precompute a table of memory
offsets for every border configuration of the footprint, sized roughly
(footprint pixels) x (footprint height x width) x 8 bytes -- O(r^4) in the
kernel radius. A 25 mm tool at 10 px/mm (r = 125 px) needs ~25 GB for that
table alone.

The functions here split the footprint into its rows: each row is a 1-row
scipy erosion (tiny offset table) of the input, shifted vertically by that
row's offset, and the running minimum over all rows is the 2-D erosion.
Results are identical to scipy's (default mode="reflect", origin 0) and
memory stays at a few image-sized arrays regardless of kernel size. Flat
rows use minimum_filter1d, whose cost doesn't depend on the row width, and
a row whose mirror row (same footprint and structure) exists is filtered
once and applied at both offsets.

The output is split into tiles of rows computed in parallel on a shared
thread pool: each tile needs only its own input rows plus the footprint's
half-height above and below, and scipy releases the GIL while it filters.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi

_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
# A tile at least this many footprint heights tall, so the rows it shares
# with its neighbours stay a small part of its work.
_MIN_TILE_KERNELS = 2
_MIN_TILE_ROWS = 64


def _row_erosion(padded, row, srow, kw):
    idx = np.flatnonzero(row)
    lo, hi = idx[0], idx[-1]
    if (not srow[row].any() and len(idx) == hi - lo + 1
            and lo + hi == kw - 1):
        # Flat, contiguous run centred on the kernel column.
        return ndi.minimum_filter1d(padded, hi - lo + 1, axis=1)
    return ndi.grey_erosion(padded, footprint=row[None, :],
                            structure=srow[None, :])


def _rows_erosion(padded, footprint, structure, H):
    """Erosion of the H output rows whose input is *padded*, which holds
    them plus the footprint's half-height of rows above and below."""
    kh, kw = footprint.shape
    out = None
    done = set()
    for i in range(kh):
        row = footprint[i]
        if i in done or not row.any():
            continue
        e = _row_erosion(padded, row, structure[i], kw)
        rows = [i]
        j = kh - 1 - i
        if (j != i and np.array_equal(footprint[j], row)
                and np.array_equal(structure[j][row], structure[i][row])):
            rows.append(j)
            done.add(j)
        for k in rows:
            if out is None:
                out = e[k:k + H].copy()
            else:
                np.minimum(out, e[k:k + H], out=out)
    return out


def grey_erosion(input, footprint, structure=None):
    """Same result as scipy.ndimage.grey_erosion(input, footprint=footprint,
    structure=structure) for a 2-D input and an odd-sized footprint:
    min over d in footprint of input(x + d) - structure(d).
    structure None means flat (all zeros)."""
    input = np.asarray(input)
    footprint = np.asarray(footprint, dtype=bool)
    kh, kw = footprint.shape
    if input.ndim != 2 or kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("grey_erosion needs a 2-D input and an odd-sized footprint.")
    if not footprint.any():
        raise ValueError("grey_erosion needs a footprint with at least one True element.")
    if structure is None:
        structure = np.zeros(footprint.shape, dtype=np.float64)
    else:
        structure = np.asarray(structure)
    ry = kh // 2
    H = input.shape[0]
    # numpy's "symmetric" padding is scipy's "reflect" border mode.
    padded = np.pad(input, ((ry, ry), (0, 0)), mode="symmetric")
    tile = max(_MIN_TILE_KERNELS * kh, _MIN_TILE_ROWS, -(-H // _POOL._max_workers))
    if tile >= H:
        return _rows_erosion(padded, footprint, structure, H)
    # scipy's result has the input's dtype.
    out = np.empty(input.shape, dtype=padded.dtype)

    def tile_job(r0):
        r1 = min(H, r0 + tile)
        out[r0:r1] = _rows_erosion(padded[r0:r1 + 2 * ry], footprint, structure, r1 - r0)

    list(_POOL.map(tile_job, range(0, H, tile)))
    return out


def grey_dilation(input, footprint, structure=None):
    """Same result as scipy.ndimage.grey_dilation(input, footprint=footprint,
    structure=structure) for a 2-D input and an odd-sized footprint:
    max over d in footprint of input(x - d) + structure(d)."""
    footprint = np.asarray(footprint, dtype=bool)
    if structure is not None:
        structure = np.asarray(structure)[::-1, ::-1]
    return -grey_erosion(-np.asarray(input), footprint[::-1, ::-1], structure)


def grey_opening(input, footprint, structure=None):
    """Same result as scipy.ndimage.grey_opening(input, footprint=footprint,
    structure=structure): erosion followed by dilation."""
    return grey_dilation(grey_erosion(input, footprint, structure),
                         footprint, structure)
