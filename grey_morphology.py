"""Memory-bounded 2-D grey and binary morphology with scipy.ndimage's
semantics.

scipy.ndimage.grey_erosion/grey_dilation/binary_erosion/binary_dilation
precompute a table of memory offsets for every border configuration of the footprint, sized roughly
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

When numba is installed, a float input with a non-flat structure goes to a
compiled kernel instead: one parallel pass over the output rows that takes
the minimum over every footprint offset, with the same values and the same
reflect border as scipy. banded_grey_erosion/banded_grey_dilation combine
several inputs, each with its own footprint, in that one pass.

binary_erosion/binary_dilation are the flat grey operations on a 0/1 image,
with scipy's binary border: everything outside the image is False.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi

_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
# A tile at least this many footprint heights tall, so the rows it shares
# with its neighbours stay a small part of its work.
_MIN_TILE_KERNELS = 2
_MIN_TILE_ROWS = 64

USE_NUMBA = True        # the compiled kernel when numba is installed
_numba_kernel = None    # compiled on first use; False once numba is found missing
# numba's default threading layer aborts on kernel launches from two
# threads at once; every launch already uses all cores.
_numba_lock = threading.Lock()


def _get_numba_kernel():
    global _numba_kernel
    if _numba_kernel is None:
        try:
            import numba
        except ImportError:
            _numba_kernel = False
            return None

        @numba.njit(parallel=True, cache=True)
        def stack_erosion(padded, oi, oj, ob, os_, out):
            """out[r, c] = min over offsets o of
            padded[ob[o], r + oi[o], c + oj[o]] - os_[o]."""
            H, W = out.shape
            for r in numba.prange(H):
                acc = np.full(W, np.inf)
                for o in range(oi.size):
                    # A slice, so the inner loop has no index arithmetic
                    # and compiles to SIMD.
                    seg = padded[ob[o], r + oi[o], oj[o]:oj[o] + W]
                    s = os_[o]
                    for c in range(W):
                        v = seg[c] - s
                        a = acc[c]
                        acc[c] = v if v < a else a
                out[r] = acc

        _numba_kernel = stack_erosion
    return _numba_kernel or None


def _numba_applies(inputs, footprints, structure):
    """True if the compiled kernel is available and the cheaper choice:
    float inputs and a structure that isn't flat over the footprints (flat
    rows cost scipy's row path the same whatever their width)."""
    if not USE_NUMBA or _numba_kernel is False:
        return False
    if any(a.dtype not in (np.float32, np.float64) for a in inputs):
        return False
    if structure is None or not any(structure[fp].any() for fp in footprints):
        return False
    return _get_numba_kernel() is not None


def _numba_erosion(inputs, footprints, structure):
    """min over k of grey_erosion(inputs[k], footprints[k], structure) by
    the compiled kernel, in the dtype scipy gives inputs[0]."""
    kh, kw = footprints[0].shape
    ry, rx = kh // 2, kw // 2
    padded = np.stack([np.pad(np.asarray(a, np.float64), ((ry, ry), (rx, rx)),
                              mode="symmetric") for a in inputs])
    oi, oj, ob, os_ = [], [], [], []
    for k, fp in enumerate(footprints):
        i, j = np.nonzero(fp)
        oi.append(i)
        oj.append(j)
        ob.append(np.full(i.size, k))
        os_.append(structure[i, j])
    oi, oj, ob = (np.concatenate(x).astype(np.int64) for x in (oi, oj, ob))
    os_ = np.concatenate(os_).astype(np.float64)
    out = np.empty(inputs[0].shape, np.float64)
    with _numba_lock:
        _get_numba_kernel()(padded, oi, oj, ob, os_, out)
    # Rounding is monotone, so the min of the rounded values is the
    # rounded min: the float32 result scipy gives.
    return out.astype(inputs[0].dtype, copy=False)


def _check_args(input, footprint):
    kh, kw = footprint.shape
    if input.ndim != 2 or kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("grey_erosion needs a 2-D input and an odd-sized footprint.")
    if not footprint.any():
        raise ValueError("grey_erosion needs a footprint with at least one True element.")


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
    _check_args(input, footprint)
    if structure is not None:
        structure = np.asarray(structure)
    if _numba_applies([input], [footprint], structure):
        return _numba_erosion([input], [footprint], structure)
    if structure is None:
        structure = np.zeros(footprint.shape, dtype=np.float64)
    kh = footprint.shape[0]
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


def _binary_morphology(grey_op, input, structure):
    """grey_op, flat, of the boolean *input*, where everything outside
    *input* is False. A False margin one footprint half-size wide holds
    every value the output pixels read, so grey_op's reflect border is
    never reached."""
    input = np.asarray(input, dtype=bool)
    structure = np.asarray(structure, dtype=bool)
    _check_args(input, structure)
    H, W = input.shape
    ry, rx = structure.shape[0] // 2, structure.shape[1] // 2
    padded = np.pad(input, ((ry, ry), (rx, rx)), mode="constant").astype(np.float32)
    return grey_op(padded, structure)[ry:ry + H, rx:rx + W] > 0.5


def binary_erosion(input, structure):
    """Same result as scipy.ndimage.binary_erosion(input, structure=structure)
    for a 2-D input and an odd-sized structure: True where every x + d, d in
    structure, is True in *input*; pixels outside the image count as False."""
    return _binary_morphology(grey_erosion, input, structure)


def binary_dilation(input, structure):
    """Same result as scipy.ndimage.binary_dilation(input, structure=structure)
    for a 2-D input and an odd-sized structure: True where some x - d, d in
    structure, is True in *input*."""
    return _binary_morphology(grey_dilation, input, structure)


def banded_grey_erosion(inputs, footprints, structure=None):
    """min over k of grey_erosion(inputs[k], footprints[k], structure): the
    erosion by a footprint split into parts, each part reading its own
    input. Inputs share one shape and footprints one shape."""
    inputs = [np.asarray(a) for a in inputs]
    footprints = [np.asarray(fp, dtype=bool) for fp in footprints]
    if len(inputs) != len(footprints) or not inputs:
        raise ValueError("banded_grey_erosion needs one footprint per input.")
    for a, fp in zip(inputs, footprints):
        _check_args(a, fp)
        if a.shape != inputs[0].shape or fp.shape != footprints[0].shape:
            raise ValueError("banded_grey_erosion needs equal-shaped inputs and footprints.")
    if structure is not None:
        structure = np.asarray(structure)
    if _numba_applies(inputs, footprints, structure):
        return _numba_erosion(inputs, footprints, structure)
    out = None
    for a, fp in zip(inputs, footprints):
        e = grey_erosion(a, fp, structure)
        out = e if out is None else np.minimum(out, e, out=out)
    return out


def banded_grey_dilation(inputs, footprints, structure=None):
    """max over k of grey_dilation(inputs[k], footprints[k], structure)."""
    if structure is not None:
        structure = np.asarray(structure)[::-1, ::-1]
    return -banded_grey_erosion([-np.asarray(a) for a in inputs],
                                [np.asarray(fp, dtype=bool)[::-1, ::-1] for fp in footprints],
                                structure)
