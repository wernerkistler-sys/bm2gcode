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
the minimum over the footprint offsets, with the same values and the same
reflect border as scipy. It visits the offsets by falling structure value
and stops for each block of 16 output pixels once no remaining offset can
lower it: input >= m over the block's whole window, so every remaining
value is at least m - (the largest remaining structure value). Equal
values don't change a minimum, so the result is still scipy's, bit for
bit. banded_grey_erosion/banded_grey_dilation combine several inputs, each
with its own footprint, in that one pass.

binary_erosion/binary_dilation are the flat grey operations on a 0/1 image,
with scipy's binary border: everything outside the image is False.

Every intermediate array has the input's dtype, so a float32 input needs
half the memory of a float64 one. Both scipy and the compiled kernel
subtract the structure in double precision and round once, when storing
the result; rounding is monotone, so a float32 input gives exactly the
float32 rounding of the float64 result.
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
# The compiled kernel's output blocks, offset chunks and column tiles (see
# stack_erosion); the tile is a whole number of blocks.
_BLOCK = 16
_CHUNK = 64
_TILE = 1024


def _get_numba_kernel():
    global _numba_kernel
    if _numba_kernel is None:
        try:
            import numba
        except ImportError:
            _numba_kernel = False
            return None

        @numba.njit(parallel=True, cache=True)
        def band_min(padded, out):
            """out = min over k of padded[k]."""
            nb, Hp, Wp = padded.shape
            for r in numba.prange(Hp):
                row = out[r]
                for c in range(Wp):
                    row[c] = padded[0, r, c]
                for k in range(1, nb):
                    src = padded[k, r]
                    for c in range(Wp):
                        v = src[c]
                        row[c] = v if v < row[c] else row[c]

        @numba.njit(parallel=True, cache=True)
        def block_min(V, B, span, bound):
            """bound[r, b] = min of V[r, b*B : (b + span)*B]."""
            H, Wp = V.shape
            nb = bound.shape[1]
            nch = (Wp + B - 1) // B
            for r in numba.prange(H):
                ch = np.empty(nch)
                for k in range(nch):
                    m = np.inf
                    for c in range(k * B, min(k * B + B, Wp)):
                        v = V[r, c]
                        m = v if v < m else m
                    ch[k] = m
                for b in range(nb):
                    m = np.inf
                    for k in range(b, min(b + span, nch)):
                        m = ch[k] if ch[k] < m else m
                    bound[r, b] = m

        @numba.njit(parallel=True, cache=True)
        def stack_erosion(padded, bound, oi, oj, ob, os_, thr, out, perm,
                          B, K, TW):
            """out[r, c] = min over offsets o of
            padded[ob[o], r + oi[o], c + oj[o]] - os_[o].

            Each output row is computed in tiles of TW columns, each tile in
            blocks of B columns. The offsets come in chunks of K; thr[i] is
            the largest os_ after chunk i, and bound[r, b] is at most every
            value that block b of row r reads. After each chunk a block
            whose largest result is at most bound - thr is finished: no later
            offset gives it a smaller value. The remaining offsets run over
            the runs of unfinished blocks. perm is the order of the rows,
            interleaved so that every thread gets rows from all over the
            image."""
            H, W = out.shape
            n = oi.size
            nbt = TW // B
            for i in numba.prange(H):
                r = perm[i]
                acc = np.empty(TW)
                live = np.empty(nbt, np.bool_)
                rs = np.empty(nbt, np.int64)
                re = np.empty(nbt, np.int64)
                for t0 in range(0, W, TW):
                    tw = min(TW, W - t0)
                    nb = (tw + B - 1) // B
                    for c in range(tw):
                        acc[c] = np.inf
                    for b in range(nb):
                        live[b] = True
                    nruns = 1
                    rs[0] = 0
                    re[0] = tw
                    o = 0
                    ci = 0
                    while o < n and nruns > 0:
                        o1 = min(o + K, n)
                        for q in range(o, o1):
                            s = os_[q]
                            j = t0 + oj[q]
                            for k in range(nruns):
                                a0 = rs[k]
                                a1 = re[k]
                                # Slices, so the inner loop has no index
                                # arithmetic and compiles to SIMD.
                                seg = padded[ob[q], r + oi[q], j + a0:j + a1]
                                sub = acc[a0:a1]
                                for c in range(a1 - a0):
                                    v = seg[c] - s
                                    a = sub[c]
                                    sub[c] = v if v < a else a
                        o = o1
                        if o < n:
                            th = thr[ci]
                            nruns = 0
                            prev = False
                            b0 = t0 // B
                            for b in range(nb):
                                c0 = b * B
                                c1 = min(c0 + B, tw)
                                if live[b]:
                                    mx = acc[c0]
                                    for c in range(c0 + 1, c1):
                                        mx = acc[c] if acc[c] > mx else mx
                                    if bound[r, b0 + b] - th >= mx:
                                        live[b] = False
                                if live[b]:
                                    if prev:
                                        re[nruns - 1] = c1
                                    else:
                                        rs[nruns] = c0
                                        re[nruns] = c1
                                        nruns += 1
                                prev = live[b]
                        ci += 1
                    for c in range(tw):
                        out[r, t0 + c] = acc[c]

        _numba_kernel = (stack_erosion, band_min, block_min)
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


def _block_bound(padded, H, W, ry, rx):
    """bound[r, b]: the minimum of padded, over all inputs, in the window
    that output block b of row r reads (rows r .. r + 2ry, columns
    b*_BLOCK .. b*_BLOCK + _BLOCK - 1 + 2rx, widened to whole blocks)."""
    _, band_min, block_min = _get_numba_kernel()
    if padded.shape[0] > 1:
        P = np.empty(padded.shape[1:], padded.dtype)
        band_min(padded, P)
    else:
        P = padded[0]
    V = ndi.minimum_filter1d(P, 2 * ry + 1, axis=0)[ry:ry + H]
    bound = np.empty((H, -(-W // _BLOCK)))
    block_min(V, _BLOCK, 1 + -(-2 * rx // _BLOCK), bound)
    return bound


def _numba_erosion(inputs, footprints, structure):
    """min over k of grey_erosion(inputs[k], footprints[k], structure) by
    the compiled kernel, in the dtype scipy gives inputs[0]."""
    kh, kw = footprints[0].shape
    ry, rx = kh // 2, kw // 2
    H, W = inputs[0].shape
    dtype = np.result_type(*inputs)
    padded = np.stack([np.pad(np.asarray(a, dtype), ((ry, ry), (rx, rx)),
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
    # Offsets by falling structure value, so the ones that can lower a
    # result most come first and a block's minimum is often settled early.
    # Within a chunk of _CHUNK offsets, by input and row, so consecutive
    # offsets read nearby memory; thr holds each chunk's successors' largest
    # structure value.
    order = np.argsort(-os_, kind="stable")
    n = order.size
    starts = np.arange(_CHUNK, n, _CHUNK)
    thr = os_[order][starts]
    chunk = np.arange(n) // _CHUNK
    order = order[np.lexsort((oj[order], oi[order], ob[order], chunk))]
    oi, oj, ob, os_ = oi[order], oj[order], ob[order], os_[order]
    stack_erosion = _get_numba_kernel()[0]
    # The kernel takes the min in double precision and rounds it once, into
    # out; rounding is monotone, so that is the float32 result scipy gives.
    out = np.empty(inputs[0].shape, inputs[0].dtype)
    with _numba_lock:
        bound = _block_bound(padded, H, W, ry, rx)
        import numba
        T = numba.get_num_threads()
        perm = np.concatenate([np.arange(t, H, T) for t in range(T)])
        stack_erosion(padded, bound, oi, oj, ob, os_, thr, out, perm,
                      _BLOCK, _CHUNK, _TILE)
    return out


def _check_args(input, footprint):
    kh, kw = footprint.shape
    if input.ndim != 2 or kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("grey_erosion needs a 2-D input and an odd-sized footprint.")
    if not footprint.any():
        raise ValueError("grey_erosion needs a footprint with at least one True element.")


def _row_erosion(padded, row, srow, kw, output):
    """1-row erosion of *padded* by footprint row *row*, written into
    *output* (padded's shape and dtype), which is returned."""
    idx = np.flatnonzero(row)
    lo, hi = idx[0], idx[-1]
    if (not srow[row].any() and len(idx) == hi - lo + 1
            and lo + hi == kw - 1):
        # Flat, contiguous run centred on the kernel column.
        ndi.minimum_filter1d(padded, hi - lo + 1, axis=1, output=output)
    else:
        ndi.grey_erosion(padded, footprint=row[None, :],
                         structure=srow[None, :], output=output)
    return output


def _rows_erosion(padded, footprint, structure, H):
    """Erosion of the H output rows whose input is *padded*, which holds
    them plus the footprint's half-height of rows above and below."""
    kh, kw = footprint.shape
    out = None
    done = set()
    # Every row's erosion goes into this one buffer.
    buf = np.empty_like(padded)
    for i in range(kh):
        row = footprint[i]
        if i in done or not row.any():
            continue
        e = _row_erosion(padded, row, structure[i], kw, buf)
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
