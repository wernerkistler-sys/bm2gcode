"""The achievable-depth surface cache shared by every step's computation,
bounded by the machine's memory (no Tk code).

ErosionCache behaves like the dict the algorithms expect (``get``, ``[]``,
``in``, ``items``), keyed by tool signature, each value a tuple of canvas
arrays. Entries are kept in least-recently-used order: reading one with
``get`` or ``[]`` makes it the most recent, iterating doesn't. The cache only
saves time, since every reader recomputes on a miss, so it gives memory back
whenever the app or the system needs it:

- storing an entry evicts the least recently used others while the
  system's available memory is below MIN_AVAILABLE_FRACTION of its total,
  or while the cache holds more than MAX_FRACTION of it;
- ``make_room(nbytes)``, called before a computation, evicts them until
  ``nbytes`` more are available above that floor.

The entry just stored is never evicted by its own store. An array shared
by several entries counts once. Evicting an entry frees its arrays unless
something else, such as a step's toolpath, still holds them.

Memory comes from psutil when it is installed, else from /proc/meminfo
(Linux) or GlobalMemoryStatusEx (Windows). Where none of these work, the
cache holds at most FALLBACK_BYTES, and ``make_room`` evicts nothing more.
"""
import sys
import threading
from collections import OrderedDict

import numpy as np

MIN_AVAILABLE_FRACTION = 0.25   # of total RAM, left available after a store
MAX_FRACTION = 0.5              # of total RAM, the most the cache holds
FALLBACK_BYTES = 1 << 30        # the most it holds when memory can't be read


def system_memory():
    """(total, available) bytes of physical memory, or None when they
    can't be read."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                name, value = line.split(":", 1)
                info[name] = int(value.split()[0]) * 1024
        return info["MemTotal"], info["MemAvailable"]
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except Exception:
            pass
    return None


def _arrays(value):
    """The numpy arrays in *value*, a cache value (an array or a tuple or
    list of them, possibly nested)."""
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _arrays(v)


class ErosionCache:
    """Memory-bounded LRU mapping of tool signature -> tuple of arrays.

    *memory* is a callable returning (total, available) bytes or None;
    system_memory by default.
    """

    def __init__(self, memory=system_memory):
        self._memory = memory
        self._data = OrderedDict()
        self._lock = threading.RLock()

    # ── Mapping interface ───────────────────────────────────────────────
    def __getitem__(self, key):
        with self._lock:
            value = self._data[key]
            self._data.move_to_end(key)
            return value

    def get(self, key, default=None):
        with self._lock:
            if key not in self._data:
                return default
            return self[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            self._evict(0, keep=key)

    def __delitem__(self, key):
        with self._lock:
            del self._data[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._data

    def __len__(self):
        with self._lock:
            return len(self._data)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        """The keys, least recently used first, without using any entry."""
        with self._lock:
            return list(self._data)

    def items(self):
        """(key, value) pairs, least recently used first, without using any
        entry."""
        with self._lock:
            return list(self._data.items())

    def clear(self):
        with self._lock:
            self._data.clear()

    # ── Memory ──────────────────────────────────────────────────────────
    def nbytes(self):
        """Bytes of the arrays the cache holds, each distinct array once."""
        with self._lock:
            seen = {}
            for value in self._data.values():
                for a in _arrays(value):
                    seen[id(a)] = a.nbytes
            return sum(seen.values())

    def make_room(self, nbytes):
        """Evict least recently used entries until *nbytes* more bytes are
        available above the floor of MIN_AVAILABLE_FRACTION of total
        memory, or the cache is empty. When memory can't be read, only the
        FALLBACK_BYTES limit applies."""
        with self._lock:
            self._evict(max(0, int(nbytes)))

    def _over(self, need):
        mem = self._memory()
        if mem is None:
            return self.nbytes() > FALLBACK_BYTES
        total, available = mem
        return (available - need < MIN_AVAILABLE_FRACTION * total
                or self.nbytes() > MAX_FRACTION * total)

    def _evict(self, need, keep=None):
        """Drop the least recently used entry other than *keep* while
        _over(need)."""
        while self._over(need):
            victim = next((k for k in self._data if k != keep), None)
            if victim is None:
                return
            del self._data[victim]
