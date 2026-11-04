"""Plain assert-based tests for erosion_cache.py: least-recently-used
eviction bounded by (fake) system memory, and the algorithms' use of it.

Run with: python tests/test_erosion_cache.py
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import erosion_cache as ec
from erosion_cache import ErosionCache, system_memory

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


class FakeMemory:
    """total bytes of RAM, of which *other* are used outside the cache; the
    cache's arrays and the *held* ones (referenced elsewhere, like a step's
    toolpath) are used too, each distinct array once."""

    def __init__(self, total, other=0):
        self.total, self.other, self.held, self.cache = total, other, [], None

    def __call__(self):
        live = {id(a): a.nbytes for a in self.held}
        for value in self.cache._data.values():
            for a in ec._arrays(value):
                live[id(a)] = a.nbytes
        return self.total, self.total - self.other - sum(live.values())


def fake_cache(total, other=0):
    mem = FakeMemory(total, other)
    cache = ErosionCache(memory=mem)
    mem.cache = cache
    return cache, mem


def entry(nbytes):
    return (np.zeros(nbytes // 2, np.uint8), np.zeros(nbytes - nbytes // 2, np.uint8))


print("least-recently-used eviction")
# 1000 B of RAM: the cache holds at most 500 B, and 250 B stay available.
cache, mem = fake_cache(1000)
for k in "ABC":
    cache[k] = entry(150)
check("entries within the budget are all kept", cache.keys() == list("ABC")
      and cache.nbytes() == 450)
cache["D"] = entry(150)
check("over half of RAM: the least recently used entry goes", cache.keys() == list("BCD"))
cache.get("B")
cache["E"] = entry(150)
check("get makes an entry the most recent", cache.keys() == list("DBE"))
_ = cache["D"]
cache["F"] = entry(150)
check("[] makes an entry the most recent", cache.keys() == list("EDF"))
list(cache.items())
list(cache)
"D" in cache
cache["G"] = entry(150)
check("items, iteration and `in` don't count as use", cache.keys() == list("DFG"))
cache["F"] = entry(150)
check("storing an existing key replaces it as the most recent",
      cache.keys() == list("DGF") and len(cache) == 3)

print("the newest entry is never evicted by its own store")
cache["big"] = entry(700)
check("an entry over the budget by itself evicts all others and stays",
      cache.keys() == ["big"])

print("arrays shared by entries count once")
cache.clear()
a = np.zeros(200, np.uint8)
cache["vcarve synth"] = (a, a)
cache["other"] = (a, np.zeros(100, np.uint8))
check("one array stored twice counts once", cache.nbytes() == 300)

print("available memory")
cache, mem = fake_cache(1000, other=500)
cache["A"] = entry(100)
cache["B"] = entry(100)
check("stored while 250 B or more stay available", cache.keys() == ["A", "B"])
cache["C"] = entry(100)
check("a store that leaves less than a quarter available evicts the oldest",
      cache.keys() == ["B", "C"])
mem.other = 600
cache["D"] = entry(50)
check("other memory use counts too", cache.keys() == ["C", "D"])

print("make_room")
cache, mem = fake_cache(1000)
for k in "ABCD":
    cache[k] = entry(100)
# 600 B available: 300 more fit above the 250 B floor already.
cache.make_room(300)
check("no eviction while the requested bytes fit", cache.keys() == list("ABCD"))
cache.make_room(450)
check("evicts until the requested bytes are available above the floor",
      cache.keys() == list("BCD"))
mem.held = list(cache["C"])
cache.make_room(700)
check("an entry held elsewhere frees nothing, so eviction goes on to the next",
      cache.keys() == [])
cache["E"] = entry(100)
cache.make_room(5000)
check("make_room may empty the cache", len(cache) == 0)

print("memory that can't be read")
saved = ec.FALLBACK_BYTES
ec.FALLBACK_BYTES = 300
try:
    cache = ErosionCache(memory=lambda: None)
    for k in "ABC":
        cache[k] = entry(150)
    check("at most FALLBACK_BYTES are held", cache.keys() == ["B", "C"])
    cache.make_room(10 ** 9)
    check("make_room evicts nothing more", cache.keys() == ["B", "C"])
finally:
    ec.FALLBACK_BYTES = saved

print("clear and delete")
cache, _ = fake_cache(1000)
cache["A"] = entry(10)
cache["B"] = entry(10)
del cache["A"]
check("del removes an entry", cache.keys() == ["B"] and "A" not in cache)
cache.clear()
check("clear empties the cache", len(cache) == 0 and cache.get("B") is None)

print("threads")
cache = ErosionCache(memory=lambda: (10 ** 9, 10 ** 9))
errors = []


def hammer(tid):
    try:
        for i in range(2000):
            cache[(tid, i % 7)] = entry(16)
            cache.get((1 - tid, i % 5))
            cache.items()
    except Exception as e:      # noqa: BLE001 - any error fails the check
        errors.append(e)


threads = [threading.Thread(target=hammer, args=(t,)) for t in (0, 1)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("concurrent stores, reads and scans raise nothing", not errors and len(cache) == 14)

print("system memory")
mem_now = system_memory()
check("total and available memory are read", mem_now is not None
      and 0 < mem_now[1] <= mem_now[0])
if sys.platform.startswith("linux"):
    saved_psutil = sys.modules.get("psutil")
    sys.modules["psutil"] = None          # import psutil raises ImportError
    try:
        proc = system_memory()
    finally:
        if saved_psutil is None:
            del sys.modules["psutil"]
        else:
            sys.modules["psutil"] = saved_psutil
    check("without psutil, /proc/meminfo gives the same total within 5 %",
          proc is not None and abs(proc[0] - mem_now[0]) <= 0.05 * mem_now[0])

print("the algorithms use it like a dict")
from algorithms import compute_raster_toolpath, _find_compatible_vbit_erosion
from tool import Tool

depth = np.clip(np.add.outer(np.linspace(0, 1, 60), np.linspace(0, 0.5, 80)), 0, 1).astype(np.float32)
cache = ErosionCache()
messages = []
kw = dict(step_mm=1.0, raster_dir="Horizontal", stepdown_mm=0.0, max_depth=5.0, width_mm=40.0, height_mm=30.0,
          erosion_cache=cache, erosion_cache_key=("Ball", 3.0, None, 5.0, 40.0, 30.0, 0.0, ()))
compute_raster_toolpath(depth, 256, Tool(profile="Ball", diameter_mm=3.0), **kw,
                        progress=lambda p, m: messages.append(m))
compute_raster_toolpath(depth, 256, Tool(profile="Ball", diameter_mm=3.0), **kw,
                        progress=lambda p, m: messages.append(m))
check("a second Raster computation reuses the cached surface",
      len(cache) == 1 and any("Reusing achievable depth surface" in m for m in messages))
surf = np.zeros((4, 4), np.float32)
cache.clear()
cache[("V-Bit", 10.0, 60.0, 5.0, 40.0, 30.0, 0.0, ())] = (surf, surf)
check("_find_compatible_vbit_erosion scans it",
      _find_compatible_vbit_erosion(cache, ("V-Bit", 6.0, 60.0, 5.0, 40.0, 30.0, 0.0, ()), 6.0)
      is surf)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
