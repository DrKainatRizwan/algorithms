"""Micro-benchmarks: skip list vs. sorted array, and threaded stress runs."""

from __future__ import annotations

import bisect
import random
import threading
import time
from typing import Callable

from .concurrent_list import LockFreeSkipList
from .levels import GeometricLevels
from .skiplist import SkipList


def timed(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def compare_with_sorted_list(n: int, seed: int = 0) -> dict[str, float]:
    """Random inserts and lookups: skip list vs. bisect.insort on a list."""
    rng = random.Random(seed)
    keys = rng.sample(range(10 * n), n)
    sl = SkipList(GeometricLevels(seed=seed))
    arr: list[int] = []
    t_sl = timed(lambda: [sl.insert(k) for k in keys])
    t_arr = timed(lambda: [bisect.insort(arr, k) for k in keys])
    probes = rng.sample(keys, min(n, 2000))
    l_sl = timed(lambda: [sl.get(k) for k in probes])
    l_arr = timed(lambda: [bisect.bisect_left(arr, k) for k in probes])
    return {"n": n, "skip_insert_s": t_sl, "array_insert_s": t_arr,
            "skip_lookup_s": l_sl, "array_lookup_s": l_arr}


def threaded_stress(threads: int = 4, ops: int = 1500, key_space: int = 200, seed: int = 0) -> dict:
    """Mixed add/remove/contains from several threads on overlapping keys.

    Returns net successful adds-minus-removes per key plus the list itself so
    callers can verify that the final contents agree with the operation log.
    """
    lst = LockFreeSkipList(GeometricLevels(seed=seed))
    net = [[0] * key_space for _ in range(threads)]

    def worker(tid: int) -> None:
        rng = random.Random(seed * 1000 + tid)
        for _ in range(ops):
            k = rng.randrange(key_space)
            r = rng.random()
            if r < 0.4:
                if lst.add(k, tid):
                    net[tid][k] += 1
            elif r < 0.8:
                if lst.remove(k):
                    net[tid][k] -= 1
            else:
                lst.contains(k)

    pool = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    t0 = time.perf_counter()
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    elapsed = time.perf_counter() - t0
    total = [sum(net[t][k] for t in range(threads)) for k in range(key_space)]
    return {"list": lst, "net": total, "elapsed": elapsed,
            "ops_per_s": threads * ops / elapsed, "cas_failures": lst.cas_failures}
