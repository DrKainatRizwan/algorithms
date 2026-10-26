"""Belady's clairvoyant OPT policy: the offline upper bound on hit rate."""
from __future__ import annotations

import heapq
from typing import Hashable, Sequence

from .base import CacheStats

_INF = float("inf")


def next_use_indices(trace: Sequence[Hashable]) -> list[float]:
    """For each position i, the next index at which trace[i] recurs (inf if never)."""
    nxt: list[float] = [_INF] * len(trace)
    last: dict[Hashable, int] = {}
    for i in range(len(trace) - 1, -1, -1):
        nxt[i] = last.get(trace[i], _INF)
        last[trace[i]] = i
    return nxt


def simulate_opt(trace: Sequence[Hashable], capacity: int) -> CacheStats:
    """Replay ``trace`` evicting the key whose next use is farthest away.

    Runs in O(n log c) using a max-heap with lazy invalidation.
    """
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    nxt = next_use_indices(trace)
    stats = CacheStats()
    resident: dict[Hashable, float] = {}
    heap: list[tuple[float, int, Hashable]] = []  # (-next_use, tiebreak, key)
    for i, key in enumerate(trace):
        if key in resident:
            stats.hits += 1
        else:
            stats.misses += 1
            if len(resident) >= capacity:
                while True:
                    neg, _, victim = heapq.heappop(heap)
                    if resident.get(victim) == -neg:
                        del resident[victim]
                        stats.evictions += 1
                        break
        resident[key] = nxt[i]
        heapq.heappush(heap, (-nxt[i], i, key))
    return stats
