"""Timing helpers used by the demo and README benchmarks."""

from __future__ import annotations

import time
from typing import Callable


def timed(fn: Callable[[], object], repeat: int = 3) -> tuple[object, float]:
    """Run ``fn`` ``repeat`` times; return last result and best wall time (s)."""
    best = float("inf")
    result: object = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def matching_size(match: list[int]) -> int:
    """Edge count of a bipartite ``match_left`` array (non -1 entries)."""
    return sum(1 for m in match if m != -1)


def general_size(mate: list[int]) -> int:
    """Edge count of a general-graph ``mate`` array."""
    return sum(1 for m in mate if m != -1) // 2
