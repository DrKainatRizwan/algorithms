"""Small helpers: a memoisation wrapper that counts subproblem evaluations, and a timer."""

from __future__ import annotations

import time
from typing import Callable, Hashable


class Memo:
    """Top-down memoisation with hit/miss statistics.

    Wraps a function ``f(*args)`` of hashable arguments. ``misses`` equals the number of
    distinct subproblems solved, ``hits`` the number of reused answers.
    """

    def __init__(self, fn: Callable[..., object]) -> None:
        self.fn = fn
        self.cache: dict[Hashable, object] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, *args: Hashable) -> object:
        if args in self.cache:
            self.hits += 1
            return self.cache[args]
        self.misses += 1
        val = self.fn(*args)
        self.cache[args] = val
        return val

    def clear(self) -> None:
        self.cache.clear()
        self.hits = self.misses = 0


def timed(fn: Callable[..., object], *args: object, repeat: int = 1, **kw: object) -> tuple[object, float]:
    """Run ``fn`` ``repeat`` times and return (last result, best wall time in seconds)."""
    best = float("inf")
    out: object = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        best = min(best, time.perf_counter() - t0)
    return out, best
