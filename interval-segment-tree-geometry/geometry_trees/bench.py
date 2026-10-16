"""Random data generators and timing helpers."""
from __future__ import annotations

import random
import time
from typing import Callable

Interval = tuple[float, float]
Rect = tuple[float, float, float, float]


def random_intervals(n: int, rng: random.Random, span: float = 1000.0,
                     max_len: float = 20.0) -> list[Interval]:
    out = []
    for _ in range(n):
        lo = rng.uniform(0, span)
        out.append((lo, lo + rng.uniform(0, max_len)))
    return out


def random_rects(n: int, rng: random.Random, span: float = 1000.0,
                 max_side: float = 30.0) -> list[Rect]:
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, span), rng.uniform(0, span)
        out.append((x, y, x + rng.uniform(0, max_side), y + rng.uniform(0, max_side)))
    return out


def timeit(fn: Callable[[], object], repeat: int = 1) -> float:
    """Best wall-clock time in seconds over ``repeat`` runs."""
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best
