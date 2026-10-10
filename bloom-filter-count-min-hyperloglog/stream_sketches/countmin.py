"""Count-Min sketch (Cormode & Muthukrishnan) and a heavy-hitter tracker."""

from __future__ import annotations

import heapq
import math
from typing import Hashable

import numpy as np

from .hashing import hash64


class CountMinSketch:
    """Frequency sketch: estimate(x) >= true(x) and <= true(x) + eps*N w.p. 1-delta."""

    def __init__(self, epsilon: float = 0.001, delta: float = 0.01, seed: int = 0,
                 conservative: bool = False) -> None:
        if not (0 < epsilon < 1 and 0 < delta < 1):
            raise ValueError("epsilon and delta must be in (0, 1)")
        self.epsilon, self.delta = epsilon, delta
        self.width = math.ceil(math.e / epsilon)
        self.depth = math.ceil(math.log(1.0 / delta))
        self.seed = seed
        self.conservative = conservative
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        self.total = 0

    def _cols(self, item: Hashable) -> list[int]:
        return [hash64(item, self.seed * 1000003 + r) % self.width for r in range(self.depth)]

    def add(self, item: Hashable, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        cols = self._cols(item)
        rows = range(self.depth)
        if self.conservative:
            target = min(self.table[r, c] for r, c in zip(rows, cols)) + count
            for r, c in zip(rows, cols):
                if self.table[r, c] < target:
                    self.table[r, c] = target
        else:
            for r, c in zip(rows, cols):
                self.table[r, c] += count
        self.total += count

    def estimate(self, item: Hashable) -> int:
        return int(min(self.table[r, c] for r, c in zip(range(self.depth), self._cols(item))))

    def error_bound(self) -> float:
        """Additive error bound epsilon * N."""
        return self.epsilon * self.total

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        if (self.width, self.depth, self.seed) != (other.width, other.depth, other.seed):
            raise ValueError("incompatible sketches")
        out = CountMinSketch(self.epsilon, self.delta, self.seed, self.conservative)
        out.table = self.table + other.table
        out.total = self.total + other.total
        return out

    def inner_product(self, other: "CountMinSketch") -> int:
        """Upper-bound estimate of sum_x f(x) g(x) (join size)."""
        if self.table.shape != other.table.shape or self.seed != other.seed:
            raise ValueError("incompatible sketches")
        return int((self.table * other.table).sum(axis=1).min())

    @property
    def size_bytes(self) -> int:
        return int(self.table.nbytes)


class HeavyHitters:
    """Track the top-k frequent items using a Count-Min sketch plus a bounded heap."""

    def __init__(self, k: int, epsilon: float = 0.001, delta: float = 0.01,
                 seed: int = 0) -> None:
        self.k = k
        self.sketch = CountMinSketch(epsilon, delta, seed, conservative=True)
        self._est: dict[Hashable, int] = {}

    def add(self, item: Hashable, count: int = 1) -> None:
        self.sketch.add(item, count)
        est = self.sketch.estimate(item)
        if item in self._est or len(self._est) < self.k:
            self._est[item] = est
            return
        worst = min(self._est, key=lambda x: (self._est[x], repr(x)))
        if est > self._est[worst]:
            del self._est[worst]
            self._est[item] = est

    def top(self, n: int | None = None) -> list[tuple[Hashable, int]]:
        items = heapq.nlargest(n or self.k, self._est.items(), key=lambda kv: (kv[1], repr(kv[0])))
        return [(x, self.sketch.estimate(x)) for x, _ in items]

    def above_threshold(self, phi: float) -> list[Hashable]:
        """Tracked items whose estimate exceeds phi * N."""
        cut = phi * self.sketch.total
        return [x for x, c in self._est.items() if self.sketch.estimate(x) > cut]
