"""Segment trees for geometric sweeps."""
from __future__ import annotations

from bisect import bisect_left
from typing import Sequence


class CoverageTree:
    """Segment tree over elementary slabs of a sorted coordinate list.

    Supports ``add(l, r, +1/-1)`` of a closed-open coordinate range and
    reports the total covered length (union) in O(1).  Each node keeps a
    cover count and the covered length of its subtree; counts are never
    pushed down, which is what makes unmatched add/remove pairs safe
    (the classic area-of-union trick).
    """

    def __init__(self, coords: Sequence[float]):
        self.xs = sorted(set(coords))
        self.m = max(len(self.xs) - 1, 0)  # elementary slabs
        size = 4 * max(self.m, 1)
        self.cnt = [0] * size
        self.len = [0.0] * size

    def _pull(self, node: int, lo: int, hi: int) -> None:
        if self.cnt[node] > 0:
            self.len[node] = self.xs[hi] - self.xs[lo]
        elif hi - lo == 1:
            self.len[node] = 0.0
        else:
            self.len[node] = self.len[2 * node] + self.len[2 * node + 1]

    def add(self, l: float, r: float, delta: int) -> None:
        """Add ``delta`` to the cover count of ``[l, r)``."""
        if l >= r or self.m == 0:
            return
        a, b = self._index(l), self._index(r)
        self._update(1, 0, self.m, a, b, delta)

    def _index(self, x: float) -> int:
        i = bisect_left(self.xs, x)
        if i == len(self.xs) or self.xs[i] != x:
            raise KeyError(f"{x} is not a registered coordinate")
        return i

    def _update(self, node: int, lo: int, hi: int, a: int, b: int, d: int) -> None:
        if b <= lo or hi <= a:
            return
        if a <= lo and hi <= b:
            self.cnt[node] += d
        else:
            mid = (lo + hi) // 2
            self._update(2 * node, lo, mid, a, b, d)
            self._update(2 * node + 1, mid, hi, a, b, d)
        self._pull(node, lo, hi)

    @property
    def covered(self) -> float:
        return self.len[1] if self.m else 0.0


class MaxAddTree:
    """Lazy range-add / range-max segment tree over positions 0..n-1.

    Also tracks the argmax, which lets the sweep report *where* the maximum
    depth of overlap occurs.  All operations are O(log n).
    """

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.mx = [0] * (4 * n)
        self.pos = [0] * (4 * n)
        self.lz = [0] * (4 * n)
        self._build(1, 0, n - 1)

    def _build(self, node: int, lo: int, hi: int) -> None:
        self.pos[node] = lo
        if lo == hi:
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid)
        self._build(2 * node + 1, mid + 1, hi)

    def _pull(self, node: int) -> None:
        l, r = 2 * node, 2 * node + 1
        if self.mx[l] >= self.mx[r]:
            self.mx[node], self.pos[node] = self.mx[l] + self.lz[node], self.pos[l]
        else:
            self.mx[node], self.pos[node] = self.mx[r] + self.lz[node], self.pos[r]

    def add(self, l: int, r: int, delta: int) -> None:
        """Add ``delta`` to positions ``l..r`` inclusive."""
        if not (0 <= l <= r < self.n):
            raise IndexError("range out of bounds")
        self._add(1, 0, self.n - 1, l, r, delta)

    def _add(self, node: int, lo: int, hi: int, l: int, r: int, d: int) -> None:
        if r < lo or hi < l:
            return
        if l <= lo and hi <= r:
            self.mx[node] += d
            self.lz[node] += d
            return
        mid = (lo + hi) // 2
        self._add(2 * node, lo, mid, l, r, d)
        self._add(2 * node + 1, mid + 1, hi, l, r, d)
        self._pull(node)

    def query(self, l: int, r: int) -> tuple[int, int]:
        """(max value, argmax position) over ``l..r`` inclusive."""
        if not (0 <= l <= r < self.n):
            raise IndexError("range out of bounds")
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, node: int, lo: int, hi: int, l: int, r: int) -> tuple[int, int]:
        if l <= lo and hi <= r:
            return self.mx[node], self.pos[node]
        mid = (lo + hi) // 2
        best: tuple[int, int] | None = None
        if l <= mid:
            best = self._query(2 * node, lo, mid, l, r)
        if r > mid:
            c = self._query(2 * node + 1, mid + 1, hi, l, r)
            if best is None or c[0] > best[0]:
                best = c
        assert best is not None
        return best[0] + self.lz[node], best[1]

    @property
    def global_max(self) -> tuple[int, int]:
        return self.mx[1], self.pos[1]
