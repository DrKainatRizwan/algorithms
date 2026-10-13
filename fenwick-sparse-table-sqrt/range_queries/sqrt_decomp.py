"""Square-root decomposition structures and Mo's offline algorithm."""
from __future__ import annotations

from math import isqrt
from typing import Sequence


class SqrtDecomposition:
    """Point update, range sum/min/max with blocks of size ~sqrt(n).

    Update O(sqrt n) (min/max block rebuild), query O(sqrt n), space O(n).
    """

    def __init__(self, values: Sequence[int], block: int | None = None) -> None:
        self.a = list(values)
        self.n = len(self.a)
        self.B = block or max(1, isqrt(self.n))
        nb = (self.n + self.B - 1) // self.B
        self.bsum = [0] * nb
        self.bmin = [0] * nb
        self.bmax = [0] * nb
        for b in range(nb):
            self._rebuild(b)

    def _rebuild(self, b: int) -> None:
        chunk = self.a[b * self.B:(b + 1) * self.B]
        self.bsum[b], self.bmin[b], self.bmax[b] = sum(chunk), min(chunk), max(chunk)

    def set(self, i: int, v: int) -> None:
        if not 0 <= i < self.n:
            raise IndexError(i)
        self.a[i] = v
        self._rebuild(i // self.B)

    def _query(self, lo: int, hi: int, blocks: list[int], fn, whole):
        if not 0 <= lo < hi <= self.n:
            raise IndexError(f"invalid range [{lo}, {hi})")
        bl, br = lo // self.B, (hi - 1) // self.B
        if bl == br:
            return whole(self.a[lo:hi])
        parts = [whole(self.a[lo:(bl + 1) * self.B]), whole(self.a[br * self.B:hi])]
        if br - bl > 1:
            parts.append(whole(blocks[bl + 1:br]))
        return fn(parts)

    def sum(self, lo: int, hi: int) -> int:
        if lo == hi:
            return 0
        return self._query(lo, hi, self.bsum, sum, sum)

    def min(self, lo: int, hi: int) -> int:
        return self._query(lo, hi, self.bmin, min, min)

    def max(self, lo: int, hi: int) -> int:
        return self._query(lo, hi, self.bmax, max, max)


class SqrtRangeAdd:
    """Range add (lazy per block) and range sum/max in O(sqrt n)."""

    def __init__(self, values: Sequence[int], block: int | None = None) -> None:
        self.a = list(values)
        self.n = len(self.a)
        self.B = block or max(1, isqrt(self.n))
        self.nb = (self.n + self.B - 1) // self.B
        self.lazy = [0] * self.nb
        self.bsum = [0] * self.nb
        self.bmax = [0] * self.nb
        for b in range(self.nb):
            self._rebuild(b)

    def _rebuild(self, b: int) -> None:
        chunk = self.a[b * self.B:(b + 1) * self.B]
        self.bsum[b] = sum(chunk)
        self.bmax[b] = max(chunk)

    def _span(self, b: int) -> int:
        return min(self.n, (b + 1) * self.B) - b * self.B

    def range_add(self, lo: int, hi: int, delta: int) -> None:
        if not 0 <= lo <= hi <= self.n:
            raise IndexError(f"invalid range [{lo}, {hi})")
        i = lo
        while i < hi:
            b = i // self.B
            end = min(hi, (b + 1) * self.B)
            if i == b * self.B and end == min(self.n, (b + 1) * self.B):
                self.lazy[b] += delta
            else:
                for j in range(i, end):
                    self.a[j] += delta
                self._rebuild(b)
            i = end

    def sum(self, lo: int, hi: int) -> int:
        if not 0 <= lo <= hi <= self.n:
            raise IndexError(f"invalid range [{lo}, {hi})")
        total, i = 0, lo
        while i < hi:
            b = i // self.B
            end = min(hi, (b + 1) * self.B)
            if i == b * self.B and end == min(self.n, (b + 1) * self.B):
                total += self.bsum[b] + self.lazy[b] * self._span(b)
            else:
                total += sum(self.a[i:end]) + self.lazy[b] * (end - i)
            i = end
        return total

    def max(self, lo: int, hi: int) -> int:
        if not 0 <= lo < hi <= self.n:
            raise IndexError(f"invalid range [{lo}, {hi})")
        best, i = None, lo
        while i < hi:
            b = i // self.B
            end = min(hi, (b + 1) * self.B)
            if i == b * self.B and end == min(self.n, (b + 1) * self.B):
                m = self.bmax[b] + self.lazy[b]
            else:
                m = max(self.a[i:end]) + self.lazy[b]
            best = m if best is None else max(best, m)
            i = end
        return best


def mo_distinct_counts(values: Sequence[int], queries: Sequence[tuple[int, int]]) -> list[int]:
    """Number of distinct values in each half-open range, via Mo's algorithm.

    Queries are sorted by (block of lo, hi with alternating direction), giving
    O((n + q) sqrt n) pointer moves in total.
    """
    n = len(values)
    if not queries:
        return []
    B = max(1, int(n / max(1.0, len(queries)) ** 0.5)) if n else 1
    B = max(B, 1)
    order = sorted(range(len(queries)),
                   key=lambda q: (queries[q][0] // B,
                                  queries[q][1] if (queries[q][0] // B) % 2 == 0
                                  else -queries[q][1]))
    freq: dict[int, int] = {}
    distinct = 0
    cl, cr = 0, 0  # current window is [cl, cr)
    out = [0] * len(queries)
    for q in order:
        lo, hi = queries[q]
        if not 0 <= lo <= hi <= n:
            raise IndexError(f"invalid range [{lo}, {hi})")
        while cl > lo:
            cl -= 1
            v = values[cl]
            freq[v] = freq.get(v, 0) + 1
            distinct += freq[v] == 1
        while cr < hi:
            v = values[cr]
            freq[v] = freq.get(v, 0) + 1
            distinct += freq[v] == 1
            cr += 1
        while cl < lo:
            v = values[cl]
            freq[v] -= 1
            distinct -= freq[v] == 0
            cl += 1
        while cr > hi:
            cr -= 1
            v = values[cr]
            freq[v] -= 1
            distinct -= freq[v] == 0
        out[q] = distinct
    return out
