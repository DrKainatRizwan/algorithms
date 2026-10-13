"""Static range-query structures: sparse table and disjoint sparse table."""
from __future__ import annotations

from math import gcd
from typing import Callable, Sequence


class SparseTable:
    """Sparse table for idempotent operations (min, max, gcd).

    Build O(n log n) time and space; query O(1) by overlapping two
    power-of-two blocks, which is valid only because op(x, x) == x.
    """

    def __init__(self, values: Sequence[int], op: Callable[[int, int], int] = min) -> None:
        self.n = len(values)
        self.op = op
        self.table = [list(values)]
        k = 1
        while (1 << k) <= self.n:
            prev = self.table[-1]
            half = 1 << (k - 1)
            self.table.append([op(prev[i], prev[i + half])
                               for i in range(self.n - (1 << k) + 1)])
            k += 1

    def query(self, lo: int, hi: int) -> int:
        """Combine a[lo:hi] (non-empty)."""
        if not 0 <= lo < hi <= self.n:
            raise IndexError(f"invalid range [{lo}, {hi})")
        k = (hi - lo).bit_length() - 1
        row = self.table[k]
        return self.op(row[lo], row[hi - (1 << k)])

    @classmethod
    def gcd_table(cls, values: Sequence[int]) -> "SparseTable":
        return cls(values, gcd)


class ArgMinSparseTable:
    """Sparse table returning the index of the leftmost minimum."""

    def __init__(self, values: Sequence[int]) -> None:
        self.a = list(values)
        n = len(self.a)
        self.n = n
        self.table = [list(range(n))]
        k = 1
        while (1 << k) <= n:
            prev, half = self.table[-1], 1 << (k - 1)
            self.table.append([self._better(prev[i], prev[i + half])
                               for i in range(n - (1 << k) + 1)])
            k += 1

    def _better(self, i: int, j: int) -> int:
        return i if self.a[i] <= self.a[j] else j

    def query(self, lo: int, hi: int) -> int:
        if not 0 <= lo < hi <= self.n:
            raise IndexError(f"invalid range [{lo}, {hi})")
        k = (hi - lo).bit_length() - 1
        return self._better(self.table[k][lo], self.table[k][hi - (1 << k)])


class DisjointSparseTable:
    """Disjoint sparse table: O(1) queries for any associative operation.

    Each level stores running aggregates radiating outwards from block
    midpoints; a query [l, r] is answered by one table lookup on each side of
    the level at which l and r first differ. Build O(n log n).
    """

    def __init__(self, values: Sequence[int],
                 op: Callable[[int, int], int] = lambda x, y: x + y) -> None:
        self.op = op
        self.n = len(values)
        size = 1
        while size < max(self.n, 2):
            size <<= 1
        self.size = size
        self.levels = size.bit_length() - 1
        a = list(values) + [values[-1]] * (size - self.n) if self.n else []
        self.table: list[list[int]] = []
        for lvl in range(self.levels):
            half = 1 << lvl
            row = [0] * size
            for start in range(0, size, half * 2):
                mid = start + half
                row[mid - 1] = a[mid - 1]
                for i in range(mid - 2, start - 1, -1):
                    row[i] = op(a[i], row[i + 1])
                row[mid] = a[mid]
                for i in range(mid + 1, min(start + 2 * half, size)):
                    row[i] = op(row[i - 1], a[i])
            self.table.append(row)
        self.a = a

    def query(self, lo: int, hi: int) -> int:
        """Combine a[lo:hi] (non-empty) in order."""
        if not 0 <= lo < hi <= self.n:
            raise IndexError(f"invalid range [{lo}, {hi})")
        r = hi - 1
        if lo == r:
            return self.a[lo]
        lvl = (lo ^ r).bit_length() - 1
        return self.op(self.table[lvl][lo], self.table[lvl][r])
