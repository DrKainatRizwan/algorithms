"""Fenwick (binary indexed) trees: point, range-update and 2-D variants.

All range arguments are half-open ``[lo, hi)`` and indices are 0-based.
"""
from __future__ import annotations

from typing import Sequence


def _check_range(lo: int, hi: int, n: int) -> None:
    if not 0 <= lo <= hi <= n:
        raise IndexError(f"invalid range [{lo}, {hi}) for size {n}")


class Fenwick:
    """Point update / prefix-sum tree.

    Build is O(n) using the parent-propagation trick; update and query are
    O(log n); space is O(n).
    """

    def __init__(self, values: Sequence[int] | int) -> None:
        if isinstance(values, int):
            self.n = values
            self.t = [0] * (self.n + 1)
            return
        self.n = len(values)
        self.t = [0] + list(values)
        for i in range(1, self.n + 1):
            j = i + (i & -i)
            if j <= self.n:
                self.t[j] += self.t[i]

    def add(self, i: int, delta: int) -> None:
        """a[i] += delta."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        i += 1
        while i <= self.n:
            self.t[i] += delta
            i += i & -i

    def prefix(self, hi: int) -> int:
        """Sum of a[0:hi]."""
        if not 0 <= hi <= self.n:
            raise IndexError(hi)
        s = 0
        while hi > 0:
            s += self.t[hi]
            hi -= hi & -hi
        return s

    def sum(self, lo: int, hi: int) -> int:
        _check_range(lo, hi, self.n)
        return self.prefix(hi) - self.prefix(lo)

    def get(self, i: int) -> int:
        return self.sum(i, i + 1)

    def set(self, i: int, v: int) -> None:
        self.add(i, v - self.get(i))

    def lower_bound(self, target: int) -> int:
        """Smallest index i with prefix(i+1) >= target (non-negative data).

        Returns ``n`` if the total is below ``target``. O(log n) via binary
        lifting over the implicit tree.
        """
        pos = 0
        step = 1 << self.n.bit_length()
        rem = target
        while step:
            nxt = pos + step
            if nxt <= self.n and self.t[nxt] < rem:
                pos = nxt
                rem -= self.t[nxt]
            step >>= 1
        return pos


class RangeFenwick:
    """Range add / range sum using two Fenwick trees.

    With difference array d, prefix(x) = sum(d[i]*(x-i)) over i<x, which is
    x*sum(d) - sum(d[i]*i); each term is held in its own tree.
    """

    def __init__(self, values: Sequence[int] | int) -> None:
        base = [0] * values if isinstance(values, int) else list(values)
        self.n = len(base)
        self._pre = [0]
        for v in base:
            self._pre.append(self._pre[-1] + v)
        self.b1 = Fenwick(self.n + 1)
        self.b2 = Fenwick(self.n + 1)

    def range_add(self, lo: int, hi: int, delta: int) -> None:
        _check_range(lo, hi, self.n)
        if lo == hi:
            return
        self.b1.add(lo, delta)
        self.b2.add(lo, delta * lo)
        self.b1.add(hi, -delta)
        self.b2.add(hi, -delta * hi)

    def _prefix(self, x: int) -> int:
        return self._pre[x] + x * self.b1.prefix(x) - self.b2.prefix(x)

    def sum(self, lo: int, hi: int) -> int:
        _check_range(lo, hi, self.n)
        return self._prefix(hi) - self._prefix(lo)

    def get(self, i: int) -> int:
        return self.sum(i, i + 1)


class Fenwick2D:
    """2-D point update / rectangle sum. O(log^2 n) per operation."""

    def __init__(self, rows: int, cols: int) -> None:
        self.r, self.c = rows, cols
        self.t = [[0] * (cols + 1) for _ in range(rows + 1)]

    def add(self, r: int, c: int, delta: int) -> None:
        if not (0 <= r < self.r and 0 <= c < self.c):
            raise IndexError((r, c))
        i = r + 1
        while i <= self.r:
            j = c + 1
            row = self.t[i]
            while j <= self.c:
                row[j] += delta
                j += j & -j
            i += i & -i

    def prefix(self, r: int, c: int) -> int:
        """Sum over the rectangle [0, r) x [0, c)."""
        s = 0
        i = r
        while i > 0:
            j = c
            row = self.t[i]
            while j > 0:
                s += row[j]
                j -= j & -j
            i -= i & -i
        return s

    def sum(self, r0: int, c0: int, r1: int, c1: int) -> int:
        """Sum over [r0, r1) x [c0, c1)."""
        _check_range(r0, r1, self.r)
        _check_range(c0, c1, self.c)
        return (self.prefix(r1, c1) - self.prefix(r0, c1)
                - self.prefix(r1, c0) + self.prefix(r0, c0))
