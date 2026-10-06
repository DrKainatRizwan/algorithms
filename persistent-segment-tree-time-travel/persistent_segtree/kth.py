"""Range k-th smallest / rank queries via a persistent count tree."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Optional, Sequence


class _CNode:
    __slots__ = ("cnt", "left", "right")

    def __init__(self, cnt: int, left: Optional["_CNode"], right: Optional["_CNode"]) -> None:
        self.cnt = cnt
        self.left = left
        self.right = right


class RangeKth:
    """Static array supporting k-th smallest and rank in any ``a[l:r]``.

    Version ``i`` is a count tree over compressed values of ``a[:i]``;
    a range is answered by subtracting two versions. O(n log n) build,
    O(log n) per query.
    """

    def __init__(self, values: Sequence[int]) -> None:
        if len(values) == 0:
            raise ValueError("values must be non-empty")
        self.n = len(values)
        self.sorted_vals = sorted(set(values))
        self.m = len(self.sorted_vals)
        empty = self._empty(0, self.m)
        self.roots = [empty]
        for v in values:
            pos = bisect_left(self.sorted_vals, v)
            self.roots.append(self._insert(self.roots[-1], 0, self.m, pos))

    def _empty(self, lo: int, hi: int) -> _CNode:
        if hi - lo == 1:
            return _CNode(0, None, None)
        mid = (lo + hi) // 2
        # share a single empty child pair per level
        child_l, child_r = self._empty(lo, mid), self._empty(mid, hi)
        return _CNode(0, child_l, child_r)

    def _insert(self, node: _CNode, lo: int, hi: int, pos: int) -> _CNode:
        if hi - lo == 1:
            return _CNode(node.cnt + 1, None, None)
        mid = (lo + hi) // 2
        if pos < mid:
            return _CNode(node.cnt + 1, self._insert(node.left, lo, mid, pos), node.right)
        return _CNode(node.cnt + 1, node.left, self._insert(node.right, mid, hi, pos))

    def kth(self, left: int, right: int, k: int) -> int:
        """The k-th smallest (1-based) element of ``a[left:right]``."""
        if not 0 <= left <= right <= self.n:
            raise IndexError((left, right))
        if not 1 <= k <= right - left:
            raise ValueError("k out of range")
        a, b = self.roots[right], self.roots[left]
        lo, hi = 0, self.m
        while hi - lo > 1:
            mid = (lo + hi) // 2
            c = a.left.cnt - b.left.cnt
            if k <= c:
                a, b, hi = a.left, b.left, mid
            else:
                k -= c
                a, b, lo = a.right, b.right, mid
        return self.sorted_vals[lo]

    def count_less(self, left: int, right: int, x: int) -> int:
        """Number of elements in ``a[left:right]`` strictly less than ``x``."""
        return self._count_prefix(left, right, bisect_left(self.sorted_vals, x))

    def count_leq(self, left: int, right: int, x: int) -> int:
        """Number of elements in ``a[left:right]`` less than or equal to ``x``."""
        return self._count_prefix(left, right, bisect_right(self.sorted_vals, x))

    def _count_prefix(self, left: int, right: int, upto: int) -> int:
        """Count of compressed ranks ``< upto`` in the range."""
        if not 0 <= left <= right <= self.n:
            raise IndexError((left, right))
        a, b = self.roots[right], self.roots[left]
        lo, hi, total = 0, self.m, 0
        if upto <= 0:
            return 0
        if upto >= self.m:
            return a.cnt - b.cnt
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if upto <= mid:
                a, b, hi = a.left, b.left, mid
            else:
                total += a.left.cnt - b.left.cnt
                a, b, lo = a.right, b.right, mid
        if lo < upto:  # invariant lo < upto <= hi: the final leaf is included
            total += a.cnt - b.cnt
        return total
