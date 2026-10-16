"""Interval trees: a static centred tree and a dynamic augmented treap."""
from __future__ import annotations

import random
from bisect import bisect_left, bisect_right
from typing import Iterator, Sequence

Interval = tuple[float, float]


class CenteredIntervalTree:
    """Static centred interval tree (Edelsbrunner).

    Every node stores a centre point plus the intervals crossing it, sorted
    by left endpoint and by right endpoint.  Intervals entirely to one side
    recurse into children.  Build is O(n log n), stabbing queries are
    O(log n + k) and overlap queries O(log n + k).  Intervals are closed.
    """

    __slots__ = ("center", "by_lo", "by_hi", "left", "right", "size")

    def __init__(self, intervals: Sequence[Interval], _ids: list[int] | None = None):
        if _ids is None:
            _ids = list(range(len(intervals)))
            for lo, hi in intervals:
                if lo > hi:
                    raise ValueError(f"invalid interval ({lo}, {hi})")
        self.size = len(_ids)
        self.left: CenteredIntervalTree | None = None
        self.right: CenteredIntervalTree | None = None
        if not _ids:
            self.center = 0.0
            self.by_lo: list[tuple[float, int]] = []
            self.by_hi: list[tuple[float, int]] = []
            return
        endpoints = sorted([intervals[i][0] for i in _ids] + [intervals[i][1] for i in _ids])
        self.center = endpoints[len(endpoints) // 2]
        lefts, rights, mid = [], [], []
        for i in _ids:
            lo, hi = intervals[i]
            if hi < self.center:
                lefts.append(i)
            elif lo > self.center:
                rights.append(i)
            else:
                mid.append(i)
        self.by_lo = sorted((intervals[i][0], i) for i in mid)
        self.by_hi = sorted((intervals[i][1], i) for i in mid)
        if lefts:
            self.left = CenteredIntervalTree(intervals, lefts)
        if rights:
            self.right = CenteredIntervalTree(intervals, rights)

    def __len__(self) -> int:
        return self.size

    def stab(self, x: float) -> list[int]:
        """Indices of intervals containing point ``x`` (sorted)."""
        out: list[int] = []
        node: CenteredIntervalTree | None = self
        while node is not None and node.size:
            if x < node.center:
                for lo, i in node.by_lo:
                    if lo > x:
                        break
                    out.append(i)
                node = node.left
            elif x > node.center:
                for hi, i in reversed(node.by_hi):
                    if hi < x:
                        break
                    out.append(i)
                node = node.right
            else:
                out.extend(i for _, i in node.by_lo)
                break
        return sorted(out)

    def overlap(self, lo: float, hi: float) -> list[int]:
        """Indices of intervals intersecting ``[lo, hi]`` (sorted)."""
        if lo > hi:
            raise ValueError("empty query range")
        out: list[int] = []
        self._overlap(lo, hi, out)
        return sorted(out)

    def _overlap(self, lo: float, hi: float, out: list[int]) -> None:
        if not self.size:
            return
        if hi < self.center:
            for a, i in self.by_lo:
                if a > hi:
                    break
                out.append(i)
            if self.left:
                self.left._overlap(lo, hi, out)
        elif lo > self.center:
            for b, i in reversed(self.by_hi):
                if b < lo:
                    break
                out.append(i)
            if self.right:
                self.right._overlap(lo, hi, out)
        else:  # the query covers the centre: every crossing interval matches
            out.extend(i for _, i in self.by_lo)
            if self.left:
                self.left._overlap(lo, hi, out)
            if self.right:
                self.right._overlap(lo, hi, out)

    def height(self) -> int:
        h = 0
        for c in (self.left, self.right):
            if c is not None:
                h = max(h, c.height())
        return h + 1 if self.size else 0


class _Node:
    __slots__ = ("lo", "hi", "key", "prio", "max_hi", "left", "right")

    def __init__(self, lo: float, hi: float, key: int, prio: float):
        self.lo, self.hi, self.key, self.prio = lo, hi, key, prio
        self.max_hi = hi
        self.left: _Node | None = None
        self.right: _Node | None = None

    def pull(self) -> None:
        m = self.hi
        if self.left and self.left.max_hi > m:
            m = self.left.max_hi
        if self.right and self.right.max_hi > m:
            m = self.right.max_hi
        self.max_hi = m


def _split(t: _Node | None, k: tuple[float, int]) -> tuple[_Node | None, _Node | None]:
    """Split into nodes with (lo, key) < k and >= k."""
    if t is None:
        return None, None
    if (t.lo, t.key) < k:
        a, b = _split(t.right, k)
        t.right = a
        t.pull()
        return t, b
    a, b = _split(t.left, k)
    t.left = b
    t.pull()
    return a, t


def _merge(a: _Node | None, b: _Node | None) -> _Node | None:
    if a is None or b is None:
        return a or b
    if a.prio > b.prio:
        a.right = _merge(a.right, b)
        a.pull()
        return a
    b.left = _merge(a, b.left)
    b.pull()
    return b


class DynamicIntervalTree:
    """Augmented treap keyed by left endpoint, storing the subtree max of right
    endpoints (CLRS ch. 14).  Supports insert/remove in expected O(log n) and
    overlap queries in O(min(n, k log n)).  Each interval carries an integer id."""

    def __init__(self, seed: int = 0):
        self._root: _Node | None = None
        self._rng = random.Random(seed)
        self._items: dict[int, Interval] = {}
        self._next = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: int) -> bool:
        return key in self._items

    def insert(self, lo: float, hi: float, key: int | None = None) -> int:
        """Insert ``[lo, hi]`` and return its id."""
        if lo > hi:
            raise ValueError(f"invalid interval ({lo}, {hi})")
        if key is None:
            key = self._next
        if key in self._items:
            raise KeyError(f"duplicate id {key}")
        self._next = max(self._next, key + 1)
        node = _Node(lo, hi, key, self._rng.random())
        a, b = _split(self._root, (lo, key))
        self._root = _merge(_merge(a, node), b)
        self._items[key] = (lo, hi)
        return key

    def remove(self, key: int) -> None:
        """Delete the interval with this id."""
        lo, _ = self._items.pop(key)
        a, b = _split(self._root, (lo, key))
        _, c = _split(b, (lo, key + 1))
        self._root = _merge(a, c)

    def get(self, key: int) -> Interval:
        return self._items[key]

    def any_overlap(self, lo: float, hi: float) -> int | None:
        """Id of some interval intersecting ``[lo, hi]`` in O(log n), else None."""
        t = self._root
        while t is not None:
            if t.lo <= hi and lo <= t.hi:
                return t.key
            if t.left is not None and t.left.max_hi >= lo:
                t = t.left
            else:
                t = t.right
        return None

    def overlap(self, lo: float, hi: float) -> list[int]:
        """Sorted ids of all intervals intersecting ``[lo, hi]``."""
        out: list[int] = []
        stack = [self._root]
        while stack:
            t = stack.pop()
            if t is None or t.max_hi < lo:
                continue
            stack.append(t.left)
            if t.lo <= hi:
                if lo <= t.hi:
                    out.append(t.key)
                stack.append(t.right)
        return sorted(out)

    def stab(self, x: float) -> list[int]:
        return self.overlap(x, x)

    def __iter__(self) -> Iterator[tuple[int, Interval]]:
        for k in sorted(self._items):
            yield k, self._items[k]

    def depth(self) -> int:
        def d(t: _Node | None) -> int:
            return 0 if t is None else 1 + max(d(t.left), d(t.right))
        return d(self._root)

    def check_invariants(self) -> bool:
        """Verify BST order, heap order and max-augmentation."""
        def go(t: _Node | None) -> float:
            if t is None:
                return float("-inf")
            ml, mr = go(t.left), go(t.right)
            if t.left and ((t.left.lo, t.left.key) > (t.lo, t.key) or t.left.prio > t.prio):
                raise AssertionError("order violated")
            if t.right and ((t.right.lo, t.right.key) < (t.lo, t.key) or t.right.prio > t.prio):
                raise AssertionError("order violated")
            if t.max_hi != max(t.hi, ml, mr):
                raise AssertionError("augmentation violated")
            return t.max_hi
        go(self._root)
        return True


class SortedEndpointIndex:
    """Counting-only index: number of intervals containing x via two sorted arrays.

    #containing(x) = #(lo <= x) - #(hi < x).  O(n log n) build, O(log n) query.
    """

    def __init__(self, intervals: Sequence[Interval]):
        self._los = sorted(a for a, _ in intervals)
        self._his = sorted(b for _, b in intervals)

    def count_stab(self, x: float) -> int:
        return bisect_right(self._los, x) - bisect_left(self._his, x)

    def count_overlap(self, lo: float, hi: float) -> int:
        """Intervals intersecting [lo, hi] = n - (#hi < lo) - (#lo > hi)."""
        n = len(self._los)
        return n - bisect_left(self._his, lo) - (n - bisect_right(self._los, hi))
