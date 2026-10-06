"""Persistent range-add / range-sum tree using non-propagating (permanent) tags."""
from __future__ import annotations

from typing import Optional, Sequence


class LNode:
    """Node holding the subtree sum (including own tag) and a pending add tag."""

    __slots__ = ("total", "tag", "left", "right")

    def __init__(self, total: int, tag: int, left: Optional["LNode"],
                 right: Optional["LNode"]) -> None:
        self.total = total
        self.tag = tag
        self.left = left
        self.right = right


class PersistentLazyTree:
    """Range add, range sum, all versions retained.

    Tags are never pushed down (so no node is ever mutated or copied on a
    read); a query adds ``tag * overlap`` for every ancestor it passes.
    """

    def __init__(self, values: Sequence[int]) -> None:
        if len(values) == 0:
            raise ValueError("values must be non-empty")
        self.n = len(values)
        self.nodes_created = 0
        self.roots = [self._build(values, 0, self.n)]

    def _make(self, total: int, tag: int, l: Optional[LNode], r: Optional[LNode]) -> LNode:
        self.nodes_created += 1
        return LNode(total, tag, l, r)

    def _build(self, v: Sequence[int], lo: int, hi: int) -> LNode:
        if hi - lo == 1:
            return self._make(v[lo], 0, None, None)
        mid = (lo + hi) // 2
        l, r = self._build(v, lo, mid), self._build(v, mid, hi)
        return self._make(l.total + r.total, 0, l, r)

    @property
    def num_versions(self) -> int:
        return len(self.roots)

    def add(self, version: int, left: int, right: int, delta: int) -> int:
        """Add ``delta`` to ``a[left:right]`` starting from ``version``."""
        if not 0 <= left <= right <= self.n:
            raise IndexError((left, right))
        root = self.roots[version]
        if left < right:
            root = self._add(root, 0, self.n, left, right, delta)
        self.roots.append(root)
        return len(self.roots) - 1

    def _add(self, node: LNode, lo: int, hi: int, l: int, r: int, d: int) -> LNode:
        overlap = min(hi, r) - max(lo, l)
        if l <= lo and hi <= r:
            return self._make(node.total + d * (hi - lo), node.tag + d,
                              node.left, node.right)
        mid = (lo + hi) // 2
        left, right = node.left, node.right
        if l < mid:
            left = self._add(left, lo, mid, l, r, d)
        if r > mid:
            right = self._add(right, mid, hi, l, r, d)
        return self._make(node.total + d * overlap, node.tag, left, right)

    def sum(self, version: int, left: int, right: int) -> int:
        """Sum of ``a[left:right]`` at ``version``."""
        if not 0 <= left <= right <= self.n:
            raise IndexError((left, right))
        return self._sum(self.roots[version], 0, self.n, left, right)

    def _sum(self, node: LNode, lo: int, hi: int, l: int, r: int) -> int:
        if r <= lo or hi <= l:
            return 0
        if l <= lo and hi <= r:
            return node.total
        mid = (lo + hi) // 2
        overlap = min(hi, r) - max(lo, l)
        return (node.tag * overlap + self._sum(node.left, lo, mid, l, r)
                + self._sum(node.right, mid, hi, l, r))
