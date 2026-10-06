"""Fully persistent point-update / range-query segment tree (path copying)."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .monoid import Monoid, SUM


class Node:
    """Immutable tree node; children are shared between versions."""

    __slots__ = ("value", "left", "right")

    def __init__(self, value: Any, left: Optional["Node"] = None,
                 right: Optional["Node"] = None) -> None:
        self.value = value
        self.left = left
        self.right = right


class PersistentSegmentTree:
    """Segment tree where every update yields a new version id.

    Versions are integers; version 0 is the initial array. Updating any
    version (not only the latest) is allowed, so the history forms a tree.
    Each update allocates O(log n) nodes; old versions stay readable.
    """

    def __init__(self, values: Sequence[Any], monoid: Monoid = SUM) -> None:
        if len(values) == 0:
            raise ValueError("values must be non-empty")
        self.n = len(values)
        self.monoid = monoid
        self.nodes_created = 0
        self.roots: list[Node] = [self._build(values, 0, self.n)]

    # -- construction -------------------------------------------------
    def _make(self, value: Any, left: Optional[Node] = None,
              right: Optional[Node] = None) -> Node:
        self.nodes_created += 1
        return Node(value, left, right)

    def _build(self, vals: Sequence[Any], lo: int, hi: int) -> Node:
        if hi - lo == 1:
            return self._make(vals[lo])
        mid = (lo + hi) // 2
        l, r = self._build(vals, lo, mid), self._build(vals, mid, hi)
        return self._make(self.monoid.combine(l.value, r.value), l, r)

    # -- versions -----------------------------------------------------
    @property
    def num_versions(self) -> int:
        return len(self.roots)

    def _root(self, version: int) -> Node:
        if not 0 <= version < len(self.roots):
            raise IndexError(f"unknown version {version}")
        return self.roots[version]

    # -- operations ---------------------------------------------------
    def update(self, version: int, index: int, value: Any) -> int:
        """Set ``a[index] = value`` on top of ``version``; return new version id."""
        if not 0 <= index < self.n:
            raise IndexError(index)
        self.roots.append(self._update(self._root(version), 0, self.n, index, value))
        return len(self.roots) - 1

    def _update(self, node: Node, lo: int, hi: int, i: int, value: Any) -> Node:
        if hi - lo == 1:
            return self._make(value)
        mid = (lo + hi) // 2
        if i < mid:
            l, r = self._update(node.left, lo, mid, i, value), node.right
        else:
            l, r = node.left, self._update(node.right, mid, hi, i, value)
        return self._make(self.monoid.combine(l.value, r.value), l, r)

    def query(self, version: int, left: int, right: int) -> Any:
        """Fold the monoid over the half-open range ``[left, right)``."""
        if not 0 <= left <= right <= self.n:
            raise IndexError((left, right))
        return self._query(self._root(version), 0, self.n, left, right)

    def _query(self, node: Node, lo: int, hi: int, l: int, r: int) -> Any:
        if r <= lo or hi <= l:
            return self.monoid.identity
        if l <= lo and hi <= r:
            return node.value
        mid = (lo + hi) // 2
        return self.monoid.combine(self._query(node.left, lo, mid, l, r),
                                   self._query(node.right, mid, hi, l, r))

    def get(self, version: int, index: int) -> Any:
        """Point read at ``version``."""
        if not 0 <= index < self.n:
            raise IndexError(index)
        node, lo, hi = self._root(version), 0, self.n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if index < mid:
                node, hi = node.left, mid
            else:
                node, lo = node.right, mid
        return node.value

    def to_list(self, version: int) -> list[Any]:
        return [self.get(version, i) for i in range(self.n)]

    def _reachable(self, version: int) -> set[int]:
        seen: set[int] = set()
        stack = [self._root(version)]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            if n.left is not None:
                stack.extend((n.left, n.right))
        return seen

    def node_count(self, version: int) -> int:
        """Number of distinct nodes reachable from ``version``."""
        return len(self._reachable(version))

    def shared_nodes(self, a: int, b: int) -> int:
        """Count nodes physically shared between two versions."""
        return len(self._reachable(a) & self._reachable(b))
