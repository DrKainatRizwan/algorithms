"""Treap: a randomised BST that is simultaneously a heap on random priorities.

Implemented with split/merge, which also yields order statistics (``rank`` and
``select``) and O(log n) expected union-free ``split_at``/``concat``.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from .base import SortedMap


class TreapNode:
    __slots__ = ("key", "value", "prio", "left", "right", "size")

    def __init__(self, key: Any, value: Any, prio: float) -> None:
        self.key, self.value, self.prio = key, value, prio
        self.left: Optional[TreapNode] = None
        self.right: Optional[TreapNode] = None
        self.size = 1


def _size(n: Optional[TreapNode]) -> int:
    return n.size if n else 0


def _update(n: TreapNode) -> TreapNode:
    n.size = 1 + _size(n.left) + _size(n.right)
    return n


class Treap(SortedMap):
    """Max-heap-ordered treap with seeded priorities.

    Expected height is O(log n) regardless of insertion order because
    priorities are independent of keys.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    # -- split/merge primitives (each descent step counts as one comparison) ----
    def _split(self, n: Optional[TreapNode], key: Any):
        """Split into (keys < key, keys >= key)."""
        if n is None:
            return None, None
        self.comparisons += 1
        if n.key < key:
            a, b = self._split(n.right, key)
            n.right = a
            return _update(n), b
        a, b = self._split(n.left, key)
        n.left = b
        return a, _update(n)

    def _merge(self, a: Optional[TreapNode], b: Optional[TreapNode]):
        """Merge two treaps where every key in ``a`` < every key in ``b``."""
        if a is None:
            return b
        if b is None:
            return a
        self.comparisons += 1
        if a.prio > b.prio:
            a.right = self._merge(a.right, b)
            return _update(a)
        b.left = self._merge(a, b.left)
        return _update(b)

    def _find(self, key: Any) -> Optional[TreapNode]:
        n = self._root
        while n is not None:
            self.comparisons += 1
            if key == n.key:
                return n
            n = n.left if key < n.key else n.right
        return None

    # -- public API --------------------------------------------------------------
    def get(self, key: Any, default: Any = None) -> Any:
        n = self._find(key)
        return n.value if n else default

    def insert(self, key: Any, value: Any = None) -> bool:
        n = self._find(key)
        if n is not None:
            n.value = value
            return False
        left, right = self._split(self._root, key)
        node = TreapNode(key, value, self._rng.random())
        self._root = self._merge(self._merge(left, node), right)
        self._size += 1
        return True

    def delete(self, key: Any) -> bool:
        if self._find(key) is None:
            return False
        self._root = self._delete(self._root, key)
        self._size -= 1
        return True

    def _delete(self, n: Optional[TreapNode], key: Any) -> Optional[TreapNode]:
        assert n is not None
        self.comparisons += 1
        if key == n.key:
            self.rotations += 1  # merging children ~ rotating the node to a leaf
            return self._merge(n.left, n.right)
        if key < n.key:
            n.left = self._delete(n.left, key)
        else:
            n.right = self._delete(n.right, key)
        return _update(n)

    def rank(self, key: Any) -> int:
        """Number of stored keys strictly smaller than ``key``."""
        n, r = self._root, 0
        while n is not None:
            if key <= n.key:
                n = n.left
            else:
                r += _size(n.left) + 1
                n = n.right
        return r

    def select(self, k: int) -> Any:
        """The k-th smallest key (0-based)."""
        if not 0 <= k < self._size:
            raise IndexError(k)
        n = self._root
        while True:
            ls = _size(n.left)
            if k < ls:
                n = n.left
            elif k == ls:
                return n.key
            else:
                k -= ls + 1
                n = n.right

    def split(self, key: Any) -> tuple["Treap", "Treap"]:
        """Destructively split into (keys < key, keys >= key)."""
        a, b = self._split(self._root, key)
        left, right = Treap(self._rng.randrange(1 << 30)), Treap(self._rng.randrange(1 << 30))
        left._root, right._root = a, b
        left._size, right._size = _size(a), _size(b)
        self._root, self._size = None, 0
        return left, right

    @classmethod
    def concat(cls, a: "Treap", b: "Treap", seed: int = 0) -> "Treap":
        """Concatenate two treaps with ``max(a) < min(b)`` in O(log n) expected."""
        if a and b and not a.maximum() < b.minimum():
            raise ValueError("key ranges overlap")
        t = cls(seed)
        t._root = t._merge(a._root, b._root)
        t._size = a._size + b._size
        a._root = b._root = None
        a._size = b._size = 0
        return t

    def validate(self) -> None:
        self._check_bst_order()

        def walk(n: Optional[TreapNode]) -> int:
            if n is None:
                return 0
            for c in (n.left, n.right):
                assert c is None or c.prio <= n.prio, "heap property violated"
            s = 1 + walk(n.left) + walk(n.right)
            assert s == n.size, "stale subtree size"
            return s

        walk(self._root)
