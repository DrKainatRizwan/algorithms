"""Splay tree using Sleator & Tarjan's top-down splaying."""
from __future__ import annotations

from typing import Any, Optional

from .base import SortedMap


class SplayNode:
    __slots__ = ("key", "value", "left", "right")

    def __init__(self, key: Any, value: Any = None) -> None:
        self.key, self.value = key, value
        self.left: Optional[SplayNode] = None
        self.right: Optional[SplayNode] = None


class SplayTree(SortedMap):
    """Self-adjusting BST with O(log n) amortised cost and no balance metadata.

    Every access moves the touched key (or the last node on its search path) to
    the root, giving the static-optimality and working-set behaviour that make
    splay trees attractive for skewed workloads.
    """

    def _splay(self, key: Any) -> None:
        """Bring ``key`` (or its in-order neighbour on the search path) to the root."""
        t = self._root
        if t is None:
            return
        header = SplayNode(None)
        left_tail = right_tail = header
        while True:
            self.comparisons += 1
            if key < t.key:
                if t.left is None:
                    break
                self.comparisons += 1
                if key < t.left.key:  # zig-zig: rotate right
                    y = t.left
                    t.left, y.right = y.right, t
                    t = y
                    self.rotations += 1
                    if t.left is None:
                        break
                right_tail.left = t  # link right
                right_tail = t
                t = t.left
            elif key > t.key:
                if t.right is None:
                    break
                self.comparisons += 1
                if key > t.right.key:  # zig-zig: rotate left
                    y = t.right
                    t.right, y.left = y.left, t
                    t = y
                    self.rotations += 1
                    if t.right is None:
                        break
                left_tail.right = t  # link left
                left_tail = t
                t = t.right
            else:
                break
        left_tail.right = t.left
        right_tail.left = t.right
        t.left, t.right = header.right, header.left
        self._root = t

    def get(self, key: Any, default: Any = None) -> Any:
        self._splay(key)
        if self._root is not None and self._root.key == key:
            return self._root.value
        return default

    def insert(self, key: Any, value: Any = None) -> bool:
        if self._root is None:
            self._root = SplayNode(key, value)
            self._size = 1
            return True
        self._splay(key)
        r = self._root
        if r.key == key:
            r.value = value
            return False
        node = SplayNode(key, value)
        if key < r.key:
            node.left, node.right = r.left, r
            r.left = None
        else:
            node.right, node.left = r.right, r
            r.right = None
        self._root = node
        self._size += 1
        return True

    def delete(self, key: Any) -> bool:
        if self._root is None:
            return False
        self._splay(key)
        r = self._root
        if r.key != key:
            return False
        if r.left is None:
            self._root = r.right
        else:
            right = r.right
            self._root = r.left
            self._splay(key)  # max of left subtree surfaces (key > all of them)
            self._root.right = right
        self._size -= 1
        return True

    def validate(self) -> None:
        self._check_bst_order()
