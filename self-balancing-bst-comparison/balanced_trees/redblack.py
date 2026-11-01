"""Red-black tree (CLRS formulation with a shared NIL sentinel)."""
from __future__ import annotations

from typing import Any

from .base import SortedMap

RED, BLACK = True, False


class RBNode:
    __slots__ = ("key", "value", "color", "left", "right", "parent")

    def __init__(self, key: Any, value: Any, color: bool) -> None:
        self.key, self.value, self.color = key, value, color
        self.left = self.right = self.parent = None


class RedBlackTree(SortedMap):
    """Red-black tree guaranteeing height <= 2*log2(n+1).

    Invariants: the root is black; red nodes have black children; every
    root-to-NIL path has the same number of black nodes.  Insertion performs at
    most 2 rotations and deletion at most 3.
    """

    def __init__(self) -> None:
        super().__init__()
        nil = RBNode(None, None, BLACK)
        nil.left = nil.right = nil.parent = nil
        self._nil = nil
        self._root = nil

    # -- rotations --------------------------------------------------------------
    def _rotate_left(self, x: RBNode) -> None:
        nil = self._nil
        y = x.right
        x.right = y.left
        if y.left is not nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is nil:
            self._root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        self.rotations += 1

    def _rotate_right(self, x: RBNode) -> None:
        nil = self._nil
        y = x.left
        x.left = y.right
        if y.right is not nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is nil:
            self._root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        self.rotations += 1

    # -- search -----------------------------------------------------------------
    def _find(self, key: Any) -> RBNode:
        n = self._root
        while n is not self._nil:
            self.comparisons += 1
            if key == n.key:
                return n
            n = n.left if key < n.key else n.right
        return self._nil

    def get(self, key: Any, default: Any = None) -> Any:
        n = self._find(key)
        return default if n is self._nil else n.value

    # -- insertion --------------------------------------------------------------
    def insert(self, key: Any, value: Any = None) -> bool:
        nil = self._nil
        parent, cur = nil, self._root
        while cur is not nil:
            parent = cur
            self.comparisons += 1
            if key == cur.key:
                cur.value = value
                return False
            cur = cur.left if key < cur.key else cur.right
        z = RBNode(key, value, RED)
        z.left = z.right = nil
        z.parent = parent
        if parent is nil:
            self._root = z
        elif key < parent.key:
            parent.left = z
        else:
            parent.right = z
        self._size += 1
        self._insert_fixup(z)
        return True

    def _insert_fixup(self, z: RBNode) -> None:
        while z.parent.color == RED:
            gp = z.parent.parent
            if z.parent is gp.left:
                uncle = gp.right
                if uncle.color == RED:
                    z.parent.color = uncle.color = BLACK
                    gp.color = RED
                    z = gp
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._rotate_left(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:
                uncle = gp.left
                if uncle.color == RED:
                    z.parent.color = uncle.color = BLACK
                    gp.color = RED
                    z = gp
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self._root.color = BLACK

    # -- deletion ---------------------------------------------------------------
    def _transplant(self, u: RBNode, v: RBNode) -> None:
        if u.parent is self._nil:
            self._root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent  # intentionally sets nil.parent when v is nil

    def delete(self, key: Any) -> bool:
        nil = self._nil
        z = self._find(key)
        if z is nil:
            return False
        y, y_color = z, z.color
        if z.left is nil:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is nil:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = z.right
            while y.left is not nil:
                y = y.left
            y_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        self._size -= 1
        if y_color == BLACK:
            self._delete_fixup(x)
        nil.parent = nil
        return True

    def _delete_fixup(self, x: RBNode) -> None:
        while x is not self._root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color = RED
                        self._rotate_right(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self._rotate_left(x.parent)
                    x = self._root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self._rotate_right(x.parent)
                    x = self._root
        x.color = BLACK

    def black_height(self) -> int:
        h, n = 0, self._root
        while n is not self._nil:
            if n.color == BLACK:
                h += 1
            n = n.left
        return h + 1 if self._root is not self._nil else 0

    def validate(self) -> None:
        nil = self._nil
        self._check_bst_order()
        assert self._root.color == BLACK, "root must be black"

        def walk(n: RBNode) -> int:
            if n is nil:
                return 1
            if n.color == RED:
                assert n.left.color == BLACK and n.right.color == BLACK, "red-red edge"
            for c in (n.left, n.right):
                if c is not nil:
                    assert c.parent is n, "broken parent pointer"
            lh, rh = walk(n.left), walk(n.right)
            assert lh == rh, "black-height mismatch"
            return lh + (1 if n.color == BLACK else 0)

        walk(self._root)
