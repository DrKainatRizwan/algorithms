"""In-memory B+ tree with insertion, deletion, point and range queries.

``max_keys`` is the page capacity ``m``: every node holds at most ``m`` keys
and every non-root node at least ``m // 2``.  Keys are unique; inserting an
existing key overwrites its value.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any, Iterator, List, Optional, Tuple

from .nodes import InternalNode, LeafNode, Node


class BPlusTree:
    """A B+ tree mapping comparable keys to arbitrary values."""

    def __init__(self, max_keys: int = 32) -> None:
        if max_keys < 3:
            raise ValueError("max_keys must be at least 3")
        self.max_keys = max_keys
        self.min_keys = max_keys // 2
        self.root: Node = LeafNode()
        self._size = 0
        self.trace: Optional[List[int]] = None  # node ids touched, if enabled

    # ------------------------------------------------------------------ helpers
    def _touch(self, node: Node) -> None:
        if self.trace is not None:
            self.trace.append(node.node_id)

    def _find_leaf(self, key: Any) -> LeafNode:
        node = self.root
        self._touch(node)
        while not node.is_leaf:
            node = node.children[bisect_right(node.keys, key)]
            self._touch(node)
        return node  # type: ignore[return-value]

    def _leftmost_leaf(self) -> LeafNode:
        node = self.root
        self._touch(node)
        while not node.is_leaf:
            node = node.children[0]
            self._touch(node)
        return node  # type: ignore[return-value]

    def _rightmost_leaf(self) -> LeafNode:
        node = self.root
        self._touch(node)
        while not node.is_leaf:
            node = node.children[-1]
            self._touch(node)
        return node  # type: ignore[return-value]

    # ------------------------------------------------------------------ queries
    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        return i < len(leaf.keys) and leaf.keys[i] == key

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value stored under ``key`` or ``default``."""
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.values[i]
        return default

    def __getitem__(self, key: Any) -> Any:
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.values[i]
        raise KeyError(key)

    def min_item(self) -> Tuple[Any, Any]:
        if not self._size:
            raise KeyError("empty tree")
        leaf = self._leftmost_leaf()
        return leaf.keys[0], leaf.values[0]

    def max_item(self) -> Tuple[Any, Any]:
        if not self._size:
            raise KeyError("empty tree")
        leaf = self._rightmost_leaf()
        return leaf.keys[-1], leaf.values[-1]

    def ceiling(self, key: Any) -> Optional[Tuple[Any, Any]]:
        """Smallest item with key ``>= key``."""
        leaf: Optional[LeafNode] = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        while leaf is not None and i >= len(leaf.keys):
            leaf, i = leaf.next, 0
        return None if leaf is None else (leaf.keys[i], leaf.values[i])

    def floor(self, key: Any) -> Optional[Tuple[Any, Any]]:
        """Largest item with key ``<= key``."""
        leaf: Optional[LeafNode] = self._find_leaf(key)
        i = bisect_right(leaf.keys, key) - 1
        while leaf is not None and i < 0:
            leaf = leaf.prev
            i = len(leaf.keys) - 1 if leaf is not None else -1
        return None if leaf is None else (leaf.keys[i], leaf.values[i])

    def range(self, lo: Any = None, hi: Any = None, include_hi: bool = False) -> Iterator[Tuple[Any, Any]]:
        """Yield items with ``lo <= key < hi`` (``<= hi`` if ``include_hi``).

        ``None`` bounds are open.  One root-to-leaf descent, then a walk along
        the leaf chain.
        """
        if lo is None:
            leaf: Optional[LeafNode] = self._leftmost_leaf()
            i = 0
        else:
            leaf = self._find_leaf(lo)
            i = bisect_left(leaf.keys, lo)
        while leaf is not None:
            keys = leaf.keys
            while i < len(keys):
                k = keys[i]
                if hi is not None and (k > hi or (k == hi and not include_hi)):
                    return
                yield k, leaf.values[i]
                i += 1
            leaf = leaf.next
            if leaf is not None:
                self._touch(leaf)
            i = 0

    def items(self) -> Iterator[Tuple[Any, Any]]:
        return self.range()

    def __iter__(self) -> Iterator[Any]:
        for k, _ in self.range():
            yield k

    def reversed_items(self) -> Iterator[Tuple[Any, Any]]:
        """Iterate all items in descending key order via the ``prev`` links."""
        leaf: Optional[LeafNode] = self._rightmost_leaf() if self._size else None
        while leaf is not None:
            for k, v in zip(reversed(leaf.keys), reversed(leaf.values)):
                yield k, v
            leaf = leaf.prev

    def count_range(self, lo: Any = None, hi: Any = None) -> int:
        return sum(1 for _ in self.range(lo, hi))

    # ------------------------------------------------------------------ insert
    def insert(self, key: Any, value: Any = None) -> bool:
        """Insert or overwrite; returns True when a new key was added."""
        added, split = self._insert(self.root, key, value)
        if split is not None:
            sep, right = split
            new_root = InternalNode()
            new_root.keys = [sep]
            new_root.children = [self.root, right]
            self.root = new_root
        if added:
            self._size += 1
        return added

    __setitem__ = insert

    def _insert(self, node: Node, key: Any, value: Any):
        self._touch(node)
        if node.is_leaf:
            i = bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                node.values[i] = value
                return False, None
            node.keys.insert(i, key)
            node.values.insert(i, value)
            if len(node.keys) <= self.max_keys:
                return True, None
            return True, self._split_leaf(node)
        idx = bisect_right(node.keys, key)
        added, split = self._insert(node.children[idx], key, value)
        if split is None:
            return added, None
        sep, right = split
        node.keys.insert(idx, sep)
        node.children.insert(idx + 1, right)
        if len(node.keys) <= self.max_keys:
            return added, None
        return added, self._split_internal(node)

    def _split_leaf(self, leaf: LeafNode):
        mid = (len(leaf.keys) + 1) // 2
        right = LeafNode()
        right.keys, leaf.keys = leaf.keys[mid:], leaf.keys[:mid]
        right.values, leaf.values = leaf.values[mid:], leaf.values[:mid]
        right.next = leaf.next
        if right.next is not None:
            right.next.prev = right
        right.prev = leaf
        leaf.next = right
        return right.keys[0], right

    def _split_internal(self, node: InternalNode):
        mid = len(node.keys) // 2
        sep = node.keys[mid]
        right = InternalNode()
        right.keys = node.keys[mid + 1:]
        right.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        return sep, right

    # ------------------------------------------------------------------ delete
    def delete(self, key: Any) -> bool:
        """Remove ``key``; returns False if it was absent."""
        found = self._delete(self.root, key)
        if found:
            self._size -= 1
            if not self.root.is_leaf and not self.root.keys:
                self.root = self.root.children[0]
        return found

    def __delitem__(self, key: Any) -> None:
        if not self.delete(key):
            raise KeyError(key)

    def pop_min(self) -> Tuple[Any, Any]:
        k, v = self.min_item()
        self.delete(k)
        return k, v

    def _delete(self, node: Node, key: Any) -> bool:
        self._touch(node)
        if node.is_leaf:
            i = bisect_left(node.keys, key)
            if i >= len(node.keys) or node.keys[i] != key:
                return False
            del node.keys[i]
            del node.values[i]
            return True
        idx = bisect_right(node.keys, key)
        child = node.children[idx]
        found = self._delete(child, key)
        if found and len(child.keys) < self.min_keys:
            self._rebalance(node, idx)
        return found

    def _rebalance(self, parent: InternalNode, i: int) -> None:
        child = parent.children[i]
        left = parent.children[i - 1] if i > 0 else None
        right = parent.children[i + 1] if i + 1 < len(parent.children) else None
        if left is not None and len(left.keys) > self.min_keys:
            self._borrow_left(parent, i, left, child)
        elif right is not None and len(right.keys) > self.min_keys:
            self._borrow_right(parent, i, child, right)
        elif left is not None:
            self._merge(parent, i - 1)
        else:
            self._merge(parent, i)

    def _borrow_left(self, parent: InternalNode, i: int, left: Node, child: Node) -> None:
        self._touch(left)
        if child.is_leaf:
            child.keys.insert(0, left.keys.pop())
            child.values.insert(0, left.values.pop())
            parent.keys[i - 1] = child.keys[0]
        else:
            child.keys.insert(0, parent.keys[i - 1])
            child.children.insert(0, left.children.pop())
            parent.keys[i - 1] = left.keys.pop()

    def _borrow_right(self, parent: InternalNode, i: int, child: Node, right: Node) -> None:
        self._touch(right)
        if child.is_leaf:
            child.keys.append(right.keys.pop(0))
            child.values.append(right.values.pop(0))
            parent.keys[i] = right.keys[0]
        else:
            child.keys.append(parent.keys[i])
            child.children.append(right.children.pop(0))
            parent.keys[i] = right.keys.pop(0)

    def _merge(self, parent: InternalNode, i: int) -> None:
        """Merge ``children[i + 1]`` into ``children[i]``."""
        left, right = parent.children[i], parent.children[i + 1]
        self._touch(right)
        if left.is_leaf:
            left.keys += right.keys
            left.values += right.values
            left.next = right.next
            if right.next is not None:
                right.next.prev = left
        else:
            left.keys += [parent.keys[i]] + right.keys
            left.children += right.children
        del parent.keys[i]
        del parent.children[i + 1]

    # ------------------------------------------------------------------ stats
    def height(self) -> int:
        h, node = 1, self.root
        while not node.is_leaf:
            node = node.children[0]
            h += 1
        return h

    def nodes(self) -> Iterator[Node]:
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            if not n.is_leaf:
                stack.extend(reversed(n.children))

    def stats(self) -> dict:
        leaves = internals = 0
        for n in self.nodes():
            if n.is_leaf:
                leaves += 1
            else:
                internals += 1
        return {
            "size": self._size,
            "height": self.height(),
            "leaves": leaves,
            "internal": internals,
            "leaf_fill": self._size / (leaves * self.max_keys) if leaves else 0.0,
        }

    # ------------------------------------------------------------------ invariants
    def check(self) -> None:
        """Raise AssertionError if any structural invariant is violated."""
        leaf_depths: List[int] = []
        leaves: List[LeafNode] = []

        def walk(node: Node, lo: Any, hi: Any, depth: int, is_root: bool) -> None:
            assert node.keys == sorted(node.keys), "keys unsorted"
            assert len(node.keys) <= self.max_keys, "node overfull"
            if not is_root:
                assert len(node.keys) >= self.min_keys, "node underfull"
            for k in node.keys if node.is_leaf else []:
                assert (lo is None or k >= lo) and (hi is None or k < hi), "key out of range"
            if node.is_leaf:
                assert len(node.keys) == len(node.values)
                leaf_depths.append(depth)
                leaves.append(node)
                return
            assert len(node.children) == len(node.keys) + 1
            if is_root:
                assert len(node.children) >= 2
            bounds = [lo] + node.keys + [hi]
            for j, c in enumerate(node.children):
                walk(c, bounds[j], bounds[j + 1], depth + 1, False)

        walk(self.root, None, None, 0, True)
        assert len(set(leaf_depths)) <= 1, "leaves at different depths"
        total = 0
        for a, b in zip(leaves, leaves[1:]):
            assert a.next is b and b.prev is a, "broken leaf chain"
        if leaves:
            assert leaves[0].prev is None and leaves[-1].next is None
        for lf in leaves:
            total += len(lf.keys)
        assert total == self._size, "size mismatch"
