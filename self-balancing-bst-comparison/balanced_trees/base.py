"""Common interface and instrumentation shared by all balanced search trees."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional


class SortedMap(ABC):
    """Abstract ordered map with operation counters.

    Subclasses store nodes exposing ``key``, ``left`` and ``right``.  Absent
    children are represented by ``self._nil`` (``None`` unless overridden).

    Attributes:
        comparisons: number of key comparisons performed so far.
        rotations: number of single rotations performed so far.
    """

    _nil: Any = None

    def __init__(self) -> None:
        self.comparisons = 0
        self.rotations = 0
        self._size = 0
        self._root: Any = self._nil

    # ---- abstract operations -------------------------------------------------
    @abstractmethod
    def insert(self, key: Any, value: Any = None) -> bool:
        """Insert ``key``; return True if new, False if value was replaced."""

    @abstractmethod
    def delete(self, key: Any) -> bool:
        """Remove ``key``; return True if it was present."""

    @abstractmethod
    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value stored for ``key`` or ``default``."""

    @abstractmethod
    def validate(self) -> None:
        """Raise ``AssertionError`` if any structural invariant is violated."""

    # ---- derived operations --------------------------------------------------
    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __contains__(self, key: Any) -> bool:
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    def __iter__(self) -> Iterator[Any]:
        for k, _ in self.items():
            yield k

    def items(self) -> Iterator[tuple[Any, Any]]:
        """Iterate ``(key, value)`` in key order without recursion."""
        stack: list[Any] = []
        node = self._root
        while stack or node is not self._nil:
            while node is not self._nil:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key, node.value
            node = node.right

    def keys(self) -> list[Any]:
        return list(self)

    def height(self) -> int:
        """Number of nodes on the longest root-to-leaf path (0 when empty)."""
        if self._root is self._nil:
            return 0
        best = 0
        stack = [(self._root, 1)]
        while stack:
            node, d = stack.pop()
            best = max(best, d)
            if node.left is not self._nil:
                stack.append((node.left, d + 1))
            if node.right is not self._nil:
                stack.append((node.right, d + 1))
        return best

    def average_depth(self) -> float:
        """Mean node depth (root has depth 1)."""
        if self._root is self._nil:
            return 0.0
        total = 0
        stack = [(self._root, 1)]
        while stack:
            node, d = stack.pop()
            total += d
            for c in (node.left, node.right):
                if c is not self._nil:
                    stack.append((c, d + 1))
        return total / self._size

    def minimum(self) -> Optional[Any]:
        node = self._root
        if node is self._nil:
            return None
        while node.left is not self._nil:
            node = node.left
        return node.key

    def maximum(self) -> Optional[Any]:
        node = self._root
        if node is self._nil:
            return None
        while node.right is not self._nil:
            node = node.right
        return node.key

    def floor(self, key: Any) -> Optional[Any]:
        """Largest stored key <= ``key`` (read-only, no restructuring)."""
        node, best = self._root, None
        while node is not self._nil:
            if node.key == key:
                return node.key
            if node.key < key:
                best, node = node.key, node.right
            else:
                node = node.left
        return best

    def ceiling(self, key: Any) -> Optional[Any]:
        """Smallest stored key >= ``key`` (read-only, no restructuring)."""
        node, best = self._root, None
        while node is not self._nil:
            if node.key == key:
                return node.key
            if node.key > key:
                best, node = node.key, node.left
            else:
                node = node.right
        return best

    def range(self, lo: Any, hi: Any) -> list[Any]:
        """Keys in the closed interval ``[lo, hi]`` in order."""
        out: list[Any] = []
        stack: list[Any] = []
        node = self._root
        while stack or node is not self._nil:
            while node is not self._nil:
                stack.append(node)
                node = node.left if node.key >= lo else self._nil
            node = stack.pop()
            if node.key > hi:
                break
            if node.key >= lo:
                out.append(node.key)
            node = node.right
        return out

    def reset_counters(self) -> None:
        self.comparisons = 0
        self.rotations = 0

    def _check_bst_order(self) -> None:
        prev = None
        n = 0
        for k in self:
            assert prev is None or prev < k, "in-order keys not strictly increasing"
            prev = k
            n += 1
        assert n == self._size, f"size mismatch: {n} nodes vs recorded {self._size}"
