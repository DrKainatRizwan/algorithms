"""Sequential indexable skip list (sorted map) with order statistics.

Every forward pointer stores a *span*, the number of level-0 steps it skips,
which gives O(log n) expected ``rank`` and ``kth`` queries in addition to the
usual search/insert/delete (Pugh 1990; the span trick follows Redis zsets).
"""

from __future__ import annotations

from typing import Any, Hashable, Iterator, Optional

from .levels import GeometricLevels, LevelPolicy


class _Node:
    __slots__ = ("key", "value", "fwd", "span")

    def __init__(self, key: Any, value: Any, level: int) -> None:
        self.key = key
        self.value = value
        self.fwd: list[Optional[_Node]] = [None] * level
        self.span: list[int] = [0] * level


class SkipList:
    """Ordered map with expected O(log n) operations.

    ``comparisons`` counts key comparisons made during searches, which the
    analysis module uses to validate the theoretical cost model.
    """

    def __init__(self, policy: LevelPolicy | None = None) -> None:
        self.policy = policy or GeometricLevels()
        self._head = _Node(None, None, self.policy.max_level)
        self._level = 1
        self._len = 0
        self.comparisons = 0

    # ---------------------------------------------------------------- basics
    def __len__(self) -> int:
        return self._len

    @property
    def height(self) -> int:
        """Number of levels currently in use."""
        return self._level

    def _lt(self, node_key: Any, key: Any) -> bool:
        self.comparisons += 1
        return node_key < key

    def _find_ge(self, key: Any) -> Optional[_Node]:
        x = self._head
        for i in range(self._level - 1, -1, -1):
            while x.fwd[i] is not None and self._lt(x.fwd[i].key, key):
                x = x.fwd[i]
        return x.fwd[0]

    def get(self, key: Hashable, default: Any = None) -> Any:
        node = self._find_ge(key)
        return node.value if node is not None and node.key == key else default

    def __contains__(self, key: Hashable) -> bool:
        node = self._find_ge(key)
        return node is not None and node.key == key

    def __getitem__(self, key: Hashable) -> Any:
        node = self._find_ge(key)
        if node is None or node.key != key:
            raise KeyError(key)
        return node.value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.insert(key, value)

    def __delitem__(self, key: Hashable) -> None:
        if not self.delete(key):
            raise KeyError(key)

    # ------------------------------------------------------------- mutation
    def insert(self, key: Hashable, value: Any = None) -> bool:
        """Insert or overwrite ``key``. Returns True if the key was new."""
        update: list[_Node] = [self._head] * self.policy.max_level
        rank = [0] * self.policy.max_level
        x = self._head
        for i in range(self._level - 1, -1, -1):
            rank[i] = 0 if i == self._level - 1 else rank[i + 1]
            while x.fwd[i] is not None and self._lt(x.fwd[i].key, key):
                rank[i] += x.span[i]
                x = x.fwd[i]
            update[i] = x
        nxt = x.fwd[0]
        if nxt is not None and nxt.key == key:
            nxt.value = value
            return False

        lvl = self.policy.level_for(key)
        if lvl > self._level:
            for i in range(self._level, lvl):
                rank[i] = 0
                update[i] = self._head
                self._head.span[i] = self._len
            self._level = lvl
        node = _Node(key, value, lvl)
        for i in range(lvl):
            node.fwd[i] = update[i].fwd[i]
            update[i].fwd[i] = node
            node.span[i] = update[i].span[i] - (rank[0] - rank[i])
            update[i].span[i] = (rank[0] - rank[i]) + 1
        for i in range(lvl, self._level):
            update[i].span[i] += 1
        self._len += 1
        return True

    def delete(self, key: Hashable) -> bool:
        """Remove ``key``; returns True if it was present."""
        update: list[_Node] = [self._head] * self.policy.max_level
        x = self._head
        for i in range(self._level - 1, -1, -1):
            while x.fwd[i] is not None and self._lt(x.fwd[i].key, key):
                x = x.fwd[i]
            update[i] = x
        target = x.fwd[0]
        if target is None or target.key != key:
            return False
        for i in range(self._level):
            if update[i].fwd[i] is target:
                update[i].span[i] += target.span[i] - 1
                update[i].fwd[i] = target.fwd[i]
            else:
                update[i].span[i] -= 1
        while self._level > 1 and self._head.fwd[self._level - 1] is None:
            self._level -= 1
        self._len -= 1
        return True

    def pop_min(self) -> tuple[Any, Any]:
        first = self._head.fwd[0]
        if first is None:
            raise KeyError("pop from empty skip list")
        item = (first.key, first.value)
        self.delete(first.key)
        return item

    # ------------------------------------------------------------- ordering
    def rank(self, key: Any) -> int:
        """Number of stored keys strictly smaller than ``key``."""
        x = self._head
        r = 0
        for i in range(self._level - 1, -1, -1):
            while x.fwd[i] is not None and self._lt(x.fwd[i].key, key):
                r += x.span[i]
                x = x.fwd[i]
        return r

    def kth(self, index: int) -> tuple[Any, Any]:
        """Return the (key, value) pair at 0-based sorted position ``index``."""
        if index < 0:
            index += self._len
        if not 0 <= index < self._len:
            raise IndexError("index out of range")
        target = index + 1
        x = self._head
        pos = 0
        for i in range(self._level - 1, -1, -1):
            while x.fwd[i] is not None and pos + x.span[i] <= target:
                pos += x.span[i]
                x = x.fwd[i]
            if pos == target:
                break
        return x.key, x.value

    def floor(self, key: Any) -> Optional[Any]:
        """Largest stored key <= ``key`` (None if there is none)."""
        r = self.rank(key)
        if key in self:
            return key
        return self.kth(r - 1)[0] if r > 0 else None

    def ceiling(self, key: Any) -> Optional[Any]:
        """Smallest stored key >= ``key`` (None if there is none)."""
        node = self._find_ge(key)
        return node.key if node is not None else None

    def range(self, lo: Any, hi: Any) -> Iterator[tuple[Any, Any]]:
        """Yield items with ``lo <= key < hi`` in sorted order."""
        node = self._find_ge(lo)
        while node is not None and node.key < hi:
            yield node.key, node.value
            node = node.fwd[0]

    def __iter__(self) -> Iterator[Any]:
        node = self._head.fwd[0]
        while node is not None:
            yield node.key
            node = node.fwd[0]

    def items(self) -> Iterator[tuple[Any, Any]]:
        node = self._head.fwd[0]
        while node is not None:
            yield node.key, node.value
            node = node.fwd[0]

    # ----------------------------------------------------------- inspection
    def level_sizes(self) -> list[int]:
        """Number of nodes present on each level (index 0 = bottom)."""
        sizes = []
        for i in range(self._level):
            count, node = 0, self._head.fwd[i]
            while node is not None:
                count += 1
                node = node.fwd[i]
            sizes.append(count)
        return sizes

    def tower_heights(self) -> list[int]:
        out, node = [], self._head.fwd[0]
        while node is not None:
            out.append(len(node.fwd))
            node = node.fwd[0]
        return out

    def check_invariants(self) -> None:
        """Raise AssertionError if ordering, spans or level links are inconsistent."""
        keys = list(self)
        assert keys == sorted(keys) and len(set(keys)) == len(keys), "keys not strictly sorted"
        assert len(keys) == self._len, "length mismatch"
        position = {k: i + 1 for i, k in enumerate(keys)}
        for i in range(self._level):
            x, pos = self._head, 0
            while x.fwd[i] is not None:
                nxt = x.fwd[i]
                assert position[nxt.key] == pos + x.span[i], f"bad span at level {i}"
                x, pos = nxt, position[nxt.key]
            assert pos + x.span[i] == self._len, f"bad tail span at level {i}"
        for i in range(self._level, self.policy.max_level):
            assert self._head.fwd[i] is None, "stale link above current level"
