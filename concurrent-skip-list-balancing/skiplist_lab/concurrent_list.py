"""Lock-free-style concurrent skip list (Herlihy-Shavit / Fraser design).

Deletion is two-phase: a node is first *logically* removed by setting a mark
bit on each of its next pointers, then *physically* unlinked by any traversal
that bumps into it.  Pointer + mark pairs are updated with compare-and-set.

CPython has no hardware CAS on object references, so ``AtomicMarkableRef``
emulates it with a tiny per-reference lock.  The skip list algorithm itself
never holds a lock across more than one CAS, so its control flow is exactly
that of the lock-free algorithm; ``contains`` is wait-free (read only).
"""

from __future__ import annotations

import threading
from typing import Any, Hashable

from .levels import GeometricLevels, LevelPolicy


class AtomicMarkableRef:
    """A (reference, mark) pair that can be swapped atomically."""

    __slots__ = ("_pair", "_lock")

    def __init__(self, ref: Any = None, mark: bool = False) -> None:
        self._pair = (ref, mark)
        self._lock = threading.Lock()

    def get(self) -> tuple[Any, bool]:
        return self._pair  # tuple read is atomic under the GIL

    @property
    def ref(self) -> Any:
        return self._pair[0]

    def compare_and_set(self, exp_ref: Any, new_ref: Any, exp_mark: bool, new_mark: bool) -> bool:
        with self._lock:
            ref, mark = self._pair
            if ref is exp_ref and mark == exp_mark:
                self._pair = (new_ref, new_mark)
                return True
            return False

    def attempt_mark(self, exp_ref: Any, new_mark: bool) -> bool:
        with self._lock:
            ref, _ = self._pair
            if ref is exp_ref:
                self._pair = (ref, new_mark)
                return True
            return False


class _CNode:
    __slots__ = ("key", "value", "top", "next", "kind")

    HEAD, TAIL, DATA = 0, 1, 2

    def __init__(self, key: Any, value: Any, top: int, kind: int = 2) -> None:
        self.key = key
        self.value = value
        self.top = top  # highest level index
        self.kind = kind
        self.next = [AtomicMarkableRef(None, False) for _ in range(top + 1)]

    def precedes(self, key: Any) -> bool:
        """True if this node sorts strictly before ``key``."""
        if self.kind == _CNode.HEAD:
            return True
        if self.kind == _CNode.TAIL:
            return False
        return self.key < key

    def matches(self, key: Any) -> bool:
        return self.kind == _CNode.DATA and self.key == key


class LockFreeSkipList:
    """Concurrent sorted set/map with add, remove and wait-free contains.

    Statistics (``cas_failures``, ``snips``) are plain counters and are
    therefore approximate under contention.
    """

    def __init__(self, policy: LevelPolicy | None = None) -> None:
        self.policy = policy or GeometricLevels(seed=0)
        self._policy_lock = threading.Lock()
        top = self.policy.max_level - 1
        self._head = _CNode(None, None, top, _CNode.HEAD)
        self._tail = _CNode(None, None, top, _CNode.TAIL)
        for ref in self._head.next:
            ref._pair = (self._tail, False)
        self.cas_failures = 0
        self.snips = 0

    def _random_top(self, key: Hashable) -> int:
        with self._policy_lock:
            return self.policy.level_for(key) - 1

    @staticmethod
    def _read(node: _CNode, level: int) -> tuple[Any, bool]:
        return (None, False) if node.kind == _CNode.TAIL else node.next[level].get()

    # --------------------------------------------------------------- search
    def _find(self, key: Any, preds: list, succs: list) -> bool:
        """Fill preds/succs per level, physically unlinking marked nodes."""
        levels = self.policy.max_level
        while True:  # retry loop
            pred = self._head
            restart = False
            for level in range(levels - 1, -1, -1):
                curr = pred.next[level].ref
                while True:
                    succ, marked = self._read(curr, level)
                    while marked:
                        if not pred.next[level].compare_and_set(curr, succ, False, False):
                            self.cas_failures += 1
                            restart = True
                            break
                        self.snips += 1
                        curr = succ
                        succ, marked = self._read(curr, level)
                    if restart:
                        break
                    if curr.precedes(key):
                        pred, curr = curr, succ
                    else:
                        break
                if restart:
                    break
                preds[level] = pred
                succs[level] = curr
            if restart:
                continue
            return succs[0].matches(key)

    # ------------------------------------------------------------ operations
    def add(self, key: Hashable, value: Any = None) -> bool:
        """Insert ``key``; returns False (value untouched) if already present."""
        top = self._random_top(key)
        levels = self.policy.max_level
        preds: list = [None] * levels
        succs: list = [None] * levels
        while True:
            if self._find(key, preds, succs):
                return False
            node = _CNode(key, value, top)
            for lv in range(top + 1):
                node.next[lv]._pair = (succs[lv], False)
            if not preds[0].next[0].compare_and_set(succs[0], node, False, False):
                self.cas_failures += 1
                continue
            for lv in range(1, top + 1):
                while True:
                    ref, marked = node.next[lv].get()
                    if marked:  # concurrently removed: stop building the tower
                        return True
                    if ref is not succs[lv] and not node.next[lv].compare_and_set(ref, succs[lv], False, False):
                        continue
                    if preds[lv].next[lv].compare_and_set(succs[lv], node, False, False):
                        break
                    self.cas_failures += 1
                    self._find(key, preds, succs)
            return True

    def remove(self, key: Hashable) -> bool:
        """Remove ``key``; exactly one of several concurrent removers wins."""
        levels = self.policy.max_level
        preds: list = [None] * levels
        succs: list = [None] * levels
        if not self._find(key, preds, succs):
            return False
        victim = succs[0]
        for lv in range(victim.top, 0, -1):
            succ, marked = victim.next[lv].get()
            while not marked:
                victim.next[lv].attempt_mark(succ, True)
                succ, marked = victim.next[lv].get()
        succ, marked = victim.next[0].get()
        while True:
            i_marked_it = victim.next[0].compare_and_set(succ, succ, False, True)
            succ, marked = victim.next[0].get()
            if i_marked_it:
                self._find(key, preds, succs)  # physical cleanup
                return True
            if marked:
                return False
            self.cas_failures += 1

    def contains(self, key: Hashable) -> bool:
        """Wait-free membership test (never modifies the structure)."""
        pred = self._head
        curr = pred
        for level in range(self.policy.max_level - 1, -1, -1):
            curr = pred.next[level].ref
            while curr.kind != _CNode.TAIL:
                succ, marked = curr.next[level].get()
                while marked:
                    curr = succ
                    if curr.kind == _CNode.TAIL:
                        break
                    succ, marked = curr.next[level].get()
                if curr.kind == _CNode.TAIL:
                    break
                if curr.precedes(key):
                    pred, curr = curr, succ
                else:
                    break
        return curr.matches(key)

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Value of a live node holding ``key`` (``default`` if absent)."""
        pred = self._head
        for level in range(self.policy.max_level - 1, -1, -1):
            curr = pred.next[level].ref
            while curr.kind != _CNode.TAIL and curr.precedes(key):
                pred, curr = curr, curr.next[level].ref
            if curr.matches(key) and not curr.next[0].get()[1]:
                return curr.value
        return default

    # ------------------------------------------------------------ inspection
    def keys(self) -> list:
        """Snapshot of live keys (exact only when the list is quiescent)."""
        out = []
        node = self._head.next[0].ref
        while node.kind != _CNode.TAIL:
            succ, marked = node.next[0].get()
            if not marked:
                out.append(node.key)
            node = succ
        return out

    def __len__(self) -> int:
        return len(self.keys())

    def check_invariants(self) -> None:
        """Quiescent-state check: sorted levels, upper levels subset of level 0."""
        base = self.keys()
        assert base == sorted(base) and len(set(base)) == len(base), "level 0 not strictly sorted"
        live = set(base)
        for lv in range(1, self.policy.max_level):
            node, prev = self._head.next[lv].ref, None
            while node.kind != _CNode.TAIL:
                succ, marked = node.next[lv].get()
                if not marked:
                    assert node.key in live, f"level {lv} node missing from level 0"
                    assert prev is None or prev < node.key, f"level {lv} unsorted"
                    prev = node.key
                node = succ
