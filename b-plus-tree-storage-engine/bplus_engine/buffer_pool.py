"""LRU buffer pool simulation used to measure page-cache behaviour.

The tree records the ids of the nodes it visits (``tree.trace``).  Replaying
that trace through a pool of ``capacity`` frames gives the hit rate and the
number of simulated disk reads.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List

from .tree import BPlusTree


class BufferPool:
    """Fixed-size LRU cache of page ids."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._frames: "OrderedDict[int, None]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, page_id: int) -> bool:
        """Touch a page; returns True on a hit."""
        if page_id in self._frames:
            self._frames.move_to_end(page_id)
            self.hits += 1
            return True
        self.misses += 1
        self._frames[page_id] = None
        if len(self._frames) > self.capacity:
            self._frames.popitem(last=False)
        return False

    def replay(self, trace: Iterable[int]) -> None:
        for p in trace:
            self.access(p)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def record_trace(tree: BPlusTree, operation) -> List[int]:
    """Run ``operation(tree)`` and return the page ids it touched."""
    tree.trace = []
    try:
        operation(tree)
        return tree.trace
    finally:
        tree.trace = None
