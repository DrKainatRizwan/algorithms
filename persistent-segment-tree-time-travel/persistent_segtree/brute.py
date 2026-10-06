"""Naive reference implementations used for validation and benchmarking."""
from __future__ import annotations

from functools import reduce
from typing import Any, Sequence

from .monoid import Monoid, SUM


class SnapshotArray:
    """Stores a full copy per version: O(n) per update, O(r-l) per query."""

    def __init__(self, values: Sequence[Any], monoid: Monoid = SUM) -> None:
        self.monoid = monoid
        self.snaps = [list(values)]

    def update(self, version: int, index: int, value: Any) -> int:
        s = list(self.snaps[version])
        s[index] = value
        self.snaps.append(s)
        return len(self.snaps) - 1

    def query(self, version: int, left: int, right: int) -> Any:
        return reduce(self.monoid.combine, self.snaps[version][left:right],
                      self.monoid.identity)

    def kth(self, left: int, right: int, k: int) -> int:
        return sorted(self.snaps[0][left:right])[k - 1]
