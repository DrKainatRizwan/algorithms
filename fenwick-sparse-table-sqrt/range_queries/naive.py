"""Brute-force reference used for correctness testing and as a benchmark baseline."""
from __future__ import annotations

from typing import Sequence


class NaiveArray:
    """Plain list with O(n) range operations."""

    def __init__(self, values: Sequence[int]) -> None:
        self.a = list(values)

    def __len__(self) -> int:
        return len(self.a)

    def set(self, i: int, v: int) -> None:
        self.a[i] = v

    def add(self, i: int, delta: int) -> None:
        self.a[i] += delta

    def range_add(self, lo: int, hi: int, delta: int) -> None:
        """Add ``delta`` to the half-open interval [lo, hi)."""
        for i in range(lo, hi):
            self.a[i] += delta

    def sum(self, lo: int, hi: int) -> int:
        return sum(self.a[lo:hi])

    def min(self, lo: int, hi: int) -> int:
        return min(self.a[lo:hi])

    def max(self, lo: int, hi: int) -> int:
        return max(self.a[lo:hi])
