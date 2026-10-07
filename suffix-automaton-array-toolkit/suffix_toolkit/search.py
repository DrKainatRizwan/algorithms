"""Pattern search over a suffix array by binary search."""
from __future__ import annotations

from .suffix_array import build_suffix_array, kasai_lcp


class SuffixArrayIndex:
    """Index supporting substring queries in O(m log n)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.sa = build_suffix_array(text)
        self.lcp = kasai_lcp(text, self.sa)

    def _bound(self, pattern: str, upper: bool) -> int:
        lo, hi = 0, len(self.sa)
        m = len(pattern)
        while lo < hi:
            mid = (lo + hi) // 2
            head = self.text[self.sa[mid]:self.sa[mid] + m]
            if head < pattern or (upper and head == pattern):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def occurrence_range(self, pattern: str) -> tuple[int, int]:
        """Half-open range of suffix-array rows starting with ``pattern``."""
        return self._bound(pattern, False), self._bound(pattern, True)

    def count(self, pattern: str) -> int:
        lo, hi = self.occurrence_range(pattern)
        return hi - lo

    def find_all(self, pattern: str) -> list[int]:
        """Sorted start positions of every occurrence."""
        lo, hi = self.occurrence_range(pattern)
        return sorted(self.sa[lo:hi])

    def contains(self, pattern: str) -> bool:
        return self.count(pattern) > 0
