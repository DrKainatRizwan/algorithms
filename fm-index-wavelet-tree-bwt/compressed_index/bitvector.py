"""Bit vector with O(1) rank and O(log n) select using block counters."""
from __future__ import annotations
from typing import Sequence

BLOCK = 64


class BitVector:
    """Static bit vector supporting rank and select.

    Bits are packed into 64-bit words; a cumulative popcount is stored per word,
    so rank is one table lookup plus one masked popcount.
    """

    def __init__(self, bits: Sequence[int]) -> None:
        self.n = len(bits)
        nwords = (self.n + BLOCK - 1) // BLOCK
        self.words = [0] * nwords
        for i, b in enumerate(bits):
            if b:
                self.words[i // BLOCK] |= 1 << (i % BLOCK)
        self.cum = [0] * (nwords + 1)
        for w in range(nwords):
            self.cum[w + 1] = self.cum[w] + self.words[w].bit_count()
        self.ones = self.cum[-1]

    def __len__(self) -> int:
        return self.n

    def access(self, i: int) -> int:
        """Return bit ``i``."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        return (self.words[i // BLOCK] >> (i % BLOCK)) & 1

    def rank1(self, i: int) -> int:
        """Number of ones in positions ``[0, i)``."""
        if not 0 <= i <= self.n:
            raise IndexError(i)
        w, r = divmod(i, BLOCK)
        total = self.cum[w]
        if r:
            total += (self.words[w] & ((1 << r) - 1)).bit_count()
        return total

    def rank0(self, i: int) -> int:
        """Number of zeros in positions ``[0, i)``."""
        return i - self.rank1(i)

    def select1(self, k: int) -> int:
        """Position of the ``k``-th one (1-based)."""
        if not 1 <= k <= self.ones:
            raise ValueError("select1 out of range")
        lo, hi = 0, len(self.words)
        while lo < hi:  # first word whose cumulative count >= k
            mid = (lo + hi) // 2
            if self.cum[mid + 1] >= k:
                hi = mid
            else:
                lo = mid + 1
        need = k - self.cum[lo]
        word = self.words[lo]
        for _ in range(need - 1):
            word &= word - 1
        return lo * BLOCK + (word & -word).bit_length() - 1

    def select0(self, k: int) -> int:
        """Position of the ``k``-th zero (1-based)."""
        zeros = self.n - self.ones
        if not 1 <= k <= zeros:
            raise ValueError("select0 out of range")
        lo, hi = 0, self.n
        while lo < hi:  # smallest i with rank0(i+1) >= k
            mid = (lo + hi) // 2
            if self.rank0(mid + 1) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo
