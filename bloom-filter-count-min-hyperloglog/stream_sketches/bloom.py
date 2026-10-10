"""Bloom filter variants: standard, counting (deletable) and scalable."""

from __future__ import annotations

import math
from typing import Hashable, Iterable

import numpy as np

from .hashing import double_hash_indices


def optimal_params(capacity: int, fp_rate: float) -> tuple[int, int]:
    """Return (m bits, k hashes) minimising memory for a target false-positive rate."""
    if capacity <= 0 or not 0 < fp_rate < 1:
        raise ValueError("capacity must be positive and fp_rate in (0, 1)")
    m = math.ceil(-capacity * math.log(fp_rate) / (math.log(2) ** 2))
    k = max(1, round(m / capacity * math.log(2)))
    return m, k


class BloomFilter:
    """Classic Bloom filter over a packed bit array (no false negatives)."""

    def __init__(self, capacity: int, fp_rate: float = 0.01, seed: int = 0) -> None:
        self.capacity = capacity
        self.fp_rate = fp_rate
        self.m, self.k = optimal_params(capacity, fp_rate)
        self.seed = seed
        self._bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)
        self.count = 0  # insertions (approximate for duplicates)

    def _indices(self, item: Hashable) -> list[int]:
        return double_hash_indices(item, self.k, self.m, self.seed)

    def add(self, item: Hashable) -> bool:
        """Insert item; return True if it was definitely new."""
        new = False
        for i in self._indices(item):
            byte, bit = i >> 3, 1 << (i & 7)
            if not self._bits[byte] & bit:
                self._bits[byte] |= bit
                new = True
        if new:
            self.count += 1
        return new

    def update(self, items: Iterable[Hashable]) -> None:
        for it in items:
            self.add(it)

    def __contains__(self, item: Hashable) -> bool:
        return all(self._bits[i >> 3] & (1 << (i & 7)) for i in self._indices(item))

    def bits_set(self) -> int:
        return int(np.unpackbits(self._bits)[: self.m].sum())

    def fill_ratio(self) -> float:
        return self.bits_set() / self.m

    def expected_fp_rate(self, n: int | None = None) -> float:
        """Theoretical FP rate (1 - e^{-kn/m})^k after n insertions."""
        n = self.count if n is None else n
        return (1.0 - math.exp(-self.k * n / self.m)) ** self.k

    def estimate_cardinality(self) -> float:
        """Swamidass-Baldi estimate n ~ -(m/k) ln(1 - X/m) from the fill ratio."""
        x = self.bits_set()
        if x >= self.m:
            return float("inf")
        return -(self.m / self.k) * math.log(1.0 - x / self.m)

    def union(self, other: "BloomFilter") -> "BloomFilter":
        self._check(other)
        out = BloomFilter(self.capacity, self.fp_rate, self.seed)
        out._bits = self._bits | other._bits
        out.count = self.count + other.count
        return out

    def intersection_bits(self, other: "BloomFilter") -> "BloomFilter":
        self._check(other)
        out = BloomFilter(self.capacity, self.fp_rate, self.seed)
        out._bits = self._bits & other._bits
        return out

    def _check(self, other: "BloomFilter") -> None:
        if (self.m, self.k, self.seed) != (other.m, other.k, other.seed):
            raise ValueError("incompatible Bloom filters")

    @property
    def size_bytes(self) -> int:
        return int(self._bits.nbytes)


class CountingBloomFilter:
    """Bloom filter with 8-bit saturating counters, supporting deletion."""

    def __init__(self, capacity: int, fp_rate: float = 0.01, seed: int = 0) -> None:
        self.m, self.k = optimal_params(capacity, fp_rate)
        self.seed = seed
        self._cells = np.zeros(self.m, dtype=np.uint8)

    def _indices(self, item: Hashable) -> list[int]:
        return double_hash_indices(item, self.k, self.m, self.seed)

    def add(self, item: Hashable) -> None:
        for i in self._indices(item):
            if self._cells[i] < 255:  # saturate; saturated cells are never decremented
                self._cells[i] += 1

    def remove(self, item: Hashable) -> bool:
        """Remove item if (probably) present. Returns False if definitely absent."""
        idx = self._indices(item)
        if any(self._cells[i] == 0 for i in idx):
            return False
        for i in idx:
            if self._cells[i] < 255:
                self._cells[i] -= 1
        return True

    def __contains__(self, item: Hashable) -> bool:
        return all(self._cells[i] > 0 for i in self._indices(item))

    def min_count(self, item: Hashable) -> int:
        """Upper bound on the multiplicity of item."""
        return int(min(self._cells[i] for i in self._indices(item)))


class ScalableBloomFilter:
    """Bloom filter that grows by chaining slices with tightening FP rates.

    Slice i has capacity c0 * growth^i and FP rate p0 * r^i, so the total
    FP rate is bounded by p0 / (1 - r) (Almeida et al., 2007).
    """

    def __init__(self, initial_capacity: int = 128, fp_rate: float = 0.01,
                 growth: int = 2, ratio: float = 0.5, seed: int = 0) -> None:
        if not 0 < ratio < 1:
            raise ValueError("ratio must be in (0, 1)")
        self.initial_capacity = initial_capacity
        self.p0 = fp_rate * (1 - ratio)  # so that sum p_i <= fp_rate
        self.growth = growth
        self.ratio = ratio
        self.seed = seed
        self.slices: list[BloomFilter] = []
        self._add_slice()

    def _add_slice(self) -> None:
        i = len(self.slices)
        cap = self.initial_capacity * self.growth ** i
        self.slices.append(BloomFilter(cap, self.p0 * self.ratio ** i, self.seed + i))

    def add(self, item: Hashable) -> None:
        if item in self:
            return
        last = self.slices[-1]
        if last.count >= last.capacity:
            self._add_slice()
            last = self.slices[-1]
        last.add(item)

    def __contains__(self, item: Hashable) -> bool:
        return any(item in s for s in self.slices)

    def __len__(self) -> int:
        return sum(s.count for s in self.slices)

    @property
    def size_bytes(self) -> int:
        return sum(s.size_bytes for s in self.slices)
