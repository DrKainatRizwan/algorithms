"""HyperLogLog cardinality estimator (Flajolet et al. 2007) with small-range correction."""

from __future__ import annotations

import math
from typing import Hashable, Iterable

import numpy as np

from .hashing import hash64


class HyperLogLog:
    """Estimate distinct counts with 2^p registers; relative error ~ 1.04/sqrt(2^p)."""

    def __init__(self, p: int = 12, seed: int = 0) -> None:
        if not 4 <= p <= 18:
            raise ValueError("p must be in [4, 18]")
        self.p = p
        self.m = 1 << p
        self.seed = seed
        self.registers = np.zeros(self.m, dtype=np.uint8)

    @property
    def alpha(self) -> float:
        if self.m == 16:
            return 0.673
        if self.m == 32:
            return 0.697
        if self.m == 64:
            return 0.709
        return 0.7213 / (1 + 1.079 / self.m)

    def add(self, item: Hashable) -> None:
        h = hash64(item, self.seed)
        idx = h >> (64 - self.p)
        rest = h & ((1 << (64 - self.p)) - 1)
        # rank = position of the leftmost 1-bit in the remaining 64-p bits
        rank = (64 - self.p) - rest.bit_length() + 1
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def update(self, items: Iterable[Hashable]) -> None:
        for it in items:
            self.add(it)

    def raw_estimate(self) -> float:
        return self.alpha * self.m * self.m / float(np.sum(2.0 ** -self.registers.astype(np.float64)))

    def estimate(self) -> float:
        """Cardinality estimate; linear counting when many registers are empty."""
        e = self.raw_estimate()
        zeros = int(np.count_nonzero(self.registers == 0))
        if e <= 2.5 * self.m and zeros:
            return self.m * math.log(self.m / zeros)
        return e

    def __len__(self) -> int:
        return int(round(self.estimate()))

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Union sketch: register-wise maximum."""
        if (self.p, self.seed) != (other.p, other.seed):
            raise ValueError("incompatible HyperLogLog sketches")
        out = HyperLogLog(self.p, self.seed)
        out.registers = np.maximum(self.registers, other.registers)
        return out

    def intersection_estimate(self, other: "HyperLogLog") -> float:
        """Inclusion-exclusion: |A n B| = |A| + |B| - |A u B| (noisy for small overlaps)."""
        return max(0.0, self.estimate() + other.estimate() - self.merge(other).estimate())

    def standard_error(self) -> float:
        return 1.04 / math.sqrt(self.m)

    @property
    def size_bytes(self) -> int:
        return int(self.registers.nbytes)
