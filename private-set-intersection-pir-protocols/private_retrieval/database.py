"""Fixed-length record database used by all PIR schemes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Database:
    """``n`` records of ``L`` bytes each, stored as an ``(n, L)`` uint8 array."""

    data: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.dtype != np.uint8:
            raise ValueError("data must be a 2-D uint8 array")

    @property
    def n(self) -> int:
        return self.data.shape[0]

    @property
    def record_len(self) -> int:
        return self.data.shape[1]

    @classmethod
    def random(cls, n: int, record_len: int, seed: int = 0) -> "Database":
        """Seeded random database."""
        rng = np.random.default_rng(seed)
        return cls(rng.integers(0, 256, size=(n, record_len), dtype=np.uint8))

    @classmethod
    def from_records(cls, records: Sequence[bytes]) -> "Database":
        """Build from byte strings, zero-padding to the longest record."""
        L = max((len(r) for r in records), default=1)
        arr = np.zeros((len(records), L), dtype=np.uint8)
        for i, r in enumerate(records):
            arr[i, : len(r)] = np.frombuffer(r, dtype=np.uint8)
        return cls(arr)

    def record(self, i: int) -> bytes:
        """Record ``i`` as bytes."""
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.data[i].tobytes()

    def padded(self, size: int) -> np.ndarray:
        """Data zero-padded with extra records to ``size`` rows."""
        if size < self.n:
            raise ValueError("size smaller than database")
        out = np.zeros((size, self.record_len), dtype=np.uint8)
        out[: self.n] = self.data
        return out
