"""Level-assignment policies for skip list towers.

A node's level (tower height) is drawn from a geometric distribution:
P(level >= k) = p**(k-1).  Two policies are provided: an RNG-driven one and a
key-hash-driven one that makes the structure history independent.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Hashable

_MASK = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """Finalising mixer of the SplitMix64 generator (a good 64-bit integer hash)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def stable_hash(key: Hashable) -> int:
    """Deterministic 64-bit hash of ints and strings (independent of PYTHONHASHSEED)."""
    if isinstance(key, int):
        return splitmix64(key & _MASK)
    if isinstance(key, str):
        h = 0xCBF29CE484222325
        for byte in key.encode("utf-8"):
            h = ((h ^ byte) * 0x100000001B3) & _MASK
        return splitmix64(h)
    raise TypeError(f"HashLevels supports int and str keys, got {type(key).__name__}")


class LevelPolicy(ABC):
    """Strategy returning a tower height in ``[1, max_level]``."""

    def __init__(self, p: float = 0.5, max_level: int = 32) -> None:
        if not 0.0 < p < 1.0:
            raise ValueError("p must lie in (0, 1)")
        if max_level < 1:
            raise ValueError("max_level must be >= 1")
        self.p = p
        self.max_level = max_level

    @abstractmethod
    def level_for(self, key: Hashable) -> int:
        """Return the tower height for ``key``."""


class GeometricLevels(LevelPolicy):
    """Independent geometric levels drawn from a seeded RNG."""

    def __init__(self, p: float = 0.5, max_level: int = 32, seed: int | None = 0) -> None:
        super().__init__(p, max_level)
        self._rng = random.Random(seed)

    def level_for(self, key: Hashable = None) -> int:
        level = 1
        while level < self.max_level and self._rng.random() < self.p:
            level += 1
        return level


class HashLevels(LevelPolicy):
    """Levels derived from a hash of the key.

    The same key always receives the same tower, so the final shape of the
    list does not depend on the order of insertions and deletions.
    """

    def __init__(self, p: float = 0.5, max_level: int = 32, salt: int = 0) -> None:
        super().__init__(p, max_level)
        self.salt = salt

    def level_for(self, key: Hashable) -> int:
        h = stable_hash(key)
        level = 1
        while level < self.max_level:
            h = splitmix64(h ^ self.salt)
            if (h >> 11) / float(1 << 53) >= self.p:
                break
            level += 1
        return level
