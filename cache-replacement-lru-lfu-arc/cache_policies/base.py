"""Common interface and statistics for cache replacement policies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Iterator


@dataclass
class CacheStats:
    """Hit/miss/eviction counters for a simulation run."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate if self.requests else 0.0


class CachePolicy(ABC):
    """A fixed-capacity cache that stores keys only (hit/miss simulation).

    Subclasses implement :meth:`_access`, which returns ``True`` on a hit and
    must keep ``len(self) <= capacity`` after every call.  Evictions are
    reported through :meth:`_record_eviction`.
    """

    name = "base"

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.stats = CacheStats()
        self.last_evicted: list[Hashable] = []

    def access(self, key: Hashable) -> bool:
        """Request ``key``; returns True if it was resident (a hit)."""
        self.last_evicted = []
        hit = self._access(key)
        if hit:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        return hit

    def run(self, trace: Iterable[Hashable]) -> CacheStats:
        """Feed a whole trace through the cache and return the statistics."""
        for key in trace:
            self.access(key)
        return self.stats

    def _record_eviction(self, key: Hashable) -> None:
        self.stats.evictions += 1
        self.last_evicted.append(key)

    @abstractmethod
    def _access(self, key: Hashable) -> bool: ...

    @abstractmethod
    def __contains__(self, key: object) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def keys(self) -> Iterator[Hashable]:
        """Iterate over resident keys (order is policy specific)."""

    def reset(self) -> None:
        """Empty the cache and clear statistics."""
        self.__init__(self.capacity)  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(capacity={self.capacity}, size={len(self)})"
