"""Classic policies: FIFO, Random, CLOCK, LRU and O(1) LFU."""
from __future__ import annotations

import random
from collections import OrderedDict
from typing import Hashable, Iterator

from .base import CachePolicy


class FIFOCache(CachePolicy):
    """Evict the oldest inserted key; hits do not change order."""

    name = "FIFO"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._q: OrderedDict[Hashable, None] = OrderedDict()

    def _access(self, key: Hashable) -> bool:
        if key in self._q:
            return True
        if len(self._q) >= self.capacity:
            self._record_eviction(self._q.popitem(last=False)[0])
        self._q[key] = None
        return False

    def __contains__(self, key: object) -> bool:
        return key in self._q

    def __len__(self) -> int:
        return len(self._q)

    def keys(self) -> Iterator[Hashable]:
        return iter(self._q)


class RandomCache(CachePolicy):
    """Evict a uniformly random resident key (seeded for reproducibility)."""

    name = "Random"

    def __init__(self, capacity: int, seed: int = 0) -> None:
        super().__init__(capacity)
        self._seed = seed
        self._rng = random.Random(seed)
        self._items: list[Hashable] = []
        self._pos: dict[Hashable, int] = {}

    def reset(self) -> None:
        self.__init__(self.capacity, self._seed)  # type: ignore[misc]

    def _access(self, key: Hashable) -> bool:
        if key in self._pos:
            return True
        if len(self._items) >= self.capacity:
            i = self._rng.randrange(len(self._items))
            victim = self._items[i]
            last = self._items.pop()
            if i < len(self._items):
                self._items[i] = last
                self._pos[last] = i
            del self._pos[victim]
            self._record_eviction(victim)
        self._pos[key] = len(self._items)
        self._items.append(key)
        return False

    def __contains__(self, key: object) -> bool:
        return key in self._pos

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> Iterator[Hashable]:
        return iter(self._items)


class ClockCache(CachePolicy):
    """Second-chance (CLOCK) approximation of LRU using a reference bit ring."""

    name = "CLOCK"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._slots: list[Hashable | None] = [None] * capacity
        self._ref: list[bool] = [False] * capacity
        self._pos: dict[Hashable, int] = {}
        self._hand = 0

    def _access(self, key: Hashable) -> bool:
        i = self._pos.get(key)
        if i is not None:
            self._ref[i] = True
            return True
        while True:
            victim = self._slots[self._hand]
            if victim is None or not self._ref[self._hand]:
                break
            self._ref[self._hand] = False
            self._hand = (self._hand + 1) % self.capacity
        if victim is not None:
            del self._pos[victim]
            self._record_eviction(victim)
        self._slots[self._hand] = key
        self._ref[self._hand] = False
        self._pos[key] = self._hand
        self._hand = (self._hand + 1) % self.capacity
        return False

    def __contains__(self, key: object) -> bool:
        return key in self._pos

    def __len__(self) -> int:
        return len(self._pos)

    def keys(self) -> Iterator[Hashable]:
        return iter(self._pos)


class LRUCache(CachePolicy):
    """Least-recently-used eviction with O(1) operations (hash + linked list)."""

    name = "LRU"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._d: OrderedDict[Hashable, None] = OrderedDict()

    def _access(self, key: Hashable) -> bool:
        if key in self._d:
            self._d.move_to_end(key)
            return True
        if len(self._d) >= self.capacity:
            self._record_eviction(self._d.popitem(last=False)[0])
        self._d[key] = None
        return False

    def __contains__(self, key: object) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)

    def keys(self) -> Iterator[Hashable]:
        """Keys from least to most recently used."""
        return iter(self._d)


class LFUCache(CachePolicy):
    """O(1) least-frequently-used cache (Shah, Mitra & Matani, 2010).

    Keys live in per-frequency buckets; ``_min_freq`` tracks the lowest
    non-empty bucket.  Ties inside a bucket are broken by recency (LRU).
    With ``aging=True`` all counts are halved every ``aging_period`` requests
    so that formerly popular keys can eventually be displaced.
    """

    name = "LFU"

    def __init__(self, capacity: int, aging: bool = False, aging_period: int = 0) -> None:
        super().__init__(capacity)
        self._aging = aging
        self._period = aging_period or 10 * capacity
        self._freq: dict[Hashable, int] = {}
        self._buckets: dict[int, OrderedDict[Hashable, None]] = {}
        self._min_freq = 0
        self._ticks = 0

    def reset(self) -> None:
        self.__init__(self.capacity, self._aging, self._period)  # type: ignore[misc]

    def frequency(self, key: Hashable) -> int:
        """Current reference count of a resident key (0 if absent)."""
        return self._freq.get(key, 0)

    def _bump(self, key: Hashable) -> None:
        f = self._freq[key]
        bucket = self._buckets[f]
        del bucket[key]
        if not bucket:
            del self._buckets[f]
            if self._min_freq == f:
                self._min_freq = f + 1
        self._freq[key] = f + 1
        self._buckets.setdefault(f + 1, OrderedDict())[key] = None

    def _age(self) -> None:
        """Halve every count, preserving the recency order within buckets."""
        order = [k for f in sorted(self._buckets) for k in self._buckets[f]]
        self._buckets = {}
        for k in order:
            self._freq[k] = max(1, self._freq[k] // 2)
            self._buckets.setdefault(self._freq[k], OrderedDict())[k] = None
        self._min_freq = min(self._buckets) if self._buckets else 0

    def _access(self, key: Hashable) -> bool:
        self._ticks += 1
        if self._aging and self._ticks % self._period == 0:
            self._age()
        if key in self._freq:
            self._bump(key)
            return True
        if len(self._freq) >= self.capacity:
            bucket = self._buckets[self._min_freq]
            victim, _ = bucket.popitem(last=False)
            if not bucket:
                del self._buckets[self._min_freq]
            del self._freq[victim]
            self._record_eviction(victim)
        self._freq[key] = 1
        self._buckets.setdefault(1, OrderedDict())[key] = None
        self._min_freq = 1
        return False

    def __contains__(self, key: object) -> bool:
        return key in self._freq

    def __len__(self) -> int:
        return len(self._freq)

    def keys(self) -> Iterator[Hashable]:
        return iter(self._freq)
