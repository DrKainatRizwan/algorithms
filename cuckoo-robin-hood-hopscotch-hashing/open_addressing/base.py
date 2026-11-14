"""Shared machinery for all open-addressing tables.

:class:`OpenAddressingTable` is a ``MutableMapping`` that handles size
tracking, automatic growth, probe statistics and the public mapping API.
Concrete schemes implement slot-level ``_locate``, ``_insert`` and
``_remove`` only.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, MutableMapping

from .hashing import HashFactory, HashFunction, MultiplyShift, key_to_int


class _Sentinel:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


EMPTY = _Sentinel("EMPTY")
TOMBSTONE = _Sentinel("TOMBSTONE")


class TableFull(Exception):
    """Raised by ``_insert`` when a scheme cannot place a key at this capacity."""


@dataclass
class ProbeStats:
    """Running counters of slot inspections performed by lookups."""

    hit_probes: int = 0
    hit_count: int = 0
    miss_probes: int = 0
    miss_count: int = 0
    max_probes: int = 0

    def record(self, probes: int, hit: bool) -> None:
        if hit:
            self.hit_probes += probes
            self.hit_count += 1
        else:
            self.miss_probes += probes
            self.miss_count += 1
        if probes > self.max_probes:
            self.max_probes = probes

    @property
    def mean_hit(self) -> float:
        return self.hit_probes / self.hit_count if self.hit_count else 0.0

    @property
    def mean_miss(self) -> float:
        return self.miss_probes / self.miss_count if self.miss_count else 0.0

    def reset(self) -> None:
        self.hit_probes = self.hit_count = self.miss_probes = self.miss_count = 0
        self.max_probes = 0


class OpenAddressingTable(MutableMapping, ABC):
    """Abstract open-addressing hash table with power-of-two capacity."""

    DEFAULT_MAX_LOAD = 0.7
    MIN_CAPACITY = 8

    def __init__(
        self,
        capacity: int = 16,
        *,
        max_load: float | None = None,
        hash_factory: HashFactory = MultiplyShift,
        seed: int = 0,
    ) -> None:
        load = self.DEFAULT_MAX_LOAD if max_load is None else max_load
        if not 0.0 < load < 1.0:
            raise ValueError("max_load must lie strictly between 0 and 1")
        cap = self.MIN_CAPACITY
        while cap < capacity:
            cap <<= 1
        self.max_load = load
        self.seed = seed
        self.hash_factory = hash_factory
        self.stats = ProbeStats()
        self.resizes = 0
        self._rng = random.Random(seed)
        self._size = 0
        self._keys: list[Any] = []
        self._vals: list[Any] = []
        self._init_hash()
        self._reset(cap)

    # ------------------------------------------------------------------ hooks
    def _init_hash(self) -> None:
        self._hash: HashFunction = self.hash_factory(self._rng.getrandbits(64))

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Number of physical slots."""

    @abstractmethod
    def _reset(self, capacity: int) -> None:
        """Discard all contents and allocate storage for ``capacity`` slots."""

    @abstractmethod
    def _locate(self, key: Any) -> tuple[int, int]:
        """Return ``(slot index or -1, probes used)`` for ``key``."""

    @abstractmethod
    def _insert(self, key: Any, value: Any) -> None:
        """Insert a key known to be absent; may raise :class:`TableFull`."""

    @abstractmethod
    def _remove(self, index: int, key: Any) -> None:
        """Delete the entry stored at slot ``index``."""

    def check_invariants(self) -> None:
        """Raise ``AssertionError`` if the structure is corrupt."""
        live = sum(1 for k in self._keys if k is not EMPTY and k is not TOMBSTONE)
        assert live == self._size, f"size {self._size} != live slots {live}"

    def _occupied(self) -> int:
        """Slots counted against the load limit (live keys plus tombstones)."""
        return self._size

    def _next_capacity(self) -> int:
        return self.capacity * 2

    # --------------------------------------------------------------- internals
    def _slot(self, key: Any, h: HashFunction | None = None) -> int:
        bits = self.capacity.bit_length() - 1
        return (h or self._hash).index(key_to_int(key), bits)

    def _items(self) -> Iterator[tuple[Any, Any]]:
        for k, v in zip(self._keys, self._vals):
            if k is not EMPTY and k is not TOMBSTONE:
                yield k, v

    def _resize(self, capacity: int) -> None:
        items = list(self._items())
        while True:
            self._reset(capacity)
            self._size = 0
            try:
                for k, v in items:
                    self._insert(k, v)
                    self._size += 1
                break
            except TableFull:
                capacity *= 2
        self.resizes += 1

    # -------------------------------------------------------------- public API
    @property
    def load_factor(self) -> float:
        return self._size / self.capacity

    def probe_count(self, key: Any) -> int:
        """Probes a lookup of ``key`` would take (does not touch ``stats``)."""
        return self._locate(key)[1]

    def __getitem__(self, key: Any) -> Any:
        idx, probes = self._locate(key)
        self.stats.record(probes, idx >= 0)
        if idx < 0:
            raise KeyError(key)
        return self._vals[idx]

    def __contains__(self, key: object) -> bool:
        idx, probes = self._locate(key)
        self.stats.record(probes, idx >= 0)
        return idx >= 0

    def __setitem__(self, key: Any, value: Any) -> None:
        idx, _ = self._locate(key)
        if idx >= 0:
            self._vals[idx] = value
            return
        while True:
            if self._occupied() + 1 > self.max_load * self.capacity:
                self._resize(self._next_capacity())
                continue
            try:
                self._insert(key, value)
                break
            except TableFull:
                self._resize(self._next_capacity())
        self._size += 1

    def __delitem__(self, key: Any) -> None:
        idx, _ = self._locate(key)
        if idx < 0:
            raise KeyError(key)
        self._remove(idx, key)
        self._size -= 1

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        for k, _ in self._items():
            yield k

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(size={self._size}, capacity={self.capacity}, "
            f"load={self.load_factor:.2f})"
        )
