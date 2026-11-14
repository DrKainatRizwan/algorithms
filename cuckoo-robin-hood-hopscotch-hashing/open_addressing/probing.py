"""Linear probing (with tombstones) and Robin Hood hashing.

Both walk forward from the home slot.  Robin Hood additionally stores each
key's displacement from its home slot and lets "poorer" keys (those further
from home) steal slots from "richer" ones, which flattens the probe-length
distribution and allows early termination of unsuccessful searches.
"""

from __future__ import annotations

from typing import Any

from .base import EMPTY, TOMBSTONE, OpenAddressingTable


class LinearProbingTable(OpenAddressingTable):
    """Classic linear probing; deletions leave tombstones.

    Tombstones count towards the load limit, so a table that has seen heavy
    churn is rebuilt (at the same or doubled capacity) once they pile up.
    """

    DEFAULT_MAX_LOAD = 0.7

    def _reset(self, capacity: int) -> None:
        self._cap = capacity
        self._keys = [EMPTY] * capacity
        self._vals = [None] * capacity
        self._tombs = 0

    @property
    def capacity(self) -> int:
        return self._cap

    def _occupied(self) -> int:
        return self._size + self._tombs

    def _next_capacity(self) -> int:
        # Mostly tombstones: rebuild in place instead of growing.
        if self._size + 1 <= self.max_load * self._cap * 0.5:
            return self._cap
        return self._cap * 2

    def _locate(self, key: Any) -> tuple[int, int]:
        mask = self._cap - 1
        i = self._slot(key)
        keys = self._keys
        probes = 0
        for _ in range(self._cap):
            probes += 1
            k = keys[i]
            if k is EMPTY:
                return -1, probes
            if k is not TOMBSTONE and k == key:
                return i, probes
            i = (i + 1) & mask
        return -1, probes

    def _insert(self, key: Any, value: Any) -> None:
        mask = self._cap - 1
        i = self._slot(key)
        while self._keys[i] is not EMPTY and self._keys[i] is not TOMBSTONE:
            i = (i + 1) & mask
        if self._keys[i] is TOMBSTONE:
            self._tombs -= 1
        self._keys[i] = key
        self._vals[i] = value

    def _remove(self, index: int, key: Any) -> None:
        self._keys[index] = TOMBSTONE
        self._vals[index] = None
        self._tombs += 1

    def check_invariants(self) -> None:
        super().check_invariants()
        assert self._tombs == sum(1 for k in self._keys if k is TOMBSTONE)
        assert any(k is EMPTY for k in self._keys), "probe loops need an empty slot"
        for k, _ in self._items():
            assert self._locate(k)[0] >= 0, f"key {k!r} unreachable"


class RobinHoodTable(OpenAddressingTable):
    """Robin Hood hashing with backward-shift deletion (no tombstones)."""

    DEFAULT_MAX_LOAD = 0.9

    def _reset(self, capacity: int) -> None:
        self._cap = capacity
        self._keys = [EMPTY] * capacity
        self._vals = [None] * capacity
        self._dist = [-1] * capacity  # displacement from home slot; -1 = empty

    @property
    def capacity(self) -> int:
        return self._cap

    def _locate(self, key: Any) -> tuple[int, int]:
        mask = self._cap - 1
        i = self._slot(key)
        dist = self._dist
        keys = self._keys
        d = 0
        probes = 0
        while True:
            probes += 1
            # An empty slot (-1) or a key closer to home than we are proves absence.
            if dist[i] < d:
                return -1, probes
            if keys[i] == key:
                return i, probes
            i = (i + 1) & mask
            d += 1

    def _insert(self, key: Any, value: Any) -> None:
        mask = self._cap - 1
        i = self._slot(key)
        d = 0
        keys, vals, dist = self._keys, self._vals, self._dist
        while True:
            if keys[i] is EMPTY:
                keys[i], vals[i], dist[i] = key, value, d
                return
            if dist[i] < d:  # steal from the rich
                keys[i], key = key, keys[i]
                vals[i], value = value, vals[i]
                dist[i], d = d, dist[i]
            i = (i + 1) & mask
            d += 1

    def _remove(self, index: int, key: Any) -> None:
        mask = self._cap - 1
        keys, vals, dist = self._keys, self._vals, self._dist
        i = index
        j = (i + 1) & mask
        while dist[j] > 0:  # shift the cluster tail back by one
            keys[i], vals[i], dist[i] = keys[j], vals[j], dist[j] - 1
            i = j
            j = (j + 1) & mask
        keys[i], vals[i], dist[i] = EMPTY, None, -1

    # ----------------------------------------------------------- diagnostics
    def displacements(self) -> list[int]:
        """Displacement from the home slot of every stored key."""
        return [d for d in self._dist if d >= 0]

    def max_displacement(self) -> int:
        return max(self.displacements(), default=0)

    def displacement_variance(self) -> float:
        ds = self.displacements()
        if not ds:
            return 0.0
        mean = sum(ds) / len(ds)
        return sum((x - mean) ** 2 for x in ds) / len(ds)

    def check_invariants(self) -> None:
        super().check_invariants()
        mask = self._cap - 1
        for i, k in enumerate(self._keys):
            if k is EMPTY:
                assert self._dist[i] == -1
                continue
            assert self._dist[i] >= 0
            assert (self._slot(k) + self._dist[i]) & mask == i, "stale displacement"
            # Robin Hood property: the next slot's displacement grows by at most 1.
            nxt = (i + 1) & mask
            assert self._dist[nxt] <= self._dist[i] + 1, "Robin Hood order violated"
