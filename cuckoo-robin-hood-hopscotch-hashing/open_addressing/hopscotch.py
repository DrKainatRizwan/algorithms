"""Hopscotch hashing.

Every key lives within a fixed *neighbourhood* of ``H`` consecutive slots
starting at its home slot.  Each home slot keeps an ``H``-bit bitmap marking
which slots of its neighbourhood hold keys that hash to it, so a lookup only
inspects those flagged slots.  When the nearest free slot is further than
``H`` away, occupied slots are "hopped" backwards into the free slot until it
falls inside the neighbourhood.
"""

from __future__ import annotations

from typing import Any

from .base import EMPTY, OpenAddressingTable, TableFull
from .hashing import HashFactory, MultiplyShift


class HopscotchTable(OpenAddressingTable):
    """Hopscotch hash table with bitmap neighbourhoods."""

    DEFAULT_MAX_LOAD = 0.9

    def __init__(
        self,
        capacity: int = 64,
        *,
        neighborhood: int = 32,
        max_load: float | None = None,
        hash_factory: HashFactory = MultiplyShift,
        seed: int = 0,
    ) -> None:
        if neighborhood < 2:
            raise ValueError("neighborhood must be at least 2")
        self.H = neighborhood
        self.hops = 0
        super().__init__(
            max(capacity, 2 * neighborhood),
            max_load=max_load,
            hash_factory=hash_factory,
            seed=seed,
        )

    @property
    def capacity(self) -> int:
        return self._cap

    def _reset(self, capacity: int) -> None:
        self._cap = capacity
        self._keys = [EMPTY] * capacity
        self._vals = [None] * capacity
        self._hop = [0] * capacity

    def _locate(self, key: Any) -> tuple[int, int]:
        mask = self._cap - 1
        home = self._slot(key)
        bitmap = self._hop[home]
        probes = 0
        while bitmap:
            low = bitmap & -bitmap
            idx = (home + low.bit_length() - 1) & mask
            probes += 1
            if self._keys[idx] == key:
                return idx, probes
            bitmap ^= low
        return -1, max(probes, 1)

    def _insert(self, key: Any, value: Any) -> None:
        mask = self._cap - 1
        keys, vals, hop = self._keys, self._vals, self._hop
        home = self._slot(key)

        free, dist = home, 0
        while keys[free] is not EMPTY:
            free = (free + 1) & mask
            dist += 1
            if dist >= self._cap:
                raise TableFull

        while dist >= self.H:
            moved = False
            for back in range(self.H - 1, 0, -1):
                b = (free - back) & mask
                bitmap = hop[b]
                if not bitmap:
                    continue
                low = bitmap & -bitmap
                d = low.bit_length() - 1
                if d >= back:  # nearest resident is not before the free slot
                    continue
                src = (b + d) & mask
                keys[free], vals[free] = keys[src], vals[src]
                keys[src], vals[src] = EMPTY, None
                hop[b] = (bitmap & ~low) | (1 << back)
                dist -= back - d
                free = src
                self.hops += 1
                moved = True
                break
            if not moved:
                raise TableFull  # state is still consistent; caller will grow

        keys[free], vals[free] = key, value
        hop[home] |= 1 << dist

    def _remove(self, index: int, key: Any) -> None:
        home = self._slot(key)
        offset = (index - home) & (self._cap - 1)
        self._hop[home] &= ~(1 << offset)
        self._keys[index] = EMPTY
        self._vals[index] = None

    def check_invariants(self) -> None:
        super().check_invariants()
        mask = self._cap - 1
        flagged = 0
        for home, bitmap in enumerate(self._hop):
            assert bitmap >> self.H == 0, "bitmap wider than neighbourhood"
            for d in range(self.H):
                if bitmap >> d & 1:
                    idx = (home + d) & mask
                    k = self._keys[idx]
                    assert k is not EMPTY, "flag on an empty slot"
                    assert self._slot(k) == home, "flag points at foreign key"
                    flagged += 1
        assert flagged == self._size
