"""Bucketised cuckoo hashing.

Each key may live in one of two buckets (given by two independent hash
functions), each holding ``bucket_size`` slots.  Lookups therefore touch at
most ``2 * bucket_size`` slots -- a worst-case constant.  Insertion evicts a
random resident when both buckets are full, and the evicted key moves to its
alternate bucket (a random walk).  If the walk exceeds ``max_kicks`` the whole
table is rehashed with fresh hash functions, and enlarged if that keeps
failing.
"""

from __future__ import annotations

from typing import Any

from .base import EMPTY, OpenAddressingTable
from .hashing import HashFactory, MultiplyShift

# Safe operating loads, below the theoretical thresholds
# (about 0.5, 0.897, 0.977 for bucket sizes 1, 2, 4).
_LOAD_BY_BUCKET = {1: 0.45, 2: 0.80, 4: 0.90, 8: 0.94}


class CuckooTable(OpenAddressingTable):
    """Two-choice bucketised cuckoo hash table."""

    MIN_CAPACITY = 8
    REHASH_ATTEMPTS = 4

    def __init__(
        self,
        capacity: int = 16,
        *,
        bucket_size: int = 1,
        max_kicks: int = 500,
        max_load: float | None = None,
        hash_factory: HashFactory = MultiplyShift,
        seed: int = 0,
    ) -> None:
        if bucket_size < 1 or bucket_size & (bucket_size - 1):
            raise ValueError("bucket_size must be a power of two")
        self.bucket_size = bucket_size
        self.max_kicks = max_kicks
        self.kicks = 0
        self.rehashes = 0
        if max_load is None:
            max_load = _LOAD_BY_BUCKET.get(bucket_size, 0.9)
        super().__init__(
            max(capacity, 2 * bucket_size),
            max_load=max_load,
            hash_factory=hash_factory,
            seed=seed,
        )

    # -------------------------------------------------------------- hashing
    def _init_hash(self) -> None:
        self._h1 = self.hash_factory(self._rng.getrandbits(64))
        self._h2 = self.hash_factory(self._rng.getrandbits(64))

    def _buckets(self, key: Any) -> tuple[int, int]:
        from .hashing import key_to_int

        x = key_to_int(key)
        bits = self._nb.bit_length() - 1
        return self._h1.index(x, bits), self._h2.index(x, bits)

    # ------------------------------------------------------------- storage
    @property
    def capacity(self) -> int:
        return self._nb * self.bucket_size

    def _reset(self, capacity: int) -> None:
        nb = 2
        while nb * self.bucket_size < capacity:
            nb <<= 1
        self._nb = nb
        n = nb * self.bucket_size
        self._keys = [EMPTY] * n
        self._vals = [None] * n

    def _locate(self, key: Any) -> tuple[int, int]:
        b1, b2 = self._buckets(key)
        bs = self.bucket_size
        probes = 0
        for b in (b1,) if b1 == b2 else (b1, b2):
            for idx in range(b * bs, (b + 1) * bs):
                probes += 1
                k = self._keys[idx]
                if k is not EMPTY and k == key:
                    return idx, probes
        return -1, probes

    def _free_slot(self, bucket: int) -> int:
        bs = self.bucket_size
        for idx in range(bucket * bs, (bucket + 1) * bs):
            if self._keys[idx] is EMPTY:
                return idx
        return -1

    def _try_place(self, key: Any, value: Any) -> tuple[Any, Any] | None:
        """Random-walk insertion.  Returns the homeless pair on failure."""
        bs = self.bucket_size
        last = -1
        for _ in range(self.max_kicks):
            b1, b2 = self._buckets(key)
            for b in (b1, b2):
                idx = self._free_slot(b)
                if idx >= 0:
                    self._keys[idx], self._vals[idx] = key, value
                    return None
            options = [b for b in (b1, b2) if b != last] or [b1]
            b = options[self._rng.randrange(len(options))]
            idx = b * bs + self._rng.randrange(bs)
            self._keys[idx], key = key, self._keys[idx]
            self._vals[idx], value = value, self._vals[idx]
            self.kicks += 1
            last = b
        return key, value

    def _insert(self, key: Any, value: Any) -> None:
        homeless = self._try_place(key, value)
        if homeless is not None:
            self._rehash(homeless)

    def _rehash(self, extra: tuple[Any, Any]) -> None:
        """Rebuild with fresh hash functions (growing if repeatedly unlucky)."""
        items = list(self._items()) + [extra]
        capacity = self.capacity
        self.rehashes += 1
        while True:
            for _ in range(self.REHASH_ATTEMPTS):
                self._init_hash()
                self._reset(capacity)
                if all(self._try_place(k, v) is None for k, v in items):
                    return
            capacity *= 2

    def _remove(self, index: int, key: Any) -> None:
        self._keys[index] = EMPTY
        self._vals[index] = None

    def check_invariants(self) -> None:
        super().check_invariants()
        bs = self.bucket_size
        for idx, k in enumerate(self._keys):
            if k is EMPTY:
                continue
            assert idx // bs in self._buckets(k), f"{k!r} outside both of its buckets"
