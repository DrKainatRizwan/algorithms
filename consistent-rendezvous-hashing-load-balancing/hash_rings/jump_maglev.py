"""Jump consistent hash (Lamping & Veach) and Maglev lookup tables (Eisenbud et al.)."""
from __future__ import annotations

from typing import Sequence

from .hashing import MASK64, hash_key


def jump_hash(key: int, num_buckets: int) -> int:
    """Map a 64-bit ``key`` to a bucket in ``[0, num_buckets)`` with O(ln n) steps.

    Moves exactly ~1/(n+1) of keys when growing from n to n+1 buckets, but buckets
    can only be added/removed at the end.
    """
    if num_buckets < 1:
        raise ValueError("num_buckets must be >= 1")
    key &= MASK64
    b, j = -1, 0
    while j < num_buckets:
        b = j
        key = (key * 2862933555777941757 + 1) & MASK64
        j = int((b + 1) * (float(1 << 31) / float((key >> 33) + 1)))
    return b


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def next_prime(n: int) -> int:
    while not _is_prime(n):
        n += 1
    return n


class Maglev:
    """Maglev hashing: a fixed-size prime lookup table filled via per-node permutations."""

    def __init__(self, nodes: Sequence[str], table_size: int = 65537, seed: int = 0):
        if not nodes:
            raise ValueError("need at least one node")
        if not _is_prime(table_size):
            raise ValueError("table_size must be prime")
        if table_size < len(nodes):
            raise ValueError("table smaller than node count")
        self.nodes = list(nodes)
        self.m = table_size
        self.seed = seed
        self.table = self._populate()

    def _populate(self) -> list[int]:
        m, n = self.m, len(self.nodes)
        offsets = [hash_key(x, self.seed) % m for x in self.nodes]
        skips = [hash_key(x, self.seed + 1) % (m - 1) + 1 for x in self.nodes]
        nxt = [0] * n
        table = [-1] * m
        filled = 0
        while True:
            for i in range(n):
                c = (offsets[i] + nxt[i] * skips[i]) % m
                while table[c] >= 0:
                    nxt[i] += 1
                    c = (offsets[i] + nxt[i] * skips[i]) % m
                table[c] = i
                nxt[i] += 1
                filled += 1
                if filled == m:
                    return table

    def get_node(self, key: object) -> str:
        return self.nodes[self.table[hash_key(key, self.seed + 9) % self.m]]

    def slot_counts(self) -> dict[str, int]:
        counts = {n: 0 for n in self.nodes}
        for i in self.table:
            counts[self.nodes[i]] += 1
        return counts
