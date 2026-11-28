"""k-buckets and the routing table."""
from __future__ import annotations

from dataclasses import dataclass, field

from .ids import bucket_index


@dataclass
class KBucket:
    """A bounded LRU list of contacts (oldest first) plus a replacement cache."""

    k: int
    entries: list[int] = field(default_factory=list)
    cache: list[int] = field(default_factory=list)

    def update(self, nid: int) -> int | None:
        """Record contact with ``nid``.

        Returns the least-recently-seen entry if the bucket is full and the
        caller should ping it; otherwise ``None``.
        """
        if nid in self.entries:
            self.entries.remove(nid)
            self.entries.append(nid)
            return None
        if len(self.entries) < self.k:
            self.entries.append(nid)
            if nid in self.cache:
                self.cache.remove(nid)
            return None
        if nid in self.cache:
            self.cache.remove(nid)
        self.cache.append(nid)
        if len(self.cache) > self.k:
            self.cache.pop(0)
        return self.entries[0]

    def remove(self, nid: int) -> None:
        """Drop a dead contact, promoting the freshest cached replacement."""
        if nid in self.entries:
            self.entries.remove(nid)
            if self.cache:
                self.entries.append(self.cache.pop())
        elif nid in self.cache:
            self.cache.remove(nid)

    def __len__(self) -> int:
        return len(self.entries)


class RoutingTable:
    """Fixed array of ``bits`` k-buckets indexed by XOR-distance magnitude."""

    def __init__(self, own_id: int, bits: int, k: int) -> None:
        self.own_id = own_id
        self.bits = bits
        self.k = k
        self.buckets = [KBucket(k) for _ in range(bits)]

    def update(self, nid: int) -> int | None:
        if nid == self.own_id:
            return None
        return self.buckets[bucket_index(self.own_id, nid)].update(nid)

    def remove(self, nid: int) -> None:
        if nid != self.own_id:
            self.buckets[bucket_index(self.own_id, nid)].remove(nid)

    def contains(self, nid: int) -> bool:
        return nid != self.own_id and nid in self.buckets[bucket_index(self.own_id, nid)].entries

    def contacts(self) -> list[int]:
        return [n for b in self.buckets for n in b.entries]

    def size(self) -> int:
        return sum(len(b) for b in self.buckets)

    def closest(self, target: int, count: int, exclude: int | None = None) -> list[int]:
        """The ``count`` known contacts closest to ``target`` by XOR distance."""
        cands = [n for n in self.contacts() if n != exclude]
        cands.sort(key=lambda n: n ^ target)
        return cands[:count]

    def nonempty_buckets(self) -> int:
        return sum(1 for b in self.buckets if b.entries)
