"""Adaptive Replacement Cache (Megiddo & Modha, FAST 2003) and 2Q."""
from __future__ import annotations

from collections import OrderedDict
from typing import Hashable, Iterator

from .base import CachePolicy


class ARCCache(CachePolicy):
    """ARC: self-tuning balance between recency (T1) and frequency (T2).

    ``T1``/``T2`` hold resident keys seen once / at least twice recently;
    ``B1``/``B2`` are ghost lists remembering recent evictions from each.
    A hit in a ghost list adapts the target size ``p`` of T1.
    """

    name = "ARC"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.p = 0.0
        self.t1: OrderedDict[Hashable, None] = OrderedDict()
        self.t2: OrderedDict[Hashable, None] = OrderedDict()
        self.b1: OrderedDict[Hashable, None] = OrderedDict()
        self.b2: OrderedDict[Hashable, None] = OrderedDict()

    def _replace(self, in_b2: bool) -> None:
        if self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p)):
            k, _ = self.t1.popitem(last=False)
            self.b1[k] = None
        else:
            k, _ = self.t2.popitem(last=False)
            self.b2[k] = None
        self._record_eviction(k)

    def _access(self, key: Hashable) -> bool:
        c = self.capacity
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            return True
        if key in self.b1:
            self.p = min(float(c), self.p + max(len(self.b2) / len(self.b1), 1.0))
            self._replace(False)
            del self.b1[key]
            self.t2[key] = None
            return False
        if key in self.b2:
            self.p = max(0.0, self.p - max(len(self.b1) / len(self.b2), 1.0))
            self._replace(True)
            del self.b2[key]
            self.t2[key] = None
            return False
        # Complete miss.
        l1 = len(self.t1) + len(self.b1)
        if l1 == c:
            if len(self.t1) < c:
                self.b1.popitem(last=False)
                self._replace(False)
            else:
                k, _ = self.t1.popitem(last=False)
                self._record_eviction(k)
        else:
            total = l1 + len(self.t2) + len(self.b2)
            if total >= c:
                if total == 2 * c:
                    self.b2.popitem(last=False)
                self._replace(False)
        self.t1[key] = None
        return False

    def ghost_size(self) -> int:
        return len(self.b1) + len(self.b2)

    def __contains__(self, key: object) -> bool:
        return key in self.t1 or key in self.t2

    def __len__(self) -> int:
        return len(self.t1) + len(self.t2)

    def keys(self) -> Iterator[Hashable]:
        yield from self.t1
        yield from self.t2


class TwoQCache(CachePolicy):
    """Simplified 2Q (Johnson & Shasha, VLDB 1994).

    New keys enter a FIFO probation queue ``A1in``; keys evicted from it are
    remembered in ghost queue ``A1out``.  Only a re-reference while in
    ``A1out`` promotes a key to the main LRU queue ``Am``, which keeps one-shot
    scans out of the hot set.
    """

    name = "2Q"

    def __init__(self, capacity: int, in_fraction: float = 0.25, out_fraction: float = 0.5) -> None:
        super().__init__(capacity)
        self._in_frac, self._out_frac = in_fraction, out_fraction
        self.k_in = max(1, int(capacity * in_fraction)) if capacity > 1 else 1
        self.k_out = max(1, int(capacity * out_fraction))
        self.a1in: OrderedDict[Hashable, None] = OrderedDict()
        self.a1out: OrderedDict[Hashable, None] = OrderedDict()
        self.am: OrderedDict[Hashable, None] = OrderedDict()

    def reset(self) -> None:
        self.__init__(self.capacity, self._in_frac, self._out_frac)  # type: ignore[misc]

    def _reclaim(self) -> None:
        if len(self.a1in) + len(self.am) < self.capacity:
            return
        if len(self.a1in) > self.k_in or not self.am:
            k, _ = self.a1in.popitem(last=False)
            self.a1out[k] = None
            if len(self.a1out) > self.k_out:
                self.a1out.popitem(last=False)
        else:
            k, _ = self.am.popitem(last=False)
        self._record_eviction(k)

    def _access(self, key: Hashable) -> bool:
        if key in self.am:
            self.am.move_to_end(key)
            return True
        if key in self.a1in:
            return True
        if key in self.a1out:
            del self.a1out[key]
            self._reclaim()
            self.am[key] = None
            return False
        self._reclaim()
        self.a1in[key] = None
        return False

    def __contains__(self, key: object) -> bool:
        return key in self.am or key in self.a1in

    def __len__(self) -> int:
        return len(self.am) + len(self.a1in)

    def keys(self) -> Iterator[Hashable]:
        yield from self.a1in
        yield from self.am
