"""Classic consistent-hash ring with virtual nodes, weights and replica lookup."""
from __future__ import annotations

from bisect import bisect_right
from typing import Iterable

from .hashing import hash_key


class HashRing:
    """Karger-style consistent hashing ring.

    Each node owns ``round(vnodes * weight)`` points on a 2**64 ring. A key is
    served by the first point clockwise from its hash.
    """

    def __init__(self, nodes: Iterable[str] = (), vnodes: int = 100, seed: int = 0):
        if vnodes < 1:
            raise ValueError("vnodes must be >= 1")
        self.vnodes = vnodes
        self.seed = seed
        self._weights: dict[str, float] = {}
        self._points: list[int] = []
        self._owners: list[str] = []
        for n in nodes:
            self._insert(n, 1.0)
        self._rebuild()

    # -- membership -------------------------------------------------------
    def _count(self, weight: float) -> int:
        return max(1, round(self.vnodes * weight))

    def _insert(self, node: str, weight: float) -> None:
        if weight <= 0:
            raise ValueError("weight must be positive")
        if node in self._weights:
            raise ValueError(f"node already present: {node}")
        self._weights[node] = weight

    def _rebuild(self) -> None:
        pts: list[tuple[int, str]] = []
        for node, w in self._weights.items():
            for i in range(self._count(w)):
                pts.append((hash_key(f"{node}#{i}", self.seed), node))
        pts.sort()
        self._points = [p for p, _ in pts]
        self._owners = [n for _, n in pts]

    def add_node(self, node: str, weight: float = 1.0) -> None:
        """Add ``node``; only its own points are inserted (O(v log N))."""
        self._insert(node, weight)
        for i in range(self._count(weight)):
            p = hash_key(f"{node}#{i}", self.seed)
            idx = bisect_right(self._points, p)
            self._points.insert(idx, p)
            self._owners.insert(idx, node)

    def remove_node(self, node: str) -> None:
        """Remove ``node`` and all its virtual points."""
        if node not in self._weights:
            raise KeyError(node)
        del self._weights[node]
        keep = [(p, o) for p, o in zip(self._points, self._owners) if o != node]
        self._points = [p for p, _ in keep]
        self._owners = [o for _, o in keep]

    @property
    def nodes(self) -> list[str]:
        return list(self._weights)

    def __len__(self) -> int:
        return len(self._weights)

    # -- lookup -----------------------------------------------------------
    def get_node(self, key: object) -> str:
        """Primary node for ``key``."""
        if not self._points:
            raise LookupError("ring is empty")
        idx = bisect_right(self._points, hash_key(key, self.seed + 7)) % len(self._points)
        return self._owners[idx]

    def get_nodes(self, key: object, count: int) -> list[str]:
        """Up to ``count`` distinct nodes walking clockwise (replica set)."""
        if not self._points:
            raise LookupError("ring is empty")
        count = min(count, len(self._weights))
        start = bisect_right(self._points, hash_key(key, self.seed + 7))
        out: list[str] = []
        n = len(self._points)
        for i in range(n):
            owner = self._owners[(start + i) % n]
            if owner not in out:
                out.append(owner)
                if len(out) == count:
                    break
        return out

    def ownership(self) -> dict[str, float]:
        """Exact fraction of the hash space owned by each node."""
        total = 1 << 64
        share = {n: 0 for n in self._weights}
        pts = self._points
        for i, p in enumerate(pts):
            prev = pts[i - 1] if i else pts[-1] - total
            share[self._owners[i]] += p - prev
        return {n: s / total for n, s in share.items()}
