"""Rendezvous (highest random weight) hashing, plain and weighted."""
from __future__ import annotations

import math
from typing import Iterable

from .hashing import hash_pair, unit_float


class RendezvousHash:
    """HRW hashing: a key goes to the node with the highest score.

    Weighted variant uses score = -w / ln(u), which gives P(node i wins) = w_i / sum(w)
    exactly (Resch, 2008). Lookup is O(N); no state besides the node set.
    """

    def __init__(self, nodes: Iterable[str] = (), seed: int = 0):
        self.seed = seed
        self._weights: dict[str, float] = {}
        for n in nodes:
            self.add_node(n)

    def add_node(self, node: str, weight: float = 1.0) -> None:
        if weight <= 0:
            raise ValueError("weight must be positive")
        if node in self._weights:
            raise ValueError(f"node already present: {node}")
        self._weights[node] = weight

    def remove_node(self, node: str) -> None:
        del self._weights[node]

    @property
    def nodes(self) -> list[str]:
        return list(self._weights)

    def score(self, key: object, node: str) -> float:
        u = unit_float(hash_pair(key, node, self.seed))
        return -self._weights[node] / math.log(u)

    def get_nodes(self, key: object, count: int = 1) -> list[str]:
        """Top-``count`` nodes by score (natural replica ordering)."""
        if not self._weights:
            raise LookupError("no nodes")
        ranked = sorted(self._weights, key=lambda n: (-self.score(key, n), n))
        return ranked[:count]

    def get_node(self, key: object) -> str:
        best, best_s = "", -1.0
        for n in self._weights:
            s = self.score(key, n)
            if s > best_s or (s == best_s and n < best):
                best, best_s = n, s
        if not best:
            raise LookupError("no nodes")
        return best
