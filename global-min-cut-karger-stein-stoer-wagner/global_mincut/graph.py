"""Weighted undirected graph container, cut utilities and instance generators."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class CutResult:
    """Result of a global minimum cut computation.

    Attributes:
        value: total weight of edges crossing the cut.
        side: sorted tuple of vertices on one side of the cut.
    """

    value: float
    side: tuple[int, ...]


class Graph:
    """Dense weighted undirected graph on vertices ``0..n-1``.

    Parallel edges are merged by summing weights; self-loops are ignored.
    """

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.w = np.zeros((n, n), dtype=float)

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Add ``weight`` to edge ``{u, v}`` (self-loops are dropped)."""
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise IndexError("vertex out of range")
        if weight < 0:
            raise ValueError("weights must be non-negative")
        if u == v:
            return
        self.w[u, v] += weight
        self.w[v, u] += weight

    @classmethod
    def from_edges(cls, n: int, edges: list[tuple[int, int, float]]) -> "Graph":
        g = cls(n)
        for u, v, wt in edges:
            g.add_edge(u, v, wt)
        return g

    def edges(self) -> list[tuple[int, int, float]]:
        """Return ``(u, v, w)`` with ``u < v`` for every positive-weight edge."""
        us, vs = np.nonzero(np.triu(self.w, 1))
        return [(int(u), int(v), float(self.w[u, v])) for u, v in zip(us, vs)]

    def num_edges(self) -> int:
        return int(np.count_nonzero(np.triu(self.w, 1)))

    def total_weight(self) -> float:
        return float(np.triu(self.w, 1).sum())

    def degree(self) -> np.ndarray:
        return self.w.sum(axis=1)

    def is_connected(self) -> bool:
        if self.n <= 1:
            return True
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in np.nonzero(self.w[u])[0]:
                if int(v) not in seen:
                    seen.add(int(v))
                    stack.append(int(v))
        return len(seen) == self.n


def cut_value(g: Graph, side: set[int] | tuple[int, ...] | list[int]) -> float:
    """Weight of the cut separating ``side`` from the remaining vertices."""
    mask = np.zeros(g.n, dtype=bool)
    mask[list(side)] = True
    return float(g.w[np.ix_(mask, ~mask)].sum())


def brute_force_min_cut(g: Graph) -> CutResult:
    """Exhaustive global min cut, O(2^n * n^2); for validation on tiny graphs."""
    if g.n < 2:
        raise ValueError("need at least two vertices")
    best = CutResult(float("inf"), ())
    for k in range(1, g.n):
        for comb in combinations(range(1, g.n), k - 1):
            side = (0, *comb)
            val = cut_value(g, side)
            if val < best.value - 1e-12:
                best = CutResult(val, side)
    return best


def random_graph(n: int, p: float, seed: int, wmin: float = 1.0,
                 wmax: float = 1.0, connected: bool = True) -> Graph:
    """Erdos-Renyi graph; integer weights in ``[wmin, wmax]`` if they differ.

    If ``connected`` is set, a random Hamiltonian path is added first.
    """
    rng = np.random.default_rng(seed)
    g = Graph(n)

    def wt() -> float:
        return float(rng.integers(int(wmin), int(wmax) + 1)) if wmax > wmin else wmin

    if connected and n > 1:
        perm = rng.permutation(n)
        for a, b in zip(perm[:-1], perm[1:]):
            g.add_edge(int(a), int(b), wt())
    for u in range(n):
        for v in range(u + 1, n):
            if g.w[u, v] == 0 and rng.random() < p:
                g.add_edge(u, v, wt())
    return g


def planted_cut_graph(n1: int, n2: int, p_in: float, cross_edges: int,
                      seed: int) -> tuple[Graph, float]:
    """Two dense clusters joined by ``cross_edges`` unit edges.

    Returns the graph and the weight of the planted cut. The planted cut is
    the global minimum whenever both clusters have min degree > cross_edges,
    which the generator enforces by adding a ring inside each cluster.
    """
    rng = np.random.default_rng(seed)
    n = n1 + n2
    g = Graph(n)
    for lo, hi in ((0, n1), (n1, n)):
        size = hi - lo
        for i in range(size):  # ring guarantees connectivity
            g.add_edge(lo + i, lo + (i + 1) % size, 1.0)
        for u in range(lo, hi):
            for v in range(u + 1, hi):
                if rng.random() < p_in:
                    g.add_edge(u, v, 1.0)
    placed = 0
    while placed < cross_edges:
        u = int(rng.integers(0, n1))
        v = int(rng.integers(n1, n))
        if g.w[u, v] == 0:
            g.add_edge(u, v, 1.0)
            placed += 1
    return g, float(cross_edges)


def cycle_graph(n: int, weight: float = 1.0) -> Graph:
    g = Graph(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n, weight)
    return g


def complete_graph(n: int, weight: float = 1.0) -> Graph:
    g = Graph(n)
    for u in range(n):
        for v in range(u + 1, n):
            g.add_edge(u, v, weight)
    return g
