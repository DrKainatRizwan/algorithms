"""Weighted undirected graph container and synthetic graph generators."""
from __future__ import annotations

import numpy as np


class Graph:
    """Weighted undirected graph on nodes ``0..n-1``.

    A self-loop of weight ``w`` is stored once in ``adj[i][i]`` and counts
    twice towards the weighted degree, matching the convention used when
    Louvain aggregates communities into super-nodes.
    """

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.adj: list[dict[int, float]] = [dict() for _ in range(n)]

    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        """Add ``w`` to the weight of edge ``(u, v)`` (creating it if needed)."""
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise IndexError("node out of range")
        if w <= 0:
            raise ValueError("edge weights must be positive")
        self.adj[u][v] = self.adj[u].get(v, 0.0) + w
        if u != v:
            self.adj[v][u] = self.adj[v].get(u, 0.0) + w

    def neighbors(self, u: int) -> dict[int, float]:
        return self.adj[u]

    def degree(self, u: int) -> float:
        """Weighted degree; a self-loop contributes twice."""
        return sum(w for v, w in self.adj[u].items() if v != u) + 2.0 * self.adj[u].get(u, 0.0)

    def degrees(self) -> np.ndarray:
        return np.array([self.degree(u) for u in range(self.n)], dtype=float)

    def total_weight(self) -> float:
        """Sum of edge weights (each edge once, loops once)."""
        return float(self.degrees().sum() / 2.0)

    def edges(self) -> list[tuple[int, int, float]]:
        """All edges as ``(u, v, w)`` with ``u <= v``."""
        return [(u, v, w) for u in range(self.n) for v, w in self.adj[u].items() if u <= v]

    def num_edges(self) -> int:
        return len(self.edges())

    def adjacency_matrix(self) -> np.ndarray:
        """Dense symmetric adjacency matrix (loops placed on the diagonal as 2w)."""
        a = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                a[u, v] = 2.0 * w if u == v else w
        return a

    @classmethod
    def from_edges(cls, n: int, edges: list[tuple]) -> "Graph":
        g = cls(n)
        for e in edges:
            g.add_edge(*e)
        return g

    def copy(self) -> "Graph":
        g = Graph(self.n)
        g.adj = [dict(d) for d in self.adj]
        return g

    def remove_edge(self, u: int, v: int) -> None:
        self.adj[u].pop(v, None)
        self.adj[v].pop(u, None)

    def connected_components(self) -> list[list[int]]:
        seen = [False] * self.n
        comps: list[list[int]] = []
        for s in range(self.n):
            if seen[s]:
                continue
            seen[s] = True
            stack, comp = [s], []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in self.adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comps.append(sorted(comp))
        return comps


def planted_partition(sizes: list[int], p_in: float, p_out: float,
                      seed: int = 0) -> tuple[Graph, np.ndarray]:
    """Stochastic block model with equal in-block / out-of-block probabilities.

    Returns the graph and the ground-truth block labels.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    n = int(labels.size)
    g = Graph(n)
    probs = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    draws = rng.random((n, n))
    for u in range(n):
        for v in range(u + 1, n):
            if draws[u, v] < probs[u, v]:
                g.add_edge(u, v)
    return g, labels


def ring_of_cliques(num_cliques: int, clique_size: int) -> tuple[Graph, np.ndarray]:
    """Cliques joined in a ring by single edges (resolution-limit benchmark)."""
    if num_cliques < 3 or clique_size < 2:
        raise ValueError("need >= 3 cliques of size >= 2")
    n = num_cliques * clique_size
    g = Graph(n)
    labels = np.repeat(np.arange(num_cliques), clique_size)
    for c in range(num_cliques):
        base = c * clique_size
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                g.add_edge(base + i, base + j)
        nxt = ((c + 1) % num_cliques) * clique_size
        g.add_edge(base + clique_size - 1, nxt)
    return g, labels


def two_triangles_bridge() -> tuple[Graph, np.ndarray]:
    """Two triangles joined by one edge: the smallest clear community example."""
    g = Graph.from_edges(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
    return g, np.array([0, 0, 0, 1, 1, 1])
