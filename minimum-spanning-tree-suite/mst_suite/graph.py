"""Weighted undirected graph container and random graph generators."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Edge = tuple[int, int, float]


@dataclass
class Graph:
    """Undirected weighted multigraph stored as an edge list.

    Edges keep their insertion index, which is used as a deterministic
    tie-breaker so that every algorithm selects the same tree when weights
    are not distinct.
    """

    n: int
    edges: list[Edge] = field(default_factory=list)

    def add_edge(self, u: int, v: int, w: float) -> int:
        """Append an edge and return its index."""
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError(f"endpoint out of range: ({u}, {v})")
        self.edges.append((u, v, float(w)))
        return len(self.edges) - 1

    @property
    def m(self) -> int:
        return len(self.edges)

    def adjacency(self) -> list[list[tuple[int, float, int]]]:
        """Adjacency lists of ``(neighbour, weight, edge_index)``; self-loops skipped."""
        adj: list[list[tuple[int, float, int]]] = [[] for _ in range(self.n)]
        for i, (u, v, w) in enumerate(self.edges):
            if u == v:
                continue
            adj[u].append((v, w, i))
            adj[v].append((u, w, i))
        return adj

    def dense_matrix(self) -> np.ndarray:
        """Weight matrix with ``inf`` for missing edges (lightest parallel edge kept)."""
        mat = np.full((self.n, self.n), np.inf)
        for u, v, w in self.edges:
            if u != v and w < mat[u, v]:
                mat[u, v] = mat[v, u] = w
        return mat

    def components(self) -> int:
        """Number of connected components (simple DFS)."""
        adj = self.adjacency()
        seen = [False] * self.n
        count = 0
        for s in range(self.n):
            if seen[s]:
                continue
            count += 1
            stack = [s]
            seen[s] = True
            while stack:
                x = stack.pop()
                for y, _, _ in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        stack.append(y)
        return count

    def is_connected(self) -> bool:
        return self.n == 0 or self.components() == 1

    def total_weight(self, indices: list[int]) -> float:
        return float(sum(self.edges[i][2] for i in indices))


def random_connected_graph(n: int, m: int, seed: int = 0, integer_weights: int | None = None) -> Graph:
    """Random connected simple graph: a random spanning tree plus extra edges."""
    if n < 1:
        raise ValueError("n must be positive")
    max_m = n * (n - 1) // 2
    m = min(max(m, n - 1), max_m)
    rng = np.random.default_rng(seed)
    g = Graph(n)
    present: set[tuple[int, int]] = set()

    def weight() -> float:
        if integer_weights:
            return float(rng.integers(1, integer_weights + 1))
        return float(rng.random())

    perm = rng.permutation(n)
    for i in range(1, n):
        j = int(rng.integers(0, i))
        a, b = int(perm[i]), int(perm[j])
        present.add((min(a, b), max(a, b)))
        g.add_edge(a, b, weight())
    while g.m < m:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        key = (min(a, b), max(a, b))
        if a == b or key in present:
            continue
        present.add(key)
        g.add_edge(a, b, weight())
    return g


def grid_graph(rows: int, cols: int, seed: int = 0) -> Graph:
    """4-connected grid with random weights."""
    rng = np.random.default_rng(seed)
    g = Graph(rows * cols)
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                g.add_edge(v, v + 1, float(rng.random()))
            if r + 1 < rows:
                g.add_edge(v, v + cols, float(rng.random()))
    return g


def euclidean_complete_graph(n: int, seed: int = 0) -> tuple[Graph, np.ndarray]:
    """Complete graph on random planar points weighted by Euclidean distance."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    g = Graph(n)
    for i in range(n):
        d = np.hypot(*(pts[i + 1:] - pts[i]).T) if i + 1 < n else []
        for k, w in enumerate(d):
            g.add_edge(i, i + 1 + k, float(w))
    return g, pts


def disconnected_graph(sizes: list[int], seed: int = 0) -> Graph:
    """Disjoint union of random connected graphs, to exercise spanning forests."""
    g = Graph(sum(sizes))
    off = 0
    for k, s in enumerate(sizes):
        sub = random_connected_graph(s, 2 * s, seed=seed + k)
        for u, v, w in sub.edges:
            g.add_edge(u + off, v + off, w)
        off += s
    return g
