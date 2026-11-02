"""Overlay topologies used by the gossip protocols."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    """Undirected simple graph stored as adjacency lists."""

    n: int
    adj: list[list[int]]

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    def edge_count(self) -> int:
        return sum(len(a) for a in self.adj) // 2

    def random_neighbor(self, v: int, rng: np.random.Generator) -> int:
        """Uniform random neighbour of ``v`` (``-1`` if isolated)."""
        a = self.adj[v]
        if not a:
            return -1
        return a[int(rng.integers(len(a)))]

    def is_connected(self, removed: frozenset[int] | set[int] = frozenset()) -> bool:
        """Connectivity of the subgraph induced on nodes not in ``removed``."""
        alive = [v for v in range(self.n) if v not in removed]
        if not alive:
            return True
        seen = {alive[0]}
        q = deque([alive[0]])
        while q:
            u = q.popleft()
            for w in self.adj[u]:
                if w not in removed and w not in seen:
                    seen.add(w)
                    q.append(w)
        return len(seen) == len(alive)

    def diameter_from(self, src: int) -> int:
        """Eccentricity of ``src`` (BFS depth; -1 if graph disconnected)."""
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for w in self.adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return max(dist.values()) if len(dist) == self.n else -1


def _from_edges(n: int, edges: set[tuple[int, int]]) -> Graph:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in sorted(edges):
        adj[u].append(v)
        adj[v].append(u)
    return Graph(n, adj)


def complete_graph(n: int) -> Graph:
    """Complete graph K_n."""
    return Graph(n, [[u for u in range(n) if u != v] for v in range(n)])


def ring_lattice(n: int, k: int = 2) -> Graph:
    """Ring where each node links to ``k`` neighbours on each side."""
    if n <= 2 * k:
        raise ValueError("need n > 2k")
    edges = {(min(i, (i + d) % n), max(i, (i + d) % n)) for i in range(n) for d in range(1, k + 1)}
    return _from_edges(n, edges)


def erdos_renyi(n: int, p: float, rng: np.random.Generator) -> Graph:
    """G(n, p) random graph."""
    mask = np.triu(rng.random((n, n)) < p, 1)
    us, vs = np.nonzero(mask)
    return _from_edges(n, {(int(u), int(v)) for u, v in zip(us, vs)})


def watts_strogatz(n: int, k: int, beta: float, rng: np.random.Generator) -> Graph:
    """Small-world graph: ring lattice with each edge rewired w.p. ``beta``."""
    base = ring_lattice(n, k)
    edges = {(u, v) for u in range(n) for v in base.adj[u] if u < v}
    for (u, v) in sorted(edges):
        if rng.random() < beta:
            for _ in range(20):
                w = int(rng.integers(n))
                e = (min(u, w), max(u, w))
                if w != u and e not in edges:
                    edges.discard((u, v))
                    edges.add(e)
                    break
    return _from_edges(n, edges)


def barabasi_albert(n: int, m: int, rng: np.random.Generator) -> Graph:
    """Preferential-attachment scale-free graph with ``m`` edges per new node."""
    if not 1 <= m < n:
        raise ValueError("need 1 <= m < n")
    edges: set[tuple[int, int]] = set()
    targets = list(range(m))
    repeated: list[int] = []
    for new in range(m, n):
        for t in targets:
            edges.add((min(new, t), max(new, t)))
        repeated.extend(targets)
        repeated.extend([new] * m)
        chosen: set[int] = set()
        while len(chosen) < m:
            chosen.add(repeated[int(rng.integers(len(repeated)))])
        targets = sorted(chosen)
    return _from_edges(n, edges)
