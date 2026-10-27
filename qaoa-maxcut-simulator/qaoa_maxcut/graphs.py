"""Weighted undirected graphs, generators and classical Max-Cut baselines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Graph:
    """Simple weighted undirected graph on vertices ``0..n-1``."""

    n: int
    edges: list[tuple[int, int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: dict[tuple[int, int], float] = {}
        for u, v, w in self.edges:
            if u == v:
                raise ValueError("self loops are not allowed")
            if not (0 <= u < self.n and 0 <= v < self.n):
                raise ValueError("edge endpoint out of range")
            key = (min(u, v), max(u, v))
            seen[key] = seen.get(key, 0.0) + float(w)
        self.edges = [(u, v, w) for (u, v), w in sorted(seen.items())]

    @property
    def m(self) -> int:
        return len(self.edges)

    def degrees(self) -> np.ndarray:
        d = np.zeros(self.n, dtype=int)
        for u, v, _ in self.edges:
            d[u] += 1
            d[v] += 1
        return d

    def adjacency(self) -> np.ndarray:
        a = np.zeros((self.n, self.n))
        for u, v, w in self.edges:
            a[u, v] = a[v, u] = w
        return a

    def neighbours(self, u: int) -> set[int]:
        return {(v if a == u else a) for a, v, _ in self.edges if u in (a, v)}

    def total_weight(self) -> float:
        return float(sum(w for _, _, w in self.edges))

    def is_unweighted(self) -> bool:
        return all(abs(w - 1.0) < 1e-12 for _, _, w in self.edges)

    def triangles_on_edge(self, u: int, v: int) -> int:
        return len(self.neighbours(u) & self.neighbours(v))

    def cut_of(self, bits: np.ndarray | list[int]) -> float:
        b = np.asarray(bits)
        return float(sum(w for u, v, w in self.edges if b[u] != b[v]))


def ring_graph(n: int) -> Graph:
    """Cycle on ``n >= 3`` vertices."""
    if n < 3:
        raise ValueError("ring needs n >= 3")
    return Graph(n, [(i, (i + 1) % n, 1.0) for i in range(n)])


def complete_graph(n: int) -> Graph:
    return Graph(n, [(i, j, 1.0) for i in range(n) for j in range(i + 1, n)])


def erdos_renyi(n: int, p: float, seed: int = 0, weighted: bool = False) -> Graph:
    """G(n, p) random graph (optionally with uniform [0.5, 1.5] weights)."""
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j, float(rng.uniform(0.5, 1.5)) if weighted else 1.0))
    return Graph(n, edges)


def random_regular_graph(n: int, d: int, seed: int = 0, max_tries: int = 1000) -> Graph:
    """Random d-regular simple graph by the pairing model with rejection."""
    if n * d % 2 or d >= n:
        raise ValueError("no d-regular graph with these parameters")
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue
        keys = {(min(a, b), max(a, b)) for a, b in pairs}
        if len(keys) != len(pairs):
            continue
        return Graph(n, [(a, b, 1.0) for a, b in sorted(keys)])
    raise RuntimeError("failed to sample a regular graph")


def bit_table(n: int) -> np.ndarray:
    """Array of shape (2**n, n); row x holds bits with qubit 0 as most significant."""
    idx = np.arange(1 << n)
    return ((idx[:, None] >> (n - 1 - np.arange(n))) & 1).astype(np.int8)


def cut_values(g: Graph) -> np.ndarray:
    """Cut value of every bitstring (the diagonal of the cost Hamiltonian)."""
    bits = bit_table(g.n)
    out = np.zeros(1 << g.n)
    for u, v, w in g.edges:
        out += w * (bits[:, u] != bits[:, v])
    return out


def brute_force_maxcut(g: Graph) -> tuple[float, np.ndarray]:
    """Exact optimum by enumeration; returns (value, bits of one optimal cut)."""
    cv = cut_values(g)
    x = int(np.argmax(cv))
    return float(cv[x]), bit_table(g.n)[x].copy()


def greedy_maxcut(g: Graph, seed: int = 0) -> tuple[float, np.ndarray]:
    """Sequential greedy placement in a random vertex order."""
    rng = np.random.default_rng(seed)
    side = -np.ones(g.n, dtype=int)
    adj = g.adjacency()
    for u in rng.permutation(g.n):
        w0 = sum(adj[u, v] for v in range(g.n) if side[v] == 0)
        w1 = sum(adj[u, v] for v in range(g.n) if side[v] == 1)
        side[u] = 1 if w0 >= w1 else 0
    return g.cut_of(side), side


def local_search_maxcut(g: Graph, restarts: int = 10, seed: int = 0) -> tuple[float, np.ndarray]:
    """1-flip hill climbing from random starts (a 1/2-approximation per start)."""
    rng = np.random.default_rng(seed)
    adj = g.adjacency()
    best, best_bits = -1.0, None
    for _ in range(restarts):
        s = rng.integers(0, 2, g.n)
        improved = True
        while improved:
            improved = False
            for u in range(g.n):
                same = adj[u][s == s[u]].sum()
                diff = adj[u][s != s[u]].sum()
                if same > diff + 1e-12:
                    s[u] ^= 1
                    improved = True
        val = g.cut_of(s)
        if val > best:
            best, best_bits = val, s.copy()
    return best, best_bits
