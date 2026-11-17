"""Weighted directed graph container and synthetic road-network generators."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

INF = math.inf


@dataclass
class Graph:
    """Directed graph with non-negative edge weights.

    Parallel edges are collapsed to the cheapest one.
    """

    n: int
    out: list[dict[int, float]] = field(default_factory=list)
    inc: list[dict[int, float]] = field(default_factory=list)
    coords: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.out:
            self.out = [dict() for _ in range(self.n)]
        if not self.inc:
            self.inc = [dict() for _ in range(self.n)]

    def add_edge(self, u: int, v: int, w: float) -> None:
        """Insert edge u->v, keeping the minimum weight on duplicates."""
        if w < 0:
            raise ValueError("negative weights are not supported")
        if u == v:
            return
        if v not in self.out[u] or w < self.out[u][v]:
            self.out[u][v] = w
            self.inc[v][u] = w

    def add_undirected(self, u: int, v: int, w: float) -> None:
        self.add_edge(u, v, w)
        self.add_edge(v, u, w)

    @property
    def m(self) -> int:
        return sum(len(d) for d in self.out)

    def reversed(self) -> "Graph":
        g = Graph(self.n)
        g.out, g.inc = [dict(d) for d in self.inc], [dict(d) for d in self.out]
        g.coords = list(self.coords)
        return g

    def edges(self):
        for u in range(self.n):
            for v, w in self.out[u].items():
                yield u, v, w


def grid_road_network(rows: int, cols: int, seed: int = 0, highway_frac: float = 0.03,
                      drop_frac: float = 0.08) -> Graph:
    """Synthetic road network: a perturbed grid with fast long-range "highways".

    Local street weights are Euclidean length times a random congestion factor
    (slightly asymmetric per direction); a few long edges get a speed bonus,
    giving the hierarchical structure contraction hierarchies exploit.
    Edges are dropped symmetrically so the graph stays strongly connected
    (row 0 and column 0 are never dropped; use ``largest_scc_subgraph`` to be safe).
    """
    rng = random.Random(seed)
    n = rows * cols
    g = Graph(n)
    g.coords = [(c + rng.uniform(-0.3, 0.3), r + rng.uniform(-0.3, 0.3))
                for r in range(rows) for c in range(cols)]

    def dist(a: int, b: int) -> float:
        (x1, y1), (x2, y2) = g.coords[a], g.coords[b]
        return math.hypot(x1 - x2, y1 - y2)

    def add_road(a: int, b: int, speed: float = 1.0) -> None:
        d = dist(a, b) / speed
        g.add_edge(a, b, d * rng.uniform(1.0, 1.3))
        g.add_edge(b, a, d * rng.uniform(1.0, 1.3))

    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols and (r == 0 or rng.random() > drop_frac):
                add_road(u, u + 1)
            if r + 1 < rows and (c == 0 or rng.random() > drop_frac):
                add_road(u, u + cols)
    # Every vertex hangs off an unbroken spine (row 0 + column 0 edges are
    # kept), but interior vertices could still be isolated; reconnect them.
    for u in range(n):
        if not g.out[u]:
            add_road(u, u + 1 if u + 1 < n else u - 1)
    k = int(highway_frac * n)
    for _ in range(k):
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b and dist(a, b) < max(rows, cols) / 2:
            add_road(a, b, speed=3.0)
    return g


def largest_scc_subgraph(g: Graph) -> Graph:
    """Restrict ``g`` to its largest strongly connected component (relabelled)."""
    from .dijkstra import dijkstra_all

    best: set[int] = set()
    seen: set[int] = set()
    for s in range(g.n):
        if s in seen:
            continue
        fwd = {v for v, d in enumerate(dijkstra_all(g, s)) if d < INF}
        bwd = {v for v, d in enumerate(dijkstra_all(g.reversed(), s)) if d < INF}
        comp = fwd & bwd
        seen |= comp
        if len(comp) > len(best):
            best = comp
        if len(best) > g.n // 2:
            break
    ids = {v: i for i, v in enumerate(sorted(best))}
    h = Graph(len(ids))
    h.coords = [g.coords[v] for v in sorted(best)] if g.coords else []
    for u, v, w in g.edges():
        if u in ids and v in ids:
            h.add_edge(ids[u], ids[v], w)
    return h
