"""Minimum spanning tree / forest algorithms.

Every function returns an :class:`MSTResult` holding the chosen edge indices.
Ties are broken by edge index so results are deterministic and agree across
algorithms (the MST is then unique under the induced strict total order).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .graph import Graph
from .unionfind import UnionFind


@dataclass
class MSTResult:
    """Edge indices of a minimum spanning forest plus bookkeeping."""

    edges: list[int]
    weight: float
    components: int
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def is_spanning_tree(self) -> bool:
        return self.components == 1


def _result(g: Graph, chosen: list[int], stats: dict[str, int] | None = None) -> MSTResult:
    return MSTResult(sorted(chosen), g.total_weight(chosen), g.n - len(chosen), stats or {})


def sorted_edge_order(g: Graph) -> list[int]:
    """Edge indices ordered by (weight, index)."""
    w = np.array([e[2] for e in g.edges])
    return [int(i) for i in np.lexsort((np.arange(g.m), w))] if g.m else []


def kruskal(g: Graph, union: str = "rank", compression: str = "full") -> MSTResult:
    """Kruskal: scan edges by weight, keep those joining two components. O(m log m)."""
    uf = UnionFind(g.n, union, compression)
    chosen: list[int] = []
    for i in sorted_edge_order(g):
        u, v, _ = g.edges[i]
        if uf.union(u, v):
            chosen.append(i)
            if len(chosen) == g.n - 1:
                break
    return _result(g, chosen, {"uf_steps": uf.steps})


def prim(g: Graph) -> MSTResult:
    """Lazy Prim with a binary heap, restarted per component. O(m log n)."""
    adj = g.adjacency()
    visited = [False] * g.n
    chosen: list[int] = []
    pushes = 0
    for s in range(g.n):
        if visited[s]:
            continue
        visited[s] = True
        heap = [(w, i, v) for v, w, i in adj[s]]
        heapq.heapify(heap)
        while heap:
            w, i, v = heapq.heappop(heap)
            if visited[v]:
                continue
            visited[v] = True
            chosen.append(i)
            for x, wx, ix in adj[v]:
                if not visited[x]:
                    heapq.heappush(heap, (wx, ix, x))
                    pushes += 1
    return _result(g, chosen, {"heap_pushes": pushes})


def prim_dense(g: Graph) -> MSTResult:
    """Array-based Prim, O(n^2); preferable for dense graphs.

    Parallel edges are collapsed to the lightest one (lowest index on ties).
    """
    n = g.n
    best_w = np.full((n, n), np.inf)
    best_i = np.full((n, n), -1, dtype=np.int64)
    for i, (u, v, w) in enumerate(g.edges):
        if u != v and w < best_w[u, v]:
            best_w[u, v] = best_w[v, u] = w
            best_i[u, v] = best_i[v, u] = i
    in_tree = np.zeros(n, dtype=bool)
    chosen: list[int] = []
    for s in range(n):
        if in_tree[s]:
            continue
        in_tree[s] = True
        key = best_w[s].copy()
        via = best_i[s].copy()
        key[in_tree] = np.inf
        while True:
            masked = np.where(in_tree, np.inf, key)
            lo = masked.min()
            if not np.isfinite(lo):
                break
            cand = np.flatnonzero(masked == lo)
            v = int(cand[np.argmin(via[cand])])  # tie-break on edge index
            in_tree[v] = True
            chosen.append(int(via[v]))
            better = ((best_w[v] < key) | ((best_w[v] == key) & (best_i[v] < via))) & ~in_tree
            key[better] = best_w[v][better]
            via[better] = best_i[v][better]
    return _result(g, chosen)


def boruvka(g: Graph) -> MSTResult:
    """Boruvka: every component picks its cheapest outgoing edge each round.

    At most log2(n) rounds, each O(m alpha(n)), so O(m log n) total. Ties are
    resolved by (weight, index) which prevents cycles among simultaneous picks.
    """
    uf = UnionFind(g.n)
    chosen: list[int] = []
    rounds = 0
    while True:
        cheapest: dict[int, int] = {}
        for i, (u, v, w) in enumerate(g.edges):
            ru, rv = uf.find(u), uf.find(v)
            if ru == rv:
                continue
            for r in (ru, rv):
                j = cheapest.get(r)
                if j is None or (w, i) < (g.edges[j][2], j):
                    cheapest[r] = i
        if not cheapest:
            break
        rounds += 1
        for i in set(cheapest.values()):
            u, v, _ = g.edges[i]
            if uf.union(u, v):
                chosen.append(i)
    return _result(g, chosen, {"rounds": rounds})


def reverse_delete(g: Graph) -> MSTResult:
    """Reverse-delete: drop edges by decreasing weight unless it disconnects.

    Quadratic-ish reference implementation (a BFS per edge); useful as an
    independent cross-check of the cycle property.
    """
    keep = set(range(g.m))
    order = sorted_edge_order(g)[::-1]
    for i in order:
        u, v, _ = g.edges[i]
        if u == v:
            keep.discard(i)
            continue
        keep.discard(i)
        if not _reachable(g, keep, u, v):
            keep.add(i)
    return _result(g, list(keep))


def _reachable(g: Graph, allowed: set[int], s: int, t: int) -> bool:
    adj: dict[int, list[int]] = {}
    for i in allowed:
        u, v, _ = g.edges[i]
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = {s}
    stack = [s]
    while stack:
        x = stack.pop()
        if x == t:
            return True
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


ALGORITHMS = {
    "kruskal": kruskal,
    "prim": prim,
    "prim_dense": prim_dense,
    "boruvka": boruvka,
}
