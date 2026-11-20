"""Stoer-Wagner deterministic global minimum cut (dense and heap variants)."""
from __future__ import annotations

import heapq

import numpy as np

from .graph import CutResult, Graph


def stoer_wagner(g: Graph) -> CutResult:
    """Dense O(n^3) Stoer-Wagner using maximum-adjacency orderings.

    Each phase grows a set A by repeatedly adding the vertex most tightly
    connected to A. The last vertex t added has a "cut of the phase"
    (t versus everything else) that is a minimum s-t cut for the last two
    vertices s, t; the pair is then merged.
    """
    n = g.n
    if n < 2:
        raise ValueError("need at least two vertices")
    w = g.w.copy()
    groups: list[list[int]] = [[i] for i in range(n)]
    active = list(range(n))
    best_val = float("inf")
    best_side: list[int] = []
    while len(active) > 1:
        conn = np.zeros(n)
        in_a = np.zeros(n, dtype=bool)
        prev = last = -1
        for _ in range(len(active)):
            cand = [v for v in active if not in_a[v]]
            sel = max(cand, key=lambda v: (conn[v], -v))
            in_a[sel] = True
            prev, last = last, sel
            conn += w[sel]
        phase_cut = conn[last] - w[last, last]
        if phase_cut < best_val - 1e-12:
            best_val = float(phase_cut)
            best_side = list(groups[last])
        # merge last into prev
        groups[prev].extend(groups[last])
        w[prev, :] += w[last, :]
        w[:, prev] += w[:, last]
        w[prev, prev] = 0.0
        w[last, :] = 0.0
        w[:, last] = 0.0
        active.remove(last)
    return CutResult(best_val, tuple(sorted(best_side)))


def stoer_wagner_heap(g: Graph) -> CutResult:
    """Adjacency-dict variant with a lazy max-heap per phase.

    Runs in O(n m log n) which beats the dense version on sparse graphs.
    """
    n = g.n
    if n < 2:
        raise ValueError("need at least two vertices")
    adj: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for u, v, wt in g.edges():
        adj[u][v] = wt
        adj[v][u] = wt
    groups = {i: [i] for i in range(n)}
    best_val = float("inf")
    best_side: list[int] = []
    while len(adj) > 1:
        conn = {v: 0.0 for v in adj}
        added: set[int] = set()
        start = min(adj)
        heap = [(0.0, start)]
        order: list[int] = []
        # lazy heap; ties broken toward smaller ids, disconnected leftovers appended
        remaining = sorted(adj)
        ri = 0
        while len(order) < len(adj):
            sel = None
            while heap:
                negc, v = heapq.heappop(heap)
                if v not in added and -negc == conn[v]:
                    sel = v
                    break
            if sel is None:
                while remaining[ri] in added:
                    ri += 1
                sel = remaining[ri]
            added.add(sel)
            order.append(sel)
            for nb, wt in adj[sel].items():
                if nb not in added:
                    conn[nb] += wt
                    heapq.heappush(heap, (-conn[nb], nb))
        s, t = order[-2], order[-1]
        phase_cut = sum(adj[t].values())
        if phase_cut < best_val - 1e-12:
            best_val = phase_cut
            best_side = list(groups[t])
        groups[s].extend(groups.pop(t))
        for nb, wt in adj.pop(t).items():
            del adj[nb][t]
            if nb == s:
                continue
            adj[s][nb] = adj[s].get(nb, 0.0) + wt
            adj[nb][s] = adj[s][nb]
    return CutResult(float(best_val), tuple(sorted(best_side)))
