"""Louvain modularity optimisation (Blondel et al., 2008)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import Graph
from .metrics import modularity, relabel


@dataclass
class LouvainResult:
    labels: np.ndarray                      # final partition of original nodes
    modularity: float
    levels: list[np.ndarray] = field(default_factory=list)  # partition per level


def local_moving(g: Graph, rng: np.random.Generator, resolution: float = 1.0,
                 min_gain: float = 1e-10) -> tuple[np.ndarray, bool]:
    """Phase 1: greedily move single nodes to the best neighbouring community.

    Returns the community array and whether any node moved.
    """
    n = g.n
    m2 = 2.0 * g.total_weight()
    comm = np.arange(n)
    if m2 == 0:
        return comm, False
    deg = g.degrees()
    tot = deg.copy()
    improved = False
    moved = True
    while moved:
        moved = False
        for u in rng.permutation(n):
            cu = comm[u]
            links: dict[int, float] = {}
            for v, w in g.adj[u].items():
                if v != u:
                    links[comm[v]] = links.get(comm[v], 0.0) + w
            tot[cu] -= deg[u]
            best_c = cu
            best_gain = links.get(cu, 0.0) - resolution * tot[cu] * deg[u] / m2
            for c, k_in in links.items():
                gain = k_in - resolution * tot[c] * deg[u] / m2
                if gain > best_gain + min_gain:
                    best_gain, best_c = gain, c
            tot[best_c] += deg[u]
            if best_c != cu:
                comm[u] = best_c
                moved = improved = True
    return relabel(comm), improved


def aggregate(g: Graph, comm: np.ndarray) -> Graph:
    """Phase 2: collapse each community into a node, keeping internal weight as a loop."""
    k = int(comm.max()) + 1
    h = Graph(k)
    for u, v, w in g.edges():
        h.add_edge(int(comm[u]), int(comm[v]), w)
    return h


def louvain(g: Graph, resolution: float = 1.0, seed: int = 0,
            max_levels: int = 50) -> LouvainResult:
    """Run Louvain and return the final partition plus every hierarchy level."""
    rng = np.random.default_rng(seed)
    labels = np.arange(g.n)
    levels: list[np.ndarray] = []
    cur = g
    for _ in range(max_levels):
        comm, improved = local_moving(cur, rng, resolution)
        if not improved:
            break
        labels = comm[labels]
        levels.append(labels.copy())
        cur = aggregate(cur, comm)
    if not levels:
        levels.append(labels.copy())
    return LouvainResult(relabel(labels), modularity(g, labels, resolution), levels)
