"""Deterministic test-network generators and classic reductions."""
from __future__ import annotations

import random

from .graph import FlowNetwork
from .dinic import dinic


def random_network(n: int, m: int, max_cap: int, seed: int) -> tuple[FlowNetwork, int, int]:
    """Random digraph with a guaranteed s-t backbone path."""
    rng = random.Random(seed)
    g = FlowNetwork(n)
    s, t = 0, n - 1
    for v in range(n - 1):
        if rng.random() < 0.6:
            g.add_edge(v, v + 1, rng.randint(1, max_cap))
    for _ in range(m):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            g.add_edge(u, v, rng.randint(1, max_cap))
    return g, s, t


def layered_network(layers: int, width: int, max_cap: int, seed: int) -> tuple[FlowNetwork, int, int]:
    """Dense layered graph (hard for augmenting-path methods)."""
    rng = random.Random(seed)
    n = layers * width + 2
    g = FlowNetwork(n)
    s, t = n - 2, n - 1
    for j in range(width):
        g.add_edge(s, j, rng.randint(1, max_cap))
        g.add_edge((layers - 1) * width + j, t, rng.randint(1, max_cap))
    for l in range(layers - 1):
        for i in range(width):
            for j in range(width):
                g.add_edge(l * width + i, (l + 1) * width + j, rng.randint(1, max_cap))
    return g, s, t


def bipartite_matching(left: int, right: int, pairs: list[tuple[int, int]]) -> tuple[FlowNetwork, int, int]:
    """Unit-capacity network whose max flow is the maximum matching size."""
    g = FlowNetwork(left + right + 2)
    s, t = left + right, left + right + 1
    for i in range(left):
        g.add_edge(s, i, 1)
    for j in range(right):
        g.add_edge(left + j, t, 1)
    for i, j in pairs:
        g.add_edge(i, left + j, 1)
    return g, s, t


def matching_from_flow(g: FlowNetwork, left: int, right: int) -> list[tuple[int, int]]:
    """Read matched pairs from a solved matching network."""
    out = []
    for i in range(left):
        for a in g.adj[i]:
            if a % 2 == 0 and g.flow_on(a) == 1 and g.to[a] < left + right:
                out.append((i, g.to[a] - left))
    return out


def edge_disjoint_paths(n: int, edges: list[tuple[int, int]], s: int, t: int) -> int:
    """Max number of edge-disjoint s-t paths in a directed graph."""
    g = FlowNetwork(n)
    for u, v in edges:
        g.add_edge(u, v, 1)
    return dinic(g, s, t)
