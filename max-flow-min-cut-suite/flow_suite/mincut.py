"""Minimum cut extraction and max-flow/min-cut certificate checking."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from .graph import FlowNetwork
from .dinic import dinic


@dataclass
class MinCut:
    value: int
    source_side: set[int]
    cut_edges: list[tuple[int, int, int]]


def reachable_from(g: FlowNetwork, s: int) -> set[int]:
    """Vertices reachable from ``s`` through positive residual capacity."""
    seen = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        for a in g.adj[u]:
            v = g.to[a]
            if g.cap[a] > 0 and v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def min_cut(g: FlowNetwork, s: int, t: int,
            solver: Callable[[FlowNetwork, int, int], int] = dinic) -> MinCut:
    """Solve max flow on a copy of ``g`` and return the minimum s-t cut."""
    h = g.copy()
    h.reset()
    value = solver(h, s, t)
    side = reachable_from(h, s)
    cut = [(u, v, c) for u, v, c in g.edges() if u in side and v not in side and c > 0]
    return MinCut(value, side, cut)


def cut_capacity(g: FlowNetwork, side: set[int]) -> int:
    return sum(c for u, v, c in g.edges() if u in side and v not in side)


def verify_certificate(g: FlowNetwork, s: int, t: int, cut: MinCut) -> bool:
    """Check that the cut separates s from t and its capacity equals the flow."""
    if s not in cut.source_side or t in cut.source_side:
        return False
    return cut_capacity(g, cut.source_side) == cut.value
