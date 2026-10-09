"""Edmonds-Karp: Ford-Fulkerson with shortest (BFS) augmenting paths. O(V E^2)."""
from __future__ import annotations

from collections import deque

from .graph import FlowNetwork


def edmonds_karp(g: FlowNetwork, s: int, t: int) -> int:
    """Push a maximum flow from ``s`` to ``t`` in place and return its value."""
    if s == t:
        raise ValueError("source and sink must differ")
    total = 0
    while True:
        parent_arc = [-1] * g.n
        parent_arc[s] = -2
        q = deque([s])
        while q and parent_arc[t] == -1:
            u = q.popleft()
            for a in g.adj[u]:
                v = g.to[a]
                if g.cap[a] > 0 and parent_arc[v] == -1:
                    parent_arc[v] = a
                    q.append(v)
        if parent_arc[t] == -1:
            return total
        bottleneck = None
        v = t
        while v != s:
            a = parent_arc[v]
            bottleneck = g.cap[a] if bottleneck is None else min(bottleneck, g.cap[a])
            v = g.to[a ^ 1]
        v = t
        while v != s:
            a = parent_arc[v]
            g.cap[a] -= bottleneck
            g.cap[a ^ 1] += bottleneck
            v = g.to[a ^ 1]
        total += bottleneck
