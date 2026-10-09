"""Dinic's blocking-flow algorithm. O(V^2 E), O(E sqrt V) on unit networks."""
from __future__ import annotations

from collections import deque

from .graph import FlowNetwork


def _levels(g: FlowNetwork, s: int, t: int) -> list[int] | None:
    level = [-1] * g.n
    level[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for a in g.adj[u]:
            v = g.to[a]
            if g.cap[a] > 0 and level[v] < 0:
                level[v] = level[u] + 1
                q.append(v)
    return level if level[t] >= 0 else None


def _blocking_flow(g: FlowNetwork, s: int, t: int, level: list[int]) -> int:
    """Iterative DFS with current-arc pointers; avoids recursion limits."""
    ptr = [0] * g.n
    pushed = 0
    while True:
        path: list[int] = []
        u = s
        while u != t:
            advanced = False
            while ptr[u] < len(g.adj[u]):
                a = g.adj[u][ptr[u]]
                v = g.to[a]
                if g.cap[a] > 0 and level[v] == level[u] + 1:
                    path.append(a)
                    u = v
                    advanced = True
                    break
                ptr[u] += 1
            if not advanced:
                if u == s:
                    return pushed
                level[u] = -1  # dead end
                a = path.pop()
                u = g.to[a ^ 1]
                ptr[u] += 1
        d = min(g.cap[a] for a in path)
        for a in path:
            g.cap[a] -= d
            g.cap[a ^ 1] += d
        pushed += d


def dinic(g: FlowNetwork, s: int, t: int) -> int:
    """Push a maximum flow in place and return its value."""
    if s == t:
        raise ValueError("source and sink must differ")
    total = 0
    while True:
        level = _levels(g, s, t)
        if level is None:
            return total
        total += _blocking_flow(g, s, t, level)
