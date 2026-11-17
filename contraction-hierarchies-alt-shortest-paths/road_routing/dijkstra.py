"""Baseline shortest-path searches: Dijkstra and bidirectional Dijkstra."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

from .graph import Graph, INF


@dataclass
class QueryResult:
    """Outcome of a point-to-point query."""

    dist: float
    path: list[int]
    settled: int  # number of vertices settled (search-space size)


def dijkstra_all(g: Graph, s: int) -> list[float]:
    """Single-source distances to all vertices."""
    dist = [INF] * g.n
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in g.out[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _trace(parent: dict[int, int], t: int) -> list[int]:
    path = [t]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return path[::-1]


def dijkstra(g: Graph, s: int, t: int) -> QueryResult:
    """Point-to-point Dijkstra with early termination."""
    dist = {s: 0.0}
    parent: dict[int, int] = {}
    done: set[int] = set()
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        if u == t:
            return QueryResult(d, _trace(parent, t), len(done))
        for v, w in g.out[u].items():
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return QueryResult(INF, [], len(done))


def bidirectional_dijkstra(g: Graph, s: int, t: int) -> QueryResult:
    """Alternating forward/backward Dijkstra with the standard stopping rule.

    Stops when the sum of the two queue minima reaches the best known
    meeting distance ``mu``.
    """
    if s == t:
        return QueryResult(0.0, [s], 1)
    dist = ({s: 0.0}, {t: 0.0})
    par: tuple[dict[int, int], dict[int, int]] = ({}, {})
    done: tuple[set[int], set[int]] = (set(), set())
    pq = ([(0.0, s)], [(0.0, t)])
    adj = (g.out, g.inc)
    mu, meet = INF, -1
    side = 0
    while pq[0] and pq[1]:
        if pq[0][0][0] + pq[1][0][0] >= mu:
            break
        side = 0 if len(pq[0]) <= len(pq[1]) else 1
        d, u = heapq.heappop(pq[side])
        if u in done[side]:
            continue
        done[side].add(u)
        for v, w in adj[side][u].items():
            nd = d + w
            if nd < dist[side].get(v, INF):
                dist[side][v] = nd
                par[side][v] = u
                heapq.heappush(pq[side], (nd, v))
            if v in dist[1 - side] and nd + dist[1 - side][v] < mu:
                mu, meet = nd + dist[1 - side][v], v
    settled = len(done[0]) + len(done[1])
    if meet < 0:
        return QueryResult(INF, [], settled)
    left = _trace(par[0], meet)
    right = _trace(par[1], meet)[::-1]
    return QueryResult(mu, left + right[1:], settled)
