"""ALT: A* with landmarks and the triangle inequality (Goldberg & Harrelson)."""
from __future__ import annotations

import heapq
import random

from .dijkstra import QueryResult, _trace, dijkstra_all
from .graph import Graph, INF


class ALT:
    """Landmark-based A* preprocessing and query.

    For landmark L the triangle inequality gives the admissible, consistent
    lower bounds ``d(v,t) >= d(L,t) - d(L,v)`` and ``d(v,t) >= d(v,L) - d(t,L)``.
    """

    def __init__(self, g: Graph, k: int = 8, strategy: str = "farthest", seed: int = 0):
        self.g = g
        self.landmarks = self.select_landmarks(k, strategy, seed)
        rg = g.reversed()
        self.d_from = [dijkstra_all(g, l) for l in self.landmarks]   # d(L, v)
        self.d_to = [dijkstra_all(rg, l) for l in self.landmarks]    # d(v, L)

    def select_landmarks(self, k: int, strategy: str, seed: int) -> list[int]:
        g = self.g
        rng = random.Random(seed)
        if strategy == "random":
            return rng.sample(range(g.n), min(k, g.n))
        if strategy != "farthest":
            raise ValueError(f"unknown strategy {strategy!r}")
        first = rng.randrange(g.n)
        d0 = dijkstra_all(g, first)
        chosen = [max(range(g.n), key=lambda v: d0[v] if d0[v] < INF else -1)]
        mind = dijkstra_all(g, chosen[0])
        while len(chosen) < min(k, g.n):
            nxt = max(range(g.n), key=lambda v: mind[v] if mind[v] < INF else -1)
            chosen.append(nxt)
            dn = dijkstra_all(g, nxt)
            mind = [min(a, b) for a, b in zip(mind, dn)]
        return chosen

    def heuristic(self, v: int, t: int) -> float:
        """Best landmark lower bound on d(v, t)."""
        h = 0.0
        for df, dt in zip(self.d_from, self.d_to):
            if df[t] < INF and df[v] < INF:
                h = max(h, df[t] - df[v])
            if dt[v] < INF and dt[t] < INF:
                h = max(h, dt[v] - dt[t])
        return h

    def query(self, s: int, t: int) -> QueryResult:
        """A* from s to t using the landmark potential."""
        g = self.g
        dist = {s: 0.0}
        parent: dict[int, int] = {}
        done: set[int] = set()
        pq = [(self.heuristic(s, t), s)]
        while pq:
            _, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            if u == t:
                return QueryResult(dist[t], _trace(parent, t), len(done))
            du = dist[u]
            for v, w in g.out[u].items():
                nd = du + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd + self.heuristic(v, t), v))
        return QueryResult(INF, [], len(done))
