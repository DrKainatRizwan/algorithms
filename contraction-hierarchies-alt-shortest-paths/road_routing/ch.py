"""Contraction hierarchies (Geisberger et al.) with witness searches."""
from __future__ import annotations

import heapq

from .dijkstra import QueryResult
from .graph import Graph, INF


class ContractionHierarchy:
    """Vertex-ordering based speed-up technique for static road graphs.

    Preprocessing contracts vertices in importance order (lazy-updated
    edge-difference + deleted-neighbours priority). Contracting ``v`` adds a
    shortcut ``u->w`` whenever the path ``u->v->w`` is the only shortest one,
    checked by a bounded *witness search*. Queries run an upward forward
    search and an upward backward search and meet in the middle.
    """

    def __init__(self, g: Graph, witness_limit: int = 500):
        self.g = g
        self.n = g.n
        self.witness_limit = witness_limit
        self.rank = [-1] * g.n
        self.num_shortcuts = 0
        # edges[(u, v)] = (weight, via); via == -1 for original edges
        self.edges: dict[tuple[int, int], tuple[float, int]] = {}
        self._out = [dict() for _ in range(g.n)]  # u -> {v: w}
        self._in = [dict() for _ in range(g.n)]   # v -> {u: w}
        for u, v, w in g.edges():
            self._set(u, v, w, -1)
        self._contract_all()
        self._build_search_graphs()

    # ------------------------------------------------------------ build
    def _set(self, u: int, v: int, w: float, via: int) -> None:
        self.edges[(u, v)] = (w, via)
        self._out[u][v] = w
        self._in[v][u] = w

    def _witness(self, src: int, skip: int, limit_cost: float, contracted: list[bool]) -> dict[int, float]:
        dist = {src: 0.0}
        pq = [(0.0, src)]
        settled = 0
        done: set[int] = set()
        while pq and settled < self.witness_limit:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            settled += 1
            if d > limit_cost:
                break
            for v, w in self._out[u].items():
                if v == skip or contracted[v]:
                    continue
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def _shortcuts_for(self, v: int, contracted: list[bool]) -> list[tuple[int, int, float]]:
        ins = [(u, w) for u, w in self._in[v].items() if not contracted[u]]
        outs = [(x, w) for x, w in self._out[v].items() if not contracted[x]]
        if not ins or not outs:
            return []
        result = []
        max_out = max(w for _, w in outs)
        for u, wu in ins:
            dist = self._witness(u, v, wu + max_out, contracted)
            for x, wx in outs:
                if x == u:
                    continue
                cand = wu + wx
                if dist.get(x, INF) > cand:
                    old = self.edges.get((u, x))
                    if old is None or old[0] > cand:
                        result.append((u, x, cand))
        return result

    def _priority(self, v: int, contracted: list[bool], deleted: list[int]) -> float:
        sc = len(self._shortcuts_for(v, contracted))
        deg = sum(1 for u in self._in[v] if not contracted[u]) + \
            sum(1 for x in self._out[v] if not contracted[x])
        return sc - deg + deleted[v]

    def _contract_all(self) -> None:
        n = self.n
        contracted = [False] * n
        deleted = [0] * n
        pq = [(self._priority(v, contracted, deleted), v) for v in range(n)]
        heapq.heapify(pq)
        order = 0
        while pq:
            p, v = heapq.heappop(pq)
            newp = self._priority(v, contracted, deleted)
            if pq and newp > pq[0][0]:
                heapq.heappush(pq, (newp, v))   # lazy update
                continue
            for u, x, w in self._shortcuts_for(v, contracted):
                self._set(u, x, w, v)
                self.num_shortcuts += 1
            contracted[v] = True
            self.rank[v] = order
            order += 1
            for u in list(self._in[v]) + list(self._out[v]):
                if not contracted[u]:
                    deleted[u] += 1

    def _build_search_graphs(self) -> None:
        self.up_out: list[list[tuple[int, float]]] = [[] for _ in range(self.n)]
        self.up_in: list[list[tuple[int, float]]] = [[] for _ in range(self.n)]
        for (u, v), (w, _) in self.edges.items():
            if self.rank[v] > self.rank[u]:
                self.up_out[u].append((v, w))      # forward search goes up
            else:
                self.up_in[v].append((u, w))       # backward search: edge u->v, rank[u] > rank[v]

    # ------------------------------------------------------------ query
    def query(self, s: int, t: int, with_path: bool = True) -> QueryResult:
        """Exact shortest s-t distance (and path, unpacked from shortcuts)."""
        if s == t:
            return QueryResult(0.0, [s], 1)
        dist = ({s: 0.0}, {t: 0.0})
        par: tuple[dict[int, int], dict[int, int]] = ({}, {})
        pq = ([(0.0, s)], [(0.0, t)])
        adj = (self.up_out, self.up_in)
        done: tuple[set[int], set[int]] = (set(), set())
        mu, meet = INF, -1
        while pq[0] or pq[1]:
            # Upward searches may only be stopped per direction once min >= mu.
            for side in (0, 1):
                q = pq[side]
                if not q:
                    continue
                d, u = heapq.heappop(q)
                if u in done[side]:
                    continue
                if d >= mu:
                    q.clear()
                    continue
                done[side].add(u)
                if u in dist[1 - side] and d + dist[1 - side][u] < mu:
                    mu, meet = d + dist[1 - side][u], u
                for v, w in adj[side][u]:
                    nd = d + w
                    if nd < dist[side].get(v, INF):
                        dist[side][v] = nd
                        par[side][v] = u
                        heapq.heappush(q, (nd, v))
        settled = len(done[0]) + len(done[1])
        if meet < 0:
            return QueryResult(INF, [], settled)
        if not with_path:
            return QueryResult(mu, [], settled)
        fwd = [meet]
        while fwd[-1] in par[0]:
            fwd.append(par[0][fwd[-1]])
        fwd.reverse()
        bwd = [meet]
        while bwd[-1] in par[1]:
            bwd.append(par[1][bwd[-1]])
        full = fwd + bwd[1:]
        path = [full[0]]
        for a, b in zip(full, full[1:]):
            path.extend(self._unpack(a, b)[1:])
        return QueryResult(mu, path, settled)

    def _unpack(self, u: int, v: int) -> list[int]:
        """Expand a (possibly shortcut) edge into original vertices, iteratively."""
        out: list[int] = []
        stack = [(u, v)]
        while stack:
            a, b = stack.pop()
            via = self.edges[(a, b)][1]
            if via < 0:
                if not out:
                    out.append(a)
                out.append(b)
            else:
                stack.append((via, b))
                stack.append((a, via))
        return out

    def stats(self) -> dict[str, float]:
        return {"vertices": self.n, "original_edges": self.g.m,
                "shortcuts": self.num_shortcuts,
                "max_up_degree": max((len(x) for x in self.up_out), default=0)}
