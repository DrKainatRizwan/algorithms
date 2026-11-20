"""Gomory-Hu cut tree via Gusfield's algorithm with a Dinic max-flow."""
from __future__ import annotations

from collections import deque

import numpy as np

from .graph import CutResult, Graph


def max_flow(g: Graph, s: int, t: int) -> tuple[float, set[int]]:
    """Dinic max flow on an undirected dense graph.

    Returns the flow value and the set of vertices on the source side of a
    minimum s-t cut.
    """
    if s == t:
        raise ValueError("source and sink must differ")
    n = g.n
    cap = g.w.copy()
    flow = 0.0
    eps = 1e-12
    while True:
        level = [-1] * n
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in np.nonzero(cap[u] > eps)[0]:
                v = int(v)
                if level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        if level[t] < 0:
            return flow, {v for v in range(n) if level[v] >= 0}
        ptr = [0] * n
        nbrs = [list(map(int, np.nonzero(cap[u] > eps)[0])) for u in range(n)]

        def dfs(u: int, pushed: float) -> float:
            if u == t:
                return pushed
            while ptr[u] < len(nbrs[u]):
                v = nbrs[u][ptr[u]]
                if level[v] == level[u] + 1 and cap[u, v] > eps:
                    d = dfs(v, min(pushed, cap[u, v]))
                    if d > eps:
                        cap[u, v] -= d
                        cap[v, u] += d
                        return d
                ptr[u] += 1
            return 0.0

        while True:
            pushed = dfs(s, float("inf"))
            if pushed <= eps:
                break
            flow += pushed


class GomoryHuTree:
    """Equivalent-flow cut tree: min s-t cut equals the lightest tree-path edge."""

    def __init__(self, g: Graph) -> None:
        n = g.n
        if n < 2:
            raise ValueError("need at least two vertices")
        self.n = n
        self.parent = [0] * n
        self.weight = [0.0] * n
        for s in range(1, n):
            t = self.parent[s]
            f, side = max_flow(g, s, t)
            self.weight[s] = f
            for v in range(s + 1, n):
                if v in side and self.parent[v] == t:
                    self.parent[v] = s
            if self.parent[t] in side:  # Gusfield's swap step
                self.parent[s] = self.parent[t]
                self.parent[t] = s
                self.weight[s] = self.weight[t]
                self.weight[t] = f
        self._adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
        for v in range(1, n):
            p = self.parent[v]
            self._adj[v].append((p, self.weight[v]))
            self._adj[p].append((v, self.weight[v]))

    def tree_edges(self) -> list[tuple[int, int, float]]:
        return [(v, self.parent[v], self.weight[v]) for v in range(1, self.n)
                if v != self.parent[v]]

    def min_cut_value(self, s: int, t: int) -> float:
        """Min s-t cut value = minimum edge weight on the tree path."""
        if s == t:
            raise ValueError("vertices must differ")
        prev = {s: (-1, float("inf"))}
        stack = [s]
        while stack:
            u = stack.pop()
            if u == t:
                break
            for v, wt in self._adj[u]:
                if v not in prev:
                    prev[v] = (u, min(prev[u][1], wt))
                    stack.append(v)
        return prev[t][1]

    def global_min_cut(self) -> CutResult:
        """The lightest tree edge, and the component it cuts off, is a global min cut."""
        edges = self.tree_edges()
        v, p, wt = min(edges, key=lambda e: e[2])
        seen = {v}
        stack = [v]
        while stack:
            u = stack.pop()
            for x, _ in self._adj[u]:
                if x in seen or {u, x} == {v, p}:
                    continue
                seen.add(x)
                stack.append(x)
        return CutResult(float(wt), tuple(sorted(seen)))
