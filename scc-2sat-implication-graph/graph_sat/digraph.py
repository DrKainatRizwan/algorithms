"""A small adjacency-list directed graph with iterative traversals."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator


class Digraph:
    """Directed multigraph on vertices ``0 .. n-1``.

    All traversals are iterative so that very deep graphs (long paths) do not
    hit Python's recursion limit.
    """

    def __init__(self, n: int, edges: Iterable[tuple[int, int]] = ()) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.adj: list[list[int]] = [[] for _ in range(n)]
        self.m = 0
        for u, v in edges:
            self.add_edge(u, v)

    def add_edge(self, u: int, v: int) -> None:
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise IndexError(f"edge ({u}, {v}) out of range for n={self.n}")
        self.adj[u].append(v)
        self.m += 1

    def add_vertex(self) -> int:
        self.adj.append([])
        self.n += 1
        return self.n - 1

    def edges(self) -> Iterator[tuple[int, int]]:
        for u in range(self.n):
            for v in self.adj[u]:
                yield u, v

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adj[u]

    def reverse(self) -> "Digraph":
        """Return the transpose graph."""
        g = Digraph(self.n)
        for u, v in self.edges():
            g.adj[v].append(u)
        g.m = self.m
        return g

    def bfs_order(self, source: int) -> list[int]:
        seen = [False] * self.n
        seen[source] = True
        order, queue = [], deque([source])
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.adj[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
        return order

    def reachable_from(self, source: int) -> set[int]:
        return set(self.bfs_order(source))

    def shortest_path(self, s: int, t: int) -> list[int] | None:
        """Fewest-edge path from ``s`` to ``t`` or ``None``."""
        if s == t:
            return [s]
        parent = {s: -1}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if v not in parent:
                    parent[v] = u
                    if v == t:
                        path = [t]
                        while parent[path[-1]] != -1:
                            path.append(parent[path[-1]])
                        return path[::-1]
                    queue.append(v)
        return None

    def topological_order(self) -> list[int] | None:
        """Kahn's algorithm; ``None`` if the graph has a cycle."""
        indeg = [0] * self.n
        for _, v in self.edges():
            indeg[v] += 1
        queue = deque(u for u in range(self.n) if indeg[u] == 0)
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        return order if len(order) == self.n else None

    def is_dag(self) -> bool:
        return self.topological_order() is not None

    def __repr__(self) -> str:
        return f"Digraph(n={self.n}, m={self.m})"
