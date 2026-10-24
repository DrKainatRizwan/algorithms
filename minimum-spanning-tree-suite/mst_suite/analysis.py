"""Post-hoc analysis of a spanning tree: verification, path maxima,
second-best MST and edge sensitivity (replacement edges)."""
from __future__ import annotations

from dataclasses import dataclass

from .graph import Graph
from .unionfind import UnionFind


class TreePaths:
    """Rooted spanning forest with binary-lifting path-maximum queries.

    ``path_max(u, v)`` returns ``(weight, edge_index)`` of the heaviest edge on
    the tree path, in O(log n) after O(n log n) preprocessing.
    """

    def __init__(self, g: Graph, tree_edges: list[int]) -> None:
        n = g.n
        self.g = g
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for i in tree_edges:
            u, v, _ = g.edges[i]
            adj[u].append((v, i))
            adj[v].append((u, i))
        self.parent = [-1] * n
        self.parent_edge = [-1] * n
        self.depth = [0] * n
        self.root = list(range(n))
        self.order: list[int] = []
        seen = [False] * n
        for s in range(n):
            if seen[s]:
                continue
            seen[s] = True
            stack = [s]
            while stack:
                x = stack.pop()
                self.order.append(x)
                for y, i in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        self.parent[y] = x
                        self.parent_edge[y] = i
                        self.depth[y] = self.depth[x] + 1
                        self.root[y] = s
                        stack.append(y)
        self.levels = max(1, n.bit_length())
        self.up = [self.parent[:]]
        self.mx: list[list[int]] = [self.parent_edge[:]]
        for k in range(1, self.levels):
            pu, pm = self.up[-1], self.mx[-1]
            nu, nm = [-1] * n, [-1] * n
            for x in range(n):
                mid = pu[x]
                if mid == -1:
                    continue
                nu[x] = pu[mid]
                nm[x] = self._heavier(pm[x], pm[mid])
            self.up.append(nu)
            self.mx.append(nm)

    def _key(self, i: int) -> tuple[float, int]:
        return (self.g.edges[i][2], i)

    def _heavier(self, a: int, b: int) -> int:
        if a == -1:
            return b
        if b == -1:
            return a
        return a if self._key(a) >= self._key(b) else b

    def connected(self, u: int, v: int) -> bool:
        return self.root[u] == self.root[v]

    def path_max(self, u: int, v: int) -> int:
        """Index of the heaviest tree edge on the u-v path (-1 if u == v)."""
        if not self.connected(u, v):
            raise ValueError("vertices are in different trees")
        best = -1
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = self.depth[u] - self.depth[v]
        for k in range(self.levels):
            if diff >> k & 1:
                best = self._heavier(best, self.mx[k][u])
                u = self.up[k][u]
        if u == v:
            return best
        for k in reversed(range(self.levels)):
            if self.up[k][u] != self.up[k][v]:
                best = self._heavier(best, self._heavier(self.mx[k][u], self.mx[k][v]))
                u, v = self.up[k][u], self.up[k][v]
        return self._heavier(best, self._heavier(self.parent_edge[u], self.parent_edge[v]))


def is_spanning_forest(g: Graph, edges: list[int]) -> bool:
    """True if ``edges`` is acyclic and has the same components as ``g``."""
    uf = UnionFind(g.n)
    for i in edges:
        u, v, _ = g.edges[i]
        if not uf.union(u, v):
            return False
    return uf.components == g.components()


def is_minimum_spanning_forest(g: Graph, edges: list[int]) -> bool:
    """Cycle-property certificate: no non-tree edge is lighter than the path max."""
    if not is_spanning_forest(g, edges):
        return False
    tp = TreePaths(g, edges)
    chosen = set(edges)
    for i, (u, v, w) in enumerate(g.edges):
        if i in chosen or u == v:
            continue
        j = tp.path_max(u, v)
        if g.edges[j][2] > w + 1e-12:
            return False
    return True


def bottleneck(g: Graph, tree_edges: list[int], u: int, v: int) -> float:
    """Minimax path weight between u and v (the MST path maximum)."""
    if u == v:
        return 0.0
    return g.edges[TreePaths(g, tree_edges).path_max(u, v)][2]


@dataclass
class SecondBest:
    weight: float
    removed: int
    added: int


def second_best_mst(g: Graph, tree_edges: list[int]) -> SecondBest | None:
    """Lightest spanning tree differing from the MST by one edge swap.

    Returns ``None`` when the graph has no non-tree edge connecting a tree path.
    """
    tp = TreePaths(g, tree_edges)
    base = g.total_weight(tree_edges)
    chosen = set(tree_edges)
    best: SecondBest | None = None
    for i, (u, v, w) in enumerate(g.edges):
        if i in chosen or u == v or not tp.connected(u, v):
            continue
        j = tp.path_max(u, v)
        cand = base + w - g.edges[j][2]
        if best is None or cand < best.weight:
            best = SecondBest(cand, j, i)
    return best


def replacement_edges(g: Graph, tree_edges: list[int]) -> dict[int, int | None]:
    """For each tree edge, the cheapest non-tree edge that can replace it.

    Edges are processed by increasing weight; each paints the still-unpainted
    tree edges along its tree path (skipped via a union-find). O(m log m).
    """
    tp = TreePaths(g, tree_edges)
    chosen = set(tree_edges)
    result: dict[int, int | None] = {i: None for i in tree_edges}
    jump = UnionFind(g.n, "naive", "full")  # merges painted vertices into parent
    cand = sorted((i for i in range(g.m) if i not in chosen and g.edges[i][0] != g.edges[i][1]),
                  key=lambda i: (g.edges[i][2], i))
    for i in cand:
        u, v, _ = g.edges[i]
        if not tp.connected(u, v):
            continue
        u, v = jump.find(u), jump.find(v)
        while u != v:
            if tp.depth[u] < tp.depth[v]:
                u, v = v, u
            result[tp.parent_edge[u]] = i
            p = tp.parent[u]
            jump.parent[u] = p  # u now represented by its parent
            u = jump.find(p)
    return result


def sensitivity(g: Graph, tree_edges: list[int]) -> dict[int, float]:
    """Largest weight each tree edge can reach before leaving the MST.

    Equals the weight of its cheapest replacement edge (``inf`` for bridges).
    """
    rep = replacement_edges(g, tree_edges)
    return {i: (g.edges[j][2] if j is not None else float("inf")) for i, j in rep.items()}
