"""Directed graph utilities: topological orderings and cycle detection."""
from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, Hashable, Iterable, Iterator, List, Optional

Node = Hashable


class CycleError(ValueError):
    """Raised when a DAG is required but a cycle exists."""

    def __init__(self, cycle: List[Node]):
        super().__init__("cycle detected: " + " -> ".join(map(str, cycle)))
        self.cycle = cycle


class DiGraph:
    """Simple insertion-ordered directed graph with adjacency lists."""

    def __init__(self, edges: Iterable[tuple[Node, Node]] = (), nodes: Iterable[Node] = ()):
        self.succ: Dict[Node, List[Node]] = {}
        self.pred: Dict[Node, List[Node]] = {}
        for n in nodes:
            self.add_node(n)
        for u, v in edges:
            self.add_edge(u, v)

    def add_node(self, n: Node) -> None:
        if n not in self.succ:
            self.succ[n] = []
            self.pred[n] = []

    def add_edge(self, u: Node, v: Node) -> None:
        self.add_node(u)
        self.add_node(v)
        if v not in self.succ[u]:
            self.succ[u].append(v)
            self.pred[v].append(u)

    @property
    def nodes(self) -> List[Node]:
        return list(self.succ)

    def edges(self) -> List[tuple[Node, Node]]:
        return [(u, v) for u, vs in self.succ.items() for v in vs]

    def __len__(self) -> int:
        return len(self.succ)

    def reverse(self) -> "DiGraph":
        return DiGraph(((v, u) for u, v in self.edges()), self.nodes)


def kahn_order(g: DiGraph, key=None) -> List[Node]:
    """Kahn's algorithm. With ``key`` the smallest available node is taken first
    (lexicographically smallest order); otherwise FIFO. Raises CycleError."""
    indeg = {n: len(g.pred[n]) for n in g.nodes}
    out: List[Node] = []
    if key is None:
        q = deque(n for n in g.nodes if indeg[n] == 0)
        while q:
            u = q.popleft()
            out.append(u)
            for v in g.succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    else:
        heap = [(key(n), i, n) for i, n in enumerate(g.nodes) if indeg[n] == 0]
        heapq.heapify(heap)
        counter = len(heap) + len(g)
        while heap:
            _, _, u = heapq.heappop(heap)
            out.append(u)
            for v in g.succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    counter += 1
                    heapq.heappush(heap, (key(v), counter, v))
    if len(out) != len(g):
        raise CycleError(find_cycle(g) or [])
    return out


def find_cycle(g: DiGraph) -> Optional[List[Node]]:
    """Return a cycle as ``[a, b, ..., a]`` or None if the graph is acyclic (iterative DFS)."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in g.nodes}
    parent: Dict[Node, Node] = {}
    for root in g.nodes:
        if color[root] != WHITE:
            continue
        color[root] = GREY
        stack = [(root, iter(g.succ[root]))]
        while stack:
            u, it = stack[-1]
            for v in it:
                if color[v] == WHITE:
                    color[v] = GREY
                    parent[v] = u
                    stack.append((v, iter(g.succ[v])))
                    break
                if color[v] == GREY:
                    cyc = [u]
                    while cyc[-1] != v:
                        cyc.append(parent[cyc[-1]])
                    cyc.reverse()
                    return cyc + [v]
            else:
                color[u] = BLACK
                stack.pop()
    return None


def dfs_order(g: DiGraph) -> List[Node]:
    """Topological order from reverse DFS post-order. Raises CycleError."""
    cyc = find_cycle(g)
    if cyc:
        raise CycleError(cyc)
    seen = set()
    post: List[Node] = []
    for root in g.nodes:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(g.succ[root]))]
        while stack:
            u, it = stack[-1]
            for v in it:
                if v not in seen:
                    seen.add(v)
                    stack.append((v, iter(g.succ[v])))
                    break
            else:
                post.append(u)
                stack.pop()
    post.reverse()
    return post


def longest_levels(g: DiGraph) -> Dict[Node, int]:
    """Level (longest path length in edges from any source) of every node."""
    level = {n: 0 for n in g.nodes}
    for u in kahn_order(g):
        for v in g.succ[u]:
            level[v] = max(level[v], level[u] + 1)
    return level


def all_orders(g: DiGraph, limit: Optional[int] = None) -> Iterator[List[Node]]:
    """Enumerate all topological orders by backtracking (optionally capped by ``limit``)."""
    if find_cycle(g):
        return
    indeg = {n: len(g.pred[n]) for n in g.nodes}
    avail = [n for n in g.nodes if indeg[n] == 0]
    cur: List[Node] = []
    emitted = [0]

    def rec(avail: List[Node]) -> Iterator[List[Node]]:
        if limit is not None and emitted[0] >= limit:
            return
        if not avail:
            if len(cur) == len(g):
                emitted[0] += 1
                yield list(cur)
            return
        for i, u in enumerate(avail):
            nxt = avail[:i] + avail[i + 1:]
            for v in g.succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    nxt.append(v)
            cur.append(u)
            yield from rec(nxt)
            cur.pop()
            for v in g.succ[u]:
                indeg[v] += 1

    yield from rec(avail)


def count_orders(g: DiGraph) -> int:
    """Count topological orders with a bitmask DP over downsets (n <= ~20)."""
    nodes = g.nodes
    n = len(nodes)
    if n > 22:
        raise ValueError("count_orders supports at most 22 nodes")
    if find_cycle(g):
        return 0
    idx = {v: i for i, v in enumerate(nodes)}
    predmask = [0] * n
    for v in nodes:
        for p in g.pred[v]:
            predmask[idx[v]] |= 1 << idx[p]
    dp = {0: 1}
    for _ in range(n):
        nd: Dict[int, int] = {}
        for mask, c in dp.items():
            for i in range(n):
                if not mask >> i & 1 and predmask[i] & mask == predmask[i]:
                    nd[mask | 1 << i] = nd.get(mask | 1 << i, 0) + c
        dp = nd
    return dp.get((1 << n) - 1, 0)
