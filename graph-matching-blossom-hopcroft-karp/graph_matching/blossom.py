"""Maximum cardinality matching in general graphs (Edmonds' blossom algorithm)."""

from __future__ import annotations

from collections import deque

Adj = list[list[int]]


def greedy_matching(adj: Adj) -> list[int]:
    """Simple greedy maximal matching, used as a warm start and baseline."""
    mate = [-1] * len(adj)
    for u, nbrs in enumerate(adj):
        if mate[u] != -1:
            continue
        for v in nbrs:
            if mate[v] == -1 and v != u:
                mate[u] = v
                mate[v] = u
                break
    return mate


def edmonds_blossom(adj: Adj, warm_start: bool = True) -> list[int]:
    """Maximum matching in a general graph, O(V^3).

    Grows an alternating BFS forest from each exposed vertex, contracting odd
    cycles (blossoms) implicitly through a ``base`` array.  Returns ``mate``
    with ``mate[u]`` the partner of ``u`` or ``-1``.
    """
    n = len(adj)
    mate = greedy_matching(adj) if warm_start else [-1] * n
    parent = [-1] * n
    base = list(range(n))
    in_queue = [False] * n
    in_blossom = [False] * n

    def lca(a: int, b: int) -> int:
        used = [False] * n
        while True:
            a = base[a]
            used[a] = True
            if mate[a] == -1:
                break
            a = parent[mate[a]]
        while True:
            b = base[b]
            if used[b]:
                return b
            b = parent[mate[b]]

    def mark_path(v: int, b: int, child: int) -> None:
        while base[v] != b:
            in_blossom[base[v]] = in_blossom[base[mate[v]]] = True
            parent[v] = child
            child = mate[v]
            v = parent[mate[v]]

    def find_path(root: int) -> int:
        nonlocal in_queue
        in_queue = [False] * n
        for i in range(n):
            parent[i] = -1
            base[i] = i
        in_queue[root] = True
        q: deque[int] = deque([root])
        while q:
            v = q.popleft()
            for to in adj[v]:
                if to == v or base[v] == base[to] or mate[v] == to:
                    continue
                if to == root or (mate[to] != -1 and parent[mate[to]] != -1):
                    # odd cycle: contract the blossom
                    cur = lca(v, to)
                    for i in range(n):
                        in_blossom[i] = False
                    mark_path(v, cur, to)
                    mark_path(to, cur, v)
                    for i in range(n):
                        if in_blossom[base[i]]:
                            base[i] = cur
                            if not in_queue[i]:
                                in_queue[i] = True
                                q.append(i)
                elif parent[to] == -1:
                    parent[to] = v
                    if mate[to] == -1:
                        return to
                    in_queue[mate[to]] = True
                    q.append(mate[to])
        return -1

    for root in range(n):
        if mate[root] != -1:
            continue
        v = find_path(root)
        while v != -1:  # flip the augmenting path
            pv = parent[v]
            ppv = mate[pv]
            mate[v] = pv
            mate[pv] = v
            v = ppv
    return mate
