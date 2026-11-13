"""Maximum matching in bipartite graphs.

Left vertices are ``0..n_left-1`` and right vertices ``0..n_right-1``.  The
graph is given as an adjacency list ``adj[u]`` of right neighbours.
"""

from __future__ import annotations

from collections import deque

Adj = list[list[int]]


def kuhn_matching(adj: Adj, n_right: int) -> list[int]:
    """Augmenting-path matching (Kuhn), O(V*E).

    Returns ``match_left`` where ``match_left[u]`` is the matched right vertex
    or ``-1``.
    """
    n_left = len(adj)
    match_right = [-1] * n_right
    match_left = [-1] * n_left

    def try_augment(u: int, seen: list[bool]) -> bool:
        stack = [(u, iter(adj[u]))]
        path: list[tuple[int, int]] = []
        while stack:
            node, it = stack[-1]
            advanced = False
            for v in it:
                if seen[v]:
                    continue
                seen[v] = True
                if match_right[v] == -1:
                    path.append((node, v))
                    for a, b in path:
                        match_right[b] = a
                        match_left[a] = b
                    return True
                path.append((node, v))
                stack.append((match_right[v], iter(adj[match_right[v]])))
                advanced = True
                break
            if not advanced:
                stack.pop()
                if path:
                    path.pop()
        return False

    for u in range(n_left):
        try_augment(u, [False] * n_right)
    return match_left


def hopcroft_karp(adj: Adj, n_right: int) -> list[int]:
    """Hopcroft-Karp maximum matching, O(E * sqrt(V)).

    Alternates a BFS that layers the graph by shortest augmenting-path length
    with a DFS that finds a maximal set of vertex-disjoint shortest paths.
    """
    n_left = len(adj)
    INF = float("inf")
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    dist = [0.0] * n_left

    def bfs() -> bool:
        q: deque[int] = deque()
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                w = match_right[v]
                if w == -1:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return found

    def dfs(root: int) -> bool:
        # iterative DFS restricted to the layered graph
        stack = [root]
        iters = {root: iter(adj[root])}
        via: list[int] = []  # right vertex used to descend from stack[i] to stack[i+1]
        while stack:
            u = stack[-1]
            descended = False
            for v in iters[u]:
                w = match_right[v]
                if w == -1:
                    via.append(v)
                    for a, b in zip(stack, via):
                        match_left[a] = b
                        match_right[b] = a
                    return True
                if dist[w] == dist[u] + 1:
                    via.append(v)
                    stack.append(w)
                    iters[w] = iter(adj[w])
                    descended = True
                    break
            if not descended:
                dist[u] = INF
                stack.pop()
                if via:
                    via.pop()
        return False

    while bfs():
        for u in range(n_left):
            if match_left[u] == -1:
                dfs(u)
    return match_left


def konig_vertex_cover(adj: Adj, n_right: int, match_left: list[int]) -> tuple[set[int], set[int]]:
    """Minimum vertex cover from a maximum matching (Konig's theorem).

    Returns ``(left_cover, right_cover)`` with size equal to the matching size.
    """
    n_left = len(adj)
    match_right = [-1] * n_right
    for u, v in enumerate(match_left):
        if v != -1:
            match_right[v] = u
    visited_l = [False] * n_left
    visited_r = [False] * n_right
    q: deque[int] = deque(u for u in range(n_left) if match_left[u] == -1)
    for u in q:
        visited_l[u] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if visited_r[v] or match_left[u] == v:
                continue
            visited_r[v] = True
            w = match_right[v]
            if w != -1 and not visited_l[w]:
                visited_l[w] = True
                q.append(w)
    left = {u for u in range(n_left) if not visited_l[u]}
    right = {v for v in range(n_right) if visited_r[v]}
    return left, right


def hall_violator(adj: Adj, n_right: int, match_left: list[int]) -> set[int] | None:
    """Return a set S of left vertices with |N(S)| < |S| if no perfect left
    matching exists (a Hall's theorem certificate), else ``None``."""
    n_left = len(adj)
    free = [u for u in range(n_left) if match_left[u] == -1]
    if not free:
        return None
    match_right = [-1] * n_right
    for u, v in enumerate(match_left):
        if v != -1:
            match_right[v] = u
    seen_l = {free[0]}
    q: deque[int] = deque([free[0]])
    seen_r: set[int] = set()
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in seen_r:
                continue
            seen_r.add(v)
            w = match_right[v]
            if w != -1 and w not in seen_l:
                seen_l.add(w)
                q.append(w)
    return seen_l
