"""FIFO push-relabel with gap heuristic. O(V^3)."""
from __future__ import annotations

from collections import deque

from .graph import FlowNetwork


def push_relabel(g: FlowNetwork, s: int, t: int) -> int:
    """Push a maximum flow in place and return its value.

    Heights may grow up to ``2n - 1`` so that excess which cannot reach the
    sink drains back to the source; the result is therefore a true flow, not
    just a preflow.
    """
    if s == t:
        raise ValueError("source and sink must differ")
    n = g.n
    height = [0] * n
    excess = [0] * n
    height[s] = n
    count = [0] * (2 * n + 1)
    count[0] = n - 1
    count[n] += 1
    cur = [0] * n
    active: deque[int] = deque()
    in_q = [False] * n

    def enqueue(v: int) -> None:
        if v not in (s, t) and not in_q[v] and excess[v] > 0:
            in_q[v] = True
            active.append(v)

    def push(u: int, a: int) -> None:
        v = g.to[a]
        d = min(excess[u], g.cap[a])
        g.cap[a] -= d
        g.cap[a ^ 1] += d
        excess[u] -= d
        excess[v] += d
        enqueue(v)

    for a in g.adj[s]:
        d = g.cap[a]
        if d > 0:
            v = g.to[a]
            g.cap[a] -= d
            g.cap[a ^ 1] += d
            excess[v] += d
            excess[s] -= d
            enqueue(v)

    def relabel(u: int) -> None:
        old = height[u]
        best = 2 * n
        for a in g.adj[u]:
            if g.cap[a] > 0:
                best = min(best, height[g.to[a]] + 1)
        count[old] -= 1
        height[u] = min(best, 2 * n)
        count[height[u]] += 1
        # gap heuristic: nobody at `old` => vertices above it cannot reach t
        if count[old] == 0 and old < n:
            for w in range(n):
                if w != s and old < height[w] < n:
                    count[height[w]] -= 1
                    height[w] = n + 1
                    count[n + 1] += 1
                    cur[w] = 0

    while active:
        u = active.popleft()
        in_q[u] = False
        while excess[u] > 0:
            if cur[u] == len(g.adj[u]):
                relabel(u)
                cur[u] = 0
                if height[u] >= 2 * n:
                    break
                continue
            a = g.adj[u][cur[u]]
            if g.cap[a] > 0 and height[u] == height[g.to[a]] + 1:
                push(u, a)
            else:
                cur[u] += 1
        if excess[u] > 0:
            enqueue(u)
    return excess[t]
