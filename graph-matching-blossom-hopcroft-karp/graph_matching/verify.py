"""Independent checkers and exponential-time reference solvers."""

from __future__ import annotations

from functools import lru_cache


def is_matching(adj: list[list[int]], mate: list[int]) -> bool:
    """Check ``mate`` is a symmetric matching using only existing edges."""
    for u, v in enumerate(mate):
        if v == -1:
            continue
        if not (0 <= v < len(mate)) or mate[v] != u or v == u or v not in adj[u]:
            return False
    return True


def is_vertex_cover(edges: list[tuple[int, int]], cover: set[int]) -> bool:
    return all(u in cover or v in cover for u, v in edges)


def brute_force_max_matching(adj: list[list[int]]) -> int:
    """Exact maximum matching size by bitmask recursion (n <= ~20)."""
    n = len(adj)
    nbr = [sum(1 << v for v in a if v != u) for u, a in enumerate(adj)]

    @lru_cache(maxsize=None)
    def go(mask: int) -> int:
        if mask == 0:
            return 0
        u = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << u)
        best = go(rest)  # leave u unmatched
        cand = nbr[u] & rest
        while cand:
            v = (cand & -cand).bit_length() - 1
            cand &= cand - 1
            best = max(best, 1 + go(rest & ~(1 << v)))
        return best

    return go((1 << n) - 1)
