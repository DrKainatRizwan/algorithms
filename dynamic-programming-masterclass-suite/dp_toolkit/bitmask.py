"""Bitmask DP: Held-Karp TSP, assignment, sum-over-subsets, Hamiltonian paths, set cover."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def held_karp(dist: np.ndarray) -> tuple[float, list[int]]:
    """Exact TSP tour from city 0 in O(2^n n^2) time and O(2^n n) space."""
    n = len(dist)
    if n == 1:
        return 0.0, [0]
    inf = float("inf")
    full = 1 << (n - 1)  # subsets of cities 1..n-1
    dp = np.full((full, n - 1), inf)
    par = np.full((full, n - 1), -1, dtype=int)
    for j in range(n - 1):
        dp[1 << j, j] = dist[0, j + 1]
    for mask in range(1, full):
        for j in range(n - 1):
            if not mask >> j & 1 or dp[mask, j] == inf:
                continue
            for k in range(n - 1):
                if mask >> k & 1:
                    continue
                nm = mask | 1 << k
                c = dp[mask, j] + dist[j + 1, k + 1]
                if c < dp[nm, k]:
                    dp[nm, k], par[nm, k] = c, j
    mask = full - 1
    totals = [dp[mask, j] + dist[j + 1, 0] for j in range(n - 1)]
    j = int(np.argmin(totals))
    best = float(totals[j])
    tour = []
    while j != -1:
        tour.append(j + 1)
        pj = par[mask, j]
        mask ^= 1 << j
        j = int(pj)
    return best, [0] + tour[::-1]


def assignment_dp(cost: np.ndarray) -> tuple[float, list[int]]:
    """Minimum-cost assignment of n workers to n jobs by DP over job subsets, O(2^n n)."""
    n = len(cost)
    inf = float("inf")
    dp = [inf] * (1 << n)
    pick = [-1] * (1 << n)
    dp[0] = 0.0
    for mask in range(1 << n):
        if dp[mask] == inf:
            continue
        i = bin(mask).count("1")
        if i >= n:
            continue
        for j in range(n):
            if not mask >> j & 1:
                nm = mask | 1 << j
                c = dp[mask] + cost[i, j]
                if c < dp[nm]:
                    dp[nm], pick[nm] = c, j
    assign = [0] * n
    mask = (1 << n) - 1
    for i in range(n - 1, -1, -1):
        j = pick[mask]
        assign[i] = j
        mask ^= 1 << j
    return dp[-1], assign


def sum_over_subsets(f: Sequence[int], bits: int) -> list[int]:
    """Zeta transform: ``g[m] = sum of f[s] for all s subset of m`` in O(2^b * b)."""
    g = list(f)
    for b in range(bits):
        for m in range(1 << bits):
            if m >> b & 1:
                g[m] += g[m ^ (1 << b)]
    return g


def count_hamiltonian_paths(adj: Sequence[Sequence[int]]) -> int:
    """Count directed Hamiltonian paths (all start/end vertices), O(2^n n^2)."""
    n = len(adj)
    dp = [[0] * n for _ in range(1 << n)]
    for v in range(n):
        dp[1 << v][v] = 1
    for mask in range(1, 1 << n):
        for v in range(n):
            c = dp[mask][v]
            if not c:
                continue
            for u in range(n):
                if adj[v][u] and not mask >> u & 1:
                    dp[mask | 1 << u][u] += c
    return sum(dp[(1 << n) - 1])


def min_set_cover(universe_size: int, sets: Sequence[Sequence[int]]) -> list[int] | None:
    """Minimum number of sets covering ``range(universe_size)`` via DP on covered masks."""
    full = (1 << universe_size) - 1
    masks = [sum(1 << e for e in s) for s in sets]
    INF = 10**9
    dp = [INF] * (full + 1)
    prev: list[tuple[int, int]] = [(-1, -1)] * (full + 1)
    dp[0] = 0
    for m in range(full + 1):
        if dp[m] == INF:
            continue
        for i, s in enumerate(masks):
            nm = m | s
            if nm != m and dp[m] + 1 < dp[nm]:
                dp[nm], prev[nm] = dp[m] + 1, (m, i)
    if dp[full] == INF:
        return None
    out, m = [], full
    while m:
        m, i = prev[m]
        out.append(i)
    return sorted(out)
