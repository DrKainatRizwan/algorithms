"""Interval DP: matrix chain, optimal BST (Knuth), burst balloons, polygon triangulation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def matrix_chain(dims: Sequence[int]) -> tuple[int, np.ndarray]:
    """Minimum scalar multiplications for a chain; matrix i is ``dims[i] x dims[i+1]``.

    Returns (cost, split table). O(n^3) time, O(n^2) space.
    """
    n = len(dims) - 1
    if n < 1:
        return 0, np.zeros((0, 0), dtype=int)
    cost = np.zeros((n, n), dtype=np.int64)
    split = np.zeros((n, n), dtype=int)
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            best, arg = None, i
            for k in range(i, j):
                c = cost[i, k] + cost[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                if best is None or c < best:
                    best, arg = c, k
            cost[i, j], split[i, j] = best, arg
    return int(cost[0, n - 1]), split


def parenthesize(split: np.ndarray, i: int = 0, j: int | None = None) -> str:
    """Render the optimal parenthesisation (matrices named A0, A1, ...)."""
    if j is None:
        j = split.shape[0] - 1
    if i == j:
        return f"A{i}"
    k = split[i, j]
    return f"({parenthesize(split, i, k)} x {parenthesize(split, k + 1, j)})"


def optimal_bst(p: Sequence[float], q: Sequence[float]) -> tuple[float, list[list[int]]]:
    """Optimal binary search tree with Knuth's O(n^2) root-monotonicity speed-up.

    ``p[i]`` are successful-search probabilities (n keys), ``q[i]`` the n+1 gap
    probabilities. Returns (expected comparisons, root table with 1-based keys).
    """
    n = len(p)
    if len(q) != n + 1:
        raise ValueError("q must have len(p)+1 entries")
    e = [[0.0] * (n + 1) for _ in range(n + 2)]
    w = [[0.0] * (n + 1) for _ in range(n + 2)]
    root = [[0] * (n + 1) for _ in range(n + 2)]
    for i in range(1, n + 2):
        e[i][i - 1] = q[i - 1]
        w[i][i - 1] = q[i - 1]
    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            w[i][j] = w[i][j - 1] + p[j - 1] + q[j]
            lo = root[i][j - 1] if length > 1 else i
            hi = root[i + 1][j] if length > 1 else j
            best, arg = math.inf, lo
            for r in range(lo, hi + 1):
                t = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if t < best:
                    best, arg = t, r
            e[i][j], root[i][j] = best, arg
    return e[1][n], root


def burst_balloons(nums: Sequence[int]) -> tuple[int, list[int]]:
    """Maximum coins from bursting all balloons; also returns an optimal burst order. O(n^3)."""
    a = [1, *nums, 1]
    n = len(a)
    dp = [[0] * n for _ in range(n)]
    last = [[-1] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):
                v = dp[i][k] + dp[k][j] + a[i] * a[k] * a[j]
                if v > dp[i][j] or last[i][j] < 0:
                    dp[i][j], last[i][j] = v, k
    order: list[int] = []

    def walk(i: int, j: int) -> None:
        if j - i < 2:
            return
        k = last[i][j]
        walk(i, k)
        walk(k, j)
        order.append(k - 1)  # k is burst last within (i, j)

    walk(0, n - 1)
    return dp[0][n - 1], order


def polygon_triangulation(points: Sequence[tuple[float, float]]) -> float:
    """Minimum total perimeter triangulation of a convex polygon, O(n^3)."""
    n = len(points)
    if n < 3:
        return 0.0

    def d(a: int, b: int) -> float:
        return math.dist(points[a], points[b])

    dp = [[0.0] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            dp[i][j] = min(dp[i][k] + dp[k][j] + d(i, k) + d(k, j) + d(i, j) for k in range(i + 1, j))
    return dp[0][n - 1]
