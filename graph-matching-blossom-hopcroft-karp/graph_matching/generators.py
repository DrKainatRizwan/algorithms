"""Seeded random instance generators."""

from __future__ import annotations

import random


def random_bipartite(n_left: int, n_right: int, p: float, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[v for v in range(n_right) if rng.random() < p] for _ in range(n_left)]


def random_graph(n: int, p: float, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].append(v)
                adj[v].append(u)
    return adj


def random_preferences(n: int, seed: int = 0) -> tuple[list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    a = [rng.sample(range(n), n) for _ in range(n)]
    b = [rng.sample(range(n), n) for _ in range(n)]
    return a, b
