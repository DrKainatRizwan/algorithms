"""TSP instances, tour evaluation and exact solvers for small sizes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    """A Euclidean TSP instance: city coordinates and the distance matrix."""

    coords: np.ndarray
    dist: np.ndarray

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    @staticmethod
    def from_coords(coords: np.ndarray) -> "Instance":
        coords = np.asarray(coords, dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        return Instance(coords, np.sqrt((diff ** 2).sum(-1)))


def random_instance(n: int, seed: int = 0) -> Instance:
    """Uniform random cities in the unit square."""
    rng = np.random.default_rng(seed)
    return Instance.from_coords(rng.random((n, 2)))


def circle_instance(n: int) -> Instance:
    """Cities on a circle; the optimal tour is the circle order (known length)."""
    ang = 2 * np.pi * np.arange(n) / n
    return Instance.from_coords(np.stack([np.cos(ang), np.sin(ang)], axis=1))


def tour_length(inst: Instance, tour: np.ndarray | list[int]) -> float:
    """Length of the closed tour."""
    t = np.asarray(tour)
    return float(inst.dist[t, np.roll(t, -1)].sum())


def is_valid_tour(inst: Instance, tour: np.ndarray | list[int]) -> bool:
    """True when tour is a permutation of all cities."""
    t = np.asarray(tour)
    return t.shape == (inst.n,) and bool(np.array_equal(np.sort(t), np.arange(inst.n)))


def held_karp(inst: Instance) -> tuple[float, list[int]]:
    """Exact O(n^2 2^n) dynamic programme; practical up to ~14 cities."""
    n = inst.n
    if n <= 3:
        t = list(range(n))
        return tour_length(inst, t), t
    d = inst.dist
    m = n - 1  # city n-1 is the fixed start
    size = 1 << m
    dp = np.full((size, m), np.inf)
    parent = np.full((size, m), -1, dtype=int)
    for j in range(m):
        dp[1 << j, j] = d[n - 1, j]
    for mask in range(1, size):
        for j in range(m):
            if not (mask >> j) & 1 or not np.isfinite(dp[mask, j]):
                continue
            base = dp[mask, j]
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                c = base + d[j, k]
                if c < dp[nm, k]:
                    dp[nm, k] = c
                    parent[nm, k] = j
    full = size - 1
    totals = dp[full] + d[:m, n - 1]
    last = int(np.argmin(totals))
    best = float(totals[last])
    tour = [n - 1]
    mask, j = full, last
    path = []
    while j != -1:
        path.append(j)
        pj = int(parent[mask, j])
        mask ^= 1 << j
        j = pj
    tour.extend(reversed(path))
    return best, tour


def brute_force(inst: Instance) -> tuple[float, list[int]]:
    """Exhaustive search over (n-1)! tours; for validation only."""
    best, best_t = np.inf, None
    for perm in itertools.permutations(range(1, inst.n)):
        t = [0, *perm]
        L = tour_length(inst, t)
        if L < best:
            best, best_t = L, t
    return float(best), best_t
