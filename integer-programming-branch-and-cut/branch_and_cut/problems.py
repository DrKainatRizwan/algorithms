"""Seeded generators for classic integer-programming benchmark families."""

from __future__ import annotations

import numpy as np

from .model import MIP


def knapsack(n: int, seed: int = 0, tightness: float = 0.5, multi: int = 1) -> MIP:
    """Correlated (multi-)knapsack, maximise value -> minimise -value, binary variables."""
    rng = np.random.default_rng(seed)
    w = rng.integers(10, 60, size=(multi, n)).astype(float)
    v = w.mean(axis=0).round() + rng.integers(1, 15, size=n)
    cap = np.floor(tightness * w.sum(axis=1))
    return MIP(-v.astype(float), w, cap, lb=np.zeros(n), ub=np.ones(n), name=f"knapsack-{n}-{multi}-s{seed}")


def set_cover(n_sets: int, n_items: int, seed: int = 0, density: float = 0.15) -> MIP:
    """Minimum-cost set cover with a guaranteed feasible cover."""
    rng = np.random.default_rng(seed)
    M = (rng.random((n_items, n_sets)) < density).astype(float)
    for i in range(n_items):
        if M[i].sum() == 0:
            M[i, rng.integers(n_sets)] = 1.0
    cost = rng.integers(1, 10, size=n_sets).astype(float)
    return MIP(cost, -M, -np.ones(n_items), lb=np.zeros(n_sets), ub=np.ones(n_sets), name=f"setcover-{n_sets}x{n_items}-s{seed}")


def facility_location(n_fac: int, n_cust: int, seed: int = 0) -> MIP:
    """Uncapacitated facility location: binary open vars y_i and continuous assignment x_ij."""
    rng = np.random.default_rng(seed)
    fp, cp = rng.random((n_fac, 2)), rng.random((n_cust, 2))
    dist = np.linalg.norm(fp[:, None] - cp[None], axis=2)
    fixed = rng.integers(3, 7, size=n_fac).astype(float)
    nx = n_fac * n_cust
    c = np.concatenate([(10 * dist).ravel(), fixed])
    eq = np.zeros((n_cust, nx + n_fac))
    for j in range(n_cust):
        eq[j, [i * n_cust + j for i in range(n_fac)]] = 1.0
    ub_rows = np.zeros((nx, nx + n_fac))
    for i in range(n_fac):
        for j in range(n_cust):
            r = i * n_cust + j
            ub_rows[r, r] = 1.0
            ub_rows[r, nx + i] = -1.0
    integer = np.concatenate([np.zeros(nx, bool), np.ones(n_fac, bool)])
    ub = np.ones(nx + n_fac)
    return MIP(c, ub_rows, np.zeros(nx), eq, np.ones(n_cust), np.zeros(nx + n_fac), ub, integer,
               name=f"facloc-{n_fac}x{n_cust}-s{seed}")


def assignment_side(n: int, seed: int = 0) -> MIP:
    """n x n assignment with one knapsack side constraint on the chosen cells."""
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 30, size=(n, n)).astype(float)
    w = rng.integers(1, 10, size=(n, n)).astype(float)
    eq = np.zeros((2 * n, n * n))
    for i in range(n):
        eq[i, i * n:(i + 1) * n] = 1.0
        eq[n + i, i::n] = 1.0
    budget = np.floor(w.min(axis=1).sum() + 0.35 * (w.max(axis=1) - w.min(axis=1)).sum())
    return MIP(cost.ravel(), w.reshape(1, -1), np.array([budget]), eq, np.ones(2 * n),
               np.zeros(n * n), np.ones(n * n), name=f"assign-{n}-s{seed}")


def random_ip(n: int, m: int, seed: int = 0, ub: int = 4) -> MIP:
    """Small random pure-integer program with bounded variables (brute-forceable)."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-3, 8, size=(m, n)).astype(float)
    x0 = rng.integers(0, ub + 1, size=n)
    b = A @ x0 + rng.integers(0, 4, size=m)
    c = rng.integers(-9, 4, size=n).astype(float)
    return MIP(c, A, b, lb=np.zeros(n), ub=np.full(n, float(ub)), name=f"randip-{n}x{m}-s{seed}")
