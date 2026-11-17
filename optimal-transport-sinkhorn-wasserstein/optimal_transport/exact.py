"""Exact discrete optimal transport via successive shortest augmenting paths.

The transportation problem is a min-cost flow on the bipartite network
``source -> sources -> sinks -> sink``.  We keep a dense residual graph, use
Dijkstra with Johnson potentials after an initial Bellman-Ford pass, and push
the bottleneck amount along each shortest path.
"""
from __future__ import annotations

import numpy as np

from .cost import check_problem, transport_cost

_EPS = 1e-14


def emd(a: np.ndarray, b: np.ndarray, C: np.ndarray, return_plan: bool = True):
    """Solve ``min_P <P, C>`` s.t. ``P 1 = a``, ``P^T 1 = b``, ``P >= 0``.

    Returns ``(cost, plan)`` (or just ``cost`` if ``return_plan`` is False).
    Marginals are normalised to unit mass.
    """
    a, b, C = check_problem(a, b, C)
    n, m = C.shape
    N = n + m + 2
    s, t = n + m, n + m + 1
    cap = np.zeros((N, N))
    cost = np.full((N, N), np.inf)
    for i in range(n):
        cap[s, i] = a[i]
        cost[s, i] = 0.0
        cost[i, s] = 0.0
    for j in range(m):
        cap[n + j, t] = b[j]
        cost[n + j, t] = 0.0
        cost[t, n + j] = 0.0
    for i in range(n):
        for j in range(m):
            cap[i, n + j] = np.inf
            cost[i, n + j] = C[i, j]
            cost[n + j, i] = -C[i, j]
    flow = np.zeros((N, N))

    # Initial potentials = shortest distances in the initial (acyclic) residual graph,
    # which makes every reduced cost non-negative.
    pot = np.zeros(N)
    pot[:n] = 0.0
    pot[n:n + m] = C.min(axis=0)
    pot[t] = pot[n:n + m].min()
    pot[s] = 0.0

    sent = 0.0
    guard = 0
    while sent < 1.0 - 1e-12 and guard < 100 * N * N:
        guard += 1
        dist = np.full(N, np.inf)
        prev = np.full(N, -1, dtype=int)
        done = np.zeros(N, dtype=bool)
        dist[s] = 0.0
        resid = cap - flow > _EPS
        for _ in range(N):
            u = -1
            best = np.inf
            for k in range(N):
                if not done[k] and dist[k] < best:
                    best, u = dist[k], k
            if u < 0:
                break
            done[u] = True
            for v in np.nonzero(resid[u] & ~done)[0]:
                rc = cost[u, v] + pot[u] - pot[v]
                nd = dist[u] + max(rc, 0.0)
                if nd < dist[v] - 1e-15:
                    dist[v] = nd
                    prev[v] = u
        if not np.isfinite(dist[t]):
            break
        pot[np.isfinite(dist)] += dist[np.isfinite(dist)]
        # bottleneck
        delta = 1.0 - sent
        v = t
        while v != s:
            u = prev[v]
            delta = min(delta, cap[u, v] - flow[u, v])
            v = u
        v = t
        while v != s:
            u = prev[v]
            flow[u, v] += delta
            flow[v, u] -= delta
            v = u
        sent += delta

    plan = np.maximum(flow[:n, n:n + m], 0.0)
    val = transport_cost(plan, C)
    return (val, plan) if return_plan else val


def wasserstein_exact(x: np.ndarray, y: np.ndarray, a: np.ndarray | None = None,
                      b: np.ndarray | None = None, p: float = 2.0) -> float:
    """``W_p`` between two weighted point clouds using the exact solver."""
    from .cost import pairwise_cost

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = np.ones(len(x)) if a is None else a
    b = np.ones(len(y)) if b is None else b
    C = pairwise_cost(x, y, p=p)
    return float(emd(a, b, C, return_plan=False) ** (1.0 / p))


def plan_is_feasible(P: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    """Check non-negativity and marginal constraints of a plan."""
    a = np.asarray(a, float) / np.sum(a)
    b = np.asarray(b, float) / np.sum(b)
    return bool(P.min() >= -tol and np.abs(P.sum(1) - a).max() < tol and np.abs(P.sum(0) - b).max() < tol)
