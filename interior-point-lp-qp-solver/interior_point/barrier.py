"""Log-barrier method (Newton centering) with a phase-I feasibility search.

Solves ``min 0.5 x'Qx + c'x  s.t.  Gx <= h`` for strictly feasible problems.
This is the textbook path-following scheme and serves as an independent
cross-check of the primal-dual solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class BarrierResult:
    x: np.ndarray
    objective: float
    status: str
    newton_steps: int
    outer_iters: int
    gap_bound: float


def _centering(Q, c, G, h, x, t, max_newton=60, tol=1e-10,
               stop: Optional[Callable[[np.ndarray], bool]] = None):
    """Minimise ``t*f(x) - sum log(h - Gx)`` by damped Newton; returns (x, steps)."""
    n = x.size
    steps = 0
    for _ in range(max_newton):
        sl = h - G @ x
        grad = t * (Q @ x + c) + G.T @ (1.0 / sl)
        hess = t * Q + G.T @ (G / sl[:, None] ** 2) + 1e-12 * np.eye(n)
        dx = -np.linalg.solve(hess, grad)
        lam2 = float(-grad @ dx)
        if lam2 / 2.0 <= tol:
            break
        # backtracking line search keeping strict feasibility
        phi = lambda z: t * (0.5 * z @ Q @ z + c @ z) - np.sum(np.log(h - G @ z))
        step = 1.0
        while np.any(h - G @ (x + step * dx) <= 0):
            step *= 0.5
        f0 = phi(x)
        while phi(x + step * dx) > f0 - 0.25 * step * lam2 and step > 1e-14:
            step *= 0.5
        x = x + step * dx
        steps += 1
        if stop is not None and stop(x):
            break
    return x, steps


def barrier_solve(Q: np.ndarray, c: np.ndarray, G: np.ndarray, h: np.ndarray,
                  x0: np.ndarray, tol: float = 1e-8, t0: float = 1.0, mu: float = 10.0,
                  stop: Optional[Callable[[np.ndarray], bool]] = None) -> BarrierResult:
    """Barrier method from a strictly feasible ``x0`` (``G x0 < h``)."""
    if np.any(G @ x0 >= h):
        raise ValueError("x0 must be strictly feasible")
    m = h.size
    x, t, total, outer = x0.copy(), t0, 0, 0
    while True:
        x, k = _centering(Q, c, G, h, x, t, stop=stop)
        total += k
        outer += 1
        if stop is not None and stop(x):
            return BarrierResult(x, float(0.5 * x @ Q @ x + c @ x), "stopped", total, outer, m / t)
        if m / t < tol or outer > 60:
            break
        t *= mu
    return BarrierResult(x, float(0.5 * x @ Q @ x + c @ x), "optimal", total, outer, m / t)


def find_strictly_feasible(G: np.ndarray, h: np.ndarray) -> Optional[np.ndarray]:
    """Phase I: minimise ``s`` subject to ``Gx - h <= s`` until ``s < 0``.

    Returns a strictly feasible point or ``None`` if the system is infeasible.
    """
    m, n = G.shape
    G1 = np.hstack([G, -np.ones((m, 1))])
    G1 = np.vstack([G1, np.concatenate([np.zeros(n), [1.0]])])  # cap s above
    x = np.zeros(n)
    s0 = float(np.max(G @ x - h)) + 1.0
    h1 = np.concatenate([h, [s0 + 1.0]])
    z0 = np.concatenate([x, [s0]])
    c1 = np.zeros(n + 1)
    c1[-1] = 1.0
    Q1 = np.zeros((n + 1, n + 1))
    res = barrier_solve(Q1, c1, G1, h1, z0, tol=1e-9, t0=1.0,
                        stop=lambda z: z[-1] < -1e-6)
    if res.x[-1] < 0:
        return res.x[:n]
    return None
