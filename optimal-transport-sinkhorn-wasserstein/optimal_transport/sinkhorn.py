"""Entropic optimal transport: Sinkhorn scaling, log-domain stabilisation,
epsilon scaling and Sinkhorn divergences.

Solves ``min_P <P, C> - eps * H(P)`` with ``H(P) = -sum P (log P - 1)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cost import check_problem, transport_cost


@dataclass
class SinkhornResult:
    """Output of a Sinkhorn solve."""
    plan: np.ndarray
    cost: float          # <P, C>, the transport cost of the regularised plan
    f: np.ndarray        # dual potential on the source side
    g: np.ndarray        # dual potential on the target side
    n_iter: int
    marginal_error: float


def _logsumexp(M: np.ndarray, axis: int) -> np.ndarray:
    mx = M.max(axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    out = np.log(np.exp(M - mx).sum(axis=axis, keepdims=True)) + mx
    return np.squeeze(out, axis=axis)


def sinkhorn(a, b, C, eps: float = 0.05, max_iter: int = 2000, tol: float = 1e-9,
             f0: np.ndarray | None = None, g0: np.ndarray | None = None) -> SinkhornResult:
    """Log-domain Sinkhorn iterations (stable for small ``eps``)."""
    a, b, C = check_problem(a, b, C)
    if eps <= 0:
        raise ValueError("eps must be positive")
    la, lb = np.log(np.maximum(a, 1e-300)), np.log(np.maximum(b, 1e-300))
    f = np.zeros(a.size) if f0 is None else f0.copy()
    g = np.zeros(b.size) if g0 is None else g0.copy()
    err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        f = -eps * _logsumexp((g[None, :] - C) / eps + lb[None, :], axis=1)
        g = -eps * _logsumexp((f[:, None] - C) / eps + la[:, None], axis=0)
        if it % 5 == 0 or it == max_iter:
            P = np.exp((f[:, None] + g[None, :] - C) / eps) * a[:, None] * b[None, :]
            err = float(np.abs(P.sum(1) - a).sum())
            if err < tol:
                break
    P = np.exp((f[:, None] + g[None, :] - C) / eps) * a[:, None] * b[None, :]
    err = float(np.abs(P.sum(1) - a).sum() + np.abs(P.sum(0) - b).sum())
    return SinkhornResult(P, transport_cost(P, C), f, g, it, err)


def sinkhorn_classic(a, b, C, eps: float = 0.1, max_iter: int = 5000, tol: float = 1e-10) -> SinkhornResult:
    """Textbook scaling form ``P = diag(u) K diag(v)`` with ``K = exp(-C/eps)``.

    Numerically fragile for small ``eps``; kept as a reference implementation.
    """
    a, b, C = check_problem(a, b, C)
    K = np.exp(-C / eps)
    u = np.ones(a.size)
    v = np.ones(b.size)
    it = 0
    for it in range(1, max_iter + 1):
        u = a / (K @ v)
        v = b / (K.T @ u)
        if it % 5 == 0:
            P = u[:, None] * K * v[None, :]
            if np.abs(P.sum(1) - a).sum() < tol:
                break
    P = u[:, None] * K * v[None, :]
    err = float(np.abs(P.sum(1) - a).sum() + np.abs(P.sum(0) - b).sum())
    return SinkhornResult(P, transport_cost(P, C), eps * np.log(u), eps * np.log(v), it, err)


def sinkhorn_eps_scaling(a, b, C, eps: float = 0.005, eps_start: float | None = None,
                         factor: float = 0.5, iters_per_stage: int = 200, tol: float = 1e-9) -> SinkhornResult:
    """Anneal ``eps`` geometrically from a large value, warm-starting duals each stage."""
    a, b, C = check_problem(a, b, C)
    cur = float(C.max()) if eps_start is None else eps_start
    f = g = None
    total = 0
    while True:
        cur = max(cur, eps)
        res = sinkhorn(a, b, C, cur, max_iter=iters_per_stage, tol=tol, f0=f, g0=g)
        f, g = res.f, res.g
        total += res.n_iter
        if cur <= eps:
            break
        cur *= factor
    res = sinkhorn(a, b, C, eps, max_iter=5000, tol=tol, f0=f, g0=g)
    res.n_iter += total
    return res


def regularised_ot_value(a, b, C, eps: float, res: SinkhornResult | None = None) -> float:
    """Dual value ``<f,a> + <g,b>`` of the entropic problem (equals the primal at convergence)."""
    a, b, C = check_problem(a, b, C)
    res = res or sinkhorn(a, b, C, eps)
    return float(res.f @ a + res.g @ b)


def sinkhorn_divergence(a, b, Cab, Caa, Cbb, eps: float = 0.05) -> float:
    """Debiased divergence ``S = OT_e(a,b) - (OT_e(a,a) + OT_e(b,b)) / 2``."""
    ab = regularised_ot_value(a, b, Cab, eps)
    aa = regularised_ot_value(a, a, Caa, eps)
    bb = regularised_ot_value(b, b, Cbb, eps)
    return float(ab - 0.5 * (aa + bb))
