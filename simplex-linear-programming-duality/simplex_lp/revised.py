"""Revised primal simplex that re-solves with the basis matrix each iteration.

Only the basis ``B`` is factorised (via ``numpy.linalg.solve``); the full
tableau is never formed.  Pricing computes ``y = B^-T c_B`` and the reduced
costs ``r = c - A^T y``.  Phase 1 uses one artificial per row.  It is slower
per pivot on tiny problems than the tableau but is the natural starting point
for sparse/large-scale variants and serves as an independent cross-check.
"""

from __future__ import annotations

import numpy as np

from .tableau import RULES, StdResult


def _optimise(c: np.ndarray, A: np.ndarray, b: np.ndarray, basis: list[int], rule: str,
              max_iter: int, tol: float, stats: dict, allowed: int) -> tuple[str, np.ndarray | None]:
    m = A.shape[0]
    stall = 0
    while stats["iters"] < max_iter:
        B = A[:, basis]
        y = np.linalg.solve(B.T, c[basis])
        r = c[:allowed] - A[:, :allowed].T @ y
        for j in basis:
            if j < allowed:
                r[j] = 0.0
        bland = rule == "bland" or (rule == "hybrid" and stall >= 20)
        neg = np.nonzero(r < -tol)[0]
        if neg.size == 0:
            return "optimal", None
        j = int(neg[0]) if bland else int(neg[np.argmin(r[neg])])
        d = np.linalg.solve(B, A[:, j])
        xb = np.linalg.solve(B, b)
        pos = np.nonzero(d > tol)[0]
        if pos.size == 0:
            ray = np.zeros(A.shape[1])
            ray[j] = 1.0
            ray[basis] = -d
            return "unbounded", ray
        ratios = np.maximum(xb[pos], 0.0) / d[pos]
        best = ratios.min()
        ties = pos[ratios <= best + tol * max(1.0, abs(best))]
        row = int(min(ties, key=lambda i: basis[i])) if bland else int(ties[0])
        stall = stall + 1 if best <= tol else 0
        if best <= tol:
            stats["degenerate"] += 1
        basis[row] = j
        stats["iters"] += 1
    return "iteration_limit", None


def revised_simplex(c: np.ndarray, A: np.ndarray, b: np.ndarray, rule: str = "hybrid",
                    max_iter: int = 50_000, tol: float = 1e-9) -> StdResult:
    """Two-phase revised simplex for ``min c^T x, A x = b, x >= 0``."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
    c, A, b = np.asarray(c, float), np.asarray(A, float), np.asarray(b, float)
    m, n = A.shape
    sign = np.where(b < 0, -1.0, 1.0)
    As, bs = A * sign[:, None], b * sign
    stats = {"iters": 0, "degenerate": 0}
    res = StdResult(status="optimal")
    if m == 0:
        if np.any(c < -tol):
            res.status = "unbounded"
        else:
            res.x, res.objective, res.y, res.reduced_costs = np.zeros(n), 0.0, np.zeros(0), c.copy()
        return res

    Aa = np.hstack([As, np.eye(m)])
    c1 = np.concatenate([np.zeros(n), np.ones(m)])
    basis = list(range(n, n + m))
    status, _ = _optimise(c1, Aa, bs, basis, rule, max_iter, tol, stats, n + m)
    res.phase1_iterations = stats["iters"]
    xb = np.linalg.solve(Aa[:, basis], bs)
    if status != "optimal":
        res.status, res.iterations = status, stats["iters"]
        return res
    if float(c1[basis] @ xb) > 1e-7 * max(1.0, float(np.abs(bs).max())):
        y1 = np.linalg.solve(Aa[:, basis].T, c1[basis])
        res.status, res.iterations = "infeasible", stats["iters"]
        res.farkas = y1 * sign
        return res
    # Replace zero-level artificials by structural columns where possible.
    kept = list(range(m))
    for i in range(m):
        if basis[i] >= n:
            Binv_row = np.linalg.solve(Aa[:, basis].T, np.eye(m)[i])
            cand = np.nonzero(np.abs(Binv_row @ As) > 1e-7)[0]
            cand = [j for j in cand if j not in basis]
            if cand:
                basis[i] = int(cand[0])
    redundant = [i for i in range(m) if basis[i] >= n]
    if redundant:
        kept = [i for i in range(m) if i not in redundant]
        basis = [basis[i] for i in kept]
    Ak, bk = As[kept], bs[kept]
    status, ray = _optimise(c, Ak, bk, basis, rule, max_iter, tol, stats, n)
    res.iterations, res.degenerate_pivots = stats["iters"], stats["degenerate"]
    res.basis, res.kept_rows = list(basis), kept
    if status != "optimal":
        res.status, res.ray = status, ray
        return res
    B = Ak[:, basis]
    x = np.zeros(n)
    x[basis] = np.linalg.solve(B, bk)
    yk = np.linalg.solve(B.T, c[basis])
    y = np.zeros(m)
    y[kept] = yk
    res.x, res.objective = x, float(c @ x)
    res.y = y * sign
    res.reduced_costs = c - A.T @ res.y
    return res
