"""Dense-tableau two-phase primal simplex with selectable pivot rules.

The solver works on ``min c^T x, A x = b, x >= 0``.  Pivot rules:

``bland``    smallest-index entering variable and smallest-label leaving
             variable among ratio-test ties; provably never cycles.
``dantzig``  most negative reduced cost; ties in the ratio test go to the
             first row.  Fast in practice but can cycle on degenerate LPs.
``hybrid``   Dantzig until ``stall_limit`` consecutive degenerate pivots, then
             Bland until a non-degenerate pivot is made.

Final primal/dual quantities are recomputed from the optimal basis with a
linear solve, so accumulated pivoting error does not leak into certificates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RULES = ("bland", "dantzig", "hybrid")


@dataclass
class StdResult:
    """Outcome of a standard-form solve."""

    status: str  # optimal | infeasible | unbounded | cycling | iteration_limit
    x: np.ndarray | None = None
    objective: float | None = None
    y: np.ndarray | None = None
    reduced_costs: np.ndarray | None = None
    basis: list[int] = field(default_factory=list)
    iterations: int = 0
    phase1_iterations: int = 0
    degenerate_pivots: int = 0
    ray: np.ndarray | None = None
    farkas: np.ndarray | None = None
    kept_rows: list[int] = field(default_factory=list)


def _pivot(T: np.ndarray, basis: list[int], row: int, col: int) -> None:
    T[row] /= T[row, col]
    for i in range(T.shape[0]):
        if i != row and T[i, col] != 0.0:
            T[i] -= T[i, col] * T[row]
    basis[row] = col


def _run_phase(T: np.ndarray, basis: list[int], n_cols: int, rule: str, max_iter: int,
               tol: float, detect_cycles: bool, stats: dict) -> tuple[str, np.ndarray | None]:
    """Iterate pivots on the tableau until optimal/unbounded/cycling/limit."""
    m = T.shape[0] - 1
    seen = {tuple(sorted(basis))}
    stall = 0
    while True:
        if stats["iters"] >= max_iter:
            return "iteration_limit", None
        r = T[m, :n_cols]
        use_bland = rule == "bland" or (rule == "hybrid" and stall >= stats["stall_limit"])
        neg = np.nonzero(r < -tol)[0]
        if neg.size == 0:
            return "optimal", None
        j = int(neg[0]) if use_bland else int(neg[np.argmin(r[neg])])
        col = T[:m, j]
        pos = np.nonzero(col > tol)[0]
        if pos.size == 0:
            ray = np.zeros(n_cols)
            ray[j] = 1.0
            for i in range(m):
                ray[basis[i]] = -col[i]
            return "unbounded", ray
        ratios = T[pos, -1] / col[pos]
        best = ratios.min()
        ties = pos[ratios <= best + tol * max(1.0, abs(best))]
        if use_bland:
            row = int(min(ties, key=lambda i: basis[i]))
        else:
            row = int(ties[0])
        degenerate = T[row, -1] <= tol
        _pivot(T, basis, row, j)
        stats["iters"] += 1
        if degenerate:
            stats["degenerate"] += 1
            stall += 1
        else:
            stall = 0
        if detect_cycles:
            key = tuple(sorted(basis))
            if rule == "hybrid" and not use_bland:
                seen = {key}  # Dantzig phase: only watch once Bland takes over
            elif key in seen:
                return "cycling", None
            elif degenerate:
                seen.add(key)
            else:
                seen = {key}


def _finish(c: np.ndarray, A: np.ndarray, b: np.ndarray, basis: list[int],
            kept: list[int], m_full: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Recompute x, objective, duals and reduced costs from a basis."""
    n = A.shape[1]
    Ak = A[kept]
    B = Ak[:, basis]
    xb = np.linalg.solve(B, b[kept])
    x = np.zeros(n)
    x[basis] = xb
    yk = np.linalg.solve(B.T, c[basis])
    y = np.zeros(m_full)
    y[kept] = yk
    return x, float(c @ x), y, c - A.T @ y


def simplex_standard(c: np.ndarray, A: np.ndarray, b: np.ndarray, rule: str = "hybrid",
                     basis: list[int] | None = None, max_iter: int = 50_000,
                     tol: float = 1e-9, detect_cycles: bool = True,
                     stall_limit: int = 20) -> StdResult:
    """Solve ``min c^T x, A x = b, x >= 0`` by the two-phase tableau simplex.

    If ``basis`` is given it must be a feasible basis (``B^-1 b >= 0``) and
    phase 1 is skipped.  Rows with negative ``b`` are negated internally; the
    returned duals refer to the rows as supplied.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
    c = np.asarray(c, float)
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m, n = A.shape
    sign = np.where(b < 0, -1.0, 1.0)
    As, bs = A * sign[:, None], b * sign
    stats = {"iters": 0, "degenerate": 0, "stall_limit": stall_limit}
    res = StdResult(status="optimal")
    if m == 0:
        if np.any(c < -tol):
            res.status = "unbounded"
            j = int(np.argmin(c))
            res.ray = np.zeros(n)
            res.ray[j] = 1.0
        else:
            res.x, res.objective, res.y, res.reduced_costs = np.zeros(n), 0.0, np.zeros(0), c.copy()
        return res

    kept = list(range(m))
    if basis is None:
        # Phase 1: artificial variables n..n+m-1, minimise their sum.
        T = np.zeros((m + 1, n + m + 1))
        T[:m, :n] = As
        T[:m, n:n + m] = np.eye(m)
        T[:m, -1] = bs
        T[m, :n] = -As.sum(axis=0)
        T[m, -1] = -bs.sum()
        basis = list(range(n, n + m))
        status, _ = _run_phase(T, basis, n + m, rule, max_iter, tol, detect_cycles, stats)
        res.phase1_iterations = stats["iters"]
        if status in ("iteration_limit", "cycling"):
            res.status, res.iterations = status, stats["iters"]
            return res
        scale = max(1.0, float(np.abs(bs).max()))
        if -T[m, -1] > 1e-7 * scale:
            res.status = "infeasible"
            res.iterations = stats["iters"]
            art_reduced = T[m, n:n + m]
            res.farkas = (1.0 - art_reduced) * sign
            return res
        # Drive remaining (zero-level) artificials out of the basis.
        drop: list[int] = []
        for i in range(m):
            if basis[i] >= n:
                cand = np.nonzero(np.abs(T[i, :n]) > 1e-7)[0]
                if cand.size:
                    _pivot(T, basis, i, int(cand[0]))
                else:
                    drop.append(i)
        keep_idx = [i for i in range(m) if i not in drop]
        kept = keep_idx
        T = np.vstack([T[keep_idx][:, list(range(n)) + [n + m]], np.zeros((1, n + 1))])
        basis = [basis[i] for i in keep_idx]
    else:
        basis = list(basis)
        B = As[:, basis]
        T = np.zeros((m + 1, n + 1))
        sol = np.linalg.solve(B, np.hstack([As, bs[:, None]]))
        T[:m] = sol
        if np.any(T[:m, -1] < -1e-9):
            raise ValueError("supplied basis is not primal feasible")

    # Phase 2 reduced-cost row.
    mk = len(kept)
    cb = c[basis]
    T[mk, :n] = c - cb @ T[:mk, :n]
    T[mk, -1] = -cb @ T[:mk, -1]
    status, ray = _run_phase(T, basis, n, rule, max_iter, tol, detect_cycles, stats)
    res.iterations = stats["iters"]
    res.degenerate_pivots = stats["degenerate"]
    res.basis = list(basis)
    res.kept_rows = kept
    if status != "optimal":
        res.status = status
        res.ray = ray
        return res
    x, obj, y, rc = _finish(c, As, bs, basis, kept, m)
    res.x, res.objective, res.reduced_costs = x, obj, rc
    res.y = y * sign
    return res
