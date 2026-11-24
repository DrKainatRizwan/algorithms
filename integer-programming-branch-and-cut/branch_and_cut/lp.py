"""Dense two-phase primal simplex returning the optimal tableau (needed for cut generation)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


@dataclass
class TableauInfo:
    """Final simplex tableau in the shifted space y = x - lb."""

    T: np.ndarray              # m x (ncols + 1), last column is the right-hand side
    basis: list[int]
    n: int                     # number of structural columns
    rows: list[tuple[np.ndarray, str, float]]   # normalised rows (a, kind, rhs) of the y-space system
    slack_col: dict[int, int]  # row index in `rows` -> slack/surplus column
    row_of_tableau: list[int]  # tableau row -> index into `rows`


@dataclass
class LPResult:
    status: str                # 'optimal' | 'infeasible' | 'unbounded'
    x: np.ndarray | None = None
    obj: float = np.inf
    iterations: int = 0
    info: TableauInfo | None = field(default=None, repr=False)


def _pivot(T: np.ndarray, r: int, c: int) -> None:
    T[r] /= T[r, c]
    col = T[:, c].copy()
    col[r] = 0.0
    T -= np.outer(col, T[r])


def _run(T: np.ndarray, basis: list[int], cost: np.ndarray, allowed: int, max_iter: int) -> tuple[str, int]:
    """Minimise cost over tableau T (rows B^-1[A|b]); only columns < allowed may enter."""
    m = len(basis)
    it = 0
    degenerate = 0
    while it < max_iter:
        cb = cost[basis]
        red = cost[:allowed] - cb @ T[:, :allowed]
        bland = degenerate > 15
        if bland:
            cand = np.flatnonzero(red < -1e-9)
            if cand.size == 0:
                return "optimal", it
            e = int(cand[0])
        else:
            e = int(np.argmin(red))
            if red[e] >= -1e-9:
                return "optimal", it
        col = T[:, e]
        pos = col > EPS
        if not pos.any():
            return "unbounded", it
        ratios = np.full(m, np.inf)
        ratios[pos] = T[pos, -1] / col[pos]
        rmin = ratios.min()
        ties = np.flatnonzero(ratios <= rmin + 1e-12)
        r = int(min(ties, key=lambda i: basis[i])) if bland else int(ties[np.argmax(col[ties])])
        degenerate = degenerate + 1 if rmin <= 1e-12 else 0
        _pivot(T, r, e)
        T[r, -1] = max(T[r, -1], 0.0) if abs(T[r, -1]) < 1e-11 else T[r, -1]
        basis[r] = e
        it += 1
    return "iteration_limit", it


def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=None, ub=None,
             want_tableau: bool = False, max_iter: int = 20000) -> LPResult:
    """Solve min c.x s.t. A_ub x <= b_ub, A_eq x = b_eq, lb <= x <= ub (lb finite)."""
    c = np.asarray(c, float)
    n = c.size
    A_ub = np.zeros((0, n)) if A_ub is None else np.atleast_2d(np.asarray(A_ub, float))
    b_ub = np.zeros(0) if b_ub is None else np.asarray(b_ub, float).ravel()
    A_eq = np.zeros((0, n)) if A_eq is None else np.atleast_2d(np.asarray(A_eq, float))
    b_eq = np.zeros(0) if b_eq is None else np.asarray(b_eq, float).ravel()
    lb = np.zeros(n) if lb is None else np.asarray(lb, float)
    ub = np.full(n, np.inf) if ub is None else np.asarray(ub, float)
    if np.any(ub < lb - 1e-9):
        return LPResult("infeasible")

    raw: list[tuple[np.ndarray, str, float]] = []
    for a, b in zip(A_ub, b_ub - A_ub @ lb if A_ub.size else b_ub):
        raw.append((a.copy(), "le", float(b)))
    for j in np.flatnonzero(np.isfinite(ub)):
        e = np.zeros(n)
        e[j] = 1.0
        raw.append((e, "le", float(ub[j] - lb[j])))
    for a, b in zip(A_eq, b_eq - A_eq @ lb if A_eq.size else b_eq):
        raw.append((a.copy(), "eq", float(b)))

    rows: list[tuple[np.ndarray, str, float]] = []
    for a, kind, b in raw:
        if b < 0:
            a, b = -a, -b
            kind = {"le": "ge", "ge": "le", "eq": "eq"}[kind]
        rows.append((a, kind, b))
    m = len(rows)
    if m == 0:
        if np.any(c < -1e-12):
            return LPResult("unbounded")
        return LPResult("optimal", lb.copy(), float(c @ lb))

    n_slack = sum(1 for _, k, _ in rows if k != "eq")
    n_art = sum(1 for _, k, _ in rows if k != "le")
    ncols = n + n_slack + n_art
    T = np.zeros((m, ncols + 1))
    basis: list[int] = []
    slack_col: dict[int, int] = {}
    s_next, a_next = n, n + n_slack
    for i, (a, kind, b) in enumerate(rows):
        T[i, :n] = a
        T[i, -1] = b
        if kind == "le":
            T[i, s_next] = 1.0
            slack_col[i] = s_next
            basis.append(s_next)
            s_next += 1
        else:
            if kind == "ge":
                T[i, s_next] = -1.0
                slack_col[i] = s_next
                s_next += 1
            T[i, a_next] = 1.0
            basis.append(a_next)
            a_next += 1

    total_iter = 0
    art_start = n + n_slack
    if n_art:
        w = np.zeros(ncols)
        w[art_start:] = 1.0
        status, it = _run(T, basis, w, ncols, max_iter)
        total_iter += it
        if status != "optimal" or float(w[basis] @ T[:, -1]) > 1e-7:
            return LPResult("infeasible", iterations=total_iter)
        keep = list(range(m))
        for r in range(m):
            if basis[r] >= art_start:
                nz = np.flatnonzero(np.abs(T[r, :art_start]) > 1e-9)
                if nz.size:
                    j = int(nz[np.argmax(np.abs(T[r, nz]))])
                    _pivot(T, r, j)
                    basis[r] = j
                else:
                    keep.remove(r)
        T = T[keep]
        basis = [basis[r] for r in keep]
        T = np.hstack([T[:, :art_start], T[:, -1:]])
        row_map = keep
    else:
        row_map = list(range(m))

    cost = np.zeros(art_start)
    cost[:n] = c
    status, it = _run(T, basis, cost, art_start, max_iter)
    total_iter += it
    if status != "optimal":
        return LPResult(status, iterations=total_iter)
    y = np.zeros(art_start)
    y[basis] = T[:, -1]
    x = lb + y[:n]
    info = TableauInfo(T, basis, n, rows, slack_col, row_map) if want_tableau else None
    return LPResult("optimal", x, float(c @ x), total_iter, info)

