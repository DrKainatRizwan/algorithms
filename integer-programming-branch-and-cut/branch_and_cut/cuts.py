"""Cutting planes: Gomory mixed-integer cuts from the simplex tableau and knapsack cover cuts."""

from __future__ import annotations

import numpy as np

from .lp import LPResult
from .model import MIP

Cut = tuple[np.ndarray, float]  # (a, b) meaning a.x <= b


def _frac(v: float) -> float:
    return v - np.floor(v)


def gomory_cuts(mip: MIP, res: LPResult, min_frac: float = 0.02, max_cuts: int = 20,
                max_dyn: float = 1e6) -> list[Cut]:
    """Gomory mixed-integer (GMI) cuts for every fractional integer basic variable.

    Each tableau row x_B + sum a_j x_j = b with fractional b gives a valid inequality
    sum g_j x_j >= 1 over non-basic columns; slack columns are substituted back so the
    cut is expressed in the original variables as ``a.x <= b``.
    """
    info = res.info
    if info is None or res.x is None:
        return []
    n = info.n
    T, basis = info.T, info.basis
    ncols = T.shape[1] - 1
    int_col = np.zeros(ncols, bool)
    for j in range(n):
        int_col[j] = bool(mip.integer[j]) and abs(mip.lb[j] - round(mip.lb[j])) < 1e-9
    nonbasic = np.ones(ncols, bool)
    nonbasic[basis] = False
    slack_to_row = {col: i for i, col in info.slack_col.items()}

    cuts: list[tuple[float, np.ndarray, float]] = []
    for r, bj in enumerate(basis):
        if bj >= n or not int_col[bj]:
            continue
        rhs = T[r, -1]
        f0 = _frac(rhs)
        if f0 < min_frac or f0 > 1 - min_frac:
            continue
        g = np.zeros(n)
        const = 0.0
        for j in np.flatnonzero(nonbasic):
            a = T[r, j]
            if abs(a) < 1e-11:
                continue
            if int_col[j]:
                fj = _frac(a)
                coef = fj / f0 if fj <= f0 else (1 - fj) / (1 - f0)
            else:
                coef = a / f0 if a >= 0 else -a / (1 - f0)
            if coef == 0.0:
                continue
            if j < n:
                g[j] += coef
            else:
                arow, kind, brhs = info.rows[slack_to_row[j]]
                if kind == "le":      # s = b - a.y
                    g -= coef * arow
                    const += coef * brhs
                else:                 # surplus: s = a.y - b
                    g += coef * arow
                    const -= coef * brhs
        # g.y + const >= 1, y = x - lb   ->   -g.x <= -(1 - const) - g.lb
        a_cut = -g
        b_cut = -(1.0 - const) - float(g @ mip.lb)
        nz = np.abs(a_cut[np.abs(a_cut) > 1e-9])
        if nz.size == 0 or nz.max() / nz.min() > max_dyn:
            continue
        viol = float(a_cut @ res.x - b_cut)
        scale = float(np.linalg.norm(a_cut))
        if viol / scale < 1e-6:
            continue
        cuts.append((viol / scale, a_cut, b_cut))
    cuts.sort(key=lambda t: -t[0])
    return [(a, b) for _, a, b in cuts[:max_cuts]]


def cover_cuts(mip: MIP, x: np.ndarray, tol: float = 1e-6) -> list[Cut]:
    """Minimal cover inequalities sum_{j in C} x_j <= |C| - 1 for binary knapsack rows."""
    binary = mip.integer & (mip.lb == 0) & (mip.ub == 1)
    out: list[Cut] = []
    for a, b in zip(mip.A_ub, mip.b_ub):
        support = np.flatnonzero(np.abs(a) > 1e-12)
        if support.size < 2 or not np.all(binary[support]) or np.any(a[support] < 0) or b < 0:
            continue
        if a[support].sum() <= b + 1e-9:
            continue
        # greedy cover: prefer items the LP point already sets high
        order = sorted(support, key=lambda j: ((1 - x[j]) / a[j], -a[j]))
        cover, weight = [], 0.0
        for j in order:
            cover.append(j)
            weight += a[j]
            if weight > b + 1e-9:
                break
        # make the cover minimal
        for j in sorted(cover, key=lambda j: x[j]):
            if weight - a[j] > b + 1e-9:
                cover.remove(j)
                weight -= a[j]
        if weight <= b + 1e-9:
            continue
        if sum(x[j] for j in cover) > len(cover) - 1 + tol:
            row = np.zeros(mip.n)
            row[cover] = 1.0
            out.append((row, float(len(cover) - 1)))
    return out
