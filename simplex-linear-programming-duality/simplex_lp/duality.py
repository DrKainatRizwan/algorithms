"""Duality, optimality certificates and sensitivity analysis.

* :func:`build_dual` constructs the dual LP of a minimisation problem.
* :func:`check_optimality` verifies primal feasibility, dual feasibility and
  complementary slackness (the KKT conditions), which certifies optimality.
* :func:`check_farkas` / :func:`check_unbounded_ray` verify infeasibility and
  unboundedness certificates.
* :func:`rhs_ranging` / :func:`cost_ranging` give the classic sensitivity
  intervals over which the optimal basis stays optimal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import INF, LinearProgram
from .solver import LPSolution


def build_dual(lp: LinearProgram) -> LinearProgram:
    """Dual of ``lp`` (taken in minimisation form) as a maximisation LP.

    Supported variable bounds are ``x >= 0`` and free ``x``; move any other
    bound into an explicit constraint first.  The dual variables are ordered
    as the rows of ``[A_ub; A_eq]``; ``y_ub <= 0`` and ``y_eq`` is free.
    """
    p = lp.as_minimisation()
    for lo, hi in p.bounds:
        if hi != INF or lo not in (0.0, -INF):
            raise ValueError("build_dual supports only x >= 0 or free variables")
    m_ub, m_eq = p.A_ub.shape[0], p.A_eq.shape[0]
    A = np.vstack([p.A_ub, p.A_eq])
    b = np.concatenate([p.b_ub, p.b_eq])
    # dual feasibility: A_j^T y <= c_j (x_j >= 0) or == c_j (x_j free)
    free = [j for j, (lo, _) in enumerate(p.bounds) if lo == -INF]
    pos = [j for j, (lo, _) in enumerate(p.bounds) if lo == 0.0]
    bounds = [(-INF, 0.0)] * m_ub + [(-INF, INF)] * m_eq
    return LinearProgram(
        b, A_ub=A.T[pos] if pos else None, b_ub=p.c[pos] if pos else None,
        A_eq=A.T[free] if free else None, b_eq=p.c[free] if free else None,
        bounds=bounds, maximize=True, name=lp.name + "-dual")


@dataclass
class OptimalityReport:
    """Residuals of the KKT conditions for a candidate primal-dual pair."""

    primal_violation: float
    dual_violation: float
    complementarity: float
    duality_gap: float

    def ok(self, tol: float = 1e-6) -> bool:
        return max(self.primal_violation, self.dual_violation,
                   self.complementarity, abs(self.duality_gap)) <= tol


def check_optimality(lp: LinearProgram, x: np.ndarray, duals: np.ndarray) -> OptimalityReport:
    """Verify KKT conditions for ``x`` and shadow prices ``duals``.

    ``duals`` follow the convention of :class:`LPSolution` (derivative of the
    stated objective with respect to each right-hand side).
    """
    sense = -1.0 if lp.maximize else 1.0
    y = sense * np.asarray(duals, float)  # multipliers of the min form
    c = sense * lp.c
    m_ub = lp.A_ub.shape[0]
    A = np.vstack([lp.A_ub, lp.A_eq])
    b = np.concatenate([lp.b_ub, lp.b_eq])
    primal = lp.max_violation(x)
    dual_v = 0.0
    comp = 0.0
    if m_ub:
        dual_v = max(dual_v, float(np.max(y[:m_ub])))  # y_ub <= 0
        slack = lp.b_ub - lp.A_ub @ x
        comp = max(comp, float(np.max(np.abs(y[:m_ub] * slack))))
    r = c - A.T @ y if A.shape[0] else c.copy()
    for j, (lo, hi) in enumerate(lp.bounds):
        at_lo = np.isfinite(lo) and abs(x[j] - lo) <= 1e-7
        at_hi = np.isfinite(hi) and abs(x[j] - hi) <= 1e-7
        if at_lo and at_hi:
            continue
        if at_lo:
            dual_v = max(dual_v, -r[j])
        elif at_hi:
            dual_v = max(dual_v, r[j])
        else:
            comp = max(comp, abs(r[j]))
    # dual objective: b^T y + bound contributions of the reduced costs
    dual_obj = float(b @ y)
    for j, (lo, hi) in enumerate(lp.bounds):
        if r[j] > 0 and np.isfinite(lo):
            dual_obj += r[j] * lo
        elif r[j] < 0 and np.isfinite(hi):
            dual_obj += r[j] * hi
    gap = float(c @ x) - dual_obj
    return OptimalityReport(primal, dual_v, comp, gap)


def check_farkas(sol: LPSolution, tol: float = 1e-7) -> bool:
    """Check the Farkas certificate returned for an infeasible LP.

    On the standard form ``A u = b, u >= 0`` the vector ``y`` proves
    infeasibility when ``y^T A <= 0`` componentwise while ``y^T b > 0``: any
    feasible ``u`` would give ``0 < y^T b = y^T A u <= 0``.
    """
    if sol.status != "infeasible" or sol.farkas is None:
        return False
    std, y = sol.std, np.asarray(sol.farkas, float)
    return bool(np.max(std.A.T @ y) <= tol and float(std.b @ y) > tol)


def check_unbounded_ray(lp: LinearProgram, ray: np.ndarray, tol: float = 1e-7) -> bool:
    """Check ``ray`` is a feasible improving recession direction."""
    d = np.asarray(ray, float)
    sense = -1.0 if lp.maximize else 1.0
    if sense * float(lp.c @ d) >= -tol:
        return False
    if lp.A_ub.shape[0] and np.max(lp.A_ub @ d) > tol:
        return False
    if lp.A_eq.shape[0] and np.max(np.abs(lp.A_eq @ d)) > tol:
        return False
    for dj, (lo, hi) in zip(d, lp.bounds):
        if dj < -tol and np.isfinite(lo):
            return False
        if dj > tol and np.isfinite(hi):
            return False
    return True


def _basis_data(sol: LPSolution):
    if not sol.optimal:
        raise ValueError("sensitivity analysis needs an optimal solution")
    std, raw = sol.std, sol.raw
    sign = np.where(std.b < 0, -1.0, 1.0)
    As = std.A * sign[:, None]
    kept, basis = raw.kept_rows, raw.basis
    Binv = np.linalg.inv(As[kept][:, basis])
    return std, sign, As, kept, basis, Binv


def rhs_ranging(sol: LPSolution) -> list[tuple[float, float]]:
    """Allowable decrease/increase of each original right-hand side.

    For row ``i`` the returned ``(lo, hi)`` bound the change ``delta`` in
    ``b_i`` for which the current basis stays feasible (hence optimal), so
    the objective changes by exactly ``duals[i] * delta``.
    """
    std, sign, As, kept, basis, Binv = _basis_data(sol)
    bs = std.b * sign
    xb = Binv @ bs[kept]
    out = []
    for i in range(std.n_orig_rows):
        if i not in kept:
            out.append((0.0, 0.0))
            continue
        col = Binv[:, kept.index(i)] * sign[i]
        lo, hi = -INF, INF
        for xbi, ci in zip(xb, col):
            if ci > 1e-12:
                lo = max(lo, -xbi / ci)
            elif ci < -1e-12:
                hi = min(hi, -xbi / ci)
        out.append((float(lo), float(hi)))
    return out


def cost_ranging(sol: LPSolution, lp: LinearProgram) -> list[tuple[float, float] | None]:
    """Range of each original cost coefficient keeping the basis optimal.

    Only variables mapped to a single non-negative column (finite lower bound)
    are supported; others yield ``None``.
    """
    std, sign, As, kept, basis, Binv = _basis_data(sol)
    n = As.shape[1]
    y = np.zeros(As.shape[0])
    y[kept] = np.linalg.solve(As[kept][:, basis].T, std.c[basis])
    r = std.c - As.T @ y
    alpha = Binv @ As[kept]
    out: list[tuple[float, float] | None] = []
    for j in range(lp.n_vars):
        nz = np.nonzero(std.T[j])[0]
        if len(nz) != 1 or std.T[j, nz[0]] != 1.0:
            out.append(None)
            continue
        k = int(nz[0])
        ck = std.c[k]
        if k not in basis:
            lo, hi = ck - r[k], INF
        else:
            p = basis.index(k)
            lo, hi = -INF, INF
            for q in range(n):
                if q in basis:
                    continue
                a = alpha[p, q]
                if a > 1e-12:
                    hi = min(hi, r[q] / a)
                elif a < -1e-12:
                    lo = max(lo, r[q] / a)
            lo, hi = ck + lo, ck + hi
        if std.sense < 0:
            lo, hi = -hi, -lo
        out.append((float(lo), float(hi)))
    return out
