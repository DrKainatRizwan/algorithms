"""High-level entry points for general-form LPs and QPs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .barrier import barrier_solve, find_strictly_feasible
from .core import IPMResult, solve_standard
from .problem import GeneralProblem, to_standard_form


@dataclass
class Solution:
    x: np.ndarray
    objective: float
    status: str
    iterations: int
    max_violation: float
    raw: IPMResult

    @property
    def success(self) -> bool:
        return self.status == "optimal"


def solve(problem: GeneralProblem, tol: float = 1e-9, max_iter: int = 100) -> Solution:
    """Solve a general-form problem with the primal-dual predictor-corrector method."""
    red = to_standard_form(problem)
    res = solve_standard(red.std, tol=tol, max_iter=max_iter)
    x = red.recover(res.x)
    return Solution(x, problem.objective(x), res.status, res.iterations,
                    problem.max_violation(x), res)


def solve_lp(c, A_eq=None, b_eq=None, A_ub=None, b_ub=None, lb=None, ub=None, **kw) -> Solution:
    """Convenience wrapper for linear programs (default bounds ``x >= 0``)."""
    return solve(GeneralProblem(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub), **kw)


def solve_qp(Q, c, A_eq=None, b_eq=None, A_ub=None, b_ub=None, lb=None, ub=None, **kw) -> Solution:
    """Convenience wrapper for convex quadratic programs (default bounds ``x >= 0``)."""
    return solve(GeneralProblem(c=c, Q=Q, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, lb=lb, ub=ub), **kw)


def solve_barrier(Q: Optional[np.ndarray], c: np.ndarray, G: np.ndarray, h: np.ndarray,
                  tol: float = 1e-8) -> Solution:
    """Solve ``min 0.5x'Qx + c'x, Gx <= h`` with the barrier method (phase I included)."""
    c = np.asarray(c, float)
    Qm = np.zeros((c.size, c.size)) if Q is None else np.asarray(Q, float)
    G, h = np.asarray(G, float), np.asarray(h, float)
    x0 = find_strictly_feasible(G, h)
    if x0 is None:
        raise ValueError("problem has no strictly feasible point")
    r = barrier_solve(Qm, c, G, h, x0, tol=tol)
    viol = float(max(np.max(G @ r.x - h), 0.0))
    dummy = IPMResult(r.x, np.zeros(0), np.zeros(0), r.status, r.newton_steps,
                      r.objective, 0.0, 0.0, r.gap_bound)
    return Solution(r.x, r.objective, r.status, r.newton_steps, viol, dummy)
