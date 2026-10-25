"""High-level entry point: solve a general-form :class:`LinearProgram`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import LinearProgram, StandardForm
from .revised import revised_simplex
from .tableau import StdResult, simplex_standard


@dataclass
class LPSolution:
    """Solution of a general-form LP in terms of the original variables.

    ``duals[i]`` is the shadow price d(objective)/d(rhs_i) for the ``i``-th
    original row (inequalities first, then equalities).  ``reduced_costs`` are
    ``c - A^T y`` in the minimisation form of the problem.
    """

    status: str
    x: np.ndarray | None = None
    objective: float | None = None
    duals: np.ndarray | None = None
    reduced_costs: np.ndarray | None = None
    iterations: int = 0
    std: StandardForm | None = None
    raw: StdResult | None = None
    ray: np.ndarray | None = None
    farkas: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    @property
    def optimal(self) -> bool:
        return self.status == "optimal"


def solve(lp: LinearProgram, method: str = "tableau", rule: str = "hybrid",
          max_iter: int = 50_000) -> LPSolution:
    """Solve ``lp`` with the chosen method (``tableau`` or ``revised``) and rule."""
    std = lp.to_standard()
    if method == "tableau":
        raw = simplex_standard(std.c, std.A, std.b, rule=rule, max_iter=max_iter)
    elif method == "revised":
        raw = revised_simplex(std.c, std.A, std.b, rule=rule, max_iter=max_iter)
    else:
        raise ValueError(f"unknown method {method!r}")
    sol = LPSolution(status=raw.status, iterations=raw.iterations, std=std, raw=raw)
    n_rows = std.n_orig_rows
    if raw.status == "optimal":
        sol.x = std.recover(raw.x)
        sol.objective = std.objective(raw.x)
        y_min = raw.y[:n_rows]
        sol.duals = std.sense * y_min
        sol.reduced_costs = (std.sense * lp.c) - np.vstack([lp.A_ub, lp.A_eq]).T @ y_min \
            if n_rows else std.sense * lp.c
    elif raw.status == "unbounded" and raw.ray is not None:
        sol.ray = std.T @ raw.ray[: std.n_struct]
    elif raw.status == "infeasible" and raw.farkas is not None:
        sol.farkas = raw.farkas
    return sol
