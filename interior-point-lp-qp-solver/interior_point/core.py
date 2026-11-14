"""Mehrotra predictor-corrector primal-dual interior-point method.

Solves the standard-form convex QP ``min 0.5 x'Qx + c'x, Ax=b, x>=0`` (an LP when
``Q = 0``) from an infeasible start. Each iteration factors the symmetric
augmented KKT system ``[[Q + S/X, -A'], [A, 0]]``; sign-unrestricted variables
(``StandardForm.free``) get no complementarity term and a tiny proximal
regulariser instead, so they never need to be split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .problem import StandardForm


@dataclass
class IPMResult:
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    status: str
    iterations: int
    objective: float
    primal_res: float
    dual_res: float
    gap: float
    history: List[dict] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.status == "optimal"


def _max_step(v: np.ndarray, dv: np.ndarray) -> float:
    neg = dv < 0
    return float(np.min(-v[neg] / dv[neg])) if neg.any() else np.inf


def _starting_point(sf: StandardForm):
    n, m = sf.n, sf.m
    scale = max(1.0, np.linalg.norm(sf.b), np.linalg.norm(sf.c))
    return np.full(n, scale), np.zeros(m), np.full(n, scale)


def solve_standard(sf: StandardForm, tol: float = 1e-9, max_iter: int = 100,
                   record: bool = True, reg: float = 1e-11) -> IPMResult:
    """Run the predictor-corrector method on a :class:`StandardForm` problem."""
    Q, c, A, b = sf.Q, sf.c, sf.A, sf.b
    n, m = sf.n, sf.m
    free = sf.free
    bd = ~free  # variables with a complementarity condition
    nb = max(int(bd.sum()), 1)
    x, y, s = _starting_point(sf)
    x = np.where(free, 0.0, x)
    s = np.where(free, 0.0, s)
    xs = np.where(free, 1.0, x)  # safe divisor
    bnorm, cnorm = 1.0 + np.linalg.norm(b), 1.0 + np.linalg.norm(c)
    history: List[dict] = []
    status = "max_iter"
    it = 0
    for it in range(max_iter + 1):
        rp = A @ x - b
        rd = Q @ x + c - A.T @ y - s
        mu = float(x[bd] @ s[bd]) / nb
        pobj = 0.5 * x @ Q @ x + c @ x
        dobj = -0.5 * x @ Q @ x + b @ y
        pres, dres = np.linalg.norm(rp) / bnorm, np.linalg.norm(rd) / cnorm
        gap = abs(pobj - dobj) / (1.0 + abs(pobj))
        if record:
            history.append(dict(iter=it, mu=mu, pres=pres, dres=dres, gap=gap, obj=float(pobj)))
        if max(pres, dres, gap) < tol:
            status = "optimal"
            break
        if it == max_iter:
            break
        if not np.all(np.isfinite(x)) or max(np.linalg.norm(x), np.linalg.norm(s)) > 1e14:
            status = "infeasible_or_unbounded"
            break

        xs = np.where(free, 1.0, x)
        d = np.where(free, reg, s / xs)
        K = np.zeros((n + m, n + m))
        K[:n, :n] = Q + np.diag(d)
        K[:n, n:] = -A.T
        K[n:, :n] = A
        K[n:, n:] = -reg * np.eye(m)
        Kinv = np.linalg.inv(K)

        def direction(rc: np.ndarray):
            rc = np.where(free, 0.0, rc)
            sol = Kinv @ np.concatenate([-(rc / xs + rd), -rp])
            dx, dy = sol[:n], sol[n:]
            ds = np.where(free, 0.0, (-rc - s * dx) / xs)
            return dx, dy, ds

        def step_to_boundary(dx, ds):
            return min(_max_step(x[bd], dx[bd]), _max_step(s[bd], ds[bd]))

        # predictor (affine scaling) step
        dx_a, dy_a, ds_a = direction(x * s)
        a_aff = min(1.0, step_to_boundary(dx_a, ds_a))
        mu_aff = float((x + a_aff * dx_a)[bd] @ (s + a_aff * ds_a)[bd]) / nb
        sigma = (mu_aff / mu) ** 3
        # corrector step
        dx, dy, ds = direction(x * s + dx_a * ds_a - sigma * mu)
        alpha = min(1.0, 0.995 * step_to_boundary(dx, ds))
        x, y, s = x + alpha * dx, y + alpha * dy, s + alpha * ds
        if alpha < 1e-12:
            status = "stalled"
            break
    return IPMResult(x, y, s, status, it, float(sf.objective(x)),
                     float(pres), float(dres), float(gap), history)
