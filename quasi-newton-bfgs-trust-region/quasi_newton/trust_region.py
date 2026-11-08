"""Trust-region Newton methods: Steihaug-CG, dogleg and the basic TR iteration."""
from __future__ import annotations

import numpy as np

from .functions import Problem
from .quasi_newton import Result


def _boundary_tau(z: np.ndarray, d: np.ndarray, delta: float) -> float:
    """Positive tau with ||z + tau d|| = delta."""
    a, b, c = d @ d, 2 * z @ d, z @ z - delta ** 2
    return (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)


def steihaug_cg(B: np.ndarray, g: np.ndarray, delta: float, tol_rel: float = 1e-8,
                max_iter: int | None = None) -> np.ndarray:
    """Approximately minimise g'p + 0.5 p'Bp subject to ||p|| <= delta (handles indefinite B)."""
    n = len(g)
    z = np.zeros(n)
    r = g.copy()
    d = -r
    if np.linalg.norm(r) < 1e-14:
        return z
    eps = tol_rel * np.linalg.norm(g)
    for _ in range(max_iter or 2 * n):
        Bd = B @ d
        dBd = d @ Bd
        if dBd <= 0:
            return z + _boundary_tau(z, d, delta) * d
        alpha = (r @ r) / dBd
        z_new = z + alpha * d
        if np.linalg.norm(z_new) >= delta:
            return z + _boundary_tau(z, d, delta) * d
        r_new = r + alpha * Bd
        if np.linalg.norm(r_new) < eps:
            return z_new
        beta = (r_new @ r_new) / (r @ r)
        d = -r_new + beta * d
        z, r = z_new, r_new
    return z


def dogleg(B: np.ndarray, g: np.ndarray, delta: float) -> np.ndarray:
    """Powell's dogleg step; requires B positive definite."""
    pB = -np.linalg.solve(B, g)
    if np.linalg.norm(pB) <= delta:
        return pB
    pU = -(g @ g) / (g @ B @ g) * g
    if np.linalg.norm(pU) >= delta:
        return -delta * g / np.linalg.norm(g)
    diff = pB - pU
    a, b, c = diff @ diff, 2 * pU @ diff, pU @ pU - delta ** 2
    t = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return pU + t * diff


def trust_region(p: Problem, x0=None, subproblem: str = "steihaug", tol: float = 1e-6,
                 max_iter: int = 500, delta0: float = 1.0, delta_max: float = 100.0,
                 eta: float = 0.15) -> Result:
    """Trust-region Newton (N&W Alg. 4.1) using exact Hessians.

    subproblem: 'steihaug' (any Hessian) or 'dogleg' (Hessian is shifted to be PD if needed).
    """
    x = np.array(p.x0 if x0 is None else x0, dtype=float)
    fx, gx = p.value(x), p.gradient(x)
    delta = delta0
    hist = [fx]
    for k in range(max_iter):
        gn = float(np.linalg.norm(gx, np.inf))
        if gn <= tol:
            return Result(x, fx, gn, k, p.n_f, p.n_g, True, hist, f"tr-{subproblem}")
        B = p.hessian(x)
        if subproblem == "dogleg":
            lam_min = np.linalg.eigvalsh(B)[0]
            if lam_min <= 1e-8:
                B = B + (1e-8 - lam_min + 1e-6) * np.eye(len(x))
            step = dogleg(B, gx, delta)
        elif subproblem == "steihaug":
            step = steihaug_cg(B, gx, delta)
        else:
            raise ValueError(f"unknown subproblem {subproblem!r}")
        pred = -(gx @ step + 0.5 * step @ B @ step)
        f_new = p.value(x + step)
        rho = (fx - f_new) / pred if pred > 0 else -1.0
        sn = np.linalg.norm(step)
        if rho < 0.25:
            delta = 0.25 * sn
        elif rho > 0.75 and sn >= delta * (1 - 1e-9):
            delta = min(2 * delta, delta_max)
        if rho > eta:
            x = x + step
            fx, gx = f_new, p.gradient(x)
            hist.append(fx)
        if delta < 1e-14:
            break
    gn = float(np.linalg.norm(gx, np.inf))
    return Result(x, fx, gn, k + 1, p.n_f, p.n_g, gn <= tol, hist, "max_iter")
