"""Nonlinear conjugate gradient (Fletcher-Reeves, Polak-Ribiere+, Hestenes-Stiefel, Dai-Yuan)."""
from __future__ import annotations

import numpy as np

from .functions import Problem
from .linesearch import strong_wolfe
from .quasi_newton import Result


def _beta(kind: str, g_new, g_old, d_old) -> float:
    y = g_new - g_old
    if kind == "fr":
        return float(g_new @ g_new / (g_old @ g_old))
    if kind == "pr+":
        return max(0.0, float(g_new @ y / (g_old @ g_old)))
    if kind == "hs":
        den = float(d_old @ y)
        return max(0.0, float(g_new @ y) / den) if den != 0 else 0.0
    if kind == "dy":
        den = float(d_old @ y)
        return float(g_new @ g_new) / den if den != 0 else 0.0
    raise ValueError(f"unknown beta rule {kind!r}")


def nonlinear_cg(p: Problem, x0=None, kind: str = "pr+", tol: float = 1e-6,
                 max_iter: int = 2000, restart: bool = True) -> Result:
    """Nonlinear CG with strong-Wolfe (c2=0.1) search and periodic restart every n steps."""
    x = np.array(p.x0 if x0 is None else x0, dtype=float)
    n = len(x)
    fx, gx = p.value(x), p.gradient(x)
    d = -gx
    hist = [fx]
    for k in range(max_iter):
        gn = float(np.linalg.norm(gx, np.inf))
        if gn <= tol:
            return Result(x, fx, gn, k, p.n_f, p.n_g, True, hist, f"cg-{kind}")
        if gx @ d >= 0:
            d = -gx
        a0 = min(1.0, 1.0 / np.linalg.norm(gx)) if k == 0 else 1.0
        a, f_new, g_new = strong_wolfe(p, x, d, fx, gx, alpha0=a0, c2=0.1)
        if a is None:
            if np.allclose(d, -gx):
                return Result(x, fx, gn, k, p.n_f, p.n_g, False, hist, "line search failed")
            d = -gx
            continue
        beta = _beta(kind, g_new, gx, d)
        x = x + a * d
        d = -g_new + beta * d
        if restart and (k + 1) % n == 0:
            d = -g_new
        gx, fx = g_new, f_new
        hist.append(fx)
    gn = float(np.linalg.norm(gx, np.inf))
    return Result(x, fx, gn, max_iter, p.n_f, p.n_g, gn <= tol, hist, "max_iter")
