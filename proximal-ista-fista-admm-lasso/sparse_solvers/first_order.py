"""Proximal gradient methods: ISTA, FISTA (with restart / backtracking), group lasso."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .problems import SolverResult, lasso_objective, lipschitz_constant
from .proximal import soft_threshold, prox_group_l2

Array = np.ndarray


def proximal_gradient(f: Callable[[Array], float], grad: Callable[[Array], Array],
                      prox: Callable[[Array, float], Array], g: Callable[[Array], float],
                      x0: Array, step: float, accelerated: bool = False,
                      backtrack: bool = False, restart: bool = False,
                      max_iter: int = 2000, tol: float = 1e-10) -> SolverResult:
    """Minimise f(x) + g(x) with f smooth, g prox-friendly.

    ``prox(v, t)`` must return argmin_x g(x) + ||x - v||^2 / (2t).  With
    ``accelerated`` Nesterov/FISTA momentum is used; ``restart`` applies the
    function-value adaptive restart of O'Donoghue & Candes.  With ``backtrack``
    the step is shrunk until the quadratic upper bound holds.
    """
    x = x0.copy()
    yk = x.copy()
    t_k = 1.0
    hist = [f(x) + g(x)]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        gy = grad(yk)
        fy = f(yk)
        while True:
            x_new = prox(yk - step * gy, step)
            if not backtrack:
                break
            d = x_new - yk
            if f(x_new) <= fy + gy @ d + (d @ d) / (2 * step) + 1e-14 * max(1.0, abs(fy)):
                break
            step *= 0.5
        obj = f(x_new) + g(x_new)
        if accelerated:
            if restart and obj > hist[-1]:
                t_k = 1.0          # momentum reset; keep the (valid) prox step
                yk = x_new.copy()
            else:
                t_new = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
                yk = x_new + ((t_k - 1) / t_new) * (x_new - x)
                t_k = t_new
        else:
            yk = x_new
        change = np.linalg.norm(x_new - x)
        x = x_new
        hist.append(obj)
        if change <= tol * max(1.0, np.linalg.norm(x)):
            converged = True
            break
    return SolverResult(x, hist[-1], it, converged, hist)


def _lasso_parts(X: Array, y: Array, lam: float):
    f = lambda w: 0.5 * float(np.sum((X @ w - y) ** 2))
    grad = lambda w: X.T @ (X @ w - y)
    g = lambda w: lam * float(np.abs(w).sum())
    prox = lambda v, t: soft_threshold(v, lam * t)
    return f, grad, prox, g


def ista(X: Array, y: Array, lam: float, w0: Array | None = None,
         max_iter: int = 5000, tol: float = 1e-10) -> SolverResult:
    """Iterative shrinkage-thresholding with constant step 1/L."""
    f, grad, prox, g = _lasso_parts(X, y, lam)
    L = lipschitz_constant(X)
    x0 = np.zeros(X.shape[1]) if w0 is None else w0
    return proximal_gradient(f, grad, prox, g, x0, 1.0 / L, False, False, False, max_iter, tol)


def fista(X: Array, y: Array, lam: float, w0: Array | None = None,
          restart: bool = True, backtrack: bool = False, max_iter: int = 5000,
          tol: float = 1e-10) -> SolverResult:
    """FISTA (Beck & Teboulle 2009) with optional adaptive restart."""
    f, grad, prox, g = _lasso_parts(X, y, lam)
    L = lipschitz_constant(X)
    step = 1.0 / L if not backtrack else 4.0 / L
    x0 = np.zeros(X.shape[1]) if w0 is None else w0
    return proximal_gradient(f, grad, prox, g, x0, step, True, backtrack, restart, max_iter, tol)


def group_lasso(X: Array, y: Array, lam: float, groups: list[np.ndarray],
                max_iter: int = 5000, tol: float = 1e-10) -> SolverResult:
    """Group lasso 0.5||Xw-y||^2 + lam * sum_g sqrt(|g|) ||w_g||_2 via FISTA."""
    weights = [np.sqrt(len(g)) for g in groups]
    f = lambda w: 0.5 * float(np.sum((X @ w - y) ** 2))
    grad = lambda w: X.T @ (X @ w - y)

    def g_fn(w: Array) -> float:
        return lam * sum(wt * np.linalg.norm(w[gr]) for wt, gr in zip(weights, groups))

    def prox(v: Array, t: float) -> Array:
        out = v.copy()
        for wt, gr in zip(weights, groups):
            out[gr] = prox_group_l2(v[gr], lam * wt * t, [np.arange(len(gr))])
        return out

    L = lipschitz_constant(X)
    return proximal_gradient(f, grad, prox, g_fn, np.zeros(X.shape[1]), 1.0 / L,
                             True, False, True, max_iter, tol)
