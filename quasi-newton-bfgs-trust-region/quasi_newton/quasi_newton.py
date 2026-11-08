"""BFGS, DFP, SR1 (inverse-Hessian form with line search) and limited-memory BFGS."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from .functions import Problem
from .linesearch import backtracking, strong_wolfe


@dataclass
class Result:
    """Outcome of an optimisation run."""

    x: np.ndarray
    f: float
    grad_norm: float
    iterations: int
    n_f: int
    n_g: int
    converged: bool
    history: List[float] = field(default_factory=list)
    message: str = ""


def _update_bfgs(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    rho = 1.0 / (y @ s)
    I = np.eye(len(s))
    V = I - rho * np.outer(s, y)
    return V @ H @ V.T + rho * np.outer(s, s)


def _update_dfp(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    Hy = H @ y
    return H + np.outer(s, s) / (y @ s) - np.outer(Hy, Hy) / (y @ Hy)


def _update_sr1(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = s - H @ y
    den = r @ y
    if abs(den) < 1e-8 * np.linalg.norm(r) * np.linalg.norm(y):
        return H
    return H + np.outer(r, r) / den


def _dense_quasi_newton(p: Problem, update, name: str, x0=None, tol=1e-6, max_iter=500,
                        wolfe: bool = True, require_curvature: bool = True) -> Result:
    x = np.array(p.x0 if x0 is None else x0, dtype=float)
    n = len(x)
    fx, gx = p.value(x), p.gradient(x)
    H = np.eye(n)
    hist = [fx]
    first = True
    for k in range(max_iter):
        gn = float(np.linalg.norm(gx, np.inf))
        if gn <= tol:
            return Result(x, fx, gn, k, p.n_f, p.n_g, True, hist, name)
        d = -H @ gx
        if gx @ d >= 0:  # lost positive definiteness (SR1) -> reset
            H = np.eye(n)
            d = -gx
        a0 = min(1.0, 1.0 / np.linalg.norm(gx)) if first else 1.0
        if wolfe:
            a, f_new, g_new = strong_wolfe(p, x, d, fx, gx, alpha0=a0)
            if a is None:
                if np.allclose(H, np.eye(n)):
                    return Result(x, fx, gn, k, p.n_f, p.n_g, False, hist, "line search failed")
                H = np.eye(n)
                continue
        else:
            a, f_new = backtracking(p, x, d, fx, gx, alpha0=a0)
            g_new = p.gradient(x + a * d)
        first = False
        s = a * d
        y = g_new - gx
        x, fx, gx = x + s, f_new, g_new
        hist.append(fx)
        if k == 0 or (H is not None and np.allclose(H, np.eye(n))):
            # Scale initial matrix (N&W eq. 6.20) after the first step.
            if y @ y > 0 and y @ s > 0:
                H = (y @ s) / (y @ y) * np.eye(n)
        if require_curvature and y @ s <= 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            continue  # skip update that would destroy positive definiteness
        H = update(H, s, y)
    gn = float(np.linalg.norm(gx, np.inf))
    return Result(x, fx, gn, max_iter, p.n_f, p.n_g, gn <= tol, hist, "max_iter")


def bfgs(p: Problem, x0=None, tol: float = 1e-6, max_iter: int = 500, wolfe: bool = True) -> Result:
    """BFGS with inverse-Hessian update and strong-Wolfe (or Armijo) line search."""
    return _dense_quasi_newton(p, _update_bfgs, "bfgs", x0, tol, max_iter, wolfe)


def dfp(p: Problem, x0=None, tol: float = 1e-6, max_iter: int = 500) -> Result:
    """Davidon-Fletcher-Powell: the dual of BFGS (known to be less robust)."""
    return _dense_quasi_newton(p, _update_dfp, "dfp", x0, tol, max_iter, True)


def sr1(p: Problem, x0=None, tol: float = 1e-6, max_iter: int = 500) -> Result:
    """Symmetric rank-one inverse update with Armijo backtracking (no curvature requirement)."""
    return _dense_quasi_newton(p, _update_sr1, "sr1", x0, tol, max_iter, False, False)


class LBFGSMemory:
    """Stores the last m correction pairs and applies the two-loop recursion."""

    def __init__(self, m: int = 10) -> None:
        self.m = m
        self.pairs: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=m)

    def add(self, s: np.ndarray, y: np.ndarray) -> bool:
        sy = float(s @ y)
        if sy <= 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
            return False
        self.pairs.append((s, y, 1.0 / sy))
        return True

    def direction(self, g: np.ndarray) -> np.ndarray:
        """Return -H_k g via the two-loop recursion (N&W Alg. 7.4)."""
        q = g.copy()
        alphas = []
        for s, y, rho in reversed(self.pairs):
            a = rho * (s @ q)
            alphas.append(a)
            q -= a * y
        if self.pairs:
            s, y, _ = self.pairs[-1]
            q *= (s @ y) / (y @ y)
        for (s, y, rho), a in zip(self.pairs, reversed(alphas)):
            b = rho * (y @ q)
            q += (a - b) * s
        return -q


def lbfgs(p: Problem, x0=None, m: int = 10, tol: float = 1e-6, max_iter: int = 1000) -> Result:
    """Limited-memory BFGS with strong-Wolfe line search; O(mn) memory and work per step."""
    x = np.array(p.x0 if x0 is None else x0, dtype=float)
    fx, gx = p.value(x), p.gradient(x)
    mem = LBFGSMemory(m)
    hist = [fx]
    for k in range(max_iter):
        gn = float(np.linalg.norm(gx, np.inf))
        if gn <= tol:
            return Result(x, fx, gn, k, p.n_f, p.n_g, True, hist, "lbfgs")
        d = mem.direction(gx)
        if gx @ d >= 0:
            mem.pairs.clear()
            d = -gx
        a0 = min(1.0, 1.0 / np.linalg.norm(gx)) if not mem.pairs else 1.0
        a, f_new, g_new = strong_wolfe(p, x, d, fx, gx, alpha0=a0)
        if a is None:
            if not mem.pairs:
                return Result(x, fx, gn, k, p.n_f, p.n_g, False, hist, "line search failed")
            mem.pairs.clear()
            continue
        s = a * d
        mem.add(s, g_new - gx)
        x, fx, gx = x + s, f_new, g_new
        hist.append(fx)
    gn = float(np.linalg.norm(gx, np.inf))
    return Result(x, fx, gn, max_iter, p.n_f, p.n_g, gn <= tol, hist, "max_iter")
