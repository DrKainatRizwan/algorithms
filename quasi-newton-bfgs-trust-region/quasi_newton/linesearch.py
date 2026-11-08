"""Line searches: Armijo backtracking and strong-Wolfe (Nocedal & Wright Alg. 3.5/3.6)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .functions import Problem


def backtracking(p: Problem, x: np.ndarray, d: np.ndarray, fx: float, gx: np.ndarray,
                 alpha0: float = 1.0, rho: float = 0.5, c1: float = 1e-4,
                 max_iter: int = 60) -> Tuple[float, float]:
    """Return (alpha, f(x+alpha d)) satisfying the Armijo condition."""
    slope = float(gx @ d)
    if slope >= 0:
        raise ValueError("not a descent direction")
    a = alpha0
    for _ in range(max_iter):
        fa = p.value(x + a * d)
        if fa <= fx + c1 * a * slope:
            return a, fa
        a *= rho
    return a, fa


def _cubic_min(a0, f0, d0, a1, f1, d1, lo, hi):
    """Minimiser of the cubic interpolating two points with derivatives, safeguarded."""
    d1_ = d0 + d1 - 3 * (f0 - f1) / (a0 - a1)
    disc = d1_ ** 2 - d0 * d1
    if disc >= 0:
        d2 = np.sign(a1 - a0) * np.sqrt(disc)
        denom = d1 - d0 + 2 * d2
        if denom != 0:
            a = a1 - (a1 - a0) * (d1 + d2 - d1_) / denom
            if min(lo, hi) < a < max(lo, hi):
                return a
    return 0.5 * (lo + hi)


def strong_wolfe(p: Problem, x: np.ndarray, d: np.ndarray, fx: float, gx: np.ndarray,
                 alpha0: float = 1.0, c1: float = 1e-4, c2: float = 0.9,
                 alpha_max: float = 1e3, max_iter: int = 30
                 ) -> Tuple[Optional[float], float, np.ndarray]:
    """Find alpha with sufficient decrease and |g(x+ad)'d| <= c2 |g'd|.

    Returns (alpha, f_new, g_new); alpha is None on failure.
    """
    slope0 = float(gx @ d)
    if slope0 >= 0:
        raise ValueError("not a descent direction")

    def phi(a):
        xa = x + a * d
        return p.value(xa), p.gradient(xa)

    def zoom(lo, f_lo, dg_lo, hi, f_hi, dg_hi):
        for _ in range(max_iter):
            a = _cubic_min(lo, f_lo, dg_lo, hi, f_hi, dg_hi, lo, hi)
            # keep away from the bracket ends
            w = abs(hi - lo)
            if abs(a - lo) < 0.1 * w or abs(a - hi) < 0.1 * w:
                a = 0.5 * (lo + hi)
            fa, ga = phi(a)
            dga = float(ga @ d)
            if fa > fx + c1 * a * slope0 or fa >= f_lo:
                hi, f_hi, dg_hi = a, fa, dga
            else:
                if abs(dga) <= -c2 * slope0:
                    return a, fa, ga
                if dga * (hi - lo) >= 0:
                    hi, f_hi, dg_hi = lo, f_lo, dg_lo
                lo, f_lo, dg_lo = a, fa, dga
            if abs(hi - lo) < 1e-16:
                break
        return None, f_lo, gx

    a_prev, f_prev, dg_prev = 0.0, fx, slope0
    a = alpha0
    for i in range(max_iter):
        fa, ga = phi(a)
        dga = float(ga @ d)
        if fa > fx + c1 * a * slope0 or (i > 0 and fa >= f_prev):
            return zoom(a_prev, f_prev, dg_prev, a, fa, dga)
        if abs(dga) <= -c2 * slope0:
            return a, fa, ga
        if dga >= 0:
            return zoom(a, fa, dga, a_prev, f_prev, dg_prev)
        a_prev, f_prev, dg_prev = a, fa, dga
        a = min(2.0 * a, alpha_max)
        if a_prev >= alpha_max:
            break
    return None, fx, gx
