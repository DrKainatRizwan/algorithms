"""Optimisers for QAOA parameters: Adam ascent, Nelder-Mead, INTERP warm starts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class OptResult:
    x: np.ndarray
    value: float
    iterations: int
    history: list[float] = field(default_factory=list)


def adam_ascent(f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray],
                x0: np.ndarray, lr: float = 0.05, iters: int = 200, tol: float = 1e-9,
                b1: float = 0.9, b2: float = 0.999) -> OptResult:
    """Adam gradient *ascent* on ``f``."""
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    hist = [f(x)]
    for t in range(1, iters + 1):
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
        x = x + step
        hist.append(f(x))
        if abs(hist[-1] - hist[-2]) < tol and np.linalg.norm(g) < 1e-5:
            break
    return OptResult(x, hist[-1], len(hist) - 1, hist)


def nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray, step: float = 0.3,
                iters: int = 300, tol: float = 1e-10) -> OptResult:
    """Derivative-free Nelder-Mead *maximisation* of ``f``."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    pts = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] += step
        pts.append(p)
    vals = [-f(p) for p in pts]
    hist: list[float] = []
    it = 0
    for it in range(1, iters + 1):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        hist.append(-vals[0])
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = -f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = -f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * ((xr if fr < vals[-1] else pts[-1]) - c)
            fc = -f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = -f(pts[i])
    i = int(np.argmin(vals))
    return OptResult(pts[i], -vals[i], it, hist)


def interp_init(params: np.ndarray) -> np.ndarray:
    """INTERP warm start (Zhou et al. 2020): lift depth-p optimum to a depth-(p+1) guess."""
    params = np.asarray(params, dtype=float)
    p = params.size // 2

    def lift(a: np.ndarray) -> np.ndarray:
        out = np.zeros(p + 1)
        for i in range(p + 1):
            if i > 0:
                out[i] += i / p * a[i - 1]
            if i < p:
                out[i] += (p - i) / p * a[i]
        return out

    return np.concatenate([lift(params[:p]), lift(params[p:])])


def multistart(f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray],
               dim: int, starts: int = 8, seed: int = 0, lr: float = 0.05,
               iters: int = 150) -> OptResult:
    """Best of several random Adam runs (angles drawn from [0, pi) x [0, pi/2))."""
    rng = np.random.default_rng(seed)
    best: OptResult | None = None
    p = dim // 2
    for _ in range(starts):
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        r = adam_ascent(f, grad, x0, lr=lr, iters=iters)
        if best is None or r.value > best.value:
            best = r
    assert best is not None
    return best
