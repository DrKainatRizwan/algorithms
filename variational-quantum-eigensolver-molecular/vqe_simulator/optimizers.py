"""Classical optimisers written from scratch.

All optimisers minimise ``f`` and return an :class:`OptResult` with the
energy history (one entry per iteration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class OptResult:
    x: np.ndarray
    fun: float
    nfev: int
    history: list[float] = field(default_factory=list)


def gradient_descent(f: Callable, grad: Callable, x0, lr: float = 0.2,
                     iters: int = 200, tol: float = 1e-9) -> OptResult:
    x = np.array(x0, dtype=float)
    hist: list[float] = []
    for _ in range(iters):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - lr * g
        hist.append(f(x))
    return OptResult(x, f(x), len(hist), hist)


def adam(f: Callable, grad: Callable, x0, lr: float = 0.1, iters: int = 200,
         b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> OptResult:
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    hist = []
    for t in range(1, iters + 1):
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        hist.append(f(x))
    return OptResult(x, hist[-1], iters, hist)


def spsa(f: Callable, x0, iters: int = 300, a: float = 0.2, c: float = 0.1,
         seed: int = 0, A: float = 20.0, alpha: float = 0.602,
         gamma: float = 0.101) -> OptResult:
    """Simultaneous-perturbation stochastic approximation (2 evals/step)."""
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    hist = []
    for k in range(iters):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        g = (f(x + ck * delta) - f(x - ck * delta)) / (2 * ck) * delta
        x = x - ak * g
        hist.append(f(x))
    return OptResult(x, hist[-1], 3 * iters, hist)


def nelder_mead(f: Callable, x0, iters: int = 400, step: float = 0.4,
                tol: float = 1e-10) -> OptResult:
    """Nelder-Mead simplex search."""
    x0 = np.array(x0, dtype=float)
    n = x0.size
    pts = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    nfev = n + 1
    hist = []
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        hist.append(vals[0])
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = f(xr)
        nfev += 1
        if fr < vals[0]:
            xe = centroid + 2 * (centroid - pts[-1])
            fe = f(xe)
            nfev += 1
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            if fr < vals[-1]:
                xc = centroid + 0.5 * (xr - centroid)
            else:
                xc = centroid + 0.5 * (pts[-1] - centroid)
            fc = f(xc)
            nfev += 1
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
                nfev += n
    i = int(np.argmin(vals))
    return OptResult(pts[i], vals[i], nfev, hist)


def rotosolve(f: Callable, x0, sweeps: int = 10) -> OptResult:
    """Coordinate-wise analytic minimisation (Ostaszewski et al.).

    For gates ``exp(-i theta P/2)`` the energy is ``A sin(theta + phi) + C``
    in each single parameter, so three evaluations locate the exact minimum.
    Valid only when each parameter appears in one gate.
    """
    x = np.array(x0, dtype=float)
    hist, nfev = [], 0
    for _ in range(sweeps):
        for i in range(x.size):
            t0 = x[i]
            e0 = f(x)
            x[i] = t0 + np.pi / 2
            ep = f(x)
            x[i] = t0 - np.pi / 2
            em = f(x)
            nfev += 3
            x[i] = t0 - np.pi / 2 - np.arctan2(2 * e0 - ep - em, ep - em)
            x[i] = (x[i] + np.pi) % (2 * np.pi) - np.pi
        hist.append(f(x))
        nfev += 1
    return OptResult(x, hist[-1], nfev, hist)
