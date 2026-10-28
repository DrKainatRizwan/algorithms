"""Derivative-free Nelder-Mead simplex search with optional box bounds."""

from __future__ import annotations

from typing import Callable

import numpy as np


def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step: float | np.ndarray = 0.1,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    max_iter: int = 200,
    xtol: float = 1e-8,
    ftol: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Minimise ``f`` starting at ``x0``. Points are clipped into ``bounds``."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    lo, hi = (None, None) if bounds is None else (np.asarray(bounds[0], float), np.asarray(bounds[1], float))

    def clip(x: np.ndarray) -> np.ndarray:
        return x if lo is None else np.clip(x, lo, hi)

    def ev(x: np.ndarray) -> float:
        v = float(f(x))
        return v if np.isfinite(v) else 1e300

    steps = np.broadcast_to(np.asarray(step, float), (n,))
    simplex = [clip(x0)]
    for i in range(n):
        p = x0.copy()
        p[i] += steps[i]
        if lo is not None and p[i] > hi[i]:
            p[i] = x0[i] - steps[i]
        simplex.append(clip(p))
    vals = [ev(p) for p in simplex]

    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < ftol and max(np.abs(p - simplex[0]).max() for p in simplex[1:]) < xtol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = clip(centroid + (centroid - simplex[-1]))
        fr = ev(xr)
        if fr < vals[0]:
            xe = clip(centroid + 2.0 * (centroid - simplex[-1]))
            fe = ev(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            if fr < vals[-1]:
                xc = clip(centroid + 0.5 * (xr - centroid))
            else:
                xc = clip(centroid + 0.5 * (simplex[-1] - centroid))
            fc = ev(xc)
            if fc < min(fr, vals[-1]):
                simplex[-1], vals[-1] = xc, fc
            else:  # shrink toward the best vertex
                for i in range(1, n + 1):
                    simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                    vals[i] = ev(simplex[i])
    best = int(np.argmin(vals))
    return simplex[best], vals[best]
