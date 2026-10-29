"""Particle Swarm Optimisation with inertia / constriction variants and several topologies."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .result import BatchObjective, OptimizeResult, check_bounds


def _neighbour_best(pbest_x: np.ndarray, pbest_f: np.ndarray, topology: str, k: int) -> np.ndarray:
    """Return, for each particle, the best personal-best position in its neighbourhood."""
    n = pbest_x.shape[0]
    if topology == "global":
        return np.broadcast_to(pbest_x[np.argmin(pbest_f)], pbest_x.shape)
    if topology == "ring":
        # neighbourhood = self plus k particles on each side (wrap-around)
        offs = np.arange(-k, k + 1)
        idx = (np.arange(n)[:, None] + offs[None, :]) % n
        best = idx[np.arange(n), np.argmin(pbest_f[idx], axis=1)]
        return pbest_x[best]
    if topology == "von_neumann":
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        ids = np.arange(n)
        r, c = ids // cols, ids % cols
        nbrs = [ids]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = (r + dr) % rows, (c + dc) % cols
            nb = rr * cols + cc
            nbrs.append(np.where(nb < n, nb, ids))
        idx = np.stack(nbrs, axis=1)
        best = idx[ids, np.argmin(pbest_f[idx], axis=1)]
        return pbest_x[best]
    raise ValueError(f"unknown topology {topology!r}")


class PSO:
    """Particle swarm optimiser (minimisation) for box-constrained problems.

    Velocity update: v <- w v + c1 r1 (pbest - x) + c2 r2 (nbest - x), optionally scaled by
    Clerc-Kennedy constriction.  ``variant`` is ``"inertia"``, ``"decay"`` (linearly decreasing
    inertia from ``w`` to ``w_end``) or ``"constriction"``.  ``boundary`` is ``"clip"`` (position
    clipped, velocity zeroed), ``"reflect"`` or ``"random"`` (resample out-of-range coordinates).
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], lower, upper, dim: int | None = None,
                 n_particles: int = 30, max_iter: int = 200, w: float = 0.729, w_end: float = 0.4,
                 c1: float = 1.49445, c2: float = 1.49445, variant: str = "inertia",
                 topology: str = "global", ring_k: int = 1, boundary: str = "clip",
                 vmax_frac: float = 0.5, tol: float | None = None, max_nfev: int | None = None,
                 seed: int | None = 0):
        if variant not in ("inertia", "decay", "constriction"):
            raise ValueError(f"unknown variant {variant!r}")
        if boundary not in ("clip", "reflect", "random"):
            raise ValueError(f"unknown boundary {boundary!r}")
        if n_particles < 2:
            raise ValueError("need at least two particles")
        self.lower, self.upper = check_bounds(lower, upper, dim)
        self.dim = self.lower.size
        self.obj = BatchObjective(func, max_nfev)
        self.n, self.max_iter = n_particles, max_iter
        self.w, self.w_end, self.c1, self.c2 = w, w_end, c1, c2
        self.variant, self.topology, self.ring_k, self.boundary = variant, topology, ring_k, boundary
        self.vmax = vmax_frac * (self.upper - self.lower)
        self.tol, self.rng = tol, np.random.default_rng(seed)
        if variant == "constriction":
            phi = c1 + c2
            if phi <= 4:
                raise ValueError("constriction requires c1 + c2 > 4")
            self.chi = 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))
        else:
            self.chi = 1.0

    def _inertia(self, it: int) -> float:
        if self.variant == "decay":
            return self.w + (self.w_end - self.w) * it / max(self.max_iter - 1, 1)
        if self.variant == "constriction":
            return 1.0
        return self.w

    def _apply_bounds(self, x: np.ndarray, v: np.ndarray) -> None:
        lo, hi = self.lower, self.upper
        if self.boundary == "clip":
            out = (x < lo) | (x > hi)
            np.clip(x, lo, hi, out=x)
            v[out] = 0.0
        elif self.boundary == "reflect":
            span = hi - lo
            y = np.mod(x - lo, 2 * span)
            refl = y > span
            x[:] = lo + np.where(refl, 2 * span - y, y)
            v[(x != lo) & refl] *= -1
        else:
            out = (x < lo) | (x > hi)
            fresh = lo + self.rng.random(x.shape) * (hi - lo)
            x[out] = fresh[out]
            v[out] = 0.0

    def run(self) -> OptimizeResult:
        rng, lo, hi = self.rng, self.lower, self.upper
        x = lo + rng.random((self.n, self.dim)) * (hi - lo)
        v = (rng.random((self.n, self.dim)) * 2 - 1) * self.vmax * 0.1
        f = self.obj(x)
        pbest_x, pbest_f = x.copy(), f.copy()
        history = [float(pbest_f.min())]
        msg, it = "max iterations reached", 0
        for it in range(1, self.max_iter + 1):
            if self.obj.exhausted:
                msg = "evaluation budget exhausted"
                break
            nbest = _neighbour_best(pbest_x, pbest_f, self.topology, self.ring_k)
            r1, r2 = rng.random(x.shape), rng.random(x.shape)
            w = self._inertia(it - 1)
            v = self.chi * (w * v + self.c1 * r1 * (pbest_x - x) + self.c2 * r2 * (nbest - x))
            np.clip(v, -self.vmax, self.vmax, out=v)
            x = x + v
            self._apply_bounds(x, v)
            f = self.obj(x)
            better = f < pbest_f
            pbest_x[better], pbest_f[better] = x[better], f[better]
            history.append(float(pbest_f.min()))
            if self.tol is not None and history[-1] <= self.tol:
                msg = "target tolerance reached"
                break
        g = int(np.argmin(pbest_f))
        return OptimizeResult(pbest_x[g].copy(), float(pbest_f[g]), self.obj.nfev, it, history,
                              success=True, message=msg)


def pso_minimize(func, lower, upper, dim: int | None = None, **kwargs) -> OptimizeResult:
    """Convenience wrapper: ``PSO(...).run()``."""
    return PSO(func, lower, upper, dim, **kwargs).run()
