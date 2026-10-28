"""Sequential and batch Bayesian optimisation loop."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .acquisition import expected_improvement, probability_of_improvement, upper_confidence_bound
from .gp import GaussianProcess
from .kernels import Matern52
from .numopt import nelder_mead


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sample of ``n`` points in the unit cube."""
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


class BayesianOptimizer:
    """Minimise a black-box function over a box using a GP surrogate.

    The search runs in the unit cube; ``ask``/``tell`` map to and from the
    original bounds. Acquisition is maximised by random screening followed by
    Nelder-Mead polishing of the best candidates. Batches use the
    constant-liar heuristic (fantasised observations at the current best).
    """

    def __init__(self, bounds: tuple[np.ndarray, np.ndarray], acquisition: str = "ei",
                 n_init: int = 5, seed: int = 0, xi: float = 0.01, beta: float = 4.0,
                 noise: float = 1e-6, optimize_every: int = 1, n_candidates: int = 1000):
        self.lo = np.asarray(bounds[0], float)
        self.hi = np.asarray(bounds[1], float)
        self.d = self.lo.size
        if acquisition not in ("ei", "pi", "ucb"):
            raise ValueError(f"unknown acquisition {acquisition!r}")
        self.acq_name, self.n_init, self.xi, self.beta = acquisition, n_init, xi, beta
        self.noise, self.optimize_every, self.n_candidates = noise, optimize_every, n_candidates
        self.rng = np.random.default_rng(seed)
        self._init_pts = latin_hypercube(n_init, self.d, self.rng)
        self._init_used = 0
        self.U: list[np.ndarray] = []   # unit-cube inputs
        self.y: list[float] = []
        self.gp: GaussianProcess | None = None

    # -------------------------------------------------------------- mapping
    def to_unit(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.asarray(u, float) * (self.hi - self.lo)

    # ------------------------------------------------------------ surrogate
    def _fit(self, U: np.ndarray, y: np.ndarray, optimize: bool) -> GaussianProcess:
        kernel = Matern52(np.full(self.d, 0.3), 1.0) if self.gp is None else self.gp.kernel.clone()
        gp = GaussianProcess(kernel, noise=self.noise if self.gp is None else self.gp.noise)
        return gp.fit(U, y, optimize=optimize, n_restarts=2, seed=len(y))

    def _score(self, gp: GaussianProcess, U: np.ndarray, best: float) -> np.ndarray:
        mu, std = gp.predict(U)
        if self.acq_name == "ei":
            return expected_improvement(mu, std, best, self.xi * max(gp._y_std, 1e-12))
        if self.acq_name == "pi":
            return probability_of_improvement(mu, std, best, self.xi * max(gp._y_std, 1e-12))
        return upper_confidence_bound(mu, std, self.beta)

    def _maximize_acquisition(self, gp: GaussianProcess, best: float) -> np.ndarray:
        cand = self.rng.random((self.n_candidates, self.d))
        if self.U:  # also explore around the incumbent
            inc = self.U[int(np.argmin(self.y))]
            cand = np.vstack([cand, np.clip(inc + 0.05 * self.rng.standard_normal((100, self.d)), 0, 1)])
        scores = self._score(gp, cand, best)
        top = np.argsort(-scores)[:3]
        ones = (np.zeros(self.d), np.ones(self.d))
        best_u, best_s = cand[top[0]], scores[top[0]]
        for i in top:
            u, neg = nelder_mead(lambda z: -float(self._score(gp, z[None, :], best)[0]),
                                 cand[i], step=0.05, bounds=ones, max_iter=60)
            if -neg > best_s:
                best_u, best_s = u, -neg
        return best_u

    # ------------------------------------------------------------ interface
    def ask(self, batch_size: int = 1) -> np.ndarray:
        """Return ``batch_size`` points (original coordinates) to evaluate next."""
        out = []
        U_aug, y_aug = list(self.U), list(self.y)
        for _ in range(batch_size):
            if self._init_used < self.n_init:
                u = self._init_pts[self._init_used]
                self._init_used += 1
            else:
                opt = len(y_aug) % self.optimize_every == 0
                gp = self._fit(np.array(U_aug), np.array(y_aug), optimize=opt)
                if len(out) == 0:
                    self.gp = gp
                u = self._maximize_acquisition(gp, min(y_aug))
                U_aug.append(u)
                y_aug.append(min(y_aug))  # constant liar
            out.append(self.from_unit(u))
        return np.array(out)

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        """Record observed objective values."""
        X = np.atleast_2d(X)
        for x, v in zip(X, np.ravel(y)):
            self.U.append(np.clip(self.to_unit(x), 0.0, 1.0))
            self.y.append(float(v))

    def minimize(self, f: Callable[[np.ndarray], float], n_iter: int = 20, batch_size: int = 1):
        """Run ``n_iter`` rounds after the initial design; returns (x_best, y_best)."""
        X = self.ask(self.n_init)
        self.tell(X, [f(x) for x in X])
        for _ in range(n_iter):
            X = self.ask(batch_size)
            self.tell(X, [f(x) for x in X])
        return self.best

    @property
    def best(self) -> tuple[np.ndarray, float]:
        i = int(np.argmin(self.y))
        return self.from_unit(self.U[i]), self.y[i]

    def history(self) -> np.ndarray:
        """Running minimum of observed values."""
        return np.minimum.accumulate(np.array(self.y))


def random_search(f: Callable[[np.ndarray], float], bounds, n: int, seed: int = 0) -> np.ndarray:
    """Baseline: running minimum of ``n`` uniform random evaluations."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    vals = [f(lo + rng.random(lo.size) * (hi - lo)) for _ in range(n)]
    return np.minimum.accumulate(vals)
