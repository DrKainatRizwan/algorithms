"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

Follows Hansen's tutorial (2016): weighted recombination, cumulative step-size
adaptation (CSA), rank-one and rank-mu covariance updates.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .result import Result


class CMAES:
    """Ask/tell implementation of (mu/mu_w, lambda)-CMA-ES."""

    def __init__(self, x0, sigma0: float, popsize: int | None = None, seed: int = 0):
        self.mean = np.asarray(x0, dtype=float).copy()
        n = self.n = len(self.mean)
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        self.sigma = float(sigma0)
        self.rng = np.random.default_rng(seed)
        self.lam = popsize if popsize is not None else 4 + int(3 * np.log(n))
        if self.lam < 4:
            raise ValueError("popsize must be at least 4")
        self.mu = self.lam // 2
        w = np.log(self.lam / 2 + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        mueff = self.mueff
        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.cs = (mueff + 2) / (n + mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + mueff)
        self.cmu = min(1 - self.c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.C = np.eye(n)
        self.generation = 0
        self._eigen_gen = 0
        self._last_z: np.ndarray | None = None
        self._last_x: np.ndarray | None = None

    # ------------------------------------------------------------------
    def ask(self) -> np.ndarray:
        """Sample lambda candidate solutions, shape (lambda, n)."""
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * self.D) @ self.B.T
        x = self.mean + self.sigma * y
        self._last_x = x
        return x

    def tell(self, x: np.ndarray, fvals) -> None:
        """Update the search distribution from evaluated candidates."""
        x = np.asarray(x, dtype=float)
        f = np.asarray(fvals, dtype=float)
        if x.shape != (self.lam, self.n) or f.shape != (self.lam,):
            raise ValueError("tell() expects lambda candidates with lambda fitness values")
        n = self.n
        order = np.argsort(f, kind="stable")[: self.mu]
        xsel = x[order]
        old_mean = self.mean
        self.mean = self.weights @ xsel
        y_w = (self.mean - old_mean) / self.sigma
        # C^{-1/2} y_w
        invsqrt_y = self.B @ ((self.B.T @ y_w) / self.D)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrt_y
        self.generation += 1
        hsig = (
            np.linalg.norm(self.ps)
            / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation))
            / self.chi_n
            < 1.4 + 2 / (n + 1)
        )
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
        ys = (xsel - old_mean) / self.sigma
        rank_mu = (ys.T * self.weights) @ ys
        delta_h = (1 - hsig) * self.cc * (2 - self.cc)
        self.C = (
            (1 + self.c1 * delta_h - self.c1 - self.cmu * self.weights.sum()) * self.C
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * rank_mu
        )
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self._update_eigen()

    def _update_eigen(self) -> None:
        # Decompose lazily; cheap enough to do each generation for small n.
        self.C = (self.C + self.C.T) / 2
        d2, B = np.linalg.eigh(self.C)
        d2 = np.maximum(d2, 1e-30)
        self.D = np.sqrt(d2)
        self.B = B

    @property
    def condition_number(self) -> float:
        return float((self.D.max() / self.D.min()) ** 2)


def cmaes_minimize(
    func: Callable[[np.ndarray], float],
    x0,
    sigma0: float,
    popsize: int | None = None,
    max_evals: int = 100000,
    ftarget: float | None = None,
    tol_sigma: float = 1e-12,
    seed: int = 0,
) -> Result:
    """Minimise ``func`` with CMA-ES until a target, budget or stagnation stop."""
    es = CMAES(x0, sigma0, popsize, seed)
    best_x, best_f = es.mean.copy(), np.inf
    evals, history = 0, []
    message = "max_evals reached"
    while evals + es.lam <= max_evals:
        X = es.ask()
        F = np.array([func(x) for x in X])
        evals += es.lam
        es.tell(X, F)
        i = int(np.argmin(F))
        if F[i] < best_f:
            best_f, best_x = float(F[i]), X[i].copy()
        history.append(best_f)
        if ftarget is not None and best_f <= ftarget:
            message = "target reached"
            break
        if es.sigma * es.D.max() < tol_sigma:
            message = "step size collapsed"
            break
        if es.condition_number > 1e14:
            message = "ill-conditioned covariance"
            break
    return Result(best_x, best_f, evals, es.generation, message, history)
