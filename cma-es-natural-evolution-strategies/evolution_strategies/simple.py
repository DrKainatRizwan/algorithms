"""Baseline strategies: (1+1)-ES with the 1/5th success rule and the cross-entropy method."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .result import Result


class OnePlusOneES:
    """(1+1)-ES with isotropic Gaussian mutation and 1/5th success rule."""

    def __init__(self, x0, sigma0: float, seed: int = 0):
        self.x = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma0)
        self.rng = np.random.default_rng(seed)
        self.n = len(self.x)

    def minimize(self, func: Callable[[np.ndarray], float], max_evals: int = 10000,
                 ftarget: float | None = None) -> Result:
        fx = func(self.x)
        history = [fx]
        evals = 1
        message = "max_evals reached"
        # exp(1/3) on success, exp(-1/12) on failure: balances at success rate 1/5.
        up, down = np.exp(1.0 / 3.0), np.exp(-1.0 / 12.0)
        while evals < max_evals:
            y = self.x + self.sigma * self.rng.standard_normal(self.n)
            fy = func(y)
            evals += 1
            if fy <= fx:
                self.x, fx = y, fy
                self.sigma *= up
            else:
                self.sigma *= down
            history.append(fx)
            if ftarget is not None and fx <= ftarget:
                message = "target reached"
                break
        return Result(self.x.copy(), float(fx), evals, evals - 1, message, history)


class CrossEntropyMethod:
    """Cross-entropy method with a diagonal Gaussian and smoothed updates."""

    def __init__(self, x0, sigma0: float, popsize: int = 50, elite_frac: float = 0.2,
                 smoothing: float = 0.7, noise: float = 0.5,
                 noise_decay: int = 30, seed: int = 0):
        self.mean = np.asarray(x0, dtype=float).copy()
        self.std = np.full(len(self.mean), float(sigma0))
        self.lam = popsize
        self.n_elite = max(2, int(round(popsize * elite_frac)))
        self.alpha = smoothing
        self.noise = noise  # extra std added early on to avoid premature collapse
        self.noise_decay = noise_decay
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def ask(self) -> np.ndarray:
        return self.mean + self.std * self.rng.standard_normal((self.lam, len(self.mean)))

    def tell(self, x: np.ndarray, f: np.ndarray) -> None:
        elite = x[np.argsort(f, kind="stable")[: self.n_elite]]
        self.mean = self.alpha * elite.mean(axis=0) + (1 - self.alpha) * self.mean
        extra = self.noise * max(0.0, 1.0 - self.t / self.noise_decay)
        self.std = self.alpha * elite.std(axis=0) + (1 - self.alpha) * self.std + extra
        self.t += 1

    def minimize(self, func: Callable[[np.ndarray], float], max_evals: int = 20000,
                 ftarget: float | None = None) -> Result:
        best_x, best_f = self.mean.copy(), np.inf
        evals, it, history = 0, 0, []
        message = "max_evals reached"
        while evals + self.lam <= max_evals:
            X = self.ask()
            F = np.array([func(x) for x in X])
            evals += self.lam
            it += 1
            self.tell(X, F)
            i = int(np.argmin(F))
            if F[i] < best_f:
                best_f, best_x = float(F[i]), X[i].copy()
            history.append(best_f)
            if ftarget is not None and best_f <= ftarget:
                message = "target reached"
                break
        return Result(best_x, best_f, evals, it, message, history)
