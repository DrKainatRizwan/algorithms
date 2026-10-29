"""Differential Evolution: classic DE/x/y/z strategies and jDE self-adaptation."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .result import BatchObjective, OptimizeResult, check_bounds

STRATEGIES = ("rand1", "best1", "current_to_best1", "rand2", "best2")


def _distinct_indices(rng: np.random.Generator, n: int, count: int) -> np.ndarray:
    """For each i in range(n) draw ``count`` distinct indices, all different from i. Shape (n, count)."""
    if n < count + 1:
        raise ValueError(f"population size must be at least {count + 1}")
    # random permutation trick: rank random keys with i excluded
    keys = rng.random((n, n))
    keys[np.arange(n), np.arange(n)] = np.inf
    return np.argsort(keys, axis=1)[:, :count]


class DifferentialEvolution:
    """Differential evolution (minimisation) with binomial / exponential crossover.

    ``strategy`` is one of ``rand1, best1, current_to_best1, rand2, best2``. With ``adaptive=True``
    each individual carries its own (F, CR) pair, regenerated with small probability following
    jDE (Brest et al., 2006). Out-of-range mutant coordinates are bounced back between the
    violated bound and the parent.
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], lower, upper, dim: int | None = None,
                 pop_size: int = 40, max_iter: int = 300, F: float = 0.6, CR: float = 0.9,
                 strategy: str = "rand1", crossover: str = "bin", adaptive: bool = False,
                 tau1: float = 0.1, tau2: float = 0.1, tol: float | None = None,
                 max_nfev: int | None = None, seed: int | None = 0):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
        if crossover not in ("bin", "exp"):
            raise ValueError("crossover must be 'bin' or 'exp'")
        if not 0.0 <= CR <= 1.0:
            raise ValueError("CR must lie in [0, 1]")
        self.lower, self.upper = check_bounds(lower, upper, dim)
        self.dim = self.lower.size
        self.obj = BatchObjective(func, max_nfev)
        self.n, self.max_iter, self.F, self.CR = pop_size, max_iter, F, CR
        self.strategy, self.crossover, self.adaptive = strategy, crossover, adaptive
        self.tau1, self.tau2, self.tol = tau1, tau2, tol
        self.rng = np.random.default_rng(seed)
        _distinct_indices(self.rng, self.n, 5 if strategy.endswith("2") else 3)  # validate size

    def _mutate(self, pop: np.ndarray, fit: np.ndarray, F: np.ndarray) -> np.ndarray:
        best = pop[np.argmin(fit)]
        F = F[:, None]
        if self.strategy in ("rand1", "best1", "current_to_best1"):
            r = _distinct_indices(self.rng, self.n, 3)
            if self.strategy == "rand1":
                return pop[r[:, 0]] + F * (pop[r[:, 1]] - pop[r[:, 2]])
            if self.strategy == "best1":
                return best + F * (pop[r[:, 0]] - pop[r[:, 1]])
            return pop + F * (best - pop) + F * (pop[r[:, 0]] - pop[r[:, 1]])
        r = _distinct_indices(self.rng, self.n, 5)
        d1 = pop[r[:, 0]] + pop[r[:, 1]] - pop[r[:, 2]] - pop[r[:, 3]]
        if self.strategy == "rand2":
            return pop[r[:, 4]] + F * d1
        return best + F * d1

    def _crossover(self, pop: np.ndarray, mutant: np.ndarray, CR: np.ndarray) -> np.ndarray:
        n, d = pop.shape
        jrand = self.rng.integers(0, d, size=n)
        if self.crossover == "bin":
            mask = self.rng.random((n, d)) < CR[:, None]
            mask[np.arange(n), jrand] = True
        else:
            # exponential: copy a contiguous (circular) run starting at jrand
            mask = np.zeros((n, d), dtype=bool)
            run = np.ones(n, dtype=int)
            more = self.rng.random((n, d)) < CR[:, None]
            for k in range(1, d):
                cont = more[:, k - 1] & (run == k)
                run = run + cont
            cols = (jrand[:, None] + np.arange(d)[None, :]) % d
            mask[np.arange(n)[:, None], cols] = np.arange(d)[None, :] < run[:, None]
        return np.where(mask, mutant, pop)

    def _repair(self, trial: np.ndarray, parent: np.ndarray) -> np.ndarray:
        lo, hi = self.lower, self.upper
        trial = np.where(trial < lo, (lo + parent) / 2, trial)
        trial = np.where(trial > hi, (hi + parent) / 2, trial)
        return trial

    def run(self) -> OptimizeResult:
        rng, lo, hi = self.rng, self.lower, self.upper
        pop = lo + rng.random((self.n, self.dim)) * (hi - lo)
        fit = self.obj(pop)
        Fv = np.full(self.n, self.F)
        CRv = np.full(self.n, self.CR)
        history = [float(fit.min())]
        msg, it = "max iterations reached", 0
        for it in range(1, self.max_iter + 1):
            if self.obj.exhausted:
                msg = "evaluation budget exhausted"
                break
            if self.adaptive:
                Fv = np.where(rng.random(self.n) < self.tau1, 0.1 + 0.9 * rng.random(self.n), Fv)
                CRv = np.where(rng.random(self.n) < self.tau2, rng.random(self.n), CRv)
            mutant = self._mutate(pop, fit, Fv)
            trial = self._repair(self._crossover(pop, mutant, CRv), pop)
            tfit = self.obj(trial)
            win = tfit <= fit
            pop[win], fit[win] = trial[win], tfit[win]
            if self.adaptive:
                # parameters survive only with successful trials; losers revert to defaults next draw
                Fv = np.where(win, Fv, np.where(rng.random(self.n) < 0.5, self.F, Fv))
            history.append(float(fit.min()))
            if self.tol is not None and history[-1] <= self.tol:
                msg = "target tolerance reached"
                break
        g = int(np.argmin(fit))
        return OptimizeResult(pop[g].copy(), float(fit[g]), self.obj.nfev, it, history,
                              success=True, message=msg)


def de_minimize(func, lower, upper, dim: int | None = None, **kwargs) -> OptimizeResult:
    """Convenience wrapper: ``DifferentialEvolution(...).run()``."""
    return DifferentialEvolution(func, lower, upper, dim, **kwargs).run()
