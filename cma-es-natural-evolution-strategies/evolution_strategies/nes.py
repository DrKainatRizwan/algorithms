"""Natural Evolution Strategies: exponential NES (xNES) and separable NES (SNES).

Both follow Wierstra et al., "Natural Evolution Strategies" (JMLR 2014): the
search distribution is updated along the natural gradient of expected utility,
with rank-based fitness shaping.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .result import Result, rank_utilities


def expm_sym(m: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix via eigendecomposition."""
    w, v = np.linalg.eigh((m + m.T) / 2)
    return (v * np.exp(w)) @ v.T


class _NESBase:
    n: int
    lam: int
    mean: np.ndarray
    rng: np.random.Generator

    def ask(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def tell(self, x: np.ndarray, f: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def minimize(
        self,
        func: Callable[[np.ndarray], float],
        max_evals: int = 100000,
        ftarget: float | None = None,
    ) -> Result:
        """Run ask/tell cycles until the target or evaluation budget is hit."""
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


class XNES(_NESBase):
    """Exponential NES with full covariance sigma^2 B B^T, det(B) = 1."""

    def __init__(self, x0, sigma0: float, popsize: int | None = None, seed: int = 0):
        self.mean = np.asarray(x0, dtype=float).copy()
        n = self.n = len(self.mean)
        self.lam = popsize or 4 + int(3 * np.log(n))
        self.rng = np.random.default_rng(seed)
        self.sigma = float(sigma0)
        self.B = np.eye(n)
        self.eta_mu = 1.0
        self.eta_sigma = 0.6 * (3 + np.log(n)) / (n * np.sqrt(n))
        self.eta_B = self.eta_sigma
        self.utilities = rank_utilities(self.lam)
        self._z: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        self._z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * self._z @ self.B.T

    def tell(self, x: np.ndarray, f: np.ndarray) -> None:
        if self._z is None:
            raise RuntimeError("ask() must be called before tell()")
        z = self._z[np.argsort(f, kind="stable")]
        u = self.utilities
        n = self.n
        g_delta = u @ z
        g_M = (z.T * u) @ z - u.sum() * np.eye(n)
        g_sigma = np.trace(g_M) / n
        g_B = g_M - g_sigma * np.eye(n)
        self.mean = self.mean + self.eta_mu * self.sigma * self.B @ g_delta
        self.sigma *= np.exp(0.5 * self.eta_sigma * g_sigma)
        self.B = self.B @ expm_sym(0.5 * self.eta_B * g_B)
        self._z = None

    @property
    def covariance(self) -> np.ndarray:
        return self.sigma ** 2 * self.B @ self.B.T


class SNES(_NESBase):
    """Separable NES: diagonal covariance, O(n) per sample."""

    def __init__(self, x0, sigma0: float, popsize: int | None = None, seed: int = 0):
        self.mean = np.asarray(x0, dtype=float).copy()
        n = self.n = len(self.mean)
        self.lam = popsize or 4 + int(3 * np.log(n))
        self.rng = np.random.default_rng(seed)
        self.sigma = np.full(n, float(sigma0))
        self.eta_mu = 1.0
        self.eta_sigma = (3 + np.log(n)) / (5 * np.sqrt(n))
        self.utilities = rank_utilities(self.lam)
        self._z: np.ndarray | None = None

    def ask(self) -> np.ndarray:
        self._z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * self._z

    def tell(self, x: np.ndarray, f: np.ndarray) -> None:
        if self._z is None:
            raise RuntimeError("ask() must be called before tell()")
        z = self._z[np.argsort(f, kind="stable")]
        u = self.utilities
        self.mean = self.mean + self.eta_mu * self.sigma * (u @ z)
        self.sigma = self.sigma * np.exp(0.5 * self.eta_sigma * (u @ (z * z - 1.0)))
        self._z = None
