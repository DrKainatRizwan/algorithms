"""Gradient-free and Langevin Metropolis-Hastings samplers."""
from __future__ import annotations

import numpy as np

from .targets import Target


class _Chain:
    """Shared bookkeeping for single-chain samplers."""

    def __init__(self, target: Target, seed: int = 0) -> None:
        self.target = target
        self.rng = np.random.default_rng(seed)
        self.accepted = 0
        self.proposed = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.proposed, 1)


class RandomWalkMetropolis(_Chain):
    """Random-walk Metropolis with isotropic Gaussian proposals.

    Optimal scale for Gaussians is about ``2.38 / sqrt(d)`` (Roberts et al. 1997).
    """

    def __init__(self, target: Target, step: float | None = None, seed: int = 0) -> None:
        super().__init__(target, seed)
        self.step = 2.38 / np.sqrt(target.dim) if step is None else step

    def sample(self, n: int, x0: np.ndarray, burn: int = 0) -> np.ndarray:
        x = np.asarray(x0, float).copy()
        lp = self.target.log_prob(x)
        out = np.empty((n, x.size))
        self.accepted = self.proposed = 0
        for i in range(n + burn):
            prop = x + self.step * self.rng.normal(size=x.size)
            lp_prop = self.target.log_prob(prop)
            self.proposed += 1
            if np.log(self.rng.random()) < lp_prop - lp:
                x, lp = prop, lp_prop
                self.accepted += 1
            if i >= burn:
                out[i - burn] = x
        return out


class AdaptiveMetropolis(_Chain):
    """Haario-Saksman-Tamminen adaptive Metropolis.

    The proposal covariance is ``(2.38^2 / d) * (Cov_t + eps I)`` where ``Cov_t``
    is the running empirical covariance of the chain.
    """

    def __init__(self, target: Target, adapt_start: int = 200, eps: float = 1e-6, seed: int = 0) -> None:
        super().__init__(target, seed)
        self.adapt_start = adapt_start
        self.eps = eps
        self.cov_: np.ndarray | None = None

    def sample(self, n: int, x0: np.ndarray, burn: int = 0) -> np.ndarray:
        d = self.target.dim
        x = np.asarray(x0, float).copy()
        lp = self.target.log_prob(x)
        scale = 2.38 ** 2 / d
        mean = x.copy()
        cov = np.zeros((d, d))
        t = 1
        chol = np.linalg.cholesky(scale * (0.1 * np.eye(d)))
        out = np.empty((n, d))
        self.accepted = self.proposed = 0
        for i in range(n + burn):
            prop = x + chol @ self.rng.normal(size=d)
            lp_prop = self.target.log_prob(prop)
            self.proposed += 1
            if np.log(self.rng.random()) < lp_prop - lp:
                x, lp = prop, lp_prop
                self.accepted += 1
            # Recursive mean/covariance update (Haario et al. 2001).
            t += 1
            delta = x - mean
            cov = cov * (t - 2) / (t - 1) + np.outer(delta, delta) / t
            mean = mean + delta / t
            if t >= self.adapt_start:
                chol = np.linalg.cholesky(scale * (cov + self.eps * np.eye(d)))
            if i >= burn:
                out[i - burn] = x
        self.cov_ = cov
        return out


class MALA(_Chain):
    """Metropolis-adjusted Langevin algorithm."""

    def __init__(self, target: Target, step: float = 0.5, seed: int = 0) -> None:
        super().__init__(target, seed)
        self.step = step

    def _log_q(self, a: np.ndarray, b: np.ndarray, grad_b: np.ndarray) -> float:
        """log density of proposing ``a`` from ``b``."""
        mu = b + 0.5 * self.step ** 2 * grad_b
        return float(-0.5 * np.sum((a - mu) ** 2) / self.step ** 2)

    def sample(self, n: int, x0: np.ndarray, burn: int = 0) -> np.ndarray:
        x = np.asarray(x0, float).copy()
        lp = self.target.log_prob(x)
        g = self.target.grad_log_prob(x)
        out = np.empty((n, x.size))
        self.accepted = self.proposed = 0
        for i in range(n + burn):
            prop = x + 0.5 * self.step ** 2 * g + self.step * self.rng.normal(size=x.size)
            lp_prop = self.target.log_prob(prop)
            g_prop = self.target.grad_log_prob(prop)
            log_alpha = (lp_prop + self._log_q(x, prop, g_prop)) - (lp + self._log_q(prop, x, g))
            self.proposed += 1
            if np.log(self.rng.random()) < log_alpha:
                x, lp, g = prop, lp_prop, g_prop
                self.accepted += 1
            if i >= burn:
                out[i - burn] = x
        return out
