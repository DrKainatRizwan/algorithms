"""Exact Gaussian process regression via Cholesky factorisation."""

from __future__ import annotations

import numpy as np

from .kernels import Kernel, Matern52
from .numopt import nelder_mead


def _solve_chol(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (L L^T) x = b using two triangular solves (generic solve; sizes are small)."""
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


class GaussianProcess:
    """GP regressor with zero-mean prior on standardised targets.

    Targets are standardised internally so that the signal variance and noise
    hyper-parameters are on a sensible scale for any objective.
    """

    def __init__(self, kernel: Kernel | None = None, noise: float = 1e-6, jitter: float = 1e-10):
        self.kernel = kernel if kernel is not None else Matern52()
        self.noise = float(noise)
        self.jitter = jitter
        self.X: np.ndarray | None = None
        self._y_mean = 0.0
        self._y_std = 1.0

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = False,
            n_restarts: int = 3, seed: int = 0) -> "GaussianProcess":
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        if X.shape[0] != y.size:
            raise ValueError("X and y have inconsistent lengths")
        self.X = X
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) if y.std() > 1e-12 else 1.0
        self._y = (y - self._y_mean) / self._y_std
        if optimize:
            self._optimize_hyperparameters(n_restarts, seed)
        self._factorize()
        return self

    def _factorize(self) -> None:
        K = self.kernel(self.X, self.X)
        K[np.diag_indices_from(K)] += self.noise + self.jitter
        # Escalate jitter if the matrix is numerically indefinite.
        extra = 0.0
        for _ in range(8):
            try:
                self._L = np.linalg.cholesky(K + extra * np.eye(len(K)))
                break
            except np.linalg.LinAlgError:
                extra = max(extra * 10, 1e-10)
        else:
            raise np.linalg.LinAlgError("covariance not positive definite")
        self._alpha = _solve_chol(self._L, self._y)

    # ------------------------------------------------------- marginal lik.
    def log_marginal_likelihood(self) -> float:
        """log p(y | X, theta) of the standardised targets."""
        n = len(self._y)
        return float(-0.5 * self._y @ self._alpha - np.log(np.diag(self._L)).sum()
                     - 0.5 * n * np.log(2 * np.pi))

    def _neg_lml(self, theta: np.ndarray) -> float:
        k_old, n_old = self.kernel.get_log_params(), self.noise
        try:
            self.kernel.set_log_params(theta[:-1])
            self.noise = float(np.exp(theta[-1]))
            self._factorize()
            return -self.log_marginal_likelihood()
        except np.linalg.LinAlgError:
            return 1e300
        finally:
            self.kernel.set_log_params(k_old)
            self.noise = n_old

    def _optimize_hyperparameters(self, n_restarts: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        d = self.X.shape[1]
        ard = self.kernel.lengthscale.size > 1
        nls = d if ard else 1
        lo = np.concatenate([np.full(nls, np.log(1e-2)), [np.log(1e-2)], [np.log(1e-8)]])
        hi = np.concatenate([np.full(nls, np.log(1e2)), [np.log(1e2)], [np.log(1.0)]])
        start = np.concatenate([self.kernel.get_log_params(), [np.log(max(self.noise, 1e-8))]])
        starts = [np.clip(start, lo, hi)]
        for _ in range(n_restarts):
            starts.append(rng.uniform(lo, hi))
        best_t, best_v = None, np.inf
        for s in starts:
            t, v = nelder_mead(self._neg_lml, s, step=0.5, bounds=(lo, hi), max_iter=150)
            if v < best_v:
                best_t, best_v = t, v
        self.kernel.set_log_params(best_t[:-1])
        self.noise = float(np.exp(best_t[-1]))

    # ------------------------------------------------------------- predict
    def predict(self, Xs: np.ndarray, return_std: bool = True):
        """Posterior mean (and standard deviation) at ``Xs`` in original units."""
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = self.kernel(self.X, Xs)
        mu = Ks.T @ self._alpha
        mean = mu * self._y_std + self._y_mean
        if not return_std:
            return mean
        v = np.linalg.solve(self._L, Ks)
        var = self.kernel.diag(Xs) - (v ** 2).sum(0)
        std = np.sqrt(np.maximum(var, 1e-18)) * self._y_std
        return mean, std

    def sample_posterior(self, Xs: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Joint posterior draws at ``Xs``; shape (n_samples, len(Xs))."""
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = self.kernel(self.X, Xs)
        mu = Ks.T @ self._alpha
        v = np.linalg.solve(self._L, Ks)
        cov = self.kernel(Xs, Xs) - v.T @ v
        cov[np.diag_indices_from(cov)] += 1e-8
        Lc = np.linalg.cholesky(cov)
        z = rng.standard_normal((len(Xs), n_samples))
        return (mu[:, None] + Lc @ z).T * self._y_std + self._y_mean
