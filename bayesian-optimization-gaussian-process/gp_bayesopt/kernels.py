"""Stationary covariance kernels with automatic relevance determination (ARD)."""

from __future__ import annotations

import numpy as np


def _scaled_dist(x1: np.ndarray, x2: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """Euclidean distance after dividing each dimension by its length-scale."""
    a = x1 / ls
    b = x2 / ls
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(d2, 0.0))


class Kernel:
    """Base class: k(x, x') = variance * f(r), r the scaled distance."""

    name = "kernel"

    def __init__(self, lengthscale: float | np.ndarray = 1.0, variance: float = 1.0):
        self.lengthscale = np.atleast_1d(np.asarray(lengthscale, dtype=float))
        self.variance = float(variance)

    def _profile(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        ls = np.broadcast_to(self.lengthscale, (x1.shape[1],))
        return self.variance * self._profile(_scaled_dist(x1, x2, ls))

    def diag(self, x: np.ndarray) -> np.ndarray:
        """Prior variance at each input (stationary => constant)."""
        return np.full(np.atleast_2d(x).shape[0], self.variance)

    # Hyper-parameters are exposed in log space for unconstrained optimisation.
    def get_log_params(self) -> np.ndarray:
        return np.log(np.concatenate([self.lengthscale, [self.variance]]))

    def set_log_params(self, theta: np.ndarray) -> None:
        p = np.exp(np.asarray(theta, dtype=float))
        self.lengthscale = p[:-1].copy()
        self.variance = float(p[-1])

    def clone(self) -> "Kernel":
        return type(self)(self.lengthscale.copy(), self.variance)


class RBF(Kernel):
    """Squared-exponential kernel, infinitely differentiable sample paths."""

    name = "rbf"

    def _profile(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * r ** 2)


class Matern32(Kernel):
    """Matern nu=3/2 kernel (once differentiable sample paths)."""

    name = "matern32"

    def _profile(self, r: np.ndarray) -> np.ndarray:
        s = np.sqrt(3.0) * r
        return (1.0 + s) * np.exp(-s)


class Matern52(Kernel):
    """Matern nu=5/2 kernel (twice differentiable); the usual BO default."""

    name = "matern52"

    def _profile(self, r: np.ndarray) -> np.ndarray:
        s = np.sqrt(5.0) * r
        return (1.0 + s + s ** 2 / 3.0) * np.exp(-s)
