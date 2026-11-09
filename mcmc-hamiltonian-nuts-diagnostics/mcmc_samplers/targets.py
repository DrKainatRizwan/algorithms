"""Log-density targets with analytic gradients.

Every target exposes ``log_prob(x)`` and ``grad_log_prob(x)`` for a 1-D array
``x`` and a ``dim`` attribute.  Densities are unnormalised unless stated.
"""
from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


class Target:
    """Abstract target distribution."""

    dim: int

    def log_prob(self, x: FloatArray) -> float:
        raise NotImplementedError

    def grad_log_prob(self, x: FloatArray) -> FloatArray:
        raise NotImplementedError

    def numerical_grad(self, x: FloatArray, h: float = 1e-6) -> FloatArray:
        """Central finite-difference gradient, used for testing."""
        g = np.zeros_like(x, dtype=float)
        for i in range(x.size):
            e = np.zeros_like(x, dtype=float)
            e[i] = h
            g[i] = (self.log_prob(x + e) - self.log_prob(x - e)) / (2 * h)
        return g


class CorrelatedGaussian(Target):
    """Zero-mean (or shifted) multivariate normal with covariance ``cov``."""

    def __init__(self, cov: FloatArray, mean: FloatArray | None = None) -> None:
        self.cov = np.asarray(cov, dtype=float)
        self.dim = self.cov.shape[0]
        self.mean = np.zeros(self.dim) if mean is None else np.asarray(mean, float)
        self.prec = np.linalg.inv(self.cov)

    @classmethod
    def ill_conditioned(cls, dim: int, ratio: float = 100.0, seed: int = 0) -> "CorrelatedGaussian":
        """Random rotation of a spectrum log-spaced between 1 and ``ratio``."""
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        eig = np.logspace(0, np.log10(ratio), dim)
        return cls((q * eig) @ q.T)

    def log_prob(self, x: FloatArray) -> float:
        d = x - self.mean
        return float(-0.5 * d @ self.prec @ d)

    def grad_log_prob(self, x: FloatArray) -> FloatArray:
        return -self.prec @ (x - self.mean)


class Banana(Target):
    """Haario's banana: ``y' = y + b (x^2 - a^2)`` applied to a Gaussian."""

    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a, self.b, self.dim = a, b, 2

    def log_prob(self, x: FloatArray) -> float:
        u = x[1] + self.b * (x[0] ** 2 - self.a ** 2)
        return float(-0.5 * (x[0] / self.a) ** 2 - 0.5 * u ** 2)

    def grad_log_prob(self, x: FloatArray) -> FloatArray:
        u = x[1] + self.b * (x[0] ** 2 - self.a ** 2)
        return np.array([-x[0] / self.a ** 2 - u * 2 * self.b * x[0], -u])


class GaussianMixture(Target):
    """Isotropic Gaussian mixture, normalised weights."""

    def __init__(self, means: FloatArray, sigma: float = 1.0, weights: FloatArray | None = None) -> None:
        self.means = np.asarray(means, dtype=float)
        self.k, self.dim = self.means.shape
        self.sigma = sigma
        w = np.ones(self.k) / self.k if weights is None else np.asarray(weights, float)
        self.log_w = np.log(w / w.sum())

    def _component_logs(self, x: FloatArray) -> FloatArray:
        d2 = ((x - self.means) ** 2).sum(axis=1)
        return self.log_w - 0.5 * d2 / self.sigma ** 2

    def log_prob(self, x: FloatArray) -> float:
        z = self._component_logs(x)
        m = z.max()
        return float(m + np.log(np.exp(z - m).sum()))

    def grad_log_prob(self, x: FloatArray) -> FloatArray:
        z = self._component_logs(x)
        r = np.exp(z - z.max())
        r /= r.sum()
        return (r[:, None] * (self.means - x)).sum(axis=0) / self.sigma ** 2


class NealsFunnel(Target):
    """Neal's funnel: ``v ~ N(0, 9)``, ``x_i | v ~ N(0, exp(v))``."""

    def __init__(self, dim: int = 3) -> None:
        self.dim = dim

    def log_prob(self, x: FloatArray) -> float:
        v = x[0]
        return float(-v ** 2 / 18 - 0.5 * (self.dim - 1) * v - 0.5 * np.exp(-v) * (x[1:] ** 2).sum())

    def grad_log_prob(self, x: FloatArray) -> FloatArray:
        v = x[0]
        s = (x[1:] ** 2).sum()
        g = np.empty(self.dim)
        g[0] = -v / 9 - 0.5 * (self.dim - 1) + 0.5 * np.exp(-v) * s
        g[1:] = -np.exp(-v) * x[1:]
        return g


class BayesianLogisticRegression(Target):
    """Posterior of logistic regression with an N(0, prior_var) prior."""

    def __init__(self, X: FloatArray, y: FloatArray, prior_var: float = 10.0) -> None:
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        self.prior_var = prior_var
        self.dim = self.X.shape[1]

    @staticmethod
    def synthetic(n: int = 200, dim: int = 4, seed: int = 0):
        """Return ``(target, true_beta)`` for simulated data."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, dim))
        beta = rng.normal(size=dim)
        p = 1 / (1 + np.exp(-X @ beta))
        y = (rng.random(n) < p).astype(float)
        return BayesianLogisticRegression(X, y), beta

    def log_prob(self, b: FloatArray) -> float:
        eta = self.X @ b
        ll = np.sum(self.y * eta - np.logaddexp(0.0, eta))
        return float(ll - 0.5 * b @ b / self.prior_var)

    def grad_log_prob(self, b: FloatArray) -> FloatArray:
        p = 1 / (1 + np.exp(-(self.X @ b)))
        return self.X.T @ (self.y - p) - b / self.prior_var
