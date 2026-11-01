"""Synthetic tabular data and small helpers."""
from __future__ import annotations
import numpy as np


def make_tabular(n: int = 600, d: int = 8, informative: int = 4, seed: int = 0,
                 nonlinear: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y, true_weights). Only the first ``informative`` features matter.

    y is a Bernoulli draw whose logit is linear plus (optionally) one interaction
    term between features 0 and 1.
    """
    if not 1 <= informative <= d:
        raise ValueError("informative must be in [1, d]")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[:informative] = rng.uniform(1.0, 2.5, informative) * rng.choice([-1, 1], informative)
    logit = X @ w
    if nonlinear and d >= 2:
        logit = logit + 1.5 * X[:, 0] * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y, w


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Xs, mean, std) with zero-variance columns left unscaled."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def train_test_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.25, seed: int = 0):
    """Deterministic shuffled split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    k = int(len(X) * (1 - test_frac))
    a, b = idx[:k], idx[k:]
    return X[a], X[b], y[a], y[b]
