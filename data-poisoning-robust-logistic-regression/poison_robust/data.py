"""Synthetic data generation and splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    """Train / validation / test partitions (validation is the attacker's objective set)."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def make_blobs(n: int, d: int = 2, separation: float = 2.0, noise: float = 1.0,
               rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian classes with means ``+/- separation/2`` along a fixed direction.

    Returns ``(X, y)`` with balanced labels in {0, 1}.
    """
    rng = rng or np.random.default_rng(0)
    y = np.zeros(n, dtype=int)
    y[n // 2:] = 1
    direction = np.ones(d) / np.sqrt(d)
    means = np.where(y[:, None] == 1, 0.5, -0.5) * separation * direction[None, :]
    X = means + noise * rng.standard_normal((n, d))
    perm = rng.permutation(n)
    return X[perm], y[perm]


def train_val_test_split(n_train: int = 300, n_val: int = 300, n_test: int = 1000, d: int = 2,
                         separation: float = 2.5, noise: float = 1.0, seed: int = 0) -> Dataset:
    """Draw three independent samples from the same blob distribution."""
    rng = np.random.default_rng(seed)
    parts = [make_blobs(n, d, separation, noise, rng) for n in (n_train, n_val, n_test)]
    return Dataset(parts[0][0], parts[0][1], parts[1][0], parts[1][1], parts[2][0], parts[2][1])
