"""Small synthetic classification datasets, all driven by an explicit seed."""

from __future__ import annotations

import numpy as np


def make_blobs(n_per_class: int = 100, n_classes: int = 3, dim: int = 2,
               spread: float = 0.6, radius: float = 2.5, seed: int = 0
               ) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blobs whose centres lie on a circle (first two dims)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        ang = 2 * np.pi * c / n_classes
        centre = np.zeros(dim)
        centre[0], centre[1] = radius * np.cos(ang), radius * np.sin(ang)
        xs.append(centre + spread * rng.standard_normal((n_per_class, dim)))
        ys.append(np.full(n_per_class, c))
    X, y = np.vstack(xs), np.concatenate(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_spirals(n_per_class: int = 150, n_classes: int = 2, noise: float = 0.15,
                 turns: float = 1.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved spirals: a non-linearly separable benchmark."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        t = np.linspace(0.15, 1.0, n_per_class)
        ang = 2 * np.pi * (turns * t + c / n_classes)
        r = 3.0 * t
        pts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
        xs.append(pts + noise * rng.standard_normal(pts.shape))
        ys.append(np.full(n_per_class, c))
    X, y = np.vstack(xs), np.concatenate(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_rings(n_per_class: int = 150, n_classes: int = 2, noise: float = 0.12,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Concentric rings of increasing radius."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        ang = rng.uniform(0, 2 * np.pi, n_per_class)
        r = (c + 1) * 1.2 + noise * rng.standard_normal(n_per_class)
        xs.append(np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1))
        ys.append(np.full(n_per_class, c))
    X, y = np.vstack(xs), np.concatenate(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_test_split(X: np.ndarray, y: np.ndarray, test_fraction: float = 0.3,
                     seed: int = 0):
    """Shuffle and split into (X_train, y_train, X_test, y_test)."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(round(test_fraction * len(y))))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def make_fragile(n: int = 600, dim: int = 10, eta: float = 0.3, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Robust-vs-non-robust features toy problem (after Tsipras et al., 2019).

    With label sign s in {-1, +1}: feature 0 is ``s + N(0, 1)`` (weak but
    large-margin), the remaining ``dim - 1`` features are ``s * eta + N(0, 0.15^2)``
    (individually tiny margin, jointly very predictive).  A naturally trained
    model exploits the tiny-margin features, which an L-inf attack with
    ``eps > eta`` can flip simultaneously.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = 2.0 * y - 1.0
    X = np.empty((n, dim))
    X[:, 0] = s + rng.standard_normal(n)
    X[:, 1:] = s[:, None] * eta + 0.15 * rng.standard_normal((n, dim - 1))
    return X, y
