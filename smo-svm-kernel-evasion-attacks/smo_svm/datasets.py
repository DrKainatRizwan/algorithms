"""Small seeded synthetic datasets."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def make_blobs(n: int = 200, centers: Array | None = None, std: float = 0.8, seed: int = 0) -> tuple[Array, Array]:
    """Isotropic Gaussian clusters; labels are cluster indices."""
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = np.array([[-2.0, -2.0], [2.0, 2.0]])
    centers = np.asarray(centers, dtype=float)
    k = len(centers)
    y = np.arange(n) % k
    X = centers[y] + std * rng.standard_normal((n, centers.shape[1]))
    return X, y


def make_moons(n: int = 200, noise: float = 0.15, seed: int = 0) -> tuple[Array, Array]:
    """Two interleaving half circles."""
    rng = np.random.default_rng(seed)
    m = n // 2
    th = np.linspace(0, np.pi, m)
    a = np.column_stack([np.cos(th), np.sin(th)])
    b = np.column_stack([1 - np.cos(th), 0.5 - np.sin(th)])
    X = np.vstack([a, b]) + noise * rng.standard_normal((2 * m, 2))
    y = np.r_[np.zeros(m, int), np.ones(m, int)]
    return X, y


def make_circles(n: int = 200, noise: float = 0.08, factor: float = 0.4, seed: int = 0) -> tuple[Array, Array]:
    """Concentric circles; inner radius ``factor`` times the outer."""
    rng = np.random.default_rng(seed)
    m = n // 2
    th = rng.uniform(0, 2 * np.pi, 2 * m)
    r = np.r_[np.ones(m), factor * np.ones(m)]
    X = np.column_stack([r * np.cos(th), r * np.sin(th)]) + noise * rng.standard_normal((2 * m, 2))
    y = np.r_[np.zeros(m, int), np.ones(m, int)]
    return X, y


def make_xor(n: int = 200, std: float = 0.3, seed: int = 0) -> tuple[Array, Array]:
    """Four Gaussian clusters at the corners of a square with XOR labels."""
    rng = np.random.default_rng(seed)
    corners = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], float)
    idx = np.arange(n) % 4
    X = corners[idx] + std * rng.standard_normal((n, 2))
    return X, (idx >= 2).astype(int)


def train_test_split(X: Array, y: Array, test_size: float = 0.3, seed: int = 0):
    """Shuffle and split into ``X_tr, X_te, y_tr, y_te``."""
    perm = np.random.default_rng(seed).permutation(len(X))
    cut = int(round(len(X) * (1 - test_size)))
    tr, te = perm[:cut], perm[cut:]
    return X[tr], X[te], y[tr], y[te]
