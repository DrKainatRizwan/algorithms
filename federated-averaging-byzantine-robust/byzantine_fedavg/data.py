"""Synthetic datasets and client partitioning schemes."""

from __future__ import annotations

import numpy as np


def make_blobs_dataset(
    n_samples: int = 3000,
    n_features: int = 10,
    n_classes: int = 4,
    spread: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian blobs with well separated random centres."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 3.0, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    X = centres[y] + rng.normal(0.0, spread, size=(n_samples, n_features))
    return X, y


def partition_iid(
    n_samples: int, n_clients: int, seed: int = 0
) -> list[np.ndarray]:
    """Shuffle indices and split them into near-equal client shards."""
    if n_clients <= 0 or n_clients > n_samples:
        raise ValueError("n_clients must be in [1, n_samples]")
    rng = np.random.default_rng(seed)
    return [s.copy() for s in np.array_split(rng.permutation(n_samples), n_clients)]


def partition_label_skew(
    y: np.ndarray, n_clients: int, alpha: float = 0.5, seed: int = 0
) -> list[np.ndarray]:
    """Dirichlet label-skew partition (smaller ``alpha`` => more non-IID)."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    shards: list[list[int]] = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        props = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
        for k, part in enumerate(np.split(idx, cuts)):
            shards[k].extend(part.tolist())
    # Guarantee every client owns at least one sample.
    for k in range(n_clients):
        if not shards[k]:
            donor = max(range(n_clients), key=lambda j: len(shards[j]))
            shards[k].append(shards[donor].pop())
    return [np.array(sorted(s), dtype=int) for s in shards]
