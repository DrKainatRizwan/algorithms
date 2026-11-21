"""Synthetic data generators (seeded)."""
from __future__ import annotations

import numpy as np


def gaussian_clusters(n: int, dim: int, clusters: int = 10, spread: float = 0.5,
                      seed: int = 0) -> np.ndarray:
    """Mixture of isotropic Gaussians with random centres in [-5, 5]^dim."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-5, 5, size=(clusters, dim))
    labels = rng.integers(0, clusters, size=n)
    return centres[labels] + spread * rng.standard_normal((n, dim))


def uniform_cube(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Uniform points in the unit hypercube."""
    return np.random.default_rng(seed).random((n, dim))
