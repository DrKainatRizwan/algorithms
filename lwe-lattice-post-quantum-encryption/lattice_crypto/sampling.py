"""Error and secret distributions used by LWE-style schemes."""
from __future__ import annotations

import numpy as np


def uniform_mod(rng: np.random.Generator, q: int, shape) -> np.ndarray:
    """Uniform integers in [0, q)."""
    return rng.integers(0, q, size=shape, dtype=np.int64)


def centered_binomial(rng: np.random.Generator, eta: int, shape) -> np.ndarray:
    """Centered binomial CBD(eta): sum of eta coin differences, variance eta/2."""
    a = rng.integers(0, 2, size=(eta,) + tuple(np.atleast_1d(shape)), dtype=np.int64)
    b = rng.integers(0, 2, size=(eta,) + tuple(np.atleast_1d(shape)), dtype=np.int64)
    return (a.sum(axis=0) - b.sum(axis=0)).astype(np.int64)


def discrete_gaussian(rng: np.random.Generator, sigma: float, shape, tail: float = 6.0) -> np.ndarray:
    """Discrete Gaussian on Z with std ``sigma`` via inverse-CDF on a truncated support."""
    if sigma <= 0:
        return np.zeros(shape, dtype=np.int64)
    bound = int(np.ceil(tail * sigma))
    support = np.arange(-bound, bound + 1)
    w = np.exp(-(support.astype(float) ** 2) / (2 * sigma * sigma))
    w /= w.sum()
    return rng.choice(support, size=shape, p=w).astype(np.int64)


def center(x: np.ndarray, q: int) -> np.ndarray:
    """Map residues mod q to the symmetric range (-q/2, q/2]."""
    x = np.asarray(x, dtype=np.int64) % q
    return np.where(x > q // 2, x - q, x)
