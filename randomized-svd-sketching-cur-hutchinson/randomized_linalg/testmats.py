"""Synthetic test matrices."""
from __future__ import annotations

import numpy as np


def _orth(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((n, k)))[0]


def low_rank_plus_noise(m: int, n: int, rank: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Exact rank-``rank`` matrix with singular values 10..1 plus Gaussian noise of scale ``noise``."""
    s = np.linspace(10, 1, rank)
    return (_orth(m, rank, rng) * s) @ _orth(n, rank, rng).T + noise * rng.standard_normal((m, n))


def decaying_spectrum(m: int, n: int, decay: str, rng: np.random.Generator, rate: float = 0.5) -> np.ndarray:
    """Matrix with prescribed singular values: 'poly' (i^-rate*2), 'exp' (e^-rate i) or 'flat'."""
    r = min(m, n)
    i = np.arange(1, r + 1, dtype=float)
    s = {"poly": i ** (-2 * rate), "exp": np.exp(-rate * i), "flat": np.ones(r)}[decay]
    return (_orth(m, r, rng) * s) @ _orth(n, r, rng).T


def ill_conditioned_ls(n: int, d: int, cond: float, rng: np.random.Generator):
    """Least-squares problem with cond(A) = ``cond``; returns (A, b, x_true) with noisy b."""
    s = np.logspace(0, -np.log10(cond), d)
    A = (_orth(n, d, rng) * s) @ _orth(d, d, rng).T
    x = rng.standard_normal(d)
    return A, A @ x + 1e-3 * rng.standard_normal(n), x
