"""DP-SGD for logistic regression (per-example clipping + Gaussian noise)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .accountant import RDPAccountant


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def dp_logistic_regression(X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
                           lr: float, clip: float, noise_multiplier: float,
                           delta: float, rng: np.random.Generator
                           ) -> Tuple[np.ndarray, float, float]:
    """Train with DP-SGD. Returns ``(weights_with_bias, epsilon, delta)``.

    ``noise_multiplier=0`` disables noise (and reports ``inf`` epsilon).
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    q = batch_size / n
    steps = int(epochs * n / batch_size)
    for _ in range(steps):
        mask = rng.random(n) < q                      # Poisson sampling
        xb, yb = Xb[mask], y[mask]
        if len(xb):
            g = (_sigmoid(xb @ w) - yb)[:, None] * xb
            norms = np.linalg.norm(g, axis=1, keepdims=True)
            g = g / np.maximum(1.0, norms / clip)
            gsum = g.sum(axis=0)
        else:
            gsum = np.zeros_like(w)
        gsum = gsum + rng.normal(0.0, noise_multiplier * clip, size=w.shape)
        w -= lr * gsum / batch_size
    if noise_multiplier > 0:
        acc = RDPAccountant()
        acc.compose_subsampled_gaussian(noise_multiplier, q, steps)
        eps = acc.epsilon(delta)
    else:
        eps = float("inf")
    return w, eps, delta
