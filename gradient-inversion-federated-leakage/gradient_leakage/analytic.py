"""Closed-form leakage: a single example is readable from a dense layer gradient.

For ``a = W1 x + b1`` the gradients satisfy ``gW1[i] = gb1[i] * x``, hence
``x = gW1[i] / gb1[i]`` for any row with non-zero bias gradient.
"""
from __future__ import annotations

import numpy as np


def recover_label(grads: dict[str, np.ndarray]) -> int:
    """Label of a single example: ``dz = p - onehot`` is negative only at the label."""
    return int(np.argmin(grads["b2"]))


def analytic_reconstruct(grads: dict[str, np.ndarray], eps: float = 1e-12) -> np.ndarray:
    """Reconstruct a single input from first-layer gradients.

    Uses a weighted least-squares fit over all hidden rows, which equals the
    ratio estimator on clean gradients and is more stable under noise.
    """
    gW, gb = grads["W1"], grads["b1"]
    denom = float(gb @ gb)
    if denom < eps:
        return np.zeros(gW.shape[1])
    return (gb @ gW) / denom
