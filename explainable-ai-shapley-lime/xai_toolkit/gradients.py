"""Gradient-based attribution: integrated gradients."""
from __future__ import annotations
from typing import Callable
import numpy as np


def integrated_gradients(grad_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                         baseline: np.ndarray, steps: int = 64) -> np.ndarray:
    """Integrated gradients with the midpoint rule.

    ``grad_fn`` maps (m, d) inputs to (m, d) gradients of a scalar output.
    Satisfies completeness: sum(attr) ~= f(x) - f(baseline) as steps grows.
    """
    x, baseline = np.asarray(x, float), np.asarray(baseline, float)
    alphas = (np.arange(steps) + 0.5) / steps
    pts = baseline + alphas[:, None] * (x - baseline)
    return (x - baseline) * grad_fn(pts).mean(axis=0)
