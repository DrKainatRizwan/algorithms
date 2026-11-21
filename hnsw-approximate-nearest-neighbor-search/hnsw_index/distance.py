"""Distance functions. Every metric maps (vector, matrix) -> distances to rows."""
from __future__ import annotations

from typing import Callable

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], np.ndarray]


def l2_sq(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from ``q`` to every row of ``X``."""
    d = X - q
    return np.einsum("ij,ij->i", d, d)


def cosine_dist(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cosine distance ``1 - cos(q, x)``; zero vectors get distance 1."""
    nq = np.linalg.norm(q)
    nx = np.linalg.norm(X, axis=1)
    denom = nq * nx
    denom = np.where(denom == 0, 1.0, denom)
    return 1.0 - (X @ q) / denom


def neg_inner(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Negative inner product (maximum inner product search as minimisation)."""
    return -(X @ q)


_METRICS: dict[str, Metric] = {"l2": l2_sq, "cosine": cosine_dist, "ip": neg_inner}


def get_metric(name: str) -> Metric:
    """Look up a metric by name: ``'l2'``, ``'cosine'`` or ``'ip'``."""
    try:
        return _METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}; choose from {sorted(_METRICS)}")
