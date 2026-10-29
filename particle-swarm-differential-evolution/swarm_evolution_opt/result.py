"""Common result container and budget-tracked objective wrapper."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class OptimizeResult:
    """Outcome of an optimisation run."""

    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    history: list[float] = field(default_factory=list)  # best-so-far per iteration
    success: bool = False
    message: str = ""


class BatchObjective:
    """Wraps a vectorised objective f(X[n, d]) -> [n] and counts evaluations."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], max_nfev: int | None = None):
        self.func = func
        self.nfev = 0
        self.max_nfev = max_nfev

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        vals = np.asarray(self.func(X), dtype=float).reshape(-1)
        if vals.shape[0] != X.shape[0]:
            raise ValueError("objective must return one value per row")
        self.nfev += X.shape[0]
        return vals

    @property
    def exhausted(self) -> bool:
        return self.max_nfev is not None and self.nfev >= self.max_nfev


def check_bounds(lower, upper, dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast and validate box bounds."""
    lo = np.atleast_1d(np.asarray(lower, dtype=float))
    hi = np.atleast_1d(np.asarray(upper, dtype=float))
    if dim is not None:
        lo = np.broadcast_to(lo, (dim,)).copy()
        hi = np.broadcast_to(hi, (dim,)).copy()
    if lo.shape != hi.shape:
        raise ValueError("lower and upper must have the same shape")
    if np.any(hi <= lo):
        raise ValueError("upper bounds must exceed lower bounds")
    return lo, hi
