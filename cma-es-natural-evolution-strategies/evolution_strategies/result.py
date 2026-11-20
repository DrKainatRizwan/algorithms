"""Common result container and helpers shared by all optimisers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    """Outcome of an optimisation run."""

    x: np.ndarray
    fun: float
    evals: int
    iterations: int
    message: str = ""
    history: list[float] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.message.startswith("target")


def rank_utilities(n: int) -> np.ndarray:
    """Log-rank utilities used by NES (Wierstra et al., 2014).

    Returns zero-sum weights, best individual first.
    """
    ranks = np.arange(1, n + 1)
    u = np.maximum(0.0, np.log(n / 2.0 + 1.0) - np.log(ranks))
    u = u / u.sum() - 1.0 / n
    return u
