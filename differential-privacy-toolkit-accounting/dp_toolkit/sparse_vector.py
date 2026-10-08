"""Sparse vector technique (AboveThreshold)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def above_threshold(query_values: Sequence[float], threshold: float, epsilon: float,
                    rng: np.random.Generator, sensitivity: float = 1.0) -> Optional[int]:
    """Return the index of the first query answered above ``threshold`` or None.

    Satisfies epsilon-DP regardless of how many queries are inspected
    (Dwork & Roth, Algorithm 1).
    """
    rho = rng.laplace(0.0, 2.0 * sensitivity / epsilon)
    for i, v in enumerate(query_values):
        nu = rng.laplace(0.0, 4.0 * sensitivity / epsilon)
        if v + nu >= threshold + rho:
            return i
    return None
