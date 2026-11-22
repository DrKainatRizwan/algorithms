"""Stochastic trace estimation: Hutchinson and variance-reduced Hutch++ (Meyer et al. 2021)."""
from __future__ import annotations

from typing import Callable

import numpy as np

MatVec = Callable[[np.ndarray], np.ndarray]


def hutchinson(matvec: MatVec, n: int, num_queries: int, rng: np.random.Generator,
               dist: str = "rademacher") -> float:
    """Unbiased estimate of tr(A) from ``num_queries`` products with random probes."""
    if dist == "rademacher":
        G = rng.choice([-1.0, 1.0], size=(n, num_queries))
    elif dist == "gaussian":
        G = rng.standard_normal((n, num_queries))
    else:
        raise ValueError("dist must be 'rademacher' or 'gaussian'")
    return float(np.sum(G * matvec(G)) / num_queries)


def hutchpp(matvec: MatVec, n: int, num_queries: int, rng: np.random.Generator) -> float:
    """Hutch++ for PSD matrices: exact trace on a sketched top subspace plus Hutchinson on the residual.

    Needs O(1/eps) queries for a (1±eps) estimate versus O(1/eps^2) for Hutchinson.
    """
    k = max(num_queries // 3, 1)
    S = rng.choice([-1.0, 1.0], size=(n, k))
    Q, _ = np.linalg.qr(matvec(S))
    t_low = float(np.sum(Q * matvec(Q)))
    G = rng.choice([-1.0, 1.0], size=(n, k))
    G = G - Q @ (Q.T @ G)
    t_res = float(np.sum(G * matvec(G)) / k)
    return t_low + t_res
