"""Test problem generators."""
from __future__ import annotations

import numpy as np

from .sparse import CSRMatrix


def poisson_2d(n: int) -> CSRMatrix:
    """5-point Laplacian on an n x n grid (SPD, size n^2)."""
    r, c, v = [], [], []
    for i in range(n):
        for j in range(n):
            k = i * n + j
            r.append(k); c.append(k); v.append(4.0)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    r.append(k); c.append(ii * n + jj); v.append(-1.0)
    return CSRMatrix.from_triplets(r, c, v, (n * n, n * n))


def convection_diffusion_2d(n: int, peclet: float = 1.0) -> CSRMatrix:
    """Upwind-free central-difference convection-diffusion operator (nonsymmetric)."""
    h = 1.0 / (n + 1)
    cx = peclet * h / 2
    r, c, v = [], [], []
    for i in range(n):
        for j in range(n):
            k = i * n + j
            r.append(k); c.append(k); v.append(4.0)
            for di, dj, w in ((-1, 0, -1 - cx), (1, 0, -1 + cx), (0, -1, -1.0), (0, 1, -1.0)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    r.append(k); c.append(ii * n + jj); v.append(w)
    return CSRMatrix.from_triplets(r, c, v, (n * n, n * n))


def random_spd(n: int, density: float = 0.05, seed: int = 0) -> CSRMatrix:
    """Random sparse SPD matrix, strictly diagonally dominant."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, n)) * (rng.random((n, n)) < density)
    s = np.triu(m, 1)
    s = s + s.T
    s[np.diag_indices(n)] = np.abs(s).sum(axis=1) + 1.0
    return CSRMatrix.from_dense(s)
