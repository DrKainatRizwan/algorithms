"""Oblivious subspace embeddings: Gaussian, SRHT, CountSketch and sparse sign sketches."""
from __future__ import annotations

import math
import numpy as np


def fwht(x: np.ndarray) -> np.ndarray:
    """Unnormalised fast Walsh-Hadamard transform along axis 0 (length must be a power of two).

    Runs in O(n log n) per column.
    """
    a = np.array(x, dtype=float, copy=True)
    n = a.shape[0]
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    h = 1
    while h < n:
        a = a.reshape(n // (2 * h), 2, h, *a.shape[1:])
        top = a[:, 0] + a[:, 1]
        bot = a[:, 0] - a[:, 1]
        a = np.stack([top, bot], axis=1).reshape(n, *x.shape[1:])
        h *= 2
    return a


def jl_min_dim(n_points: int, eps: float) -> int:
    """Johnson-Lindenstrauss target dimension guaranteeing (1±eps) distortion of pairwise distances."""
    if not 0 < eps < 1:
        raise ValueError("eps must be in (0,1)")
    return int(math.ceil(4 * math.log(n_points) / (eps ** 2 / 2 - eps ** 3 / 3)))


class GaussianSketch:
    """Dense i.i.d. N(0, 1/m) sketch of shape (m, n). Cost O(m n d) to apply to n x d input."""

    def __init__(self, m: int, n: int, rng: np.random.Generator):
        self.m, self.n = m, n
        self.S = rng.standard_normal((m, n)) / math.sqrt(m)

    def apply(self, A: np.ndarray) -> np.ndarray:
        return self.S @ A


class SRHTSketch:
    """Subsampled randomised Hadamard transform: sqrt(1/m) * P H D, zero-padded to a power of two.

    Cost O(n d log n) to apply to an n x d input.
    """

    def __init__(self, m: int, n: int, rng: np.random.Generator):
        self.m, self.n = m, n
        self.N = 1 << max(1, (n - 1).bit_length())
        if m > self.N:
            raise ValueError("sketch size exceeds padded dimension")
        self.signs = rng.choice([-1.0, 1.0], size=n)
        self.rows = rng.choice(self.N, size=m, replace=False)

    def apply(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        shape = (self.N,) + A.shape[1:]
        padded = np.zeros(shape)
        padded[: self.n] = (self.signs.reshape((-1,) + (1,) * (A.ndim - 1))) * A
        return fwht(padded)[self.rows] / math.sqrt(self.m)


class CountSketch:
    """CountSketch (Clarkson-Woodruff): each input row hashed to one output row with a random sign.

    Cost O(nnz(A)) to apply.
    """

    def __init__(self, m: int, n: int, rng: np.random.Generator):
        self.m, self.n = m, n
        self.h = rng.integers(0, m, size=n)
        self.s = rng.choice([-1.0, 1.0], size=n)

    def apply(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        out = np.zeros((self.m,) + A.shape[1:])
        np.add.at(out, self.h, self.s.reshape((-1,) + (1,) * (A.ndim - 1)) * A)
        return out


class SparseSignSketch:
    """Sparse sign embedding with ``zeta`` nonzeros (±1/sqrt(zeta)) per input coordinate."""

    def __init__(self, m: int, n: int, rng: np.random.Generator, zeta: int = 4):
        zeta = min(zeta, m)
        self.m, self.n, self.zeta = m, n, zeta
        self.idx = np.stack([rng.choice(m, size=zeta, replace=False) for _ in range(n)])
        self.sgn = rng.choice([-1.0, 1.0], size=(n, zeta))

    def apply(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        out = np.zeros((self.m,) + A.shape[1:])
        for j in range(self.zeta):
            sg = self.sgn[:, j].reshape((-1,) + (1,) * (A.ndim - 1))
            np.add.at(out, self.idx[:, j], sg * A)
        return out / math.sqrt(self.zeta)


_KINDS = {"gaussian": GaussianSketch, "srht": SRHTSketch, "countsketch": CountSketch,
          "sparse_sign": SparseSignSketch}


def make_sketch(kind: str, m: int, n: int, rng: np.random.Generator):
    """Factory returning a sketch object with an ``apply`` method."""
    try:
        return _KINDS[kind](m, n, rng)
    except KeyError:
        raise ValueError(f"unknown sketch kind {kind!r}; choose from {sorted(_KINDS)}") from None
