"""Discrete-emission hidden Markov model container and sampler."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HMM:
    """A discrete HMM with N hidden states and M observation symbols.

    Attributes:
        pi: initial state distribution, shape (N,).
        A: transition matrix, ``A[i, j] = P(z_{t+1}=j | z_t=i)``, shape (N, N).
        B: emission matrix, ``B[i, k] = P(x_t=k | z_t=i)``, shape (N, M).
    """

    pi: np.ndarray
    A: np.ndarray
    B: np.ndarray

    def __post_init__(self) -> None:
        self.pi = np.asarray(self.pi, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.validate()

    @property
    def n_states(self) -> int:
        return self.pi.shape[0]

    @property
    def n_symbols(self) -> int:
        return self.B.shape[1]

    def validate(self, tol: float = 1e-8) -> None:
        """Raise ``ValueError`` if shapes or stochasticity constraints fail."""
        n = self.pi.shape[0]
        if self.pi.ndim != 1 or self.A.shape != (n, n) or self.B.ndim != 2 \
                or self.B.shape[0] != n:
            raise ValueError("inconsistent HMM parameter shapes")
        for name, arr in (("pi", self.pi), ("A", self.A), ("B", self.B)):
            if np.any(arr < -tol):
                raise ValueError(f"{name} has negative entries")
        if abs(self.pi.sum() - 1) > tol:
            raise ValueError("pi must sum to one")
        if not np.allclose(self.A.sum(axis=1), 1, atol=tol):
            raise ValueError("rows of A must sum to one")
        if not np.allclose(self.B.sum(axis=1), 1, atol=tol):
            raise ValueError("rows of B must sum to one")

    def check_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return ``obs`` as an int array, verifying it is a valid sequence."""
        x = np.asarray(obs, dtype=int)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("observation sequence must be 1-D and non-empty")
        if x.min() < 0 or x.max() >= self.n_symbols:
            raise ValueError("observation symbol out of range")
        return x

    def sample(self, length: int, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``(states, observations)`` of the given length."""
        if length < 1:
            raise ValueError("length must be positive")
        z = np.empty(length, dtype=int)
        x = np.empty(length, dtype=int)
        z[0] = rng.choice(self.n_states, p=self.pi)
        for t in range(length):
            if t > 0:
                z[t] = rng.choice(self.n_states, p=self.A[z[t - 1]])
            x[t] = rng.choice(self.n_symbols, p=self.B[z[t]])
        return z, x

    def stationary(self) -> np.ndarray:
        """Stationary distribution of the chain (left Perron eigenvector)."""
        w, v = np.linalg.eig(self.A.T)
        k = int(np.argmin(np.abs(w - 1)))
        s = np.real(v[:, k])
        return s / s.sum()

    def copy(self) -> "HMM":
        return HMM(self.pi.copy(), self.A.copy(), self.B.copy())


def random_hmm(n_states: int, n_symbols: int, rng: np.random.Generator,
               concentration: float = 1.0) -> HMM:
    """Random HMM with Dirichlet(concentration) rows."""
    d = lambda rows, cols: rng.dirichlet(np.full(cols, concentration), size=rows)
    return HMM(d(1, n_states)[0], d(n_states, n_states), d(n_states, n_symbols))
