"""Mercer kernels with analytic gradients with respect to the first argument."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


def linear_kernel(A: Array, B: Array) -> Array:
    """Return the Gram matrix ``A @ B.T``."""
    return A @ B.T


def polynomial_kernel(A: Array, B: Array, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> Array:
    """Return ``(gamma * <a, b> + coef0) ** degree``."""
    return (gamma * (A @ B.T) + coef0) ** degree


def rbf_kernel(A: Array, B: Array, gamma: float = 1.0) -> Array:
    """Return ``exp(-gamma * ||a - b||^2)`` computed without a Python loop."""
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.exp(-gamma * np.maximum(sq, 0.0))


@dataclass(frozen=True)
class Kernel:
    """A named kernel together with its hyper-parameters.

    ``__call__`` evaluates a Gram matrix; :meth:`grad` returns the gradient of
    ``k(x, Z[i])`` with respect to ``x`` for every row of ``Z``.
    """

    name: str = "rbf"
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 1.0

    def __call__(self, A: Array, B: Array) -> Array:
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        if self.name == "linear":
            return linear_kernel(A, B)
        if self.name == "poly":
            return polynomial_kernel(A, B, self.degree, self.gamma, self.coef0)
        if self.name == "rbf":
            return rbf_kernel(A, B, self.gamma)
        raise ValueError(f"unknown kernel {self.name!r}")

    def grad(self, x: Array, Z: Array) -> Array:
        """Gradient of ``k(x, z_i)`` wrt ``x``; shape ``(len(Z), d)``."""
        x = np.asarray(x, dtype=float)
        if self.name == "linear":
            return Z.copy()
        if self.name == "poly":
            inner = self.gamma * (Z @ x) + self.coef0
            return (self.degree * self.gamma * inner ** (self.degree - 1))[:, None] * Z
        if self.name == "rbf":
            k = rbf_kernel(x[None, :], Z, self.gamma)[0]
            return (-2.0 * self.gamma * k)[:, None] * (x[None, :] - Z)
        raise ValueError(f"unknown kernel {self.name!r}")


def make_kernel(name: str | Kernel | Callable, **params) -> Kernel:
    """Build a :class:`Kernel` from a name (``linear``, ``poly``, ``rbf``)."""
    if isinstance(name, Kernel):
        return name
    if not isinstance(name, str):
        raise TypeError("kernel must be a name or Kernel instance")
    return Kernel(name=name, **params)
