"""Convex decision sets with Euclidean projections and linear minimisation oracles."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def project_simplex(y: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{x >= 0, sum x = radius}`` (sort-based, O(d log d))."""
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - radius
    k = np.arange(1, len(y) + 1)
    cond = u - css / k > 0
    rho = k[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(y - theta, 0.0)


class Domain(ABC):
    """A compact convex set ``K`` in R^d."""

    dim: int

    @abstractmethod
    def project(self, y: np.ndarray) -> np.ndarray:
        """Euclidean projection of ``y`` onto the set."""

    @abstractmethod
    def center(self) -> np.ndarray:
        """A canonical interior starting point."""

    @abstractmethod
    def diameter(self) -> float:
        """Euclidean diameter ``max ||x - y||``."""

    @abstractmethod
    def argmin_linear(self, g: np.ndarray) -> np.ndarray:
        """Minimiser of ``<g, x>`` over the set (the centre when ``g`` is ~0)."""

    def contains(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        return bool(np.linalg.norm(self.project(x) - x) <= tol)


class Ball(Domain):
    """Euclidean ball of radius ``R`` centred at the origin."""

    def __init__(self, dim: int, radius: float = 1.0) -> None:
        self.dim, self.radius = dim, float(radius)

    def project(self, y: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(y)
        return y if n <= self.radius else y * (self.radius / n)

    def project_norm(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Projection in the norm ``||v||_A = sqrt(v^T A v)`` (A positive definite).

        The KKT conditions give ``x = (A + mu I)^{-1} A y`` with ``mu >= 0`` chosen so
        that ``||x|| = R``; ``||x||`` is decreasing in ``mu`` so we bisect.
        """
        if np.linalg.norm(y) <= self.radius:
            return y
        lam, Q = np.linalg.eigh(A)
        z = Q.T @ y

        def norm_at(mu: float) -> float:
            return float(np.linalg.norm(lam * z / (lam + mu)))

        lo, hi = 0.0, 1.0
        while norm_at(hi) > self.radius:
            hi *= 2.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if norm_at(mid) > self.radius:
                lo = mid
            else:
                hi = mid
        x = Q @ (lam * z / (lam + hi))
        n = np.linalg.norm(x)
        return x * (self.radius / n) if n > self.radius else x

    def center(self) -> np.ndarray:
        return np.zeros(self.dim)

    def diameter(self) -> float:
        return 2.0 * self.radius

    def argmin_linear(self, g: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(g)
        return self.center() if n < 1e-12 else -self.radius * g / n


class Simplex(Domain):
    """Probability simplex ``{x >= 0, sum x = 1}``."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def project(self, y: np.ndarray) -> np.ndarray:
        return project_simplex(y)

    def center(self) -> np.ndarray:
        return np.full(self.dim, 1.0 / self.dim)

    def diameter(self) -> float:
        return float(np.sqrt(2.0))

    def argmin_linear(self, g: np.ndarray) -> np.ndarray:
        x = np.zeros(self.dim)
        m = g.min()
        ties = np.flatnonzero(g <= m + 1e-12)
        x[ties] = 1.0 / len(ties)
        return x


class Box(Domain):
    """Axis-aligned box ``[lo, hi]^d``."""

    def __init__(self, dim: int, lo: float = -1.0, hi: float = 1.0) -> None:
        self.dim, self.lo, self.hi = dim, float(lo), float(hi)

    def project(self, y: np.ndarray) -> np.ndarray:
        return np.clip(y, self.lo, self.hi)

    def center(self) -> np.ndarray:
        return np.full(self.dim, 0.5 * (self.lo + self.hi))

    def diameter(self) -> float:
        return float((self.hi - self.lo) * np.sqrt(self.dim))

    def argmin_linear(self, g: np.ndarray) -> np.ndarray:
        x = np.where(g > 1e-12, self.lo, np.where(g < -1e-12, self.hi, self.center()))
        return x.astype(float)


class L1Ball(Domain):
    """L1 ball of radius ``R``; projection reduces to a simplex projection."""

    def __init__(self, dim: int, radius: float = 1.0) -> None:
        self.dim, self.radius = dim, float(radius)

    def project(self, y: np.ndarray) -> np.ndarray:
        if np.abs(y).sum() <= self.radius:
            return y
        return np.sign(y) * project_simplex(np.abs(y), self.radius)

    def center(self) -> np.ndarray:
        return np.zeros(self.dim)

    def diameter(self) -> float:
        return 2.0 * self.radius

    def argmin_linear(self, g: np.ndarray) -> np.ndarray:
        x = np.zeros(self.dim)
        i = int(np.argmax(np.abs(g)))
        if abs(g[i]) > 1e-12:
            x[i] = -self.radius * np.sign(g[i])
        return x
