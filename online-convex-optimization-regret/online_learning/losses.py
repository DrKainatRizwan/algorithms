"""Per-round loss functions and exact aggregation of loss sequences."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Loss(ABC):
    """A convex loss revealed after the learner commits to a point."""

    strong_convexity: float = 0.0   # H such that f is H-strongly convex
    exp_concavity: float = 0.0      # alpha such that exp(-alpha f) is concave (0 = unknown)

    @abstractmethod
    def value(self, x: np.ndarray) -> float: ...

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray: ...


class LinearLoss(Loss):
    """``f(x) = <g, x>``."""

    def __init__(self, g: np.ndarray) -> None:
        self.g = np.asarray(g, dtype=float)

    def value(self, x: np.ndarray) -> float:
        return float(self.g @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.g


class SquaredLoss(Loss):
    """Ridge-regularised square loss ``0.5 (a.x - b)^2 + lam/2 ||x||^2``.

    ``exp_concavity`` is ``1 / B^2`` when the residual is bounded by ``B``.
    """

    def __init__(self, a: np.ndarray, b: float, lam: float = 0.0,
                 residual_bound: float = 0.0) -> None:
        self.a = np.asarray(a, dtype=float)
        self.b, self.lam = float(b), float(lam)
        self.strong_convexity = self.lam
        self.exp_concavity = 1.0 / residual_bound ** 2 if residual_bound > 0 else 0.0

    def value(self, x: np.ndarray) -> float:
        r = self.a @ x - self.b
        return 0.5 * r * r + 0.5 * self.lam * float(x @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return (self.a @ x - self.b) * self.a + self.lam * x


class LogisticLoss(Loss):
    """``log(1 + exp(-y a.x))`` with label ``y in {-1, +1}``."""

    def __init__(self, a: np.ndarray, y: float) -> None:
        self.a, self.y = np.asarray(a, dtype=float), float(y)

    def value(self, x: np.ndarray) -> float:
        return float(np.logaddexp(0.0, -self.y * (self.a @ x)))

    def grad(self, x: np.ndarray) -> np.ndarray:
        z = -self.y * (self.a @ x)
        s = 0.5 * (1.0 + np.tanh(0.5 * z))   # stable sigmoid(z)
        return -self.y * s * self.a


class QuadraticSum(Loss):
    """``0.5 x^T H x - c.x + k``: closed-form sum of many squared losses."""

    def __init__(self, H: np.ndarray, c: np.ndarray, k: float, strong: float) -> None:
        self.H, self.c, self.k = H, c, k
        self.strong_convexity = strong

    def value(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.H @ x - self.c @ x + self.k)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.H @ x - self.c

    def lipschitz_gradient(self) -> float:
        return float(np.linalg.eigvalsh(self.H)[-1])


class LogisticSum(Loss):
    """Sum of logistic losses stored as a matrix, for fast batch evaluation."""

    def __init__(self, A: np.ndarray, y: np.ndarray) -> None:
        self.A, self.y = A, y

    def value(self, x: np.ndarray) -> float:
        return float(np.logaddexp(0.0, -self.y * (self.A @ x)).sum())

    def grad(self, x: np.ndarray) -> np.ndarray:
        z = -self.y * (self.A @ x)
        s = 0.5 * (1.0 + np.tanh(0.5 * z))
        return -(self.A.T @ (self.y * s))

    def lipschitz_gradient(self) -> float:
        return float(0.25 * np.linalg.norm(self.A, 2) ** 2)


class GenericSum(Loss):
    """Fallback sum of arbitrary losses."""

    def __init__(self, losses: Sequence[Loss]) -> None:
        self.losses = list(losses)

    def value(self, x: np.ndarray) -> float:
        return float(sum(l.value(x) for l in self.losses))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return sum(l.grad(x) for l in self.losses)


def aggregate(losses: Sequence[Loss]) -> Loss:
    """Return a single loss equal to ``sum(losses)``, vectorised when possible."""
    if all(isinstance(l, LinearLoss) for l in losses):
        return LinearLoss(np.sum([l.g for l in losses], axis=0))
    if all(isinstance(l, SquaredLoss) for l in losses):
        A = np.array([l.a for l in losses])
        b = np.array([l.b for l in losses])
        lam = sum(l.lam for l in losses)
        H = A.T @ A + lam * np.eye(A.shape[1])
        return QuadraticSum(H, A.T @ b, 0.5 * float(b @ b), lam)
    if all(isinstance(l, LogisticLoss) for l in losses):
        return LogisticSum(np.array([l.a for l in losses]),
                           np.array([l.y for l in losses]))
    return GenericSum(losses)
