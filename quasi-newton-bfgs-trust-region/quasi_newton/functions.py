"""Test problems with analytic gradients and Hessians, plus counted evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

Vec = np.ndarray


@dataclass
class Problem:
    """Objective wrapper that counts function, gradient and Hessian calls."""

    name: str
    f: Callable[[Vec], float]
    grad: Callable[[Vec], Vec]
    x0: Vec
    hess: Optional[Callable[[Vec], np.ndarray]] = None
    x_star: Optional[Vec] = None
    n_f: int = field(default=0, init=False)
    n_g: int = field(default=0, init=False)
    n_h: int = field(default=0, init=False)

    def value(self, x: Vec) -> float:
        self.n_f += 1
        return float(self.f(x))

    def gradient(self, x: Vec) -> Vec:
        self.n_g += 1
        return np.asarray(self.grad(x), dtype=float)

    def hessian(self, x: Vec) -> np.ndarray:
        if self.hess is None:
            raise ValueError("problem has no Hessian")
        self.n_h += 1
        return np.asarray(self.hess(x), dtype=float)

    def reset(self) -> None:
        self.n_f = self.n_g = self.n_h = 0


def numerical_gradient(f: Callable[[Vec], float], x: Vec, h: float = 1e-6) -> Vec:
    """Central finite-difference gradient (used to validate analytic gradients)."""
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        e = np.zeros_like(g)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def rosenbrock(x0: Optional[Vec] = None) -> Problem:
    """2-D Rosenbrock banana function, minimum 0 at (1, 1)."""
    def f(x): return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    def g(x): return np.array([-400.0 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                               200.0 * (x[1] - x[0] ** 2)])
    def h(x): return np.array([[1200.0 * x[0] ** 2 - 400.0 * x[1] + 2, -400.0 * x[0]],
                               [-400.0 * x[0], 200.0]])
    return Problem("rosenbrock", f, g, np.array([-1.2, 1.0]) if x0 is None else x0, h, np.ones(2))


def extended_rosenbrock(n: int = 20) -> Problem:
    """n-dimensional chained Rosenbrock (n even), minimum 0 at all-ones."""
    if n % 2:
        raise ValueError("n must be even")
    def f(x):
        a, b = x[0::2], x[1::2]
        return float(np.sum(100.0 * (b - a ** 2) ** 2 + (1 - a) ** 2))
    def g(x):
        a, b = x[0::2], x[1::2]
        out = np.empty_like(x)
        out[0::2] = -400.0 * a * (b - a ** 2) - 2 * (1 - a)
        out[1::2] = 200.0 * (b - a ** 2)
        return out
    def h(x):
        H = np.zeros((n, n))
        for k in range(0, n, 2):
            H[k, k] = 1200.0 * x[k] ** 2 - 400.0 * x[k + 1] + 2
            H[k, k + 1] = H[k + 1, k] = -400.0 * x[k]
            H[k + 1, k + 1] = 200.0
        return H
    x0 = np.tile([-1.2, 1.0], n // 2)
    return Problem(f"ext-rosenbrock-{n}", f, g, x0, h, np.ones(n))


def quadratic(A: np.ndarray, b: np.ndarray, x0: Optional[Vec] = None) -> Problem:
    """f(x) = 0.5 x'Ax - b'x with SPD A; minimiser solves Ax = b."""
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    return Problem("quadratic", lambda x: 0.5 * x @ A @ x - b @ x, lambda x: A @ x - b,
                   np.zeros(len(b)) if x0 is None else x0, lambda x: A, np.linalg.solve(A, b))


def powell_singular(x0: Optional[Vec] = None) -> Problem:
    """Powell's singular function (Hessian singular at the solution 0)."""
    def r(x): return np.array([x[0] + 10 * x[1], np.sqrt(5) * (x[2] - x[3]),
                               (x[1] - 2 * x[2]) ** 2, np.sqrt(10) * (x[0] - x[3]) ** 2])
    def J(x):
        return np.array([[1, 10, 0, 0], [0, 0, np.sqrt(5), -np.sqrt(5)],
                         [0, 2 * (x[1] - 2 * x[2]), -4 * (x[1] - 2 * x[2]), 0],
                         [2 * np.sqrt(10) * (x[0] - x[3]), 0, 0, -2 * np.sqrt(10) * (x[0] - x[3])]])
    def f(x): return float(r(x) @ r(x))
    def g(x): return 2 * J(x).T @ r(x)
    def h(x):
        # Gauss-Newton plus second-order term of the two quadratic residuals.
        H = 2 * J(x).T @ J(x)
        d = x[1] - 2 * x[2]
        H += 2 * r(x)[2] * 2 * np.outer([0, 1, -2, 0], [0, 1, -2, 0])
        e = x[0] - x[3]
        H += 2 * r(x)[3] * 2 * np.sqrt(10) * np.outer([1, 0, 0, -1], [1, 0, 0, -1])
        return H
    return Problem("powell-singular", f, g, np.array([3.0, -1.0, 0.0, 1.0]) if x0 is None else x0,
                   h, np.zeros(4))


def beta_function(n: int = 10) -> Problem:
    """Smooth, ill-conditioned sum of exponentials: sum exp(x_i) - 2 x_i + 0.5 * x'Lx."""
    L = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    def f(x): return float(np.sum(np.exp(x) - 2 * x) + 0.5 * x @ L @ x)
    def g(x): return np.exp(x) - 2 + L @ x
    def h(x): return np.diag(np.exp(x)) + L
    return Problem(f"expchain-{n}", f, g, np.full(n, 2.0), h)


def logistic_regression(n: int = 200, d: int = 20, l2: float = 1e-2, seed: int = 0) -> Problem:
    """Regularised logistic loss on synthetic data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = np.where(X @ w + 0.5 * rng.standard_normal(n) > 0, 1.0, -1.0)
    def f(t):
        m = -y * (X @ t)
        return float(np.mean(np.logaddexp(0, m)) + 0.5 * l2 * t @ t)
    def g(t):
        s = 1 / (1 + np.exp(y * (X @ t)))
        return -(X.T @ (y * s)) / n + l2 * t
    def h(t):
        p = 1 / (1 + np.exp(-(X @ t)))
        return (X.T * (p * (1 - p))) @ X / n + l2 * np.eye(d)
    return Problem(f"logreg-{n}x{d}", f, g, np.zeros(d), h)
