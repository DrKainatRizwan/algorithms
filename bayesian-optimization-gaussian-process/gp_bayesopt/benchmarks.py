"""Standard global-optimisation test functions with known optima."""

from __future__ import annotations

import numpy as np


def forrester(x: np.ndarray) -> float:
    """1-D Forrester function on [0, 1]; global minimum -6.0207 at x~0.7572."""
    x = float(np.ravel(x)[0])
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def branin(x: np.ndarray) -> float:
    """Branin-Hoo on [-5, 10] x [0, 15]; global minimum 0.397887."""
    x1, x2 = float(x[0]), float(x[1])
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


_H3_A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]], float)
_H3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470],
                         [1091, 8732, 5547], [381, 5743, 8828]], float)
_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann3(x: np.ndarray) -> float:
    """Hartmann-3 on [0, 1]^3; global minimum -3.86278."""
    x = np.asarray(x, float)
    return float(-np.sum(_H3_ALPHA * np.exp(-np.sum(_H3_A * (x - _H3_P) ** 2, axis=1))))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock valley; minimum 0 at (1, ..., 1)."""
    x = np.asarray(x, float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def sphere(x: np.ndarray) -> float:
    """Sum of squares; minimum 0 at the origin."""
    return float(np.sum(np.asarray(x, float) ** 2))


# name -> (function, bounds (lo, hi), known optimum value)
TEST_FUNCTIONS = {
    "forrester": (forrester, (np.array([0.0]), np.array([1.0])), -6.02074),
    "branin": (branin, (np.array([-5.0, 0.0]), np.array([10.0, 15.0])), 0.397887),
    "hartmann3": (hartmann3, (np.zeros(3), np.ones(3)), -3.86278),
    "rosenbrock2": (rosenbrock, (np.full(2, -2.0), np.full(2, 2.0)), 0.0),
    "sphere3": (sphere, (np.full(3, -5.0), np.full(3, 5.0)), 0.0),
}
