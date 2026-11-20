"""Standard black-box benchmark functions with known optima."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def sphere(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def ellipsoid(x: np.ndarray) -> float:
    """Axis-parallel ill-conditioned quadratic, condition number 1e6."""
    n = len(x)
    if n == 1:
        return float(x[0] ** 2)
    w = 1e6 ** (np.arange(n) / (n - 1))
    return float(np.sum(w * x * x))


def rotated_ellipsoid(x: np.ndarray) -> float:
    """Ill-conditioned quadratic in a fixed random rotation (non-separable)."""
    n = len(x)
    q = _rotation(n)
    return ellipsoid(q @ x)


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    return float(10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    n = len(x)
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.dot(x, x) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20.0 + np.e
    )


def cigar(x: np.ndarray) -> float:
    return float(x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2))


_ROT_CACHE: dict[int, np.ndarray] = {}


def _rotation(n: int) -> np.ndarray:
    if n not in _ROT_CACHE:
        rng = np.random.default_rng(12345 + n)
        q, r = np.linalg.qr(rng.standard_normal((n, n)))
        _ROT_CACHE[n] = q * np.sign(np.diag(r))
    return _ROT_CACHE[n]


@dataclass(frozen=True)
class Problem:
    name: str
    func: Callable[[np.ndarray], float]
    optimum_x: Callable[[int], np.ndarray]
    optimum_f: float = 0.0
    multimodal: bool = False


BENCHMARKS: dict[str, Problem] = {
    "sphere": Problem("sphere", sphere, np.zeros),
    "ellipsoid": Problem("ellipsoid", ellipsoid, np.zeros),
    "rotated_ellipsoid": Problem("rotated_ellipsoid", rotated_ellipsoid, np.zeros),
    "cigar": Problem("cigar", cigar, np.zeros),
    "rosenbrock": Problem("rosenbrock", rosenbrock, np.ones),
    "rastrigin": Problem("rastrigin", rastrigin, np.zeros, multimodal=True),
    "ackley": Problem("ackley", ackley, np.zeros, multimodal=True),
}


def get_problem(name: str) -> Problem:
    """Look up a benchmark by name."""
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise KeyError(f"unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}") from None
