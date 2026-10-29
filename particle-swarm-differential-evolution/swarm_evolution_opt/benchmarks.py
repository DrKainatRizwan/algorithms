"""Classic continuous benchmark functions, vectorised over rows of an (n, d) array."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def sphere(X: np.ndarray) -> np.ndarray:
    return np.sum(X**2, axis=1)


def rastrigin(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    return 10.0 * d + np.sum(X**2 - 10.0 * np.cos(2 * np.pi * X), axis=1)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)


def ackley(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    s1 = np.sum(X**2, axis=1) / d
    s2 = np.sum(np.cos(2 * np.pi * X), axis=1) / d
    return -20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2) + 20.0 + np.e


def griewank(X: np.ndarray) -> np.ndarray:
    i = np.sqrt(np.arange(1, X.shape[1] + 1))
    return 1.0 + np.sum(X**2, axis=1) / 4000.0 - np.prod(np.cos(X / i), axis=1)


def schwefel(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    return 418.9828872724338 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


def styblinski_tang(X: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(X**4 - 16 * X**2 + 5 * X, axis=1) + 39.16616570377142 * X.shape[1]


def elliptic(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    w = 1e6 ** (np.arange(d) / max(d - 1, 1))
    return np.sum(w * X**2, axis=1)


@dataclass(frozen=True)
class Benchmark:
    """A test function with its box domain and known global minimiser."""

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    lower: float
    upper: float
    x_opt: Callable[[int], np.ndarray]
    f_opt: float = 0.0

    def bounds(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(dim, self.lower), np.full(dim, self.upper)


_REGISTRY: dict[str, Benchmark] = {
    b.name: b
    for b in [
        Benchmark("sphere", sphere, -5.12, 5.12, lambda d: np.zeros(d)),
        Benchmark("rastrigin", rastrigin, -5.12, 5.12, lambda d: np.zeros(d)),
        Benchmark("rosenbrock", rosenbrock, -2.048, 2.048, lambda d: np.ones(d)),
        Benchmark("ackley", ackley, -32.768, 32.768, lambda d: np.zeros(d)),
        Benchmark("griewank", griewank, -600.0, 600.0, lambda d: np.zeros(d)),
        Benchmark("schwefel", schwefel, -500.0, 500.0, lambda d: np.full(d, 420.9687462275036)),
        Benchmark("styblinski_tang", styblinski_tang, -5.0, 5.0, lambda d: np.full(d, -2.903534027771178)),
        Benchmark("elliptic", elliptic, -100.0, 100.0, lambda d: np.zeros(d)),
    ]
}


def list_benchmarks() -> list[str]:
    return sorted(_REGISTRY)


def get_benchmark(name: str) -> Benchmark:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown benchmark {name!r}; choose from {list_benchmarks()}") from None
