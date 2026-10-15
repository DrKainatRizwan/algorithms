"""Behavioural drift detection: running statistics, change-point detectors
(Page-Hinkley, CUSUM) and distribution comparison (PSI, two-sample KS)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class RunningStats:
    """Welford's online mean/variance."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def var(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def zscore(self, x: float, floor: float = 1e-6) -> float:
        return (x - self.mean) / max(self.std, floor)


class PageHinkley:
    """Page-Hinkley test for an upward shift in the mean.

    ``delta`` is the tolerated magnitude of change; an alarm is raised when the
    cumulative deviation exceeds the running minimum by ``threshold``.
    """

    def __init__(self, delta: float = 0.5, threshold: float = 15.0, min_samples: int = 10):
        self.delta, self.threshold, self.min_samples = delta, threshold, min_samples
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._sum = 0.0
        self._min = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self._mean += (x - self._mean) / self.n
        self._sum += x - self._mean - self.delta
        self._min = min(self._min, self._sum)
        if self.n >= self.min_samples and self._sum - self._min > self.threshold:
            self.reset()
            return True
        return False

    @property
    def statistic(self) -> float:
        return self._sum - self._min


class CUSUM:
    """Two-sided tabular CUSUM on standardised observations."""

    def __init__(self, k: float = 0.5, h: float = 5.0):
        self.k, self.h = k, h
        self.reset()

    def reset(self) -> None:
        self.pos = 0.0
        self.neg = 0.0

    def update(self, z: float) -> int:
        """Return +1 / -1 on an upward / downward alarm, else 0."""
        self.pos = max(0.0, self.pos + z - self.k)
        self.neg = max(0.0, self.neg - z - self.k)
        if self.pos > self.h:
            self.reset()
            return 1
        if self.neg > self.h:
            self.reset()
            return -1
        return 0


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with quantile bins from ``expected``.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.size == 0 or a.size == 0:
        raise ValueError("both samples must be non-empty")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        edges = np.array([e.min() - 0.5, e.max() + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / e.size
    pa = np.histogram(a, edges)[0] / a.size
    pe, pa = np.maximum(pe, eps), np.maximum(pa, eps)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def ks_statistic(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.sort(np.asarray(y, dtype=float))
    n, m = xs.size, ys.size
    if n == 0 or m == 0:
        raise ValueError("both samples must be non-empty")
    grid = np.concatenate([xs, ys])
    cdf_x = np.searchsorted(xs, grid, side="right") / n
    cdf_y = np.searchsorted(ys, grid, side="right") / m
    d = float(np.max(np.abs(cdf_x - cdf_y)))
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-9:
        return d, 1.0
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))
