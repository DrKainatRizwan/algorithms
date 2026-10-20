"""Traffic demand models: time-varying Poisson arrival rates on boundary approaches.

Approach (travel-direction) indices used across the package:
0 = southbound, 1 = northbound, 2 = eastbound, 3 = westbound.
Directions 0/1 form the "NS" group and 2/3 the "EW" group.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

NS_DIRS = (0, 1)
EW_DIRS = (2, 3)


@dataclass(frozen=True)
class Demand:
    """Arrival intensity (vehicles per second) per travel direction as a function of time."""

    rate_fn: Callable[[int, int], float]
    name: str = "custom"

    def rate(self, t: int, direction: int) -> float:
        return max(0.0, float(self.rate_fn(t, direction)))

    def sample(self, t: int, rng: np.random.Generator) -> np.ndarray:
        """Poisson arrivals for the four directions during second ``t``."""
        lam = np.array([self.rate(t, d) for d in range(4)])
        return rng.poisson(lam)


def uniform_demand(rate: float = 0.12) -> Demand:
    """Same rate on every approach."""
    return Demand(lambda t, d: rate, name=f"uniform({rate})")


def asymmetric_demand(ns: float = 0.18, ew: float = 0.06) -> Demand:
    """Heavy north-south, light east-west traffic."""
    return Demand(lambda t, d: ns if d in NS_DIRS else ew, name=f"asymmetric({ns},{ew})")


def rush_hour_demand(period: int = 600, heavy: float = 0.19, light: float = 0.05) -> Demand:
    """Dominant direction group alternates every ``period`` seconds."""

    def fn(t: int, d: int) -> float:
        ns_phase = (t // period) % 2 == 0
        is_ns = d in NS_DIRS
        return heavy if is_ns == ns_phase else light

    return Demand(fn, name=f"rush_hour({period})")
