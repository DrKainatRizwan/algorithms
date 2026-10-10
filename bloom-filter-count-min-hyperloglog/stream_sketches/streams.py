"""Synthetic stream generators and exact baselines for evaluating sketches."""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable

import numpy as np


def zipf_stream(n: int, universe: int, s: float = 1.1, seed: int = 0) -> list[str]:
    """n draws from a truncated Zipf(s) distribution over `universe` keys."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, universe + 1, dtype=np.float64)
    probs = ranks ** -s
    probs /= probs.sum()
    draws = rng.choice(universe, size=n, p=probs)
    return [f"item{d}" for d in draws]


def uniform_stream(n: int, universe: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [f"item{d}" for d in rng.integers(0, universe, size=n)]


def exact_counts(stream: Iterable[Hashable]) -> Counter:
    return Counter(stream)
