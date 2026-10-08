"""Differentially private statistical queries built on the core mechanisms."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .mechanisms import exponential_mechanism, laplace_mechanism


def private_count(data: Sequence, predicate, epsilon: float,
                  rng: np.random.Generator) -> float:
    """Counting query (sensitivity 1) with Laplace noise."""
    c = sum(1 for x in data if predicate(x))
    return laplace_mechanism(c, 1.0, epsilon, rng)


def private_sum(data: Sequence[float], lower: float, upper: float, epsilon: float,
                rng: np.random.Generator) -> float:
    """Sum of values clipped to [lower, upper]; sensitivity max(|lower|,|upper|)."""
    x = np.clip(np.asarray(data, dtype=float), lower, upper)
    return laplace_mechanism(x.sum(), max(abs(lower), abs(upper)), epsilon, rng)


def private_mean(data: Sequence[float], lower: float, upper: float, epsilon: float,
                 rng: np.random.Generator) -> float:
    """Mean via noisy sum / noisy count, splitting the budget evenly."""
    x = np.clip(np.asarray(data, dtype=float), lower, upper)
    mid = (lower + upper) / 2.0           # centre to halve sensitivity
    s = laplace_mechanism((x - mid).sum(), (upper - lower) / 2.0, epsilon / 2, rng)
    n = max(laplace_mechanism(float(len(x)), 1.0, epsilon / 2, rng), 1.0)
    return float(np.clip(s / n + mid, lower, upper))


def private_histogram(data: Sequence[float], bins: np.ndarray, epsilon: float,
                      rng: np.random.Generator, postprocess: bool = True) -> np.ndarray:
    """Histogram release; add/remove neighbours change one bin, so L1 sensitivity is 1."""
    h, _ = np.histogram(np.asarray(data, dtype=float), bins=bins)
    noisy = laplace_mechanism(h.astype(float), 1.0, epsilon, rng)
    return np.maximum(noisy, 0.0) if postprocess else noisy


def private_quantile(data: Sequence[float], q: float, lower: float, upper: float,
                     epsilon: float, rng: np.random.Generator) -> float:
    """Quantile through the exponential mechanism over the gaps between order statistics."""
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0,1]")
    x = np.sort(np.clip(np.asarray(data, dtype=float), lower, upper))
    n = len(x)
    edges = np.concatenate(([lower], x, [upper]))
    # utility of the interval i (rank i) is -|i - q n|; sensitivity 1
    scores = -np.abs(np.arange(n + 1) - q * n)
    widths = np.maximum(edges[1:] - edges[:-1], 1e-300)
    # weight by width => add log(width) into the scores
    logits = epsilon * scores / 2.0 + np.log(widths)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    i = int(rng.choice(n + 1, p=p))
    return float(rng.uniform(edges[i], edges[i + 1]))
