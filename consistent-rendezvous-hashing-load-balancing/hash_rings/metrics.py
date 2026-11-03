"""Load-balance and key-movement metrics shared by all schemes."""
from __future__ import annotations

import statistics
from typing import Callable, Iterable, Sequence


def load_counts(assign: Callable[[object], str], keys: Iterable[object], nodes: Sequence[str]) -> dict[str, int]:
    """Number of keys mapped to each node (zeros included)."""
    counts = {n: 0 for n in nodes}
    for k in keys:
        counts[assign(k)] += 1
    return counts


def peak_to_mean(counts: dict[str, int]) -> float:
    """Max load divided by mean load (1.0 is perfect)."""
    vals = list(counts.values())
    mean = sum(vals) / len(vals)
    return max(vals) / mean if mean else 0.0


def coeff_variation(counts: dict[str, int]) -> float:
    """Standard deviation over mean of node loads."""
    vals = list(counts.values())
    mean = sum(vals) / len(vals)
    return statistics.pstdev(vals) / mean if mean else 0.0


def moved_fraction(before: dict[object, str], after: dict[object, str]) -> float:
    """Fraction of keys whose assignment changed."""
    if not before:
        return 0.0
    return sum(1 for k, v in before.items() if after[k] != v) / len(before)


def snapshot(assign: Callable[[object], str], keys: Iterable[object]) -> dict[object, str]:
    return {k: assign(k) for k in keys}


def weighted_error(counts: dict[str, int], weights: dict[str, float]) -> float:
    """Max relative deviation of observed load share from the ideal weighted share."""
    total = sum(counts.values())
    wsum = sum(weights.values())
    return max(abs(counts[n] / total - w / wsum) / (w / wsum) for n, w in weights.items())
