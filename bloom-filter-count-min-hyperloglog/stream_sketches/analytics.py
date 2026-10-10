"""Evaluation helpers: empirical error of each sketch against exact answers."""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

from .bloom import BloomFilter
from .countmin import CountMinSketch
from .hyperloglog import HyperLogLog


def empirical_fp_rate(bf: BloomFilter, negatives: Sequence[Hashable]) -> float:
    """Fraction of known-absent keys reported present."""
    return sum(1 for x in negatives if x in bf) / len(negatives)


def cms_errors(cms: CountMinSketch, truth: dict) -> dict[str, float]:
    """Absolute-error summary of a Count-Min sketch over all distinct keys."""
    errs = np.array([cms.estimate(k) - c for k, c in truth.items()], dtype=np.float64)
    return {
        "min_error": float(errs.min()),
        "mean_error": float(errs.mean()),
        "max_error": float(errs.max()),
        "bound": cms.error_bound(),
        "frac_within_bound": float(np.mean(errs <= cms.error_bound())),
    }


def hll_relative_error(p: int, n: int, trials: int = 5, seed: int = 0) -> float:
    """Mean |estimate - n| / n over independent HLL sketches of n distinct keys."""
    out = []
    for t in range(trials):
        h = HyperLogLog(p, seed=seed + t)
        h.update(range(t * n, (t + 1) * n))
        out.append(abs(h.estimate() - n) / n)
    return float(np.mean(out))


def top_k_recall(found: Sequence[Hashable], truth: dict, k: int) -> float:
    """Recall of the true top-k keys among `found`."""
    true_top = {x for x, _ in sorted(truth.items(), key=lambda kv: (-kv[1], kv[0]))[:k]}
    return len(true_top & set(found)) / k
