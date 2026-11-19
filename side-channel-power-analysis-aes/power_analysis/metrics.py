"""Attack evaluation metrics: key rank, success rate, guessing entropy."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .leakage import TraceSet


def key_rank(scores: np.ndarray, true_byte: int) -> int:
    """Rank (0 = best) of the correct key byte among 256 scores."""
    return int((scores > scores[true_byte]).sum())


def success_curve(make_traces: Callable[[int, int], TraceSet],
                  attack: Callable[[TraceSet, int], np.ndarray],
                  byte: int, trace_counts: list[int],
                  n_experiments: int = 20) -> dict[str, np.ndarray]:
    """Success rate (rank 0) and guessing entropy (mean rank) vs trace count.

    ``make_traces(n, seed)`` returns a fresh TraceSet of n traces.
    """
    nmax = max(trace_counts)
    sr = np.zeros(len(trace_counts))
    ge = np.zeros(len(trace_counts))
    for e in range(n_experiments):
        full = make_traces(nmax, e)
        true = full.key[byte]
        for i, n in enumerate(trace_counts):
            r = key_rank(attack(full.subset(n), byte), true)
            sr[i] += r == 0
            ge[i] += r
    return {"traces": np.array(trace_counts), "success_rate": sr / n_experiments,
            "guessing_entropy": ge / n_experiments}


def traces_to_success(curve: dict[str, np.ndarray], threshold: float = 0.9) -> int | None:
    """Smallest trace count whose success rate meets the threshold."""
    for n, s in zip(curve["traces"], curve["success_rate"]):
        if s >= threshold:
            return int(n)
    return None
