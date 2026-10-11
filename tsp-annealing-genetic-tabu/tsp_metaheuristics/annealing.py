"""Simulated annealing with 2-opt moves and geometric cooling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .instances import Instance, tour_length
from .local_search import two_opt_delta, nearest_neighbour


@dataclass
class Result:
    """Outcome of a metaheuristic run."""

    tour: np.ndarray
    length: float
    history: list[float] = field(default_factory=list)  # best-so-far, sampled
    evaluations: int = 0


def initial_temperature(inst: Instance, rng: np.random.Generator, acceptance: float = 0.8,
                        samples: int = 200) -> float:
    """Temperature at which an average uphill 2-opt move is accepted with given probability."""
    n = inst.n
    t = rng.permutation(n)
    ups = []
    for _ in range(samples):
        i, j = sorted(rng.choice(np.arange(1, n), size=2, replace=False))
        dl = two_opt_delta(inst, t, int(i), int(j))
        if dl > 0:
            ups.append(dl)
    mean = float(np.mean(ups)) if ups else 1.0
    return -mean / math.log(acceptance)


def simulated_annealing(inst: Instance, iterations: int = 50_000, seed: int = 0,
                        t0: float | None = None, t_end_ratio: float = 1e-3,
                        start: np.ndarray | None = None) -> Result:
    """Anneal a tour using random segment reversals.

    Temperature decays geometrically from ``t0`` to ``t0 * t_end_ratio`` over the run.
    """
    rng = np.random.default_rng(seed)
    n = inst.n
    cur = nearest_neighbour(inst) if start is None else np.array(start)
    cur_len = tour_length(inst, cur)
    best, best_len = cur.copy(), cur_len
    if n < 4:
        return Result(best, best_len, [best_len], 0)
    T = initial_temperature(inst, rng) * 0.2 if t0 is None else t0
    alpha = t_end_ratio ** (1.0 / max(iterations, 1))
    history: list[float] = []
    step = max(iterations // 50, 1)
    for it in range(iterations):
        i = int(rng.integers(1, n - 1))
        j = int(rng.integers(i + 1, n))
        dl = two_opt_delta(inst, cur, i, j)
        if dl <= 0 or rng.random() < math.exp(-dl / T):
            cur[i:j + 1] = cur[i:j + 1][::-1]
            cur_len += dl
            if cur_len < best_len - 1e-12:
                best_len, best = cur_len, cur.copy()
        T *= alpha
        if it % step == 0:
            history.append(best_len)
    best_len = tour_length(inst, best)
    history.append(best_len)
    return Result(best, best_len, history, iterations)
