"""IPOP restart strategy (Auger & Hansen, 2005) for multimodal landscapes."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .cmaes import cmaes_minimize
from .result import Result


def ipop_cmaes(
    func: Callable[[np.ndarray], float],
    dim: int,
    bounds: tuple[float, float],
    sigma0: float,
    max_evals: int = 200000,
    ftarget: float | None = None,
    inc_factor: float = 2.0,
    max_restarts: int = 9,
    seed: int = 0,
) -> Result:
    """Run CMA-ES with restarts from random points, growing the population each time."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    lam = float(4 + int(3 * np.log(dim)))
    best: Result | None = None
    total, restarts = 0, 0
    history: list[float] = []
    while total < max_evals and restarts <= max_restarts:
        x0 = rng.uniform(lo, hi, dim)
        res = cmaes_minimize(func, x0, sigma0, popsize=int(lam), max_evals=max_evals - total,
                             ftarget=ftarget, seed=int(rng.integers(1 << 31)))
        total += res.evals
        if best is None or res.fun < best.fun:
            best = res
        history.append(best.fun)
        if ftarget is not None and best.fun <= ftarget:
            break
        lam *= inc_factor
        restarts += 1
    assert best is not None
    hit = ftarget is not None and best.fun <= ftarget
    return Result(best.x, best.fun, total, restarts + 1,
                  "target reached" if hit else "budget exhausted", history)
