"""Multi-seed comparison harness and a rank-sum significance test."""
from __future__ import annotations

from math import erfc, sqrt
from typing import Callable

import numpy as np

from .benchmarks import get_benchmark
from .de import DifferentialEvolution
from .pso import PSO


def make_optimizer(kind: str, func, lower, upper, seed: int, **kw):
    """Instantiate ``"pso"`` or ``"de"`` with a per-run seed."""
    if kind == "pso":
        return PSO(func, lower, upper, seed=seed, **kw)
    if kind == "de":
        return DifferentialEvolution(func, lower, upper, seed=seed, **kw)
    raise ValueError(f"unknown optimizer kind {kind!r}")


def compare_optimizers(configs: dict[str, tuple[str, dict]], benchmark: str, dim: int,
                       seeds: list[int], max_nfev: int = 20000) -> dict[str, np.ndarray]:
    """Run each (kind, kwargs) config on ``benchmark`` over ``seeds``; return final errors f - f*.

    Every optimiser gets the same evaluation budget ``max_nfev`` (iterations are set high enough
    that the budget is the binding stop criterion).
    """
    b = get_benchmark(benchmark)
    lo, hi = b.bounds(dim)
    out: dict[str, np.ndarray] = {}
    for name, (kind, kw) in configs.items():
        pop_key = "n_particles" if kind == "pso" else "pop_size"
        pop = kw.get(pop_key, 30 if kind == "pso" else 40)
        kw = {**kw, "max_iter": max_nfev // pop, "max_nfev": max_nfev}
        errs = []
        for s in seeds:
            res = make_optimizer(kind, b.func, lo, hi, s, **kw).run()
            errs.append(max(res.fun - b.f_opt, 0.0))
        out[name] = np.array(errs)
    return out


def summarize(errors: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Mean / std / median / best of error arrays."""
    return {k: {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
                "best": float(v.min())} for k, v in errors.items()}


def wilcoxon_rank_sum(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon rank-sum (Mann-Whitney) test, normal approximation with tie correction.

    Returns ``(z, p)``; negative ``z`` means ``a`` tends to be smaller than ``b``.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = a.size, b.size
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size)
    sorted_v = allv[order]
    i = 0
    tie_term = 0.0
    while i < allv.size:
        j = i
        while j + 1 < allv.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        t = j - i + 1
        tie_term += t**3 - t
        i = j + 1
    n = n1 + n2
    mu = n1 * (n + 1) / 2
    var = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return 0.0, 1.0
    z = (ranks[:n1].sum() - mu) / sqrt(var)
    return float(z), float(erfc(abs(z) / sqrt(2)))
