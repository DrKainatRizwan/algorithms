"""Side-by-side comparison of the solvers."""

from __future__ import annotations

import time

import numpy as np

from .annealing import simulated_annealing
from .genetic import genetic_algorithm
from .instances import Instance, tour_length
from .local_search import nearest_neighbour, two_opt
from .tabu import tabu_search


def compare(inst: Instance, seeds: tuple[int, ...] = (0, 1, 2), sa_iters: int = 30_000,
            ga_gens: int = 100, tabu_iters: int = 150) -> dict[str, dict[str, float]]:
    """Run every method over several seeds; report mean length and mean wall time."""
    nn = nearest_neighbour(inst)
    methods = {
        "nearest-neighbour": lambda s: nn,
        "nn+2opt": lambda s: two_opt(inst, nn),
        "annealing": lambda s: simulated_annealing(inst, sa_iters, seed=s).tour,
        "genetic": lambda s: genetic_algorithm(inst, 40, ga_gens, seed=s).tour,
        "tabu": lambda s: tabu_search(inst, tabu_iters, seed=s, candidates=300).tour,
    }
    out = {}
    for name, fn in methods.items():
        lens, t0 = [], time.perf_counter()
        for s in seeds:
            lens.append(tour_length(inst, fn(s)))
        out[name] = {"mean": float(np.mean(lens)), "best": float(np.min(lens)),
                     "seconds": (time.perf_counter() - t0) / len(seeds)}
    return out
