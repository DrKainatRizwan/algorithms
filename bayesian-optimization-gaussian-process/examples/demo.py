"""Compare Bayesian optimisation against random search on standard benchmarks."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gp_bayesopt import BayesianOptimizer, TEST_FUNCTIONS, random_search  # noqa: E402

BUDGET = 30
SEEDS = [0, 1]

t0 = time.time()
print(f"{'function':<12}{'optimum':>10}{'BO-EI':>10}{'BO-UCB':>10}{'random':>10}   (mean best of {len(SEEDS)} seeds, {BUDGET} evals)")
for name in ["forrester", "branin", "hartmann3"]:
    f, b, opt = TEST_FUNCTIONS[name]
    res = {"ei": [], "ucb": [], "rs": []}
    for s in SEEDS:
        for acq in ("ei", "ucb"):
            bo = BayesianOptimizer(b, acquisition=acq, n_init=5, seed=s, optimize_every=4)
            res[acq].append(bo.minimize(f, n_iter=BUDGET - 5)[1])
        res["rs"].append(random_search(f, b, BUDGET, seed=s)[-1])
    print(f"{name:<12}{opt:>10.4f}{np.mean(res['ei']):>10.4f}{np.mean(res['ucb']):>10.4f}{np.mean(res['rs']):>10.4f}")

f, b, opt = TEST_FUNCTIONS["branin"]
bo = BayesianOptimizer(b, n_init=5, seed=0, optimize_every=4)
x, y = bo.minimize(f, n_iter=4, batch_size=3)
print(f"\nBatch (constant liar, 3 per round) on branin: best={y:.4f} at {np.round(x, 3)}")
print(f"elapsed {time.time() - t0:.1f}s")
