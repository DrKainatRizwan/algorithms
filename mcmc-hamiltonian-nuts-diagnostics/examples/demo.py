"""Compare RWM, adaptive RWM, MALA, HMC and NUTS on three targets."""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcmc_samplers import (AdaptiveMetropolis, Banana, BayesianLogisticRegression,
                           CorrelatedGaussian, HMC, MALA, NUTS, RandomWalkMetropolis,
                           effective_sample_size, split_rhat)


def run_chains(make, target, n, warm, n_chains=4):
    draws, t0 = [], time.perf_counter()
    for c in range(n_chains):
        rng = np.random.default_rng(100 + c)
        x0 = rng.normal(size=target.dim)
        s = make(c)
        draws.append(s.sample(n, x0, **warm))
    return np.stack(draws), time.perf_counter() - t0


def report(name, target, samplers, n):
    print(f"\n== {name} (dim={target.dim}, {n} draws x 4 chains) ==")
    print(f"{'sampler':<12}{'min ESS':>9}{'ESS/s':>9}{'max Rhat':>10}{'sec':>7}")
    for label, make, warm in samplers:
        ch, sec = run_chains(make, target, n, warm)
        ess = min(effective_sample_size(ch[:, :, k]) for k in range(target.dim))
        rh = max(split_rhat(ch[:, :, k]) for k in range(target.dim))
        print(f"{label:<12}{ess:>9.0f}{ess / sec:>9.0f}{rh:>10.3f}{sec:>7.2f}")


def main():
    warnings.simplefilter("ignore", RuntimeWarning)  # overflow in rejected divergent trajectories
    n = 1000
    gauss = CorrelatedGaussian.ill_conditioned(10, 100.0, seed=0)
    suite = lambda t: [
        ("RWM", lambda c: RandomWalkMetropolis(t, seed=c), {"burn": 500}),
        ("AdaptiveMH", lambda c: AdaptiveMetropolis(t, seed=c), {"burn": 500}),
        ("MALA", lambda c: MALA(t, step=0.25, seed=c), {"burn": 500}),
        ("HMC", lambda c: HMC(t, n_leapfrog=20, jitter=0.2, seed=c), {"warmup": 300}),
        ("NUTS", lambda c: NUTS(t, seed=c), {"warmup": 300}),
    ]
    report("Ill-conditioned Gaussian, cond=100", gauss, suite(gauss), n)
    report("Haario banana", Banana(), suite(Banana()), n)
    lr, beta = BayesianLogisticRegression.synthetic(300, 5, seed=3)
    report("Bayesian logistic regression", lr, suite(lr)[1:], n)

    ch = NUTS(lr, seed=0).sample(2000, np.zeros(5), warmup=300)
    print("\nlogistic posterior mean vs truth:")
    print(" mean:", np.round(ch.mean(0), 2))
    print(" true:", np.round(beta, 2))


if __name__ == "__main__":
    main()
