"""Demo: mechanisms, composition accounting and DP-SGD."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from dp_toolkit import *

rng = np.random.default_rng(42)
t0 = time.time()

print("== Gaussian calibration (sens=1, eps=0.5, delta=1e-5) ==")
print(f"classical sigma = {classical_gaussian_sigma(1, .5, 1e-5):.3f}")
print(f"analytic  sigma = {analytic_gaussian_sigma(1, .5, 1e-5):.3f}")

print("\n== Private statistics on 10,000 ages (eps=1 each) ==")
ages = np.clip(rng.normal(40, 12, 10000), 0, 100)
print(f"count>60  true={(ages>60).sum()}  dp={private_count(ages, lambda a: a>60, 1.0, rng):.1f}")
print(f"mean      true={ages.mean():.3f}  dp={private_mean(ages, 0, 100, 1.0, rng):.3f}")
print(f"median    true={np.median(ages):.3f}  dp={private_quantile(ages, .5, 0, 100, 1.0, rng):.3f}")

print("\n== Composition of k Gaussian queries (sigma=5, delta=1e-5) ==")
print(f"{'k':>6} {'RDP eps':>10} {'basic eps':>10}")
for k in (10, 100, 1000):
    acc = RDPAccountant(); acc.compose_gaussian(5.0, steps=k)
    lo, hi = 0.0, 100.0
    for _ in range(60):
        m = (lo + hi) / 2
        if gaussian_delta(5.0, 1.0, m) > 1e-5 / k: lo = m
        else: hi = m
    print(f"{k:>6} {acc.epsilon(1e-5):>10.3f} {k*hi:>10.2f}")

print("\n== Subsampled Gaussian (q=0.01, sigma=1.1, 1000 steps) ==")
a = RDPAccountant(); a.compose_subsampled_gaussian(1.1, 0.01, 1000)
print(f"epsilon at delta=1e-5: {a.epsilon(1e-5):.3f}")

print("\n== DP-SGD logistic regression (n=5000) ==")
X = rng.normal(size=(5000, 3)); y = (X @ np.array([1.5, -2, 1]) > 0).astype(float)
Xb = np.hstack([X, np.ones((5000, 1))])
for nm in (0.0, 0.8, 2.0):
    w, eps, _ = dp_logistic_regression(X, y, 5, 250, 1.0, 1.0, nm, 1e-5, np.random.default_rng(1))
    print(f"noise={nm:<4} accuracy={((Xb@w>0)==y).mean():.3f} epsilon={eps:.2f}")
print(f"\nElapsed {time.time()-t0:.1f}s")
