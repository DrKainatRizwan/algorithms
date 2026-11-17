"""Demo: exact OT vs Sinkhorn vs sliced Wasserstein, plus barycenters."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from optimal_transport import (emd, entropic_barycenter, grid_cost, pairwise_cost, sinkhorn,
                               sinkhorn_eps_scaling, sliced_wasserstein, wasserstein_1d,
                               wasserstein_exact, wasserstein_1d_barycenter)

rng = np.random.default_rng(0)

print("== Accuracy of entropic OT vs exact (n=m=40, 2-D Gaussians) ==")
x = rng.standard_normal((40, 2)); y = rng.standard_normal((40, 2)) + [2, 1]
a = b = np.ones(40) / 40
C = pairwise_cost(x, y)
t0 = time.perf_counter(); ex, _ = emd(a, b, C); te = time.perf_counter() - t0
print(f"exact cost = {ex:.5f}  ({te*1000:.0f} ms)")
for eps in (1.0, 0.3, 0.1, 0.03, 0.01):
    t0 = time.perf_counter(); r = sinkhorn(a, b, C, eps, max_iter=20000); ts = time.perf_counter() - t0
    print(f"eps={eps:<5} cost={r.cost:.5f} gap={r.cost-ex:+.5f} iters={r.n_iter:<6} {ts*1000:.0f} ms")
r = sinkhorn_eps_scaling(a, b, C, eps=0.01)
print(f"eps-scaling eps=0.01: cost={r.cost:.5f} total iters={r.n_iter}")

print("\n== Scaling of the exact solver ==")
for n in (10, 20, 40, 60):
    x = rng.standard_normal((n, 2)); y = rng.standard_normal((n, 2))
    t0 = time.perf_counter(); emd(np.ones(n), np.ones(n), pairwise_cost(x, y)); te = time.perf_counter() - t0
    t0 = time.perf_counter(); sinkhorn(np.ones(n), np.ones(n), pairwise_cost(x, y), 0.1); ts = time.perf_counter() - t0
    print(f"n={n:<3} exact {te*1000:7.1f} ms   sinkhorn {ts*1000:6.1f} ms")

print("\n== Sliced Wasserstein (d=5, 100 points) ==")
X = rng.standard_normal((100, 5)); Y = rng.standard_normal((100, 5)) + 1
print(f"W2 exact        = {wasserstein_exact(X, Y):.4f}")
for k in (10, 100, 1000):
    print(f"SW2 ({k:>4} proj) = {sliced_wasserstein(X, Y, n_proj=k):.4f}")

print("\n== 1-D closed form ==")
u, v = rng.standard_normal(500), rng.standard_normal(500) * 2 + 3
print(f"W1 = {wasserstein_1d(u, v, p=1):.4f}, W2 = {wasserstein_1d(u, v, p=2):.4f}")

print("\n== Barycenter of two bumps on [0,1] ==")
n = 100; t = np.linspace(0, 1, n)
g = lambda m: (lambda h: h / h.sum())(np.exp(-(t - m) ** 2 / (2 * 0.04 ** 2)))
bar = entropic_barycenter(np.vstack([g(0.2), g(0.8)]), grid_cost(n, 2), eps=0.002, max_iter=3000)
print(f"entropic barycenter mean={np.sum(bar*t):.3f} std={np.sqrt(np.sum(bar*(t-np.sum(bar*t))**2)):.3f}")
z = rng.standard_normal(2000)
q = wasserstein_1d_barycenter([z - 3, z + 5], [.5, .5])
print(f"quantile barycenter mean={q.mean():.3f} std={q.std():.3f} (inputs std={z.std():.3f})")
