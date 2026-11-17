# Optimal Transport Toolkit: Exact Solver, Sinkhorn, Sliced Wasserstein and Barycenters

## Overview
From-scratch NumPy implementations of discrete optimal transport (OT):

- `exact.py` – exact Kantorovich solver (successive shortest augmenting paths, dense Dijkstra with potentials).
- `sinkhorn.py` – entropic OT: log-domain Sinkhorn, textbook scaling form, epsilon-scaling, Sinkhorn divergence.
- `one_dimensional.py` – closed-form 1-D Wasserstein via monotone coupling, quantile functions, sliced Wasserstein.
- `barycenter.py` – entropic barycenters (iterative Bregman projections), 1-D quantile barycenters, displacement interpolation, barycentric maps.
- `cost.py` – cost matrices and validation.

## How it works
Given histograms `a`, `b` and cost `C`, OT solves `min_P <P,C>` over plans with marginals `a`, `b`.

**Exact:** the problem is a min-cost flow on `source → rows → columns → sink`. Each iteration finds a shortest residual path (reduced costs keep Dijkstra valid) and pushes the bottleneck mass.

**Sinkhorn:** adding `-eps·H(P)` gives plans of form `diag(u) K diag(v)`, `K = exp(-C/eps)`, found by alternating marginal scalings. The log-domain version updates dual potentials `f, g` with log-sum-exp and stays stable for tiny `eps`. Epsilon-scaling anneals `eps` with warm starts. The debiased Sinkhorn divergence is `OT_e(a,b) − ½(OT_e(a,a)+OT_e(b,b))`.

**1-D / sliced:** in 1-D the optimal plan is monotone, so `W_p` is computed after sorting. Sliced Wasserstein averages `W_p^p` over random projections.

**Barycenters:** the entropic barycenter alternates Bregman projections onto the marginal constraints; in 1-D the barycenter is the average of quantile functions.

## Complexity
| Method | Time | Space |
|---|---|---|
| Exact (this implementation) | ~O((n+m)·V²) with V=n+m+2 (dense) | O(V²) |
| Sinkhorn | O(n·m) per iteration, ~1/eps iterations | O(n·m) |
| 1-D exact | O((n+m) log(n+m)) | O(n+m) |
| Sliced (L projections) | O(L·n log n) | O(n) |
| Entropic barycenter (k inputs) | O(k·n²) per iteration | O(k·n) |

## Usage
```python
import numpy as np
from optimal_transport import emd, sinkhorn, pairwise_cost, sliced_wasserstein

rng = np.random.default_rng(0)
x, y = rng.standard_normal((40, 2)), rng.standard_normal((40, 2)) + [2, 1]
a = b = np.ones(40) / 40
C = pairwise_cost(x, y)
cost, plan = emd(a, b, C)
res = sinkhorn(a, b, C, eps=0.05)
print(cost, res.cost, res.marginal_error)
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests (23 tests).

## Results
Output of the demo (n = m = 40, 2-D Gaussians; exact cost 3.97041, 60 ms):

| eps | Sinkhorn cost | gap to exact | iterations |
|---|---|---|---|
| 1.0 | 4.64377 | +0.67336 | 50 |
| 0.3 | 4.14713 | +0.17671 | 170 |
| 0.1 | 4.00775 | +0.03734 | 810 |
| 0.03 | 3.97712 | +0.00671 | 20000 (cap) |
| 0.01 | 3.97183 | +0.00142 | 20000 (cap) |

Epsilon-scaling reaches eps = 0.01 (cost 3.97186) in 6280 total iterations. Exact solver time grows from 1.4 ms (n=10) to 129 ms (n=60). For 5-D Gaussians with 100 points, W2 = 2.783 while SW2 ≈ 1.1, illustrating that sliced distances are a lower bound on W2 and scale differently. Barycenters of two bumps at 0.2 and 0.8 land at mean 0.500.

## References
- Kantorovich, L. V. (1942). On the translocation of masses.
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS.
- Benamou, J.-D. et al. (2015). Iterative Bregman Projections for Regularized Transportation Problems. SIAM J. Sci. Comput.
- Peyré, G. & Cuturi, M. (2019). Computational Optimal Transport. Foundations and Trends in ML.
- Schmitzer, B. (2019). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems.
- Bonneel, N. et al. (2015). Sliced and Radon Wasserstein Barycenters of Measures.
- Genevay, A., Peyré, G., Cuturi, M. (2018). Learning Generative Models with Sinkhorn Divergences.

---

**Author:** Dr. Kainat Rizwan
