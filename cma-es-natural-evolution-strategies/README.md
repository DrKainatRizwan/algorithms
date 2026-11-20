# CMA-ES and Natural Evolution Strategies for Black-Box Optimisation

## Overview
From-scratch NumPy implementations of derivative-free evolution strategies: CMA-ES (weighted recombination, cumulative step-size adaptation, rank-one and rank-mu covariance updates), exponential and separable Natural Evolution Strategies (xNES, SNES), a (1+1)-ES with the 1/5th success rule, a cross-entropy method baseline and IPOP restarts for multimodal functions. A suite of benchmarks (sphere, ellipsoid, rotated ellipsoid, cigar, Rosenbrock, Rastrigin, Ackley) with known optima is included.

## How it works
All methods maintain a Gaussian search distribution N(m, sigma^2 C), sample lambda candidates and update the distribution using only the ranking of fitness values.

- **CMA-ES**: the mean moves to the weighted average of the best mu samples. The evolution path p_sigma drives step-size control (sigma grows when steps are correlated, shrinks otherwise). C is updated by a rank-one term from the path p_c and a rank-mu term from the selected steps, so it learns the local Hessian shape.
- **xNES**: follows the natural gradient of expected utility (rank-based log utilities) on the mean, a global scale and a unit-determinant shape matrix B, updated through a symmetric matrix exponential.
- **SNES**: same idea restricted to a diagonal covariance (O(n) per sample).
- **(1+1)-ES**: single parent; sigma is multiplied by exp(1/3) on success and exp(-1/12) on failure.
- **CEM**: refits a diagonal Gaussian to the elite fraction with smoothing and decaying extra noise.
- **IPOP**: restarts CMA-ES from random points, doubling the population size each time.

## Complexity
Per generation, CMA-ES costs O(lambda n^2) for sampling and the covariance update plus O(n^3) for the eigendecomposition (done every generation here); memory is O(n^2). xNES has the same profile. SNES, CEM and (1+1)-ES are O(lambda n) time and O(n) space.

## Usage
```python
import numpy as np
from evolution_strategies import cmaes_minimize, get_problem

p = get_problem("rosenbrock")
res = cmaes_minimize(p.func, np.zeros(6), 0.5, ftarget=1e-8, seed=0)
print(res.fun, res.evals, res.message)
```

Run the tests with `python -m pytest -q` and the demo with `python examples/demo.py`.

## Results
Output of `examples/demo.py` (start point 2.0 in every coordinate, sigma0 = 1, target f <= 1e-8):

```
Median evaluations to reach f<=1e-08 (3 seeds, budget 20000); (successes/3)
problem                         CMA-ES            xNES            SNES             CEM        (1+1)-ES
sphere d=10                 1460 (3/3)      6300 (3/3)      1920 (3/3)      2600 (1/3)       689 (3/3)
rotated_ellipsoid d=8       4310 (3/3)      5930 (3/3)         - (0/3)         - (0/3)         - (0/3)
rosenbrock d=6              2556 (3/3)      3708 (3/3)         - (0/3)         - (0/3)         - (0/3)
cigar d=10                  4400 (3/3)     10150 (3/3)      3130 (3/3)         - (0/3)         - (0/3)

Rastrigin d=5: plain CMA-ES vs IPOP-CMA-ES (3 seeds)
  plain CMA-ES  median best f = 9.9496
  IPOP-CMA-ES   median best f = 1.9899
```

CMA-ES solves the rotated and ill-conditioned problems that separable methods (SNES, CEM, (1+1)-ES) cannot, because it learns the covariance. SNES and the (1+1)-ES are cheapest on separable or well-conditioned problems. IPOP restarts improve the Rastrigin result over a single run.

## References
- N. Hansen, *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772, 2016.
- N. Hansen, A. Ostermeier, "Completely Derandomized Self-Adaptation in Evolution Strategies", Evolutionary Computation 9(2), 2001.
- D. Wierstra et al., "Natural Evolution Strategies", JMLR 15, 2014.
- T. Schaul, T. Glasmachers, J. Schmidhuber, "High Dimensions and Heavy Tails for Natural Evolution Strategies", GECCO 2011.
- A. Auger, N. Hansen, "A Restart CMA Evolution Strategy with Increasing Population Size", CEC 2005.
- R. Rubinstein, D. Kroese, *The Cross-Entropy Method*, Springer, 2004.

---

**Author:** Dr. Kainat Rizwan
