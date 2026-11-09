# MCMC Sampler Suite: Metropolis-Hastings, MALA, Hamiltonian Monte Carlo and NUTS with Convergence Diagnostics

## Overview

From-scratch NumPy implementations of the main Markov chain Monte Carlo samplers, together with the diagnostics needed to judge them (autocorrelation, effective sample size, split R-hat, MCSE). The package includes analytic-gradient targets: correlated and ill-conditioned Gaussians, Haario's banana, a Gaussian mixture, Neal's funnel and a Bayesian logistic regression posterior.

Modules (`mcmc_samplers/`):

| Module | Contents |
|---|---|
| `targets.py` | Log-densities with analytic gradients and a finite-difference checker |
| `metropolis.py` | Random-walk Metropolis, Haario adaptive Metropolis, MALA |
| `hmc.py` | Leapfrog integrator, HMC, step-size heuristic, dual averaging |
| `nuts.py` | No-U-Turn Sampler (slice-sampling form) with adaptive step size |
| `diagnostics.py` | FFT autocorrelation, Geyer-truncated ESS, split R-hat, MCSE |

## How it works

- **Random-walk Metropolis** proposes `x' = x + s*z` and accepts with probability `min(1, pi(x')/pi(x))`. The adaptive variant tunes a proposal covariance `2.38^2/d * Cov(chain)`.
- **MALA** drifts along the gradient, `x' = x + (h^2/2) grad log pi(x) + h z`, and corrects with the asymmetric Hastings ratio.
- **HMC** augments the state with a momentum `p ~ N(0, I)` and simulates Hamiltonian dynamics with the reversible, volume-preserving leapfrog integrator, then applies a Metropolis test on the energy error. The step size is tuned during warm-up by Nesterov dual averaging towards a target acceptance statistic.
- **NUTS** removes the trajectory-length parameter: the path is doubled in a random direction until `(q+ - q-) . p < 0` at either end (a U-turn) or a divergence occurs. A slice variable gives a uniform choice among valid states in the tree.
- **Diagnostics** follow Vehtari et al.: ESS from the multi-chain autocorrelation estimate truncated by Geyer's initial positive sequence, and rank-free split R-hat.

## Complexity

| Sampler | Cost per draw | Memory |
|---|---|---|
| RWM | 1 density eval, O(d) | O(d) |
| Adaptive MH | 1 density eval + O(d^2) covariance update, O(d^3) Cholesky each step | O(d^2) |
| MALA | 1 density + 1 gradient | O(d) |
| HMC | L gradient evaluations | O(d) |
| NUTS | up to 2^depth gradient evaluations (recursion stores O(depth) states) | O(d * depth) |
| ESS / R-hat | O(n log n) per chain/parameter (FFT) | O(n) |

## Usage

```python
import numpy as np
from mcmc_samplers import CorrelatedGaussian, NUTS, effective_sample_size, split_rhat

target = CorrelatedGaussian.ill_conditioned(10, ratio=100.0)
chains = np.stack([NUTS(target, seed=s).sample(1000, np.zeros(10), warmup=300)
                   for s in range(4)])                  # (chains, draws, dim)
print(effective_sample_size(chains[:, :, 0]), split_rhat(chains[:, :, 0]))
```

Run `python -m pytest -q` for the tests (20) and `python examples/demo.py` for the benchmark below.

## Results

Output of `examples/demo.py` (4 chains x 1000 draws; burn-in 500 or warm-up 300; ESS is the minimum over dimensions; wall time includes warm-up):

```
== Ill-conditioned Gaussian, cond=100 (dim=10) ==
sampler       min ESS    ESS/s  max Rhat    sec
RWM                 3       49     2.272   0.05
AdaptiveMH         61      388     1.072   0.16
MALA                3       20     2.824   0.13
HMC              1351     2105     1.003   0.64
NUTS             1314      597     1.002   2.20

== Haario banana (dim=2) ==
RWM                 9      287     1.259   0.03
AdaptiveMH         36      271     1.089   0.13
MALA               20      184     1.178   0.11
HMC                 2        4     2.757   0.56
NUTS              205      275     1.023   0.75

== Bayesian logistic regression (dim=5) ==
AdaptiveMH        193      839     1.024   0.23
MALA              349     1349     1.015   0.26
HMC              2725     1587     1.000   1.72
NUTS             4612     3594     1.000   1.28
```

On the ill-conditioned Gaussian the gradient-based Hamiltonian samplers are orders of magnitude more efficient per draw than random-walk methods. On the banana, the fixed-length HMC configuration fails to converge (R-hat 2.76) because one trajectory length does not suit all regions of the density, whereas NUTS adapts its path length and converges. This is a limitation of the specific settings used, not of HMC in general.

## References

- Metropolis et al., "Equation of State Calculations by Fast Computing Machines", J. Chem. Phys. 1953.
- Hastings, "Monte Carlo Sampling Methods Using Markov Chains and Their Applications", Biometrika 1970.
- Haario, Saksman, Tamminen, "An Adaptive Metropolis Algorithm", Bernoulli 2001.
- Roberts, Rosenthal, "Optimal Scaling of Discrete Approximations to Langevin Diffusions", JRSS-B 1998.
- Neal, "MCMC Using Hamiltonian Dynamics", Handbook of MCMC, 2011.
- Hoffman, Gelman, "The No-U-Turn Sampler", JMLR 15, 2014.
- Vehtari et al., "Rank-Normalization, Folding, and Localization: An Improved R-hat", Bayesian Analysis 2021.
- Geyer, "Practical Markov Chain Monte Carlo", Statistical Science 1992.

---

**Author:** Dr. Kainat Rizwan
