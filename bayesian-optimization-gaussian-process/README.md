# Bayesian Optimization from Scratch with Gaussian Process Surrogates

## Overview
A compact NumPy implementation of Bayesian optimisation for expensive black-box minimisation: an exact Gaussian process (RBF, Matern-3/2, Matern-5/2 kernels with ARD), marginal-likelihood hyper-parameter fitting with a hand-written Nelder-Mead optimiser, Expected Improvement / Probability of Improvement / Lower Confidence Bound acquisitions, an ask/tell loop with constant-liar batch proposals, and a random-search baseline.

## How it works
1. Evaluate a Latin-hypercube initial design in the unit cube.
2. Fit a GP to the standardised observations; hyper-parameters (length-scales, signal variance, noise) maximise the log marginal likelihood `-1/2 y^T K^-1 y - 1/2 log|K| - n/2 log 2pi` via multi-start Nelder-Mead in log space.
3. Posterior: `mu(x) = k*^T K^-1 y`, `s^2(x) = k(x,x) - k*^T K^-1 k*`, computed from a Cholesky factor.
4. Maximise the acquisition (`EI = (f* - mu) Phi(z) + s phi(z)`, `z = (f* - mu)/s`) by random screening plus Nelder-Mead polishing of the top candidates.
5. Evaluate, update, repeat. For batches, each proposed point is added with a "lie" equal to the incumbent before proposing the next.

## Complexity
For `n` observations in `d` dimensions: fitting is `O(n^3)` time and `O(n^2)` space per likelihood evaluation; a prediction costs `O(n)` for the mean and `O(n^2)` for the variance per point. Each iteration needs `O(C (n^2 + n d))` for `C` acquisition candidates plus the hyper-parameter search.

## Usage
```python
from gp_bayesopt import BayesianOptimizer, TEST_FUNCTIONS

f, bounds, optimum = TEST_FUNCTIONS["branin"]
bo = BayesianOptimizer(bounds, acquisition="ei", n_init=5, seed=0)
x_best, y_best = bo.minimize(f, n_iter=25)

# ask/tell and batches
X = bo.ask(3); bo.tell(X, [f(x) for x in X])
```
Run `python examples/demo.py` for the benchmark and `python -m pytest -q` for the 21 tests.

## Results
Mean best value after 30 evaluations over 2 seeds (`python examples/demo.py`, ~12 s):

| Function | Optimum | BO (EI) | BO (LCB) | Random search |
|---|---|---|---|---|
| Forrester (1-D) | -6.0207 | -6.0185 | -6.0207 | -5.8313 |
| Branin (2-D) | 0.3979 | 0.4374 | 0.4063 | 1.7053 |
| Hartmann-3 (3-D) | -3.8628 | -3.8582 | -3.8584 | -3.4948 |

## References
- Mockus, J. (1975). On Bayesian methods for seeking the extremum.
- Jones, Schonlau & Welch (1998). Efficient Global Optimization of Expensive Black-Box Functions. *J. Global Optimization*.
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Snoek, Larochelle & Adams (2012). Practical Bayesian Optimization of Machine Learning Algorithms. NeurIPS.
- Ginsbourger, Le Riche & Carraro (2010). Kriging is well-suited to parallelize optimization.
- Nelder & Mead (1965). A simplex method for function minimization.

---

**Author:** Dr. Kainat Rizwan
