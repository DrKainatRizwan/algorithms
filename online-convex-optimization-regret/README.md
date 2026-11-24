# Online Convex Optimisation: Regret Minimisation from Scratch

## Overview
A NumPy-only library for online convex optimisation (OCO). A learner repeatedly picks a point `x_t` in a convex set, then a convex loss `f_t` is revealed. Performance is measured by regret against the best fixed point in hindsight:

`Regret_T = sum_t f_t(x_t) - min_{u in K} sum_t f_t(u)`

The package implements the main full-information algorithms, exact comparators, and the textbook regret bounds so they can be checked empirically.

Modules (`online_learning/`): `domains.py` (ball, simplex, box, L1 ball, projections, linear oracles), `losses.py` (linear, ridge-squared, logistic and exact aggregation), `learners.py` (OGD, FTL, FTRL, Hedge, EG, FPL, AdaGrad, ONS), `regret.py` (run loop, FISTA comparator, bounds, growth exponent) and `environments.py` (adversarial and stochastic streams).

## How it works
- **OGD**: `x <- Pi_K(x - eta_t grad f_t(x))`. With `eta = D/(G sqrt T)` regret is at most `D G sqrt T`; with `eta_t = 1/(H t)` on H-strongly convex losses it is `G^2 (1+ln T)/(2H)`.
- **FTL**: plays the minimiser of past (linearised) losses. It is unstable and an adversary forces `Omega(T)` regret (`ftl_adversary`).
- **FTRL**: adds a quadratic regulariser; for `||x||^2/(2 eta)` this is lazy projection `Pi_K(-eta sum g)`.
- **Hedge / EG / FPL**: multiplicative weights on the simplex (regret `sqrt(T ln N / 2)`), its mirror-descent form, and a randomised perturbed-leader variant.
- **AdaGrad**: per-coordinate steps from accumulated squared gradients (exact on boxes).
- **ONS**: Newton-like updates for exp-concave losses, `A <- A + g g^T`, `A^-1` maintained by Sherman-Morrison, with projection in the `A`-norm computed by bisection on the KKT multiplier. Regret is `O((d/alpha) log T)`.
- **Comparator**: closed form for linear losses via the linear oracle; otherwise projected FISTA with backtracking on the exactly aggregated loss.

## Complexity
| Learner | Time / round | Memory |
|---|---|---|
| OGD, FTRL, FTL (linear) | O(d) plus projection (ball O(d), simplex O(d log d)) | O(d) |
| Hedge, EG, FPL | O(N) | O(N) |
| AdaGrad | O(d) | O(d) |
| ONS | O(d^2) update, O(d^3) projection (eigendecomposition) | O(d^2) |

## Usage
```python
from online_learning import *
dom = Ball(5, 1.0)
losses = regression_stream(2000, 5, seed=3, lam=0.5)
run = run_online(OnlineGradientDescent(dom, schedule="strong"), losses)
print(regret(run, losses, dom))
```

## Results
From `python examples/demo.py` (runs in about 1 second):

| Setting | Learner | T=200 | T=800 | T=3200 | Fitted exponent |
|---|---|---|---|---|---|
| FTL adversary | FTL | 199.5 | 799.5 | 3199.5 | 1.00 |
| FTL adversary | FTRL | 7.5 | 14.6 | 28.8 | 0.48 |
| FTL adversary | OGD | 13.7 | 27.9 | 56.1 | 0.51 |
| Experts N=10 | Hedge | 8.9 | 16.5 | 31.8 | bound 15.2 / 30.3 / 60.7 |
| Ridge stream | OGD const | 0.23 | 0.77 | 2.70 | 0.88 |
| Ridge stream | OGD 1/sqrt t | 0.16 | 0.28 | 0.48 | 0.39 |
| Ridge stream | OGD 1/(Ht) | 0.18 | 0.21 | 0.24 | 0.10 |
| Regression | ONS | 0.73 | 0.87 | 1.40 | 0.23 |

On a logistic stream over the box `[-2,2]^8` with `T=3200`, AdaGrad reaches regret 23.2 against 122.0 for OGD with `1/sqrt t` steps.

Run the tests with `python -m pytest -q` (22 tests).

## References
- M. Zinkevich, *Online Convex Programming and Generalized Infinitesimal Gradient Ascent*, ICML 2003.
- E. Hazan, *Introduction to Online Convex Optimization*, Foundations and Trends in Optimization, 2016.
- E. Hazan, A. Agarwal, S. Kale, *Logarithmic Regret Algorithms for Online Convex Optimization*, Machine Learning 2007.
- Y. Freund, R. Schapire, *A Decision-Theoretic Generalization of On-Line Learning*, JCSS 1997.
- J. Duchi, E. Hazan, Y. Singer, *Adaptive Subgradient Methods*, JMLR 2011.
- A. Kalai, S. Vempala, *Efficient Algorithms for Online Decision Problems*, JCSS 2005.
- A. Beck, M. Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm*, SIAM J. Imaging Sci. 2009.

---

**Author:** Dr. Kainat Rizwan
