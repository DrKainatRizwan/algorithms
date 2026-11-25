# Proximal Methods for Sparse Optimisation: ISTA, FISTA, Coordinate Descent and ADMM

## Overview
From-scratch NumPy implementations of the main solvers for ℓ1-regularised least squares and its relatives:

- **ISTA / FISTA** (proximal gradient, Nesterov momentum, adaptive restart, backtracking) via a generic `proximal_gradient` routine
- **Cyclic coordinate descent** for the elastic net with residual updates and an active-set strategy
- **ADMM** for the generalised lasso `min ½‖Xw−y‖² + λ‖Dw‖₁` (lasso, fused lasso, trend filtering) with residual balancing
- **Group lasso** through block soft-thresholding, ℓ1-ball projection, regularisation paths, K-fold cross-validation and support-recovery metrics
- Optimality certificates: `lambda_max` and the KKT violation of any candidate solution

## How it works
The lasso objective is `F(w) = ½‖Xw−y‖² + λ‖w‖₁`. The smooth part has gradient Lipschitz constant `L = λ_max(XᵀX)` (power iteration).

- **ISTA:** `w ← S_{λ/L}(w − Xᵀ(Xw−y)/L)`, with `S` the soft-threshold; error `O(1/k)`.
- **FISTA:** the same step evaluated at an extrapolated point `y_k = x_k + (t_{k-1}−1)/t_k (x_k − x_{k-1})`; error `O(1/k²)`. Function-value restart resets `t` whenever the objective rises, which recovers linear convergence on strongly convex problems.
- **Coordinate descent:** exact minimisation along one coordinate, `w_j ← S_{λα}(x_jᵀr + ‖x_j‖² w_j) / (‖x_j‖² + λ(1−α))`, with the residual `r` maintained incrementally. The active set is swept to convergence between full sweeps.
- **ADMM:** split `z = Dw`; alternate a ridge-type linear solve (matrix inverted once, refreshed when ρ is rebalanced), a soft-threshold on `z`, and a dual update; stop on primal and dual residuals.

## Complexity
| Method | Cost per iteration | Memory |
|---|---|---|
| ISTA / FISTA | O(np) | O(np) |
| Coordinate descent | O(n) per coordinate, O(np) per sweep (O(n·nnz) on the active set) | O(np) |
| ADMM | O(p²) after an O(p³) factorisation | O(p²) |
| ℓ1-ball projection | O(p log p) | O(p) |

## Usage
```python
from sparse_solvers import make_sparse_regression, lambda_max, fista, coordinate_descent, kkt_violation

X, y, w_true = make_sparse_regression(200, 100, 8, noise=0.2, seed=0)
lam = 0.1 * lambda_max(X, y)
res = fista(X, y, lam)
print(res.iterations, kkt_violation(X, y, res.w, lam))
```
Run `python examples/demo.py` for the benchmark below and `python -m pytest -q` for the 17 tests.

## Results
Lasso, n=200, p=100, 8 true non-zeros, column correlation 0.3, λ = 0.1·λ_max (all solvers reach the same objective, 1925.899645, and 7 non-zeros):

| Solver | Iterations | Time (ms) | KKT violation |
|---|---|---|---|
| ISTA | 788 | 21.4 | 1.7e-05 |
| FISTA | 972 | 31.1 | 4.6e-06 |
| FISTA + restart | 150 | 3.5 | 1.1e-05 |
| Coordinate descent | 3 sweeps | 5.2 | 9.5e-09 |
| ADMM | 3136 | 110.8 | 3.5e-05 |

Plain FISTA oscillates on this well-conditioned problem and ends up needing more iterations than ISTA; restart fixes this. Cross-validation over a 30-point path picks a λ that gives full recall (1.00) but a dense fit (precision 0.24), the familiar tendency of CV-tuned lasso to over-select. Fused-lasso denoising of a piecewise-constant signal lowers RMSE from 0.342 to 0.149.

## References
- A. Beck, M. Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", SIAM J. Imaging Sci., 2009.
- I. Daubechies, M. Defrise, C. De Mol, "An iterative thresholding algorithm for linear inverse problems", CPAM, 2004.
- J. Friedman, T. Hastie, R. Tibshirani, "Regularization Paths for Generalized Linear Models via Coordinate Descent", J. Stat. Softw., 2010.
- S. Boyd et al., "Distributed Optimization and Statistical Learning via ADMM", Found. Trends Mach. Learn., 2011.
- B. O'Donoghue, E. Candès, "Adaptive Restart for Accelerated Gradient Schemes", Found. Comput. Math., 2015.
- J. Duchi et al., "Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions", ICML, 2008.
- R. Tibshirani, "Regression Shrinkage and Selection via the Lasso", JRSS B, 1996.

---

**Author:** Dr. Kainat Rizwan
