# Quasi-Newton Optimisation Suite: BFGS, L-BFGS, Nonlinear CG and Trust-Region Methods

## Overview
A from-scratch NumPy library for smooth unconstrained minimisation. It contains dense quasi-Newton methods (BFGS, DFP, SR1), limited-memory BFGS, four nonlinear conjugate-gradient variants and trust-region Newton methods (Steihaug-CG and dogleg), all sharing a strong-Wolfe line search with cubic interpolation. Every objective is wrapped in a `Problem` that counts function, gradient and Hessian calls so methods can be compared fairly.

Modules: `functions.py` (test problems), `linesearch.py`, `quasi_newton.py`, `nonlinear_cg.py`, `trust_region.py`, `benchmark.py`.

## How it works
- **BFGS** keeps an inverse-Hessian approximation `H` and updates it with `H+ = (I - ρ s yᵀ) H (I - ρ y sᵀ) + ρ s sᵀ`, `ρ = 1/yᵀs`. A Wolfe line search guarantees `yᵀs > 0`, so `H` stays positive definite and the secant equation `H+ y = s` holds.
- **DFP** is the dual update; **SR1** is a rank-one update that can represent indefinite curvature (used with Armijo backtracking and a safeguard).
- **L-BFGS** stores only the last `m` pairs `(s, y)` and computes `-H g` with the two-loop recursion, using `γ = sᵀy / yᵀy` as initial scaling.
- **Strong-Wolfe line search** brackets a step and zooms with safeguarded cubic interpolation (Nocedal & Wright Alg. 3.5/3.6).
- **Nonlinear CG** uses Fletcher-Reeves, Polak-Ribière+, Hestenes-Stiefel(+) or Dai-Yuan β with restarts every n steps.
- **Trust region** minimises the quadratic model inside a ball of radius Δ, either by Steihaug truncated CG (handles negative curvature by walking to the boundary) or by the dogleg path; Δ adapts to the ratio of actual to predicted reduction.

## Complexity
| Method | Time per iteration | Memory |
|---|---|---|
| BFGS / DFP / SR1 | O(n²) | O(n²) |
| L-BFGS | O(mn) | O(mn) |
| Nonlinear CG | O(n) | O(n) |
| Trust region (Steihaug) | O(n²) per CG step, up to 2n steps | O(n²) (Hessian) |
| Trust region (dogleg) | O(n³) (solve / eigenvalues) | O(n²) |

## Usage
```python
from quasi_newton import rosenbrock, bfgs, lbfgs, trust_region

p = rosenbrock()
r = bfgs(p)
print(r.x, r.f, r.iterations, p.n_f, p.n_g)
r = trust_region(rosenbrock(), subproblem="steihaug")
```

Run `python examples/demo.py` for the benchmark and `python -m pytest -q` for the tests.

## Results
Output of `examples/demo.py` (tolerance 1e-6 on the max-norm of the gradient; "N" = failed to converge in 500 iterations):

| Problem | BFGS g-evals | L-BFGS (m=5) g-evals | CG-PR+ g-evals | TR-Steihaug iters (Hessians) | DFP g-evals |
|---|---|---|---|---|---|
| Rosenbrock 2-D | 45 | 44 | 125 | 25 | 140 |
| Ext. Rosenbrock n=40 | 49 | 53 | 118 | 27 | N (500 iters, \|g\|=1e-2) |
| Exponential chain n=30 | 17 | 15 | 29 | 6 | 19 |
| Logistic regression 200x20 | 23 | 14 | 82 | 6 | 28 |

Observations: Newton-type trust-region steps need the fewest iterations but require Hessians; BFGS and L-BFGS reach similar accuracy using gradients only; CG needs more evaluations but O(n) memory; DFP is visibly less robust than BFGS on the 40-dimensional problem, matching the classical literature.

## References
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006.
- D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization", *Math. Programming* 45, 1989.
- T. Steihaug, "The conjugate gradient method and trust regions in large scale optimization", *SIAM J. Numer. Anal.* 20, 1983.
- M. J. D. Powell, "A new algorithm for unconstrained optimization", 1970 (dogleg).
- W. W. Hager and H. Zhang, "A survey of nonlinear conjugate gradient methods", 2006.

---

**Author:** Dr. Kainat Rizwan
