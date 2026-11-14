# Interior-Point Solver for Linear and Convex Quadratic Programs

## Overview

A from-scratch NumPy implementation of interior-point methods for linear programs (LP) and convex quadratic programs (QP):

- **Mehrotra predictor-corrector primal-dual method** with an infeasible start, native handling of free variables, and a general-form front end (equalities, inequalities, box bounds).
- **Log-barrier path-following method** with Newton centering, backtracking line search and a phase-I feasibility search, used as an independent cross-check.
- **Applications** that reduce to LP/QP: L1 and Chebyshev regression, long-only minimum-variance portfolios, and the soft-margin SVM dual.

Package layout: `problem.py` (containers, reduction to standard form, generators), `core.py` (primal-dual solver), `barrier.py` (barrier method and phase I), `solver.py` (public API), `applications.py`.

## How it works

Standard form: `min ½x'Qx + c'x  s.t.  Ax = b, x ≥ 0` (`Q = 0` for an LP). The KKT conditions are

```
Qx + c − A'y − s = 0,   Ax = b,   x_i s_i = 0,   x, s ≥ 0.
```

Interior-point methods apply Newton's method to the perturbed system `x_i s_i = σμ` with `μ = x's/n`, keeping `x, s > 0`. Each iteration:

1. Solve the augmented system `[[Q + S/X, −A'], [A, 0]]` for the **affine-scaling** direction (σ = 0).
2. Compute the step to the boundary and the adaptive centering parameter `σ = (μ_aff/μ)³`.
3. Solve again with the second-order corrector term `Δx_aff Δs_aff − σμe`, and step 99.5% of the way to the boundary.

General problems are reduced by shifting lower bounds, reflecting upper-only bounds, turning finite two-sided bounds and inequalities into rows with slack variables, and keeping free variables as sign-unrestricted columns (a tiny proximal term replaces their complementarity condition). Splitting free variables into `x⁺ − x⁻` was tried first and fails: the pair drifts to infinity and recovery loses all precision.

The barrier method minimises `t·f(x) − Σ log(h − Gx)` for increasing `t`; the duality gap is bounded by `m/t`. Phase I minimises `s` subject to `Gx − h ≤ s` until `s < 0`.

## Complexity

| Quantity | Cost |
|---|---|
| Iterations (theory) | O(√n · log(1/ε)); in practice 10–25 |
| Per iteration (dense KKT) | O((n+m)³) time, O((n+m)²) space |
| Barrier method | O(√m · log(1/ε)) Newton steps, each O(n³ + m n²) |

## Usage

```python
import numpy as np
from interior_point import solve_lp, solve_qp, solve_barrier

# max 3x + 5y  s.t.  x <= 4, 2y <= 12, 3x + 2y <= 18
r = solve_lp([-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])
print(r.x, r.objective, r.iterations)       # [2. 6.]  -36.0  8

# Convex QP on the probability simplex
Q = np.array([[2.0, 0.5], [0.5, 1.0]])
q = solve_qp(Q, [-1, -1], A_eq=[[1, 1]], b_eq=[1])

# Barrier method for  min c'x  s.t.  Gx <= h
b = solve_barrier(None, [1.0, 1.0], np.array([[-1.0, 0], [0, -1.0], [-1, -1]]), np.array([0, 0, -1.0]))
```

Run the tests with `python -m pytest -q` and the demo with `python examples/demo.py`.

## Results

Output of `examples/demo.py` (random feasible problems, seed 1, dense NumPy on one CPU):

| Problem | m | n | Iterations | Time |
|---|---|---|---|---|
| LP | 10 | 30 | 13 | 1.9 ms |
| LP | 50 | 150 | 18 | 17.4 ms |
| LP | 100 | 300 | 17 | 93.7 ms |
| LP | 200 | 600 | 20 | 557.8 ms |
| QP | 100 | 300 | 16 | 81.4 ms |
| QP | 200 | 600 | 17 | 468.1 ms |

Iteration counts stay almost flat as the problem grows 20×, as the theory predicts. On a 20-variable LP with 70 inequality rows the barrier method needs 123 Newton steps against 12 iterations for the primal-dual method, and both reach the same objective (−20.40813561). The textbook LP converges to `(2, 6)` with value 36 in 8 iterations; the L1 fit recovers `[1.00, 2.009]` from data with 10% gross outliers where least squares gives `[3.16, 0.683]`.

The tests verify results against vertex enumeration, KKT conditions, complementarity, a projected-gradient QP reference, barrier/primal-dual agreement, and infeasible/unbounded detection.

## References

- S. Mehrotra, "On the implementation of a primal-dual interior point method", *SIAM J. Optimization* 2(4), 1992.
- J. Nocedal and S. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006 (chs. 14, 16).
- S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004 (ch. 11).
- N. Karmarkar, "A new polynomial-time algorithm for linear programming", *Combinatorica* 4, 1984.
- M. Friedlander and D. Orban, "A primal-dual regularized interior-point method for convex quadratic programs", *Math. Prog. Comp.* 4, 2012.

---

**Author:** Dr. Kainat Rizwan
