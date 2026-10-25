# Linear Programming Solver: Simplex Method with Bland's Rule and Duality

## Overview

A from-scratch linear programming library in Python and NumPy. It solves general-form problems
(`min/max c·x` with `<=` and `=` rows and arbitrary variable bounds) using the two-phase primal
simplex method, with three pivot rules (Dantzig, Bland, and a Dantzig-to-Bland hybrid), a dense
tableau implementation and an independent revised simplex implementation. On top of the solver
sit the duality tools: dual construction, KKT optimality certificates, Farkas and unbounded-ray
certificates, and sensitivity (ranging) analysis. A set of applications (transportation, max-flow,
assignment, zero-sum games, L1 and Chebyshev regression) is included.

Modules:

| Module | Contents |
|---|---|
| `simplex_lp/model.py` | `LinearProgram`, conversion to standard form `min cᵀu, Au=b, u≥0` |
| `simplex_lp/tableau.py` | Two-phase tableau simplex, pivot rules, cycle detection |
| `simplex_lp/revised.py` | Revised simplex re-solving with the basis matrix |
| `simplex_lp/solver.py` | `solve()` returning primal solution, shadow prices, reduced costs |
| `simplex_lp/duality.py` | Dual LP, KKT/Farkas/ray certificates, RHS and cost ranging |
| `simplex_lp/problems.py` | Beale, Klee-Minty, diet, transportation, flows, games, regression |

## How it works

1. **Standard form.** Bounds are removed by the substitution `x = shift + T u` (`x = l + u`,
   `x = h - u`, or `x = u⁺ - u⁻`), finite upper bounds become rows, and every `<=` row gets a slack.
2. **Phase 1.** Rows are sign-normalised so `b >= 0`, one artificial variable is added per row and
   their sum is minimised. A positive optimum proves infeasibility, and the phase-1 duals give a
   Farkas vector `y` with `yᵀA <= 0`, `yᵀb > 0`. Artificials left in the basis at level zero are
   pivoted out, or their row is dropped as redundant.
3. **Phase 2.** The pivot loop picks an entering column with negative reduced cost, applies the
   minimum-ratio test, and pivots. A column with no positive entries yields an unbounded ray.
4. **Pivot rules.** *Dantzig* picks the most negative reduced cost: fast, but cycles on degenerate
   vertices (Beale's example). *Bland* picks the smallest-index entering and leaving variables and
   provably terminates. *Hybrid* uses Dantzig and falls back to Bland after a run of degenerate pivots.
5. **Duals and certificates.** From the final basis `B`, `y = B⁻ᵀ c_B` and reduced costs
   `r = c - Aᵀy` are recomputed with a linear solve. `check_optimality` verifies primal
   feasibility, dual feasibility, complementary slackness and a zero duality gap.
6. **Sensitivity.** With `B⁻¹` available, the right-hand-side range of row `i` is the interval of
   `δ` keeping `B⁻¹(b + δeᵢ) >= 0`; cost ranges follow from keeping all reduced costs non-negative.

## Complexity

- Tableau: `O(m(n+m))` memory; each pivot costs `O(m(n+m))`.
- Revised: `O(m²)` memory for the basis; each pivot costs `O(m³)` here (dense solve; an LU update would give `O(m²)`).
- Number of pivots: exponential in the worst case (Klee-Minty with Dantzig takes `2ⁿ-1`), but
  typically a small multiple of `m` in practice.

## Usage

```python
import numpy as np
from simplex_lp import LinearProgram, solve, rhs_ranging, check_optimality

lp = LinearProgram([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True)
sol = solve(lp, method="tableau", rule="bland")
print(sol.x, sol.objective)        # [2. 6.] 36.0
print(sol.duals)                   # shadow prices [0, 1.5, 1]
print(rhs_ranging(sol))            # allowable change of each right-hand side
print(check_optimality(lp, sol.x, sol.duals).ok())
```

Run the demo with `python examples/demo.py` and the tests with `python -m pytest -q`.

## Results

Output of `examples/demo.py`:

Beale's degenerate LP (standard-form start at the slack basis):

| Rule | Status | Pivots |
|---|---|---|
| Dantzig | cycling detected | 6 |
| Bland | optimal (-1.25) | 6 |
| Hybrid | optimal (-1.25) | 24 |

Klee-Minty cube, pivots to optimality:

| n | Dantzig | Bland | 2ⁿ-1 |
|---|---|---|---|
| 4 | 15 | 9 | 15 |
| 6 | 63 | 25 | 63 |
| 8 | 255 | 67 | 255 |

Random LPs (mean of 5 seeds, hybrid rule):

| Size | Tableau ms (pivots) | Revised ms (pivots) |
|---|---|---|
| 20x15 | 3.1 (57) | 2.6 (57) |
| 40x30 | 10.4 (134) | 10.8 (136) |
| 80x60 | 62.8 (294) | 100.7 (294) |
| 120x100 | 252.9 (740) | 640.3 (740) |

Objectives of the two implementations agree to better than `3e-14`. Other results: the CLRS
max-flow network gives 23, an L1 fit with two gross outliers recovers `[0.997, 2.007]` (true
`[1, 2]`) while ordinary least squares gives `[4.25, -1.51]`, and the explicit dual LP reproduces
the primal optimum (strong duality).

## References

- G. B. Dantzig, *Linear Programming and Extensions*, Princeton University Press, 1963.
- R. G. Bland, "New finite pivoting rules for the simplex method", *Mathematics of Operations Research* 2(2), 1977.
- E. M. L. Beale, "Cycling in the dual simplex algorithm", *Naval Research Logistics Quarterly* 2, 1955.
- V. Klee and G. J. Minty, "How good is the simplex algorithm?", in *Inequalities III*, 1972.
- D. Bertsimas and J. N. Tsitsiklis, *Introduction to Linear Optimization*, Athena Scientific, 1997.
- V. Chvátal, *Linear Programming*, W. H. Freeman, 1983.

---

**Author:** Dr. Kainat Rizwan
