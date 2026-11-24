# Mixed-Integer Programming: Branch-and-Bound with Gomory and Cover Cuts

## Overview

A from-scratch mixed-integer linear programming (MIP) solver using only NumPy. It contains a dense two-phase primal simplex that exposes its optimal tableau, Gomory mixed-integer (GMI) cuts read off that tableau, knapsack cover cuts, and a branch-and-bound driver with several branching and node-selection rules. Seeded generators for knapsack, set cover, facility location and assignment-with-side-constraint instances are included, together with a brute-force oracle used by the tests.

Problem form: `min c·x  s.t.  A_ub x ≤ b_ub,  A_eq x = b_eq,  lb ≤ x ≤ ub,  x_j ∈ ℤ for j ∈ I`.

## How it works

1. **LP relaxation** (`lp.py`): variables are shifted to `y = x - lb ≥ 0`, finite upper bounds become rows, rows are sign-normalised, and a two-phase tableau simplex is run (Dantzig pricing, switching to Bland's rule after a run of degenerate pivots). Redundant equalities are removed after phase 1.
2. **Gomory mixed-integer cuts** (`cuts.py`): for a basic integer variable with fractional value `f0`, the tableau row `x_B + Σ a_j x_j = b` yields `Σ g_j x_j ≥ 1` with `g_j = f_j/f0` or `(1-f_j)/(1-f0)` for integer columns and `a_j/f0` or `-a_j/(1-f0)` for continuous ones. Slack columns are substituted back so cuts live in the original space. Numerically unsafe cuts (large dynamism, tiny violation) are discarded.
3. **Cover cuts**: for binary rows `a·x ≤ b` with `a ≥ 0` a greedy minimal cover `C` gives `Σ_{j∈C} x_j ≤ |C| - 1`.
4. **Branch-and-bound** (`bnb.py`): cuts are added at the root only (so they stay globally valid); nodes are processed best-first or depth-first. Branching rules: first fractional, most fractional, pseudocost and strong branching. A rounding heuristic supplies incumbents, and bounds are rounded up when the objective is provably integral.

## Complexity

- One simplex pivot costs O(m·(n+m)) time; the tableau needs O(m·(n+m)) memory. Worst-case pivot counts are exponential but typically O(m) to O(m log n).
- Branch-and-bound is exponential in the number of integer variables in the worst case (O(2^k) nodes); cuts and good branching reduce the node count in practice.
- Each GMI cut costs O(n + m) per tableau row.

## Usage

```python
from branch_and_cut import SolverOptions, knapsack, solve_mip

mip = knapsack(30, seed=1, multi=3, tightness=0.4)
res = solve_mip(mip, SolverOptions(gomory_rounds=4, cover_rounds=3, branching="pseudocost"))
print(res.status, res.obj, res.nodes, res.cuts_added)
```

Run the tests with `python -m pytest -q` (46 tests, including comparisons against brute force, dynamic programming, vertex enumeration and permutation enumeration) and the demo with `python examples/demo.py`.

## Results

Output of `examples/demo.py` (rounding heuristic disabled so tree sizes are comparable; root0/root1 are root bounds before/after cuts):

```
instance                config               obj     root0     root1 cuts  nodes    lps    sec
knapsack-30-3-s1        plain            -561.00   -567.00   -567.00    0    740    740   0.88
knapsack-30-3-s1        gomory           -561.00   -567.00   -566.00   17    589    593   0.76
knapsack-30-3-s1        cover+gomory     -561.00   -567.00   -566.00   25   1158   1162   1.54
setcover-40x30-s3       plain              62.00     62.00     62.00    0      1      1   0.00
setcover-40x30-s3       gomory             62.00     62.00     62.00    0      1      1   0.00
setcover-40x30-s3       cover+gomory       62.00     62.00     62.00    0      1      1   0.00
facloc-6x10-s2          plain              34.96     34.96     34.96    0      1      1   0.01
facloc-6x10-s2          gomory             34.96     34.96     34.96    0      1      1   0.01
facloc-6x10-s2          cover+gomory       34.96     34.96     34.96    0      1      1   0.01
assign-6-s4             plain              68.00     60.00     60.00    0      9      9   0.02
assign-6-s4             gomory             68.00     60.00     68.00    6      1      2   0.00
assign-6-s4             cover+gomory       68.00     60.00     68.00    7      1      2   0.00

Branching rules on knapsack-30-3-s1
first_fractional   obj=  -561.00 nodes= 1032 lp_solves= 1032 time=1.16s
most_fractional    obj=  -561.00 nodes=  740 lp_solves=  740 time=0.88s
pseudocost         obj=  -561.00 nodes=  678 lp_solves=  678 time=0.86s
strong             obj=  -561.00 nodes=  324 lp_solves= 2130 time=2.56s
```

Observations: on the assignment problem with a side budget, four Gomory rounds lift the root bound from 60 to the optimum 68 and close the instance at the root (9 nodes down to 1). On the 3-constraint knapsack the cuts only move the root bound by one unit, and Gomory cuts reduce the node count from 740 to 589 while the added cover cuts did not help. The set-cover and facility-location instances happen to have integral LP relaxations, so they are solved at the root regardless of configuration. Strong branching halves the tree compared with most-fractional branching but costs more LP solves overall.

## References

- R. E. Gomory, "An algorithm for integer solutions to linear programs", 1958.
- A. H. Land and A. G. Doig, "An automatic method of solving discrete programming problems", Econometrica, 1960.
- M. Padberg, T. Van Roy, L. Wolsey, "Valid linear inequalities for fixed charge problems", 1985 (cover inequalities).
- T. Achterberg, T. Koch, A. Martin, "Branching rules revisited", Operations Research Letters, 2005.
- G. Cornuéjols, "Valid inequalities for mixed integer linear programs", Mathematical Programming, 2008.
- L. Wolsey, *Integer Programming*, Wiley, 1998.

---

**Author:** Dr. Kainat Rizwan
