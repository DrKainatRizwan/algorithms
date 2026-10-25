"""Demo: simplex rules, certificates, sensitivity analysis and benchmarks."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from simplex_lp import (build_dual, check_farkas, check_optimality, check_unbounded_ray,
                        cost_ranging, revised_simplex, rhs_ranging, simplex_standard, solve,
                        LinearProgram)
from simplex_lp.problems import (beale_cycling_example, diet_problem, klee_minty, l1_regression,
                                 max_flow_lp, production_planning, random_feasible_lp,
                                 solve_matrix_game, transportation)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


section("Product mix: optimum, shadow prices, ranging")
lp = production_planning()
sol = solve(lp)
print("x* =", np.round(sol.x, 4), " profit =", round(sol.objective, 4))
print("shadow prices =", np.round(sol.duals, 4))
print("rhs ranging   =", [(round(a, 2), round(b, 2)) for a, b in rhs_ranging(sol)])
print("cost ranging  =", [tuple(round(v, 2) for v in r) for r in cost_ranging(sol, lp)])
rep = check_optimality(lp, sol.x, sol.duals)
print(f"KKT residuals: primal={rep.primal_violation:.1e} dual={rep.dual_violation:.1e} "
      f"compl={rep.complementarity:.1e} gap={rep.duality_gap:.1e}")
dual_sol = solve(build_dual(lp))
print("explicit dual LP objective (min form) =", round(dual_sol.objective, 6))

section("Beale's cycling example")
c, A, b, basis = beale_cycling_example()
for rule in ("dantzig", "bland", "hybrid"):
    r = simplex_standard(c, A, b, rule=rule, basis=basis)
    print(f"{rule:8s} status={r.status:8s} iterations={r.iterations:3d} objective={r.objective}")

section("Klee-Minty cube: pivots needed")
print(" n | dantzig | bland | 2^n - 1")
for n in range(3, 9):
    d = solve(klee_minty(n), rule="dantzig").iterations
    bl = solve(klee_minty(n), rule="bland").iterations
    print(f"{n:2d} | {d:7d} | {bl:5d} | {2 ** n - 1}")

section("Certificates")
infeasible = LinearProgram([1, 1], A_ub=[[1, 1], [-1, -1]], b_ub=[1, -3])
s = solve(infeasible)
print("infeasible LP ->", s.status, "| Farkas certificate valid:", check_farkas(s))
unb = LinearProgram([-1, -1], A_ub=[[1, -1]], b_ub=[2])
s = solve(unb)
print("unbounded LP  ->", s.status, "| ray", np.round(s.ray, 3), "valid:", check_unbounded_ray(unb, s.ray))

section("Applications")
tp = transportation([20, 30], [10, 25, 15], [[8, 6, 10], [9, 12, 13]])
ts = solve(tp)
print("transportation cost =", round(ts.objective, 3), "plan =", np.round(ts.x.reshape(2, 3), 2).tolist())
edges = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12), (3, 2, 9),
         (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
print("max flow (CLRS network) =", round(solve(max_flow_lp(6, edges, 0, 5)).objective, 3))
v, p, q = solve_matrix_game([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
print("rock-paper-scissors value =", round(v, 6), "strategy =", np.round(p, 3))
v, p, q = solve_matrix_game([[3, -1], [-2, 4]])
print("2x2 game value =", round(v, 4), "row =", np.round(p, 3), "col =", np.round(q, 3))
dsol = solve(diet_problem())
print("diet cost =", round(dsol.objective, 4), "servings =", np.round(dsol.x, 3))
rng = np.random.default_rng(0)
X = np.column_stack([np.ones(40), np.linspace(0, 1, 40)])
y = 1.0 + 2.0 * X[:, 1] + rng.normal(0, 0.02, 40)
y[[3, 20]] += 30
print("L1 fit with outliers (true 1, 2) =", np.round(l1_regression(X, y), 3),
      "| OLS =", np.round(np.linalg.lstsq(X, y, rcond=None)[0], 3))

section("Benchmark: random LPs (m x n), tableau vs revised, mean of 5 seeds")
print("  size  | tableau ms (iters) | revised ms (iters) | objective agreement")
for m, n in [(20, 15), (40, 30), (80, 60), (120, 100)]:
    tt = tr = 0.0
    it_t = it_r = 0
    diff = 0.0
    for seed in range(5):
        lp = random_feasible_lp(m, n, seed)
        t0 = time.perf_counter(); a = solve(lp, "tableau", "hybrid"); tt += time.perf_counter() - t0
        t0 = time.perf_counter(); b_ = solve(lp, "revised", "hybrid"); tr += time.perf_counter() - t0
        it_t += a.iterations; it_r += b_.iterations
        diff = max(diff, abs(a.objective - b_.objective))
    print(f"{m:3d}x{n:<3d} | {tt / 5 * 1e3:8.1f} ({it_t // 5:4d})    | {tr / 5 * 1e3:8.1f} ({it_r // 5:4d})    | {diff:.1e}")
