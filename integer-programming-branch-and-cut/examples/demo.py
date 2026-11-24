"""Compare plain branch-and-bound with cut-strengthened and different branching rules."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from branch_and_cut import (SolverOptions, assignment_side, facility_location, knapsack,
                            set_cover, solve_mip)

INSTANCES = [
    knapsack(30, seed=1, multi=3, tightness=0.4),
    set_cover(40, 30, seed=3, density=0.08),
    facility_location(6, 10, seed=2),
    assignment_side(6, seed=4),
]
CONFIGS = {
    "plain": SolverOptions(rounding_heuristic=False),
    "gomory": SolverOptions(gomory_rounds=4, rounding_heuristic=False),
    "cover+gomory": SolverOptions(gomory_rounds=4, cover_rounds=3, rounding_heuristic=False),
}

print(f"{'instance':24s}{'config':14s}{'obj':>10s}{'root0':>10s}{'root1':>10s}{'cuts':>5s}{'nodes':>7s}{'lps':>7s}{'sec':>7s}")
for mip in INSTANCES:
    for name, opt in CONFIGS.items():
        r = solve_mip(mip, opt)
        print(f"{mip.name:24s}{name:14s}{r.obj:10.2f}{r.root_bound_before:10.2f}{r.root_bound_after:10.2f}"
              f"{r.cuts_added:5d}{r.nodes:7d}{r.lp_solves:7d}{r.seconds:7.2f}")

print("\nBranching rules on", INSTANCES[0].name)
for rule in ["first_fractional", "most_fractional", "pseudocost", "strong"]:
    r = solve_mip(INSTANCES[0], SolverOptions(branching=rule, rounding_heuristic=False))
    print(f"{rule:18s} obj={r.obj:9.2f} nodes={r.nodes:5d} lp_solves={r.lp_solves:5d} time={r.seconds:.2f}s")
