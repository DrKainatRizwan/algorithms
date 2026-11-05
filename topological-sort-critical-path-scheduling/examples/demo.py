"""Demo: ordering, CPM, PERT risk and resource-constrained scheduling."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from project_scheduler import *
from project_scheduler.cpm import critical_paths
from project_scheduler.pert import criticality_index


def main() -> None:
    acts = [Activity("Design", 5), Activity("Backend", 8, ("Design",)), Activity("Frontend", 6, ("Design",)),
            Activity("API docs", 2, ("Backend",)), Activity("Integration", 4, ("Backend", "Frontend")),
            Activity("Testing", 5, ("Integration",)), Activity("Release", 1, ("Testing", "API docs"))]
    r = critical_path_method(acts)
    print("== CPM ==")
    for a in r.order:
        print(f"{a:12s} ES={r.es[a]:5.1f} EF={r.ef[a]:5.1f} float={r.total_float[a]:4.1f}")
    print("duration:", r.duration, "critical paths:", critical_paths(acts))

    g = DiGraph([(p, a.name) for a in acts for p in a.predecessors], [a.name for a in acts])
    print("topological orders:", count_orders(g), "| lexicographic:", kahn_order(g, key=str))

    print("\n== PERT ==")
    pa = [PertActivity("a", 2, 4, 9), PertActivity("b", 3, 5, 10, ("a",)),
          PertActivity("c", 4, 6, 11, ("a",)), PertActivity("d", 1, 2, 4, ("b", "c"))]
    p = pert_analysis(pa)
    sim = monte_carlo_duration(pa, 20000, seed=1)
    print(f"analytic E={p.expected:.2f} sd={p.std:.2f} P(<=15)={p.prob_complete_by(15):.3f}")
    print(f"simulated mean={sim.mean():.2f} sd={sim.std():.2f} P(<=15)={(sim <= 15).mean():.3f}")
    print("criticality:", criticality_index(pa, 1000, seed=2))

    print("\n== RCPSP (60 activities, 2 resources) ==")
    big = random_project(60, seed=11, resources=("crew", "crane"), edge_prob=0.15)
    res = [Resource("crew", 5), Resource("crane", 3)]
    print(f"CPM bound {critical_path_method(big).duration}, resource-aware lower bound {resource_lower_bound(big, res):.1f}")
    for rule in ("lft", "mslk", "spt", "grd", "mts"):
        row = []
        for name, fn in (("serial", schedule_serial), ("parallel", schedule_parallel)):
            t = time.perf_counter()
            st = fn(big, res, rule)
            ms = max(st[a.name] + a.duration for a in big)
            assert not validate_schedule(big, res, st)
            row.append(f"{name}={ms:.0f} ({(time.perf_counter()-t)*1000:.0f}ms)")
        print(f"{rule:5s}", "  ".join(row))


if __name__ == "__main__":
    main()
