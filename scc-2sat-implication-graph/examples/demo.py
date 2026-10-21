"""Demo: SCC algorithms, 2-SAT solving and scaling."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph_sat import (TwoSat, random_digraph, planted_2sat, random_2sat,
                       tarjan_scc, kosaraju_scc, gabow_scc, ReachabilityIndex)
from graph_sat.scc import same_partition
from graph_sat.applications import kernel_graph_summary, schedule_two_slots


def timed(f, *a):
    t = time.perf_counter()
    r = f(*a)
    return r, (time.perf_counter() - t) * 1000


def main() -> None:
    print("== SCC algorithms on random digraphs ==")
    for n, m in [(20000, 30000), (20000, 60000), (50000, 100000)]:
        g = random_digraph(n, m, seed=1)
        res = {}
        for name, f in (("tarjan", tarjan_scc), ("kosaraju", kosaraju_scc), ("gabow", gabow_scc)):
            res[name] = timed(f, g)
        agree = same_partition(res["tarjan"][0], res["kosaraju"][0]) and same_partition(res["tarjan"][0], res["gabow"][0])
        s = kernel_graph_summary(g)
        print(f"n={n} m={m}: " + " ".join(f"{k}={v[1]:.0f}ms" for k, v in res.items())
              + f" | agree={agree} comps={s['components']} largest={s['largest']}")

    print("\n== Reachability index ==")
    g = random_digraph(1000, 1500, seed=2)
    idx, ms = timed(ReachabilityIndex, g)
    print(f"built in {ms:.0f}ms; 0 reaches {idx.reachable_count(0)} vertices")

    print("\n== 2-SAT ==")
    s = TwoSat(3)
    s.add_clause(1, 2); s.add_clause(-1, 3); s.add_clause(-2, -3)
    print("small:", s.solve())
    u = TwoSat(2)
    for a, b in [(1, 2), (1, -2), (-1, 2), (-1, -2)]:
        u.add_clause(a, b)
    r = u.solve()
    print("unsat conflict var:", r.conflict_var, "proof:", r.proof)

    print("\n== Phase transition (random 2-SAT, n=2000, 20 trials) ==")
    for ratio in (0.5, 0.8, 1.0, 1.2, 1.5):
        sat = sum(random_2sat(2000, int(2000 * ratio), seed=t).is_satisfiable() for t in range(20))
        print(f"  clauses/vars={ratio:.1f}: {sat}/20 satisfiable")

    print("\n== Scaling (planted, satisfiable) ==")
    for n in (10000, 50000, 100000):
        inst, _ = planted_2sat(n, 2 * n, seed=3)
        r, ms = timed(inst.solve)
        print(f"  vars={n} clauses={2*n}: sat={r.satisfiable} valid={inst.evaluate(r.assignment)} {ms:.0f}ms")

    print("\n== Scheduling as 2-colouring ==")
    print(schedule_two_slots(6, [(i, (i + 1) % 6) for i in range(6)]))


if __name__ == "__main__":
    main()
