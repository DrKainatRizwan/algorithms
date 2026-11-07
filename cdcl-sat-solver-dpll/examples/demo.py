"""Demo: CDCL vs DPLL on random, structured and unsatisfiable formulas."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sat_solver import DPLLSolver, check_rup_proof, evaluate, generators as gen, solve


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def row(name, cnf, dpll=True):
    c, tc = timed(lambda: solve(cnf, seed=0))
    line = f"{name:<22}{cnf.num_vars:>5}{cnf.num_clauses:>7}  {c.status:<6}"
    line += f" CDCL {tc*1000:8.1f}ms {c.stats.conflicts:>6} confl"
    if dpll:
        d, td = timed(lambda: DPLLSolver(cnf, "jw", max_decisions=300000).solve())
        line += f" | DPLL {td*1000:8.1f}ms {d.decisions:>7} dec ({d.status})"
    print(line)
    if c.status == "SAT":
        assert evaluate(cnf, c.model)


print("== Benchmarks ==")
row("php-5", gen.pigeonhole(5))
row("php-7", gen.pigeonhole(7))
row("parity-contra-12", gen.parity_contradiction(12))
row("queens-10", gen.n_queens(10))
row("3sat n=40 r=4.26", gen.random_ksat(40, 170, 3, 1))
row("3sat n=60 r=4.26", gen.random_ksat(60, 256, 3, 2))
row("planted n=150 r=4.0", gen.planted_ksat(150, 600, 3, 3)[0], dpll=False)
row("random n=200 r=4.26", gen.random_ksat(200, 852, 3, 4), dpll=False)

print("\n== Colouring a random graph (60 nodes, p=0.12) ==")
edges = gen.random_graph(60, 0.12, seed=5)
for k in (2, 3, 4, 5):
    cnf = gen.graph_coloring(edges, 60, k)
    r, t = timed(lambda: solve(cnf))
    print(f"k={k}: {r.status:<5} conflicts={r.stats.conflicts:<5} {t*1000:.1f}ms")

print("\n== Proof checking ==")
cnf = gen.pigeonhole(6)
r = solve(cnf, log_proof=True, reduce_first=50)
ok = check_rup_proof(cnf, r.proof)
print(f"php-6: {r.status}, {sum(1 for k,_ in r.proof if k=='a')} lemmas, "
      f"{r.stats.deleted} deletions, RUP check passed: {ok}")

print("\n== Restart policy ablation (php-7 + 3 random UNSAT-ish) ==")
for pol in ("luby", "geometric", "none"):
    tot = 0
    for cnf in (gen.pigeonhole(7), gen.random_ksat(70, 320, 3, 8), gen.random_ksat(70, 320, 3, 9)):
        tot += solve(cnf, restart=pol).stats.conflicts
    print(f"{pol:<10} total conflicts: {tot}")
