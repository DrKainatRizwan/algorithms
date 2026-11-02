"""Demo: dissemination speed, rumor residue, push-sum and anti-entropy."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from gossip_sim import (Replica, barabasi_albert, complete_graph, erdos_renyi,
                        mean_field_rounds, push_rounds_estimate, push_sum,
                        ring_lattice, rumor_mongering, rumor_residue,
                        run_anti_entropy, run_trials, watts_strogatz)

print("== Rounds to inform all nodes (complete graph, 15 trials) ==")
print(f"{'n':>6} {'push':>7} {'pull':>7} {'push-pull':>10} {'theory(push)':>13} {'MF push-pull':>13}")
for n in (128, 512, 1024):
    g = complete_graph(n)
    row = [run_trials(g, m, 15, seed=1).mean_rounds for m in ("push", "pull", "push-pull")]
    print(f"{n:>6} {row[0]:7.2f} {row[1]:7.2f} {row[2]:10.2f} {push_rounds_estimate(n):13.2f} "
          f"{mean_field_rounds('push-pull', n):13d}")

print("\n== Topology effect (n=300, push-pull, 10 trials) ==")
r = np.random.default_rng(0)
topos = {"complete": complete_graph(300), "G(n,0.03)": erdos_renyi(300, 0.03, r),
         "small-world": watts_strogatz(300, 3, 0.1, r), "scale-free": barabasi_albert(300, 3, r),
         "ring k=2": ring_lattice(300, 2)}
for name, g in topos.items():
    s = run_trials(g, "push-pull", 10, seed=2, max_rounds=600)
    print(f"{name:>12}: rounds={s.mean_rounds:7.2f} msgs={s.mean_messages:9.0f} done={s.completion_rate:.0%}")

print("\n== Message loss (n=1000 complete, push-pull) ==")
for loss in (0.0, 0.3, 0.6):
    s = run_trials(complete_graph(1000), "push-pull", 10, seed=3, loss=loss)
    print(f"loss={loss:.1f}: rounds={s.mean_rounds:.2f} (90%: {s.mean_rounds_90:.2f})")

print("\n== Rumor mongering residue (n=2000, feedback-coin) ==")
g = complete_graph(2000)
for k in (1, 2, 3, 4):
    sim = np.mean([rumor_mongering(g, np.random.default_rng(i), k=k).residue for i in range(3)])
    print(f"k={k}: simulated={sim:.4f} theory={rumor_residue(k):.4f}")

print("\n== Push-sum averaging (n=200) ==")
x = np.random.default_rng(5).normal(50, 10, 200)
for name, gr in (("complete", complete_graph(200)), ("small-world", watts_strogatz(200, 3, 0.2, r))):
    res = push_sum(gr, x, np.random.default_rng(6), tol=1e-6, max_rounds=3000)
    print(f"{name:>12}: rounds={res.rounds} mean={x.mean():.4f} est={res.estimates[0]:.4f}")

print("\n== Anti-entropy (64 replicas, 20 keys) ==")
for mode in ("push", "pull", "push-pull"):
    reps = [Replica(i) for i in range(64)]
    for i, rep in enumerate(reps):
        rep.put(f"key{i % 20}", f"val{i}", ts=i)
    print(f"{mode:>10}: converged in {run_anti_entropy(reps, np.random.default_rng(7), mode)} rounds")
