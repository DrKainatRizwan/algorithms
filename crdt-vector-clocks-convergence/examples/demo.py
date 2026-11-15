"""Demo: convergence behaviour of CRDTs under gossip with an unreliable network."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crdt_toolkit import (RGA, CausalBroadcaster, Cluster, GCounter, MVRegister, ORSet,
                          VectorClock, check_lattice_laws)
from crdt_toolkit.workload import apply_random_ops, orset_op, pncounter_op, rga_op
from crdt_toolkit import PNCounter


def section(t):
    print(f"\n== {t} ==")


section("Vector clocks")
a = VectorClock().incremented("x")
b = a.incremented("y")
c = a.incremented("z")
print("a vs b:", a.compare(b), "| b vs c:", b.compare(c), "| join(b,c):", b.merged(c))

section("Add-wins OR-Set vs concurrent remove")
s1, s2 = ORSet("A"), ORSet("B")
s1.add("milk"); s2.merge(s1); s2.remove("milk"); s1.add("milk")
s1.merge(s2); s2.merge(s1)
print("replica A:", sorted(s1.value()), "replica B:", sorted(s2.value()))

section("Multi-value register keeps conflicts")
m1, m2 = MVRegister("A"), MVRegister("B")
m1.assign("red"); m2.assign("blue"); m1.merge(m2)
print("siblings:", sorted(m1.value()))

section("Collaborative text (RGA)")
d1, d2 = RGA("alice"), RGA("bob")
d1.insert_text(0, "CRDT")
d2.merge(d1)
d1.insert_text(4, "s"); d2.insert_text(0, "Fun ")
d1.merge(d2); d2.merge(d1)
print(repr(d1.text()), "==", repr(d2.text()), d1 == d2)

section("Causal broadcast")
n = {k: CausalBroadcaster(k) for k in "abc"}
m_q = n["a"].broadcast("Q?"); n["b"].receive(m_q); m_a = n["b"].broadcast("A!")
print("c receives answer first ->", n["c"].receive(m_a), "| then question ->",
      [m.payload for m in n["c"].receive(m_q)])

section("Lattice laws on random OR-Set states")
rng = random.Random(0)
reps = {k: ORSet(k) for k in "abc"}
samples = []
for _ in range(3):
    apply_random_ops(reps, orset_op, 5, rng)
    samples += [r.copy() for r in reps.values()]
print(check_lattice_laws(samples))

section("Rounds to converge (mean of 10 seeds, 16-48 replicas)")
print(f"{'replicas':>8} {'mode':>9} {'drop':>5} {'rounds':>7} {'msgs':>7}")
for nrep in (16, 32, 48):
    for mode in ("push", "pushpull"):
        for drop in (0.0, 0.3):
            R = M = 0
            for seed in range(10):
                cl = Cluster(GCounter, nrep, seed=seed, mode=mode, drop_prob=drop)
                for r in cl.replicas.values():
                    r.increment(1)
                res = cl.run_until_converged(300)
                assert res.converged and cl.replicas["n0"].value() == nrep
                R += res.rounds; M += res.messages
            print(f"{nrep:>8} {mode:>9} {drop:>5} {R/10:>7.1f} {M/10:>7.0f}")

section("Chaos run: 5 replicas, 30% loss, 30% duplication, reordering")
rng = random.Random(1)
cl = Cluster(PNCounter, 5, seed=1, drop_prob=0.3, dup_prob=0.3, max_delay=5)
apply_random_ops(cl.replicas, pncounter_op, 100, rng)
res = cl.run_until_converged(400)
print(res, "value =", cl.replicas["n0"].value())
cl = Cluster(RGA, 5, seed=2, drop_prob=0.3, dup_prob=0.3, max_delay=5)
apply_random_ops(cl.replicas, rga_op, 80, rng)
res = cl.run_until_converged(400)
print(res, "document:", repr(cl.replicas["n0"].text()))
