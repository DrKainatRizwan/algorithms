# CRDT Toolkit: Vector Clocks, Convergent Data Types and Gossip Anti-Entropy

## Overview

A from-scratch library of conflict-free replicated data types (CRDTs) together with the machinery needed to study them: Lamport and vector clocks, a causal-broadcast layer, a deterministic lossy network (delay, loss, duplication, reordering, partitions) and a gossip simulator that measures how quickly replicas converge.

Included types: `GCounter`, `PNCounter`, `GSet`, `TwoPSet`, `ORSet` (add-wins), `LWWRegister`, `MVRegister`, `LWWMap` and an `RGA` sequence for collaborative text.

## How it works

* **State-based CRDTs.** The state of each type lives in a join-semilattice and `merge` is the least upper bound, so merge is commutative, associative and idempotent. Any delivery order, duplication or loss pattern therefore ends in the same state once replicas have exchanged enough information. `check_lattice_laws` verifies the three laws on sampled states.
* **OR-Set.** Every add creates a unique tag; a remove tombstones only the tags it observed, so a concurrent add wins.
* **MV-Register.** Values carry vector clocks; merge discards values dominated by another and keeps concurrent siblings.
* **RGA.** Each character has a unique `(lamport, node)` id and the id of the character it follows. Document order is a depth-first walk with siblings sorted by descending id; deletes are tombstones.
* **Causal broadcast.** A message from `s` with clock `V` is delivered once `V[s] = D[s]+1` and `V[k] <= D[k]` elsewhere; otherwise it is held back. Duplicates are dropped.
* **Gossip.** Each round every replica pushes its state to random peers (optionally with a pull reply). Expected convergence time is O(log n) rounds.

## Complexity

| Operation | Time | Space |
|---|---|---|
| Vector clock compare / merge | O(k) for k nodes | O(k) |
| GCounter / PNCounter merge | O(k) | O(k) |
| ORSet add / remove / merge | O(1) / O(tags of x) / O(total tags) | O(adds) |
| MVRegister merge | O(s²·k) for s siblings | O(s·k) |
| RGA insert / delete (index) | O(n) (document order recomputed) | O(n) incl. tombstones |
| RGA merge | O(n) | O(n) |
| Causal delivery | O(b) per receive for buffer size b | O(b) |
| Gossip round | O(n · fanout) messages | O(n · state) |

## Usage

```python
from crdt_toolkit import ORSet, RGA, Cluster, GCounter

a, b = ORSet("A"), ORSet("B")
a.add("milk"); b.merge(a); b.remove("milk"); a.add("milk")   # concurrent re-add
a.merge(b); b.merge(a)
assert a.value() == b.value() == frozenset({"milk"})

cluster = Cluster(GCounter, 16, seed=0, mode="pushpull", drop_prob=0.3)
for r in cluster.replicas.values():
    r.increment(1)
print(cluster.run_until_converged())
```

Run the demo with `python examples/demo.py` and the tests with `python -m pytest -q`.

## Results

Mean rounds to full convergence over 10 seeds for a GCounter where every replica increments once (fanout 1, delay 1-3 ticks):

| Replicas | Mode | Drop | Rounds | Messages |
|---|---|---|---|---|
| 16 | push | 0.0 | 11.7 | 187 |
| 16 | push | 0.3 | 16.9 | 270 |
| 16 | pushpull | 0.0 | 8.8 | 267 |
| 16 | pushpull | 0.3 | 11.9 | 313 |
| 32 | push | 0.0 | 14.7 | 470 |
| 32 | pushpull | 0.0 | 10.9 | 668 |
| 48 | push | 0.0 | 16.1 | 773 |
| 48 | pushpull | 0.3 | 16.9 | 1343 |

Rounds grow roughly logarithmically in cluster size; push-pull saves rounds at the cost of more messages. Chaos runs with 30% loss, 30% duplication and reordering converged for counters and for a 38-character RGA document.

## References

* Shapiro, Preguiça, Baquero, Zawirski. *A comprehensive study of Convergent and Commutative Replicated Data Types.* INRIA RR-7506, 2011.
* Roh, Jeon, Kim, Lee. *Replicated abstract data types: Building blocks for collaborative applications.* JPDC 71(3), 2011 (RGA).
* Lamport. *Time, clocks, and the ordering of events in a distributed system.* CACM 21(7), 1978.
* Fidge (1988) and Mattern (1989), vector clocks.
* Demers et al. *Epidemic algorithms for replicated database maintenance.* PODC 1987.
* Birman et al. *Lightweight causal and atomic group multicast.* ACM TOCS 9(3), 1991.

---

**Author:** Dr. Kainat Rizwan
