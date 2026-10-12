# Raft Consensus Simulator with Leader Election and Log Replication

## Overview

A deterministic, discrete-event simulator of the Raft consensus protocol written from scratch with the standard library only. A cluster of Raft servers runs on a simulated network with random delay, message loss, partitions and node crashes. A runtime checker validates Raft's safety properties (election safety, log matching, leader completeness, state machine safety) on every simulated millisecond, so fault-injection scenarios double as correctness tests.

## How it works

- **Leader election.** Followers that hear nothing for a randomised timeout become candidates, increment their term, vote for themselves and send `RequestVote`. A vote is granted at most once per term and only to a candidate whose log is at least as up to date (compared by last term, then last index). A majority of votes makes a leader; the leader appends a no-op so entries from earlier terms can be committed.
- **Log replication.** The leader sends `AppendEntries` carrying the entries after `next_index[peer]` along with a consistency check on `(prev_index, prev_term)`. Followers truncate conflicting suffixes and append. On rejection the follower returns a conflict hint (first index of the conflicting term), letting the leader back up in one round trip per term rather than per entry.
- **Commit rule.** The leader advances `commit_index` to the highest index stored on a majority *whose term equals the current term*; followers learn it through `leader_commit`.
- **State machine.** Committed entries are applied in order to a key-value store (`set` / `del`).
- **Faults.** `Cluster.crash/restart` drops volatile state while keeping `current_term`, `voted_for` and the log; `Cluster.partition/heal` cut and restore links; the network can drop a fraction of messages.

Modules: `messages` (RPC types), `node` (server logic), `network` (seeded transport), `cluster` (driver and fault injection), `invariants` (safety checker), `stats` (latency experiments).

## Complexity

- Per simulated step: O(n) node ticks plus delivery of due messages (heap, O(log m) each, m = messages in flight).
- Replication: one `AppendEntries` round trip commits a batch; messages per committed batch are O(n). Log catch-up is O(number of conflicting terms) round trips with the hint optimisation.
- The invariant checker's log-matching test is O(n^2 · L) per step in the worst case (L = log length), which is acceptable for small clusters.
- Space: O(n · L) for logs plus O(m) for in-flight messages.

## Usage

```python
from raft_sim import Cluster

c = Cluster(5, seed=3)
leader = c.wait_for_leader()
_, idx = c.submit(("set", "x", 1))
c.wait_committed(idx)

c.crash(leader.id)                       # failover
c.run_until(lambda cl: cl.stable_leader() is not None)
c.partition([0, 1], [2, 3, 4])           # network split
c.heal()
```

Run `python -m pytest -q` for the tests and `python examples/demo.py` for the demo.

## Results

Output of `examples/demo.py` (5 nodes, delay 2-15 ms, heartbeat 50 ms):

- Cold-start election with seed 3 finishes in 173 ms; a leader crash is repaired in 175 ms.
- A leader isolated with a minority keeps an uncommitted entry; after healing, that entry is discarded and all logs converge on the majority's history.
- With 20% message loss and repeated leader crashes, no invariant violation was raised over 4062 simulated ms.

Election latency over 40 seeds (ms), for different election timeout ranges:

| Timeout range | Mean | Median | p95 | Max | Failover mean |
|---------------|-----:|-------:|----:|----:|--------------:|
| 150-155 | 794 | 626 | 1997 | 2916 | 1037 |
| 150-200 | 194 | 180 | 353 | 380 | 220 |
| 150-300 | 202 | 197 | 253 | 264 | 203 |
| 300-600 | 383 | 369 | 484 | 506 | 383 |

A narrow timeout window causes repeated split votes, while a wide window pays a longer expected wait; this matches the trade-off discussed in the Raft paper.

## References

- D. Ongaro and J. Ousterhout, "In Search of an Understandable Consensus Algorithm", USENIX ATC 2014.
- D. Ongaro, "Consensus: Bridging Theory and Practice", PhD thesis, Stanford University, 2014.
- L. Lamport, "The Part-Time Parliament", ACM TOCS 16(2), 1998.

---

**Author:** Dr. Kainat Rizwan
