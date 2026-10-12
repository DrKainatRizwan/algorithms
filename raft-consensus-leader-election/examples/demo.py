"""Demo: election, replication, failover, partition healing and latency statistics."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raft_sim import Cluster, election_latency, failover_latency, summarize  # noqa: E402


def main() -> None:
    print("== 1. Cold start and replication (5 nodes) ==")
    c = Cluster(5, seed=3)
    ldr = c.wait_for_leader()
    print(f"leader {ldr.id} elected in term {ldr.current_term} at t={c.now} ms")
    last = 0
    for i in range(8):
        last = c.submit(("set", f"key{i}", i * i))[1]
    c.wait_committed(last)
    c.run(100)
    print(f"committed through index {last}; converged={c.converged()}; "
          f"state on node 4: {dict(list(c.nodes[4].state.items())[:4])} ...")

    print("\n== 2. Leader crash and failover ==")
    old = c.leader()
    c.crash(old.id)
    t0 = c.now
    c.run_until(lambda x: x.stable_leader() is not None)
    new = c.leader()
    print(f"node {old.id} crashed; node {new.id} took over (term {new.current_term}) "
          f"after {c.now - t0} ms")
    idx = c.submit(("set", "after_failover", True))[1]
    c.wait_committed(idx)
    c.restart(old.id)
    c.run_until(lambda x: x.converged())
    print(f"restarted node {old.id} caught up: log length {len(c.nodes[old.id].log)}")

    print("\n== 3. Partition: stale leader's uncommitted write is discarded ==")
    c = Cluster(5, seed=6)
    old = c.wait_for_leader()
    others = [i for i in c.nodes if i != old.id]
    c.partition([old.id, others[0]], others[1:])
    stale = old.propose(("set", "stale", 1), c.now)
    c.run(1500)
    new = max((c.nodes[i] for i in others[1:] if c.nodes[i].role == "leader"),
              key=lambda n: n.current_term)
    fresh = new.propose(("set", "fresh", 2), c.now)
    c.wait_committed(fresh, others[1:])
    print(f"old leader {old.id} (term {old.current_term}) stuck at commit {old.commit_index} "
          f"with uncommitted index {stale}; new leader {new.id} (term {new.current_term}) "
          f"committed {fresh}")
    c.heal()
    c.run_until(lambda x: x.converged() and x.stable_leader() is not None)
    print(f"after heal: converged={c.converged()}, 'stale' present="
          f"{'stale' in c.nodes[old.id].state}, fresh={c.nodes[old.id].state.get('fresh')}")

    print("\n== 4. Safety under 20% message loss with crashes (invariants checked every ms) ==")
    c = Cluster(5, seed=21, drop_rate=0.2)
    for r in range(8):
        c.run(500)
        c.submit(("set", f"r{r}", r))
        if r % 2 == 0 and c.leader():
            c.crash(c.leader().id)
        elif c.down:
            c.restart(next(iter(c.down)))
    for i in list(c.down):
        c.restart(i)
    c.net.drop_rate = 0.0
    ok = c.run_until(lambda x: x.converged() and x.stable_leader() is not None, 10000)
    print(f"no invariant violation in {c.now} ms; converged={ok}; "
          f"sent={c.net.sent} dropped={c.net.dropped}")

    print("\n== 5. Election latency vs timeout range (5 nodes, 40 seeds, ms) ==")
    from raft_sim import RaftConfig
    print(f"{'timeout range':>14} {'mean':>7} {'median':>7} {'p95':>7} {'max':>7}   failover mean")
    for lo, hi in [(150, 155), (150, 200), (150, 300), (300, 600)]:
        cfg = RaftConfig(election_min=lo, election_max=hi)
        s = summarize([election_latency(5, sd, cfg) for sd in range(40)])
        f = summarize([failover_latency(5, sd, cfg) for sd in range(40)])
        print(f"{lo:>6}-{hi:<7} {s['mean']:7.0f} {s['median']:7.0f} {s['p95']:7.0f} "
              f"{s['max']:7.0f}   {f['mean']:.0f}")


if __name__ == "__main__":
    main()
