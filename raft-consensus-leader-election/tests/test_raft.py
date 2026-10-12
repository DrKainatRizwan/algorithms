import pytest

from raft_sim import (Cluster, Entry, InvariantChecker, InvariantViolation, RaftConfig,
                      RaftNode, RequestVote, AppendEntries, apply_command,
                      election_latency, failover_latency, summarize)


def test_elects_single_leader():
    c = Cluster(5, seed=1)
    ldr = c.wait_for_leader()
    assert ldr is not None
    assert sum(1 for n in c.nodes.values() if n.role == "leader") == 1


def test_single_node_cluster_self_elects():
    c = Cluster(1, seed=0)
    assert c.wait_for_leader() is not None
    idx = c.submit(("set", "a", 1))[1]
    assert c.wait_committed(idx)
    assert c.nodes[0].state == {"a": 1}


def test_replication_and_state_machine():
    c = Cluster(5, seed=2)
    c.wait_for_leader()
    last = 0
    for i in range(10):
        last = c.submit(("set", f"k{i}", i))[1]
    c.submit(("del", "k3"))
    assert c.wait_committed(last + 1)
    c.run(200)
    assert c.converged()
    assert "k3" not in c.nodes[0].state and c.nodes[4].state["k9"] == 9


def test_follower_rejects_proposals():
    c = Cluster(3, seed=3)
    ldr = c.wait_for_leader()
    follower = next(n for n in c.nodes.values() if n is not ldr)
    assert follower.propose(("set", "x", 1), c.now) is None


def test_failover_after_leader_crash_preserves_committed():
    c = Cluster(5, seed=4)
    old = c.wait_for_leader()
    idx = c.submit(("set", "x", 42))[1]
    assert c.wait_committed(idx)
    c.crash(old.id)
    assert c.run_until(lambda x: x.stable_leader() is not None)
    new = c.leader()
    assert new.id != old.id and new.current_term > old.current_term
    assert new.log[idx - 1].command == ("set", "x", 42)


def test_no_leader_without_majority():
    c = Cluster(5, seed=5)
    c.wait_for_leader()
    for i in (0, 1, 2):
        c.crash(i)
    c.run(3000)
    live_leader = c.leader()
    assert c.stable_leader() is None
    idx_before = max(n.commit_index for n in c.nodes.values())
    if live_leader:
        live_leader.propose(("set", "z", 1), c.now)
    c.run(1000)
    assert max(n.commit_index for n in c.nodes.values() if n.id not in c.down) <= idx_before


def test_minority_partition_cannot_commit_and_log_is_repaired():
    c = Cluster(5, seed=6)
    old = c.wait_for_leader()
    others = [i for i in c.nodes if i != old.id]
    minority = [old.id, others[0]]
    majority = others[1:]
    c.partition(minority, majority)
    stale_idx = old.propose(("set", "stale", 1), c.now)
    c.run(1500)
    assert old.commit_index < stale_idx
    new = max((c.nodes[i] for i in majority if c.nodes[i].role == "leader"),
              key=lambda n: n.current_term)
    idx = new.propose(("set", "fresh", 2), c.now)
    assert c.wait_committed(idx, majority)
    c.heal()
    assert c.run_until(lambda x: x.converged() and x.stable_leader() is not None)
    for n in c.nodes.values():
        assert n.state.get("fresh") == 2 and "stale" not in n.state


def test_restarted_node_catches_up():
    c = Cluster(3, seed=7)
    c.wait_for_leader()
    victim = next(i for i in c.nodes if i != c.leader().id)
    c.crash(victim)
    for i in range(5):
        last = c.submit(("set", f"k{i}", i))[1]
    assert c.wait_committed(last)
    c.restart(victim)
    assert c.run_until(lambda x: x.converged())
    assert c.nodes[victim].state["k4"] == 4


def test_persistent_state_survives_restart():
    n = RaftNode(0, [0, 1, 2], RaftConfig(), seed=0)
    n.current_term, n.voted_for = 4, 2
    n.log = [Entry(1, ("set", "a", 1))]
    n.commit_index = 1
    n.restart(100)
    assert (n.current_term, n.voted_for, len(n.log)) == (4, 2, 1)
    assert n.commit_index == 0 and n.role == "follower"


def test_vote_granted_once_per_term_and_only_if_up_to_date():
    n = RaftNode(0, [0, 1, 2], RaftConfig(), seed=0)
    n.log = [Entry(2, None)]
    n.receive(RequestVote(1, 3, 0, 0), 0)          # stale log -> refuse
    assert n.outbox[-1][1].granted is False
    n.receive(RequestVote(1, 3, 1, 2), 0)          # up to date -> grant
    assert n.outbox[-1][1].granted is True
    n.receive(RequestVote(2, 3, 5, 3), 0)          # already voted for 1
    assert n.outbox[-1][1].granted is False


def test_append_entries_truncates_conflicts():
    n = RaftNode(0, [0, 1, 2], RaftConfig(), seed=0)
    n.current_term = 2
    n.log = [Entry(1, ("set", "a", 1)), Entry(1, ("set", "b", 2)), Entry(1, ("set", "c", 3))]
    n.receive(AppendEntries(1, 3, 1, 1, (Entry(3, ("set", "d", 4)),), 0), 0)
    assert [e.term for e in n.log] == [1, 3]
    assert n.outbox[-1][1].success and n.outbox[-1][1].match_index == 2


def test_append_entries_rejects_missing_prefix_with_hint():
    n = RaftNode(0, [0, 1, 2], RaftConfig(), seed=0)
    n.receive(AppendEntries(1, 1, 5, 1, (), 0), 0)
    reply = n.outbox[-1][1]
    assert not reply.success and reply.conflict_index == 1


def test_stale_term_append_rejected():
    n = RaftNode(0, [0, 1, 2], RaftConfig(), seed=0)
    n.current_term = 5
    n.receive(AppendEntries(1, 2, 0, 0, (Entry(2, None),), 0), 0)
    assert n.log == [] and not n.outbox[-1][1].success


def test_safety_under_message_loss_and_churn():
    c = Cluster(5, seed=11, drop_rate=0.15)
    for round_ in range(6):
        c.run(400)
        c.submit(("set", f"r{round_}", round_))
        ldr = c.leader()
        if ldr and round_ % 2 == 0:
            c.crash(ldr.id)
        elif c.down:
            c.restart(next(iter(c.down)))
    c.heal()
    for i in list(c.down):
        c.restart(i)
    c.net.drop_rate = 0.0
    assert c.run_until(lambda x: x.converged() and x.stable_leader() is not None, 8000)


def test_checker_detects_double_leader():
    c = Cluster(3, seed=1, check_invariants=False)
    c.wait_for_leader()
    chk = InvariantChecker()
    a = c.leader()
    chk.check(c)
    other = next(n for n in c.nodes.values() if n is not a)
    other.role, other.current_term = "leader", a.current_term
    with pytest.raises(InvariantViolation):
        chk.check(c)


def test_determinism():
    def run():
        c = Cluster(5, seed=9, drop_rate=0.1)
        c.wait_for_leader()
        c.run(500)
        return c.now, c.leader().id, c.net.sent
    assert run() == run()


def test_apply_command_and_stats():
    s = {}
    apply_command(s, ("set", "a", 1)); apply_command(s, None); apply_command(s, ("del", "a"))
    assert s == {}
    with pytest.raises(ValueError):
        apply_command(s, ("bogus",))
    lat = [election_latency(5, sd) for sd in range(5)]
    assert all(x is not None and x < 2000 for x in lat)
    assert failover_latency(5, 0) is not None
    assert summarize([1, 2, 3, None])["failed"] == 1
