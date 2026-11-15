import pytest

from crdt_toolkit import Cluster, GCounter, ORSet, RGA, PNCounter
from crdt_toolkit.workload import apply_random_ops, counter_op, orset_op, rga_op
import random


def test_gossip_converges_gcounter_with_expected_total():
    c = Cluster(GCounter, 6, seed=1)
    for i, r in enumerate(c.replicas.values()):
        r.increment(i + 1)
    res = c.run_until_converged()
    assert res.converged
    assert {r.value() for r in c.replicas.values()} == {21}


def test_converges_under_loss_duplication_and_reordering():
    rng = random.Random(2)
    c = Cluster(ORSet, 5, seed=2, drop_prob=0.3, dup_prob=0.3, max_delay=5)
    apply_random_ops(c.replicas, orset_op, 60, rng)
    assert c.run_until_converged(400).converged


def test_partition_blocks_then_heal_converges():
    c = Cluster(GCounter, 4, seed=3)
    c.net.partition([["n0", "n1"], ["n2", "n3"]])
    c.replicas["n0"].increment(5)
    c.replicas["n3"].increment(7)
    for _ in range(30):
        c.gossip_round()
    assert not c.converged()
    c.net.heal()
    assert c.run_until_converged().converged
    assert c.replicas["n1"].value() == 12


def test_pushpull_not_slower_than_push_on_average():
    def avg(mode):
        total = 0
        for seed in range(8):
            c = Cluster(GCounter, 16, seed=seed, mode=mode)
            for r in c.replicas.values():
                r.increment(1)
            total += c.run_until_converged().rounds
        return total / 8
    assert avg("pushpull") <= avg("push")


def test_text_editing_converges_via_gossip():
    rng = random.Random(4)
    c = Cluster(RGA, 4, seed=4, drop_prob=0.1)
    apply_random_ops(c.replicas, rga_op, 50, rng)
    assert c.run_until_converged(300).converged
    assert len({r.text() for r in c.replicas.values()}) == 1


def test_invalid_mode():
    with pytest.raises(ValueError):
        Cluster(GCounter, 3, mode="bogus")
