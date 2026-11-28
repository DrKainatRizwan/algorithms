import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from kademlia_dht import (DHTNode, KBucket, Network, RoutingTable, bucket_index,
                          build_network, churn_experiment, fit_log2_slope,
                          hash_to_id, lookup_stats, xor_distance)
from kademlia_dht.ids import random_id_in_bucket, random_unique_ids


def test_xor_metric_properties():
    rng = random.Random(0)
    for _ in range(50):
        a, b, c = (rng.getrandbits(32) for _ in range(3))
        assert xor_distance(a, a) == 0
        assert xor_distance(a, b) == xor_distance(b, a)
        assert xor_distance(a, c) <= xor_distance(a, b) + xor_distance(b, c)


def test_xor_unidirectional():
    # For a point and a distance there is exactly one point at that distance.
    a, d = 0b1011, 0b0110
    assert (a ^ d) ^ a == d


def test_bucket_index_and_random_in_bucket():
    assert bucket_index(0, 1) == 0
    assert bucket_index(0, 0b1000) == 3
    with pytest.raises(ValueError):
        bucket_index(5, 5)
    rng = random.Random(1)
    for i in range(32):
        assert bucket_index(12345, random_id_in_bucket(12345, i, rng)) == i


def test_hash_deterministic_and_bounded():
    assert hash_to_id("x", 16) == hash_to_id(b"x", 16)
    assert 0 <= hash_to_id("hello", 20) < 1 << 20
    assert hash_to_id("a") != hash_to_id("b")


def test_unique_ids():
    ids = random_unique_ids(100, 8, random.Random(0))
    assert len(set(ids)) == 100
    with pytest.raises(ValueError):
        random_unique_ids(300, 8, random.Random(0))


def test_kbucket_lru_and_cache():
    b = KBucket(2)
    assert b.update(1) is None and b.update(2) is None
    assert b.update(1) is None and b.entries == [2, 1]
    assert b.update(3) == 2          # full: oldest returned for ping
    assert b.cache == [3]
    b.remove(2)                      # dead -> replacement promoted
    assert b.entries == [1, 3] and b.cache == []


def test_routing_table_closest_sorted():
    rt = RoutingTable(0, 8, 4)
    for n in (1, 2, 3, 100, 200, 255):
        rt.update(n)
    got = rt.closest(101, 3)
    assert got == sorted(got, key=lambda n: n ^ 101)
    assert got[0] == 100
    rt.update(0)  # self ignored
    assert rt.size() == 6
    assert rt.contains(3) and not rt.contains(0)


def test_bucket_capacity_respected():
    net, nodes = build_network(120, k=4, seed=2)
    for n in nodes:
        assert all(len(b.entries) <= 4 for b in n.table.buckets)


def test_lookup_finds_true_closest():
    net, nodes = build_network(150, seed=4)
    st = lookup_stats(net, nodes, trials=60, seed=1)
    assert st.accuracy >= 0.97


def test_lookup_rounds_logarithmic():
    sizes, hops = [], []
    for n in (50, 200, 800):
        net, nodes = build_network(n, seed=3)
        sizes.append(n)
        hops.append(lookup_stats(net, nodes, 60, seed=2).mean_rounds)
    assert hops[-1] < 8
    a, _ = fit_log2_slope(sizes, hops)
    assert 0 < a < 1


def test_fit_slope_exact():
    a, b = fit_log2_slope([2, 4, 8], [1.0, 2.0, 3.0])
    assert a == pytest.approx(1.0) and b == pytest.approx(0.0)


def test_put_get_roundtrip():
    net, nodes = build_network(100, seed=5)
    assert nodes[3].put("alpha", "A") >= 1
    res = nodes[77].get("alpha")
    assert res.found and res.value == "A"
    assert not nodes[10].get("missing").found


def test_replication_factor():
    net, nodes = build_network(100, k=8, seed=6)
    nodes[0].put("beta", 42)
    kid = hash_to_id("beta")
    holders = [n for n in nodes if kid in n.storage]
    assert len(holders) >= 8
    truth = set(net.true_closest(kid, 8))
    assert len(truth & {h.id for h in holders}) >= 7


def test_dead_node_removed_from_table():
    net = Network(seed=0)
    a, b = DHTNode(1, net, 8), DHTNode(2, net, 8)
    a._touch(2)
    assert a.table.contains(2)
    net.kill(2)
    a.lookup(200)
    assert not a.table.contains(2)
    assert net.timeouts >= 1


def test_stale_entry_evicted_when_dead():
    net = Network(seed=0)
    me = DHTNode(0, net, 8, k=2)
    others = [DHTNode(n, net, 8) for n in (128, 129, 130)]  # all in bucket 7
    me._touch(128); me._touch(129)
    net.kill(128)
    me._touch(130)
    assert not me.table.contains(128) and me.table.contains(130)


def test_live_stale_entry_kept():
    net = Network(seed=0)
    me = DHTNode(0, net, 8, k=2)
    for n in (128, 129, 130):
        DHTNode(n, net, 8)
    me._touch(128); me._touch(129); me._touch(130)
    assert me.table.contains(128) and me.table.contains(129)
    assert not me.table.contains(130)


def test_churn_survival_and_k_effect():
    low = churn_experiment(150, 60, 0.8, seed=2, k=4)
    high = churn_experiment(150, 60, 0.8, seed=2, k=16)
    assert high >= low
    assert churn_experiment(150, 60, 0.3, seed=2) >= 0.95


def test_republish_restores_replicas():
    net, nodes = build_network(100, seed=8)
    nodes[0].put("gamma", 1)
    kid = hash_to_id("gamma")
    holders = [n for n in nodes if kid in n.storage]
    for h in holders[: len(holders) // 2]:
        net.kill(h.id)
    survivor = next(n for n in holders if n.id in net.online)
    survivor.republish()
    now = [n for n in nodes if kid in n.storage and n.id in net.online]
    assert len(now) >= 6


def test_deterministic():
    r1 = lookup_stats(*build_network(80, seed=9), 30, seed=1)
    r2 = lookup_stats(*build_network(80, seed=9), 30, seed=1)
    assert r1 == r2


def test_packet_loss_tolerated():
    net, nodes = build_network(100, seed=3, loss=0.0)
    nodes[0].put("delta", "d")
    net.loss = 0.1
    ok = sum(nodes[i].get("delta").found for i in range(1, 30))
    assert ok >= 25
