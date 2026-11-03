import pytest
from hash_rings import *
from hash_rings.benchmark import compare

NODES = [f"n{i}" for i in range(10)]
KEYS = [f"k{i}" for i in range(5000)]


def test_fnv_known_vector():
    assert fnv1a64(b"") == 0xCBF29CE484222325
    assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C


def test_hash_deterministic_and_seeded():
    assert hash_key("x", 1) == hash_key("x", 1)
    assert hash_key("x", 1) != hash_key("x", 2)
    assert hash_key(5) == hash_key("5")
    with pytest.raises(TypeError):
        hash_key(1.5)


def test_unit_float_open_interval():
    assert 0 < unit_float(0) < 1 and 0 < unit_float(2**64 - 1) < 1


def test_ring_empty_and_duplicates():
    r = HashRing()
    with pytest.raises(LookupError):
        r.get_node("a")
    r.add_node("a")
    with pytest.raises(ValueError):
        r.add_node("a")
    with pytest.raises(KeyError):
        r.remove_node("zz")


def test_ring_incremental_equals_rebuild():
    a = HashRing(NODES[:5], 50)
    a.add_node("n5", 2.0)
    b = HashRing(NODES[:5], 50)
    b._insert("n5", 2.0)
    b._rebuild()
    assert a._points == b._points and a._owners == b._owners


def test_ring_minimal_disruption_on_removal():
    r = HashRing(NODES, 100)
    before = snapshot(r.get_node, KEYS)
    r.remove_node("n3")
    after = snapshot(r.get_node, KEYS)
    for k in KEYS:
        if before[k] != "n3":
            assert after[k] == before[k]
        else:
            assert after[k] != "n3"


def test_ring_ownership_sums_to_one_and_vnodes_improve_balance():
    lo = HashRing(NODES, 1).ownership()
    hi = HashRing(NODES, 400).ownership()
    assert abs(sum(hi.values()) - 1) < 1e-9
    assert max(hi.values()) < max(lo.values())
    assert max(hi.values()) < 0.14


def test_ring_replicas_distinct_and_primary_first():
    r = HashRing(NODES, 20)
    reps = r.get_nodes("key", 3)
    assert len(set(reps)) == 3 and reps[0] == r.get_node("key")
    assert len(r.get_nodes("key", 99)) == 10


def test_ring_weights_proportional():
    r = HashRing(["a", "b"], 500)
    r.remove_node("b")
    r.add_node("b", 3.0)
    own = r.ownership()
    assert 2.4 < own["b"] / own["a"] < 3.6


def test_rendezvous_minimal_disruption_and_replicas():
    h = RendezvousHash(NODES)
    before = snapshot(h.get_node, KEYS)
    h.remove_node("n7")
    after = snapshot(h.get_node, KEYS)
    assert all(after[k] == before[k] for k in KEYS if before[k] != "n7")
    h2 = RendezvousHash(NODES)
    assert h2.get_nodes("q", 3)[0] == h2.get_node("q")


def test_rendezvous_weighted_share():
    h = RendezvousHash(seed=3)
    for n, w in {"a": 1, "b": 2, "c": 5}.items():
        h.add_node(n, w)
    c = load_counts(h.get_node, range(30000), h.nodes)
    assert weighted_error(c, {"a": 1, "b": 2, "c": 5}) < 0.06


def test_rendezvous_better_balanced_than_small_ring():
    h = RendezvousHash(NODES)
    r = HashRing(NODES, 5)
    ph = peak_to_mean(load_counts(h.get_node, KEYS, NODES))
    pr = peak_to_mean(load_counts(r.get_node, KEYS, NODES))
    assert ph < pr


def test_jump_hash_range_and_movement():
    for k in range(200):
        assert 0 <= jump_hash(hash_key(k), 17) < 17
    assert jump_hash(5, 1) == 0
    ks = [hash_key(k) for k in range(20000)]
    moved = sum(jump_hash(k, 10) != jump_hash(k, 11) for k in ks) / len(ks)
    assert abs(moved - 1 / 11) < 0.015
    # monotone: keys only ever move to the new bucket
    assert all(jump_hash(k, 11) == 10 for k in ks if jump_hash(k, 10) != jump_hash(k, 11))


def test_maglev_table_full_and_even():
    m = Maglev(NODES, 1009)
    assert -1 not in m.table
    counts = m.slot_counts()
    assert max(counts.values()) - min(counts.values()) <= 1
    assert m.get_node("abc") == Maglev(NODES, 1009).get_node("abc")


def test_maglev_validation_and_primes():
    with pytest.raises(ValueError):
        Maglev(NODES, 1000)
    with pytest.raises(ValueError):
        Maglev([], 7)
    assert next_prime(100) == 101


def test_maglev_low_churn():
    a = Maglev(NODES, 10007)
    b = Maglev(NODES[:-1], 10007)
    moved = moved_fraction(snapshot(a.get_node, KEYS), snapshot(b.get_node, KEYS))
    assert moved < 0.3


def test_metrics_and_compare():
    assert peak_to_mean({"a": 5, "b": 5}) == 1.0
    assert coeff_variation({"a": 5, "b": 5}) == 0.0
    rows = compare(8, 2000)
    assert len(rows) == 3
    for r in rows[:2]:
        assert abs(r["moved_on_removal"] - r["ideal_moved"]) < 1e-9
