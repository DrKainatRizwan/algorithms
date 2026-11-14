import random

import pytest

from open_addressing import (
    CuckooTable,
    HopscotchTable,
    IdentityHash,
    LinearProbingTable,
    MultiplyShift,
    RobinHoodTable,
    TabulationHash,
    key_to_int,
)
from open_addressing.benchmark import (
    churn,
    load_sweep,
    make_workload,
    probe_profile,
    random_keys,
    strided_keys,
)

FACTORIES = {
    "linear": lambda: LinearProbingTable(8, seed=1),
    "robin": lambda: RobinHoodTable(8, seed=1),
    "cuckoo1": lambda: CuckooTable(8, seed=1),
    "cuckoo4": lambda: CuckooTable(8, bucket_size=4, seed=1),
    "hopscotch": lambda: HopscotchTable(8, neighborhood=16, seed=1),
}
ALL = list(FACTORIES)


@pytest.fixture(params=ALL)
def table(request):
    return FACTORIES[request.param]()


def test_basic_mapping(table):
    table["a"] = 1
    table[42] = "x"
    table[(1, 2)] = [3]
    assert table["a"] == 1 and table[42] == "x" and table[(1, 2)] == [3]
    assert len(table) == 3 and "a" in table and "zzz" not in table
    with pytest.raises(KeyError):
        table["zzz"]
    with pytest.raises(KeyError):
        del table["zzz"]
    table.check_invariants()


def test_overwrite_does_not_grow(table):
    for _ in range(50):
        table["k"] = random.random()
    assert len(table) == 1
    assert list(table) == ["k"]


def test_growth_preserves_contents(table):
    keys = random_keys(3000, seed=3)
    for k in keys:
        table[k] = k * 2
    assert len(table) == 3000
    assert table.resizes > 0
    assert all(table[k] == 2 * k for k in keys)
    assert table.load_factor <= table.max_load + 1e-9
    table.check_invariants()


def test_model_based_random_ops(table):
    rng = random.Random(11)
    ref: dict[int, int] = {}
    for step in range(6000):
        k = rng.randrange(400)
        op = rng.random()
        if op < 0.5:
            table[k] = step
            ref[k] = step
        elif op < 0.8:
            if k in ref:
                del table[k]
                del ref[k]
            else:
                with pytest.raises(KeyError):
                    del table[k]
        else:
            assert table.get(k) == ref.get(k)
        if step % 500 == 0:
            table.check_invariants()
    assert dict(table.items()) == ref
    table.check_invariants()


def test_delete_all_then_reuse(table):
    keys = random_keys(500, seed=5)
    for k in keys:
        table[k] = 1
    for k in keys:
        del table[k]
    assert len(table) == 0 and list(table) == []
    for k in keys[:100]:
        table[k] = 2
    assert len(table) == 100 and table[keys[0]] == 2
    table.check_invariants()


def test_stats_recorded(table):
    for k in range(100):
        table[k] = k
    table.stats.reset()
    for k in range(100):
        table[k]
    assert 200 not in table
    assert table.stats.hit_count == 100 and table.stats.miss_count == 1
    assert table.stats.mean_hit >= 1.0 and table.stats.mean_miss >= 1.0


def test_invalid_parameters():
    with pytest.raises(ValueError):
        LinearProbingTable(max_load=1.0)
    with pytest.raises(ValueError):
        CuckooTable(bucket_size=3)
    with pytest.raises(ValueError):
        HopscotchTable(neighborhood=1)


def test_linear_probing_tombstone_reuse_and_cleanup():
    t = LinearProbingTable(64, seed=2)
    for k in range(40):
        t[k] = k
    for k in range(40):
        del t[k]
    assert t._tombs == 40
    for k in range(1000, 1200):  # forces tombstone-clearing rebuilds
        t[k] = k
        del t[k]
    t.check_invariants()
    assert t.capacity <= 128 and len(t) == 0


def test_robin_hood_variance_below_linear_probing():
    keys = random_keys(1800, seed=8)
    lin = LinearProbingTable(2048, max_load=0.95, seed=4)
    rh = RobinHoodTable(2048, max_load=0.95, seed=4)
    for k in keys:
        lin[k] = rh[k] = 1
    assert lin.capacity == rh.capacity == 2048
    lin_hits = [lin.probe_count(k) for k in keys]
    var = lambda xs: sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)
    assert var([d + 1 for d in rh.displacements()]) < var(lin_hits)
    assert rh.max_displacement() < max(lin_hits)
    assert rh.displacement_variance() > 0


def test_robin_hood_early_termination_for_misses():
    keys, absent = make_workload("random", 1700, seed=2)
    lin = LinearProbingTable(2048, max_load=0.95, seed=6)
    rh = RobinHoodTable(2048, max_load=0.95, seed=6)
    for k in keys:
        lin[k] = rh[k] = 1
    p_lin = probe_profile(lin, keys, absent)
    p_rh = probe_profile(rh, keys, absent)
    assert p_rh.mean_miss < p_lin.mean_miss


def test_cuckoo_lookup_bound_and_bucket_size():
    for bs in (1, 2, 4):
        t = CuckooTable(256, bucket_size=bs, seed=9)
        keys = random_keys(1000, seed=bs)
        for k in keys:
            t[k] = k
        assert max(t.probe_count(k) for k in keys) <= 2 * bs
        assert max(t.probe_count(k + 1) for k in keys) <= 2 * bs
        t.check_invariants()


def test_cuckoo_larger_buckets_reach_higher_load():
    keys = random_keys(1900, seed=21)
    small = CuckooTable(2048, max_load=0.95, seed=3)
    big = CuckooTable(2048, bucket_size=4, max_load=0.95, seed=3)
    for k in keys:
        small[k] = big[k] = 1
    assert small.capacity > 2048  # one-slot buckets cannot sustain 93% load
    assert big.capacity == 2048
    assert big.load_factor > 0.9


def test_cuckoo_forced_rehash_keeps_all_keys():
    t = CuckooTable(16, max_kicks=2, max_load=0.9, seed=5)
    keys = random_keys(60, seed=1)
    for k in keys:
        t[k] = -k
    assert t.rehashes > 0
    assert all(t[k] == -k for k in keys)
    t.check_invariants()


def test_hopscotch_neighbourhood_property_and_hops():
    t = HopscotchTable(1024, neighborhood=8, max_load=0.95, seed=2)
    keys = random_keys(930, seed=4)
    for k in keys:
        t[k] = k
    assert t.hops > 0
    mask = t.capacity - 1
    for k in keys:
        idx, _ = t._locate(k)
        assert (idx - t._slot(k)) & mask < t.H
    t.check_invariants()


def test_hopscotch_lookup_probes_bounded_by_neighbourhood():
    t = HopscotchTable(512, neighborhood=16, seed=7)
    keys = random_keys(400, seed=7)
    for k in keys:
        t[k] = k
    assert max(t.probe_count(k) for k in keys) <= 16


def test_key_to_int_deterministic():
    assert key_to_int("hello") == key_to_int("hello")
    assert key_to_int("hello") != key_to_int("hellp")
    assert key_to_int(-1) == (1 << 64) - 1
    assert key_to_int((1, "a")) == key_to_int((1, "a")) != key_to_int(("a", 1))
    assert key_to_int(b"abc") == key_to_int(bytearray(b"abc"))


@pytest.mark.parametrize("cls", [MultiplyShift, TabulationHash])
def test_hash_uniformity(cls):
    h = cls(3)
    bits = 6
    counts = [0] * (1 << bits)
    n = 64 * 400
    for x in range(n):
        counts[h.index(x * 4096, bits)] += 1
    expected = n / len(counts)
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    assert all(0 <= h.index(i, bits) < 64 for i in range(100))
    assert chi2 < 130  # df=63; generous bound


def test_identity_hash_collapses_on_strided_keys():
    keys = strided_keys(200, 1024)
    bad = RobinHoodTable(1024, hash_factory=IdentityHash, seed=0)
    good = RobinHoodTable(1024, hash_factory=TabulationHash, seed=0)
    for k in keys:
        bad[k] = good[k] = 1
    assert bad.max_displacement() > 10 * max(1, good.max_displacement())
    assert all(bad[k] == 1 for k in keys)


def test_churn_robin_hood_not_worse_than_linear():
    lin = churn(LinearProbingTable(1024, max_load=0.8, seed=1), 600, 3000, seed=1)
    rh = churn(RobinHoodTable(1024, max_load=0.8, seed=1), 600, 3000, seed=1)
    assert rh[1] <= lin[1]


def test_load_sweep_marks_infeasible_loads():
    facs = {
        "cuckoo": lambda c: CuckooTable(c, max_load=0.99, seed=1),
        "robin": lambda c: RobinHoodTable(c, max_load=0.99, seed=1),
    }
    res = load_sweep(facs, [0.3, 0.8], capacity=512, seed=2)
    assert res["cuckoo"][0.3] is not None
    assert res["cuckoo"][0.8] is None
    assert res["robin"][0.8] is not None
    assert res["robin"][0.8].mean_hit >= 1.0


def test_determinism_across_runs():
    def build():
        t = CuckooTable(64, bucket_size=2, seed=13)
        for k in random_keys(300, seed=2):
            t[k] = 1
        return t._keys, t.kicks

    assert build() == build()
