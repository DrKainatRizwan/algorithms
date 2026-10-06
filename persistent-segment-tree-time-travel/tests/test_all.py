import random

import pytest

from persistent_segtree import (GCD, MAX, MIN, SUM, PersistentLazyTree,
                                PersistentSegmentTree, RangeKth,
                                SnapshotArray, TimeTravelArray)


def rand_arr(n, seed):
    rng = random.Random(seed)
    return [rng.randint(-50, 50) for _ in range(n)]


def test_build_and_query_sum():
    a = rand_arr(37, 1)
    t = PersistentSegmentTree(a)
    assert t.query(0, 0, 37) == sum(a)
    assert t.query(0, 5, 20) == sum(a[5:20])
    assert t.query(0, 3, 3) == 0


def test_update_keeps_old_versions():
    a = [1, 2, 3, 4, 5]
    t = PersistentSegmentTree(a)
    v1 = t.update(0, 2, 100)
    assert t.to_list(0) == a
    assert t.to_list(v1) == [1, 2, 100, 4, 5]


def test_branching_history():
    t = PersistentSegmentTree([0] * 8)
    v1 = t.update(0, 0, 5)
    v2 = t.update(0, 0, 7)  # branch from root, not from v1
    v3 = t.update(v1, 1, 1)
    assert t.query(v1, 0, 8) == 5
    assert t.query(v2, 0, 8) == 7
    assert t.query(v3, 0, 8) == 6


@pytest.mark.parametrize("monoid", [SUM, MIN, MAX])
def test_random_against_snapshot(monoid):
    rng = random.Random(7)
    a = rand_arr(50, 3)
    t, ref = PersistentSegmentTree(a, monoid), SnapshotArray(a, monoid)
    for _ in range(200):
        v = rng.randrange(t.num_versions)
        if rng.random() < 0.5:
            i, x = rng.randrange(50), rng.randint(-99, 99)
            assert t.update(v, i, x) == ref.update(v, i, x)
        else:
            l = rng.randrange(51)
            r = rng.randint(l, 50)
            assert t.query(v, l, r) == ref.query(v, l, r)


def test_gcd_monoid():
    t = PersistentSegmentTree([12, 18, 24, 7], GCD)
    assert t.query(0, 0, 3) == 6
    assert t.query(0, 0, 4) == 1


def test_node_allocation_is_logarithmic():
    t = PersistentSegmentTree(list(range(1024)))
    before = t.nodes_created
    t.update(0, 500, 9)
    assert t.nodes_created - before == 11  # log2(1024) + 1


def test_structural_sharing():
    t = PersistentSegmentTree(list(range(64)))
    v = t.update(0, 10, -1)
    total = t.node_count(0)
    assert t.shared_nodes(0, v) == total - 7  # one fresh root-to-leaf path


def test_bounds_errors():
    t = PersistentSegmentTree([1, 2, 3])
    with pytest.raises(IndexError):
        t.update(0, 3, 1)
    with pytest.raises(IndexError):
        t.query(0, 2, 5)
    with pytest.raises(IndexError):
        t.query(9, 0, 1)
    with pytest.raises(ValueError):
        PersistentSegmentTree([])


def test_lazy_range_add_matches_naive():
    rng = random.Random(11)
    a = rand_arr(40, 5)
    t = PersistentLazyTree(a)
    snaps = [list(a)]
    for _ in range(150):
        v = rng.randrange(t.num_versions)
        if rng.random() < 0.5:
            l = rng.randrange(41)
            r = rng.randint(l, 40)
            d = rng.randint(-9, 9)
            nv = t.add(v, l, r, d)
            s = list(snaps[v])
            for i in range(l, r):
                s[i] += d
            snaps.append(s)
            assert nv == len(snaps) - 1
        else:
            l = rng.randrange(41)
            r = rng.randint(l, 40)
            assert t.sum(v, l, r) == sum(snaps[v][l:r])


def test_lazy_old_version_untouched():
    t = PersistentLazyTree([1] * 10)
    v = t.add(0, 0, 10, 5)
    assert t.sum(0, 0, 10) == 10
    assert t.sum(v, 0, 10) == 60
    assert t.sum(v, 3, 4) == 6


def test_kth_matches_sorted():
    rng = random.Random(2)
    a = [rng.randint(0, 30) for _ in range(80)]
    q = RangeKth(a)
    for _ in range(200):
        l = rng.randrange(80)
        r = rng.randint(l + 1, 80)
        k = rng.randint(1, r - l)
        assert q.kth(l, r, k) == sorted(a[l:r])[k - 1]


def test_rank_queries():
    rng = random.Random(4)
    a = [rng.randint(0, 20) for _ in range(60)]
    q = RangeKth(a)
    for x in (-5, 0, 7, 20, 99):
        for l, r in ((0, 60), (10, 30), (5, 5)):
            assert q.count_less(l, r, x) == sum(v < x for v in a[l:r])
            assert q.count_leq(l, r, x) == sum(v <= x for v in a[l:r])


def test_kth_invalid_k():
    with pytest.raises(ValueError):
        RangeKth([3, 1, 2]).kth(0, 3, 4)


def test_time_travel_queries():
    tt = TimeTravelArray([1, 1, 1, 1], start_time=0)
    tt.set(10, 0, 5)
    tt.set(20, 1, 7)
    tt.set(20, 2, 9)
    assert tt.query(0, 0, 4) == 4
    assert tt.query(9.9, 0, 4) == 4
    assert tt.query(10, 0, 4) == 8
    assert tt.query(25, 0, 4) == 22
    assert tt.get(15, 1) == 1


def test_time_travel_rejects_bad_timestamps():
    tt = TimeTravelArray([0, 0], start_time=5)
    with pytest.raises(ValueError):
        tt.version_at(1)
    tt.set(6, 0, 1)
    with pytest.raises(ValueError):
        tt.set(5.5, 1, 1)


def test_rollback_and_history():
    tt = TimeTravelArray([0, 0, 0])
    tt.set(1, 0, 10)
    tt.set(2, 0, 20)
    tt.rollback(1, 3)
    assert tt.get(3, 0) == 10
    assert tt.get(2, 0) == 20  # history preserved
    tt.set(4, 0, 30)
    assert tt.history(0) == [(0, 0), (1, 10), (2, 20), (3, 10), (4, 30)]
