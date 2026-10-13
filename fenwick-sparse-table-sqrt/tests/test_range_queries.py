import random
from math import gcd

import pytest

from range_queries import (ArgMinSparseTable, DisjointSparseTable, Fenwick, Fenwick2D,
                           NaiveArray, RangeFenwick, SparseTable, SqrtDecomposition,
                           SqrtRangeAdd, mo_distinct_counts)
from range_queries.bench import bench_static_min, bench_sum


def rand_range(rng, n):
    a, b = rng.randrange(n), rng.randrange(n)
    return min(a, b), max(a, b) + 1


def data(n, seed=1, hi=100):
    rng = random.Random(seed)
    return [rng.randrange(-hi, hi) for _ in range(n)]


def test_fenwick_build_matches_prefix_sums():
    a = data(37)
    f = Fenwick(a)
    for i in range(len(a) + 1):
        assert f.prefix(i) == sum(a[:i])


def test_fenwick_random_updates():
    rng = random.Random(2)
    a = data(60)
    f, ref = Fenwick(a), NaiveArray(a)
    for _ in range(300):
        i = rng.randrange(60)
        d = rng.randrange(-20, 20)
        f.add(i, d)
        ref.add(i, d)
        lo, hi = rand_range(rng, 60)
        assert f.sum(lo, hi) == ref.sum(lo, hi)


def test_fenwick_set_get_and_bounds():
    f = Fenwick([1, 2, 3])
    f.set(1, 10)
    assert f.get(1) == 10 and f.sum(0, 3) == 14
    with pytest.raises(IndexError):
        f.add(3, 1)
    with pytest.raises(IndexError):
        f.sum(2, 1)


def test_fenwick_lower_bound():
    a = [3, 0, 2, 5, 0, 1]
    f = Fenwick(a)
    assert f.lower_bound(1) == 0
    assert f.lower_bound(4) == 2
    assert f.lower_bound(10) == 3
    assert f.lower_bound(11) == 5
    assert f.lower_bound(12) == 6


def test_range_fenwick_vs_naive():
    rng = random.Random(3)
    a = data(50)
    f, ref = RangeFenwick(a), NaiveArray(a)
    for _ in range(300):
        lo, hi = rand_range(rng, 50)
        if rng.random() < 0.5:
            d = rng.randrange(-9, 10)
            f.range_add(lo, hi, d)
            ref.range_add(lo, hi, d)
        else:
            assert f.sum(lo, hi) == ref.sum(lo, hi)
    assert [f.get(i) for i in range(50)] == ref.a


def test_fenwick2d_vs_bruteforce():
    rng = random.Random(4)
    R, C = 9, 13
    grid = [[0] * C for _ in range(R)]
    f = Fenwick2D(R, C)
    for _ in range(200):
        r, c, d = rng.randrange(R), rng.randrange(C), rng.randrange(-5, 6)
        grid[r][c] += d
        f.add(r, c, d)
        r0, r1 = sorted((rng.randrange(R + 1), rng.randrange(R + 1)))
        c0, c1 = sorted((rng.randrange(C + 1), rng.randrange(C + 1)))
        expect = sum(grid[i][j] for i in range(r0, r1) for j in range(c0, c1))
        assert f.sum(r0, c0, r1, c1) == expect


@pytest.mark.parametrize("n", [1, 2, 5, 16, 33])
def test_sparse_table_min_max_gcd(n):
    rng = random.Random(n)
    a = [rng.randrange(1, 500) for _ in range(n)]
    smin, smax, sg = SparseTable(a), SparseTable(a, max), SparseTable.gcd_table(a)
    for lo in range(n):
        for hi in range(lo + 1, n + 1):
            seg = a[lo:hi]
            assert smin.query(lo, hi) == min(seg)
            assert smax.query(lo, hi) == max(seg)
            g = 0
            for x in seg:
                g = gcd(g, x)
            assert sg.query(lo, hi) == g


def test_argmin_leftmost_on_ties():
    a = [5, 2, 7, 2, 9, 2]
    t = ArgMinSparseTable(a)
    assert t.query(0, 6) == 1
    assert t.query(2, 6) == 3
    assert t.query(4, 5) == 4


@pytest.mark.parametrize("n", [1, 3, 8, 21])
def test_disjoint_sparse_table_noncommutative(n):
    rng = random.Random(n + 10)
    a = [str(rng.randrange(10)) for _ in range(n)]
    t = DisjointSparseTable(a, lambda x, y: x + y)  # string concat: order matters
    for lo in range(n):
        for hi in range(lo + 1, n + 1):
            assert t.query(lo, hi) == "".join(a[lo:hi])


def test_disjoint_sparse_table_sum():
    a = data(100)
    t = DisjointSparseTable(a)
    rng = random.Random(5)
    for _ in range(200):
        lo, hi = rand_range(rng, 100)
        assert t.query(lo, hi) == sum(a[lo:hi])


def test_sqrt_decomposition_vs_naive():
    rng = random.Random(6)
    a = data(101)
    s, ref = SqrtDecomposition(a), NaiveArray(a)
    for _ in range(400):
        if rng.random() < 0.4:
            i, v = rng.randrange(101), rng.randrange(-50, 50)
            s.set(i, v)
            ref.set(i, v)
        lo, hi = rand_range(rng, 101)
        assert s.sum(lo, hi) == ref.sum(lo, hi)
        assert s.min(lo, hi) == ref.min(lo, hi)
        assert s.max(lo, hi) == ref.max(lo, hi)


def test_sqrt_range_add_vs_naive():
    rng = random.Random(7)
    a = data(97)
    s, ref = SqrtRangeAdd(a, block=7), NaiveArray(a)
    for _ in range(400):
        lo, hi = rand_range(rng, 97)
        if rng.random() < 0.5:
            d = rng.randrange(-8, 9)
            s.range_add(lo, hi, d)
            ref.range_add(lo, hi, d)
        else:
            assert s.sum(lo, hi) == ref.sum(lo, hi)
            assert s.max(lo, hi) == ref.max(lo, hi)


def test_mo_distinct_counts():
    rng = random.Random(8)
    a = [rng.randrange(10) for _ in range(120)]
    qs = [rand_range(rng, 120) for _ in range(150)] + [(5, 5)]
    assert mo_distinct_counts(a, qs) == [len(set(a[lo:hi])) for lo, hi in qs]
    assert mo_distinct_counts(a, []) == []


def test_query_errors():
    with pytest.raises(IndexError):
        SparseTable([1, 2, 3]).query(2, 2)
    with pytest.raises(IndexError):
        DisjointSparseTable([1, 2, 3]).query(0, 4)
    with pytest.raises(IndexError):
        SqrtDecomposition([1, 2]).sum(0, 3)


def test_bench_harness_runs():
    assert set(bench_sum(200, 50)) == {"naive", "sqrt", "fenwick"}
    assert "sparse build" in bench_static_min(200, 50)
