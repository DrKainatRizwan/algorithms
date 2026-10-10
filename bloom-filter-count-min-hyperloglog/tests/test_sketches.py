import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from stream_sketches import (BloomFilter, CountingBloomFilter, CountMinSketch, HeavyHitters,
                             HyperLogLog, ScalableBloomFilter, exact_counts, hash64, zipf_stream)
from stream_sketches.analytics import cms_errors, empirical_fp_rate, top_k_recall
from stream_sketches.bloom import optimal_params


def test_hash_deterministic_and_seeded():
    assert hash64("a", 1) == hash64("a", 1)
    assert hash64("a", 1) != hash64("a", 2)
    assert hash64(1) != hash64("1")


def test_hash_bit_balance():
    ones = sum(bin(hash64(i)).count("1") for i in range(2000))
    assert abs(ones / (2000 * 64) - 0.5) < 0.01


def test_optimal_params_formula():
    m, k = optimal_params(1000, 0.01)
    assert 9500 < m < 9700 and k == 7
    with pytest.raises(ValueError):
        optimal_params(0, 0.1)


def test_bloom_no_false_negatives():
    bf = BloomFilter(2000, 0.01)
    keys = [f"k{i}" for i in range(2000)]
    bf.update(keys)
    assert all(k in bf for k in keys)


def test_bloom_fp_rate_near_target():
    bf = BloomFilter(5000, 0.01, seed=3)
    bf.update(f"in{i}" for i in range(5000))
    rate = empirical_fp_rate(bf, [f"out{i}" for i in range(20000)])
    assert rate < 0.02
    assert abs(rate - bf.expected_fp_rate()) < 0.008


def test_bloom_cardinality_estimate_and_union():
    a, b = BloomFilter(4000, 0.01), BloomFilter(4000, 0.01)
    a.update(range(0, 1500))
    b.update(range(1000, 2500))
    assert abs(a.estimate_cardinality() - 1500) / 1500 < 0.05
    u = a.union(b)
    assert abs(u.estimate_cardinality() - 2500) / 2500 < 0.05
    assert all(i in u for i in range(2500))
    with pytest.raises(ValueError):
        a.union(BloomFilter(100, 0.1))


def test_counting_bloom_remove():
    cb = CountingBloomFilter(1000, 0.01)
    cb.add("x"); cb.add("y"); cb.add("x")
    assert cb.min_count("x") >= 2
    assert cb.remove("x") and "x" in cb
    assert cb.remove("x") and "x" not in cb
    assert "y" in cb
    assert cb.remove("never-added") is False


def test_scalable_bloom_grows_and_keeps_members():
    sb = ScalableBloomFilter(initial_capacity=64, fp_rate=0.01)
    keys = [f"s{i}" for i in range(3000)]
    for k in keys:
        sb.add(k)
    assert len(sb.slices) > 3
    assert all(k in sb for k in keys)
    fp = sum(1 for i in range(10000) if f"zz{i}" in sb) / 10000
    assert fp < 0.02


def test_cms_never_underestimates_and_bound():
    stream = zipf_stream(20000, 2000, 1.1, seed=1)
    truth = exact_counts(stream)
    cms = CountMinSketch(0.005, 0.01)
    for x in stream:
        cms.add(x)
    stats = cms_errors(cms, truth)
    assert stats["min_error"] >= 0
    assert stats["frac_within_bound"] >= 0.99
    assert cms.total == 20000


def test_cms_conservative_not_worse():
    stream = zipf_stream(20000, 3000, 1.0, seed=2)
    truth = exact_counts(stream)
    plain = CountMinSketch(0.01, 0.05)
    cons = CountMinSketch(0.01, 0.05, conservative=True)
    for x in stream:
        plain.add(x); cons.add(x)
    e_plain, e_cons = cms_errors(plain, truth), cms_errors(cons, truth)
    assert e_cons["min_error"] >= 0
    assert e_cons["mean_error"] <= e_plain["mean_error"]


def test_cms_merge_equals_joint():
    a, b, j = (CountMinSketch(0.01, 0.05) for _ in range(3))
    for i in range(500):
        a.add(i % 37); j.add(i % 37)
    for i in range(300):
        b.add(i % 11); j.add(i % 11)
    m = a.merge(b)
    assert (m.table == j.table).all() and m.total == 800


def test_cms_inner_product_upper_bound():
    a, b = CountMinSketch(0.01, 0.05), CountMinSketch(0.01, 0.05)
    fa, fb = {}, {}
    for i in range(400):
        a.add(i % 20); fa[i % 20] = fa.get(i % 20, 0) + 1
        b.add(i % 30); fb[i % 30] = fb.get(i % 30, 0) + 1
    true = sum(fa[k] * fb.get(k, 0) for k in fa)
    assert a.inner_product(b) >= true


def test_heavy_hitters_recall():
    stream = zipf_stream(30000, 5000, 1.2, seed=4)
    truth = exact_counts(stream)
    hh = HeavyHitters(20, epsilon=0.002)
    for x in stream:
        hh.add(x)
    found = [x for x, _ in hh.top(10)]
    assert top_k_recall(found, truth, 10) >= 0.9
    assert all(hh.sketch.estimate(x) >= truth[x] for x in found)


@pytest.mark.parametrize("n", [50, 5000, 100000])
def test_hll_accuracy(n):
    h = HyperLogLog(p=12)
    h.update(f"u{i}" for i in range(n))
    assert abs(h.estimate() - n) / n < 4 * h.standard_error()


def test_hll_duplicates_ignored_and_empty():
    h = HyperLogLog(10)
    assert h.estimate() == 0
    for _ in range(5):
        h.update(range(300))
    assert abs(len(h) - 300) / 300 < 0.1


def test_hll_merge_and_intersection():
    a, b = HyperLogLog(12), HyperLogLog(12)
    a.update(range(0, 30000)); b.update(range(15000, 45000))
    u = a.merge(b)
    assert abs(u.estimate() - 45000) / 45000 < 0.05
    inter = a.intersection_estimate(b)
    assert abs(inter - 15000) / 15000 < 0.25
    with pytest.raises(ValueError):
        a.merge(HyperLogLog(10))


def test_hll_error_scales_with_precision():
    from stream_sketches.analytics import hll_relative_error
    lo = hll_relative_error(6, 20000, trials=8)
    hi = hll_relative_error(14, 20000, trials=8)
    assert hi < lo
    assert math.isfinite(hi)
