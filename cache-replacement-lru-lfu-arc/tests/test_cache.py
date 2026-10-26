import itertools
import random

import pytest

from cache_policies import (ARCCache, ClockCache, FIFOCache, LFUCache, LRUCache, RandomCache,
                            TwoQCache, cached, compare, hit_rate, loop_trace, make_policy,
                            mixed_trace, next_use_indices, scan_pollution_trace,
                            shifting_working_set_trace, simulate_opt, working_set_size, zipf_trace)

ALL = [FIFOCache, RandomCache, ClockCache, LRUCache, LFUCache, TwoQCache, ARCCache]


def test_lru_evicts_least_recent():
    c = LRUCache(2)
    for k in "abca":
        c.access(k)
    assert list(c.keys()) == ["c", "a"]
    assert c.stats.hits == 0 and c.stats.evictions == 2


def test_lru_hit_refreshes():
    c = LRUCache(2)
    for k in "aba":
        c.access(k)
    c.access("c")
    assert "a" in c and "b" not in c


def test_fifo_ignores_hits():
    c = FIFOCache(2)
    for k in "aab":
        c.access(k)
    c.access("c")
    assert "a" not in c and "b" in c


def test_lfu_evicts_lowest_frequency_then_oldest():
    c = LFUCache(3)
    for k in "aabbc":
        c.access(k)
    c.access("d")
    assert "c" not in c and c.frequency("a") == 2
    c.access("e")  # d (freq 1) goes
    assert "d" not in c


def test_lfu_aging_allows_displacement():
    plain, aged = LFUCache(3), LFUCache(3, aging=True, aging_period=20)
    trace = ["x"] * 200 + ["y", "z"] * 40 + ["w"]
    for k in trace:
        plain.access(k)
        aged.access(k)
    assert "x" in plain
    assert "x" not in aged


def test_clock_second_chance():
    c = ClockCache(2)
    c.access("a"); c.access("b"); c.access("a")
    c.access("c")
    assert "a" in c and "b" not in c


def test_random_is_seeded():
    t = zipf_trace(500, 50, seed=3)
    a = RandomCache(8, seed=5).run(t).hits
    b = RandomCache(8, seed=5).run(t).hits
    assert a == b


@pytest.mark.parametrize("cls", ALL)
def test_capacity_and_accounting_invariants(cls):
    rng = random.Random(1)
    c = cls(7)
    for _ in range(2000):
        k = rng.randrange(25)
        was = k in c
        assert c.access(k) == was
        assert len(c) <= 7 and k in c
        assert len(set(c.keys())) == len(c)
    assert c.stats.requests == 2000
    assert c.stats.misses - c.stats.evictions == len(c)


def test_arc_structural_invariants():
    c = ARCCache(10)
    for k in mixed_trace(5000, seed=2):
        c.access(k)
        t1, t2, b1, b2 = map(len, (c.t1, c.t2, c.b1, c.b2))
        assert t1 + t2 <= 10
        assert t1 + b1 <= 10
        assert t1 + t2 + b1 + b2 <= 20
        assert 0 <= c.p <= 10
        assert not (set(c.t1) & set(c.t2)) and not ((set(c.t1) | set(c.t2)) & (set(c.b1) | set(c.b2)))


def test_arc_adapts_p_toward_recency_on_b1_hit():
    c = ARCCache(4)
    for k in (0, 1, 2, 3, 0, 4):  # the re-reference of 0 puts it in T2
        c.access(k)
    assert 1 in c.b1 and c.p == 0
    c.access(1)
    assert c.p > 0 and 1 in c.t2


def test_arc_resists_scans_better_than_lru():
    t = scan_pollution_trace(20000, hot=50, scan_len=400, hot_burst=400, seed=1)
    assert hit_rate("ARC", t, 100) > hit_rate("LRU", t, 100) + 0.02


def test_lru_thrashes_on_loop_but_opt_does_not():
    t = loop_trace(2000, 11)
    assert hit_rate("LRU", t, 10) == 0.0
    assert hit_rate("OPT", t, 10) > 0.8


def test_lfu_beats_lru_on_stationary_zipf():
    t = zipf_trace(20000, 1000, 1.0, seed=4)
    assert hit_rate("LFU", t, 50) > hit_rate("LRU", t, 50)


def test_next_use_indices():
    assert next_use_indices([1, 2, 1, 3, 2]) == [2, 4, float("inf"), float("inf"), float("inf")]


def test_opt_small_example_matches_brute_force():
    trace = [1, 2, 3, 1, 4, 2, 5, 1, 2, 3, 4, 5]
    cap = 3

    def brute(i, cache):
        if i == len(trace):
            return 0
        k = trace[i]
        if k in cache:
            return 1 + brute(i + 1, cache)
        if len(cache) < cap:
            return brute(i + 1, cache | {k})
        return max(brute(i + 1, (cache - {v}) | {k}) for v in cache)

    assert simulate_opt(trace, cap).hits == brute(0, frozenset())


@pytest.mark.parametrize("name", ["FIFO", "Random", "CLOCK", "LRU", "LFU", "2Q", "ARC"])
def test_opt_dominates_every_policy(name):
    t = mixed_trace(4000, seed=7)
    assert hit_rate("OPT", t, 40) >= hit_rate(name, t, 40)


def test_lru_stack_inclusion_property():
    t = zipf_trace(5000, 200, 0.8, seed=9)
    rates = [hit_rate("LRU", t, c) for c in (5, 10, 20, 40, 80)]
    assert rates == sorted(rates)


def test_compare_shape_and_unknown_policy():
    t = zipf_trace(1000, 100, seed=0)
    curves = compare(t, [5, 10], ["LRU", "ARC"])
    assert set(curves) == {"LRU", "ARC", "OPT"} and all(len(v) == 2 for v in curves.values())
    with pytest.raises(ValueError):
        make_policy("nope", 3)
    with pytest.raises(ValueError):
        LRUCache(0)


def test_trace_generators_deterministic_and_shaped():
    assert zipf_trace(100, 10, seed=1) == zipf_trace(100, 10, seed=1)
    t = shifting_working_set_trace(1000, set_size=30, phases=4, seed=2)
    assert len(t) == 1000 and max(t) < 120 and min(t[-200:]) >= 90
    assert len(scan_pollution_trace(1234)) == 1234
    assert working_set_size(loop_trace(100, 10), 20) == 10


def test_cached_decorator_counts_calls_and_evicts():
    calls = []

    @cached("LRU", 2)
    def sq(x):
        calls.append(x)
        return x * x

    assert [sq(1), sq(2), sq(1), sq(3), sq(1)] == [1, 4, 1, 9, 1]
    assert calls == [1, 2, 3]
    sq(2)
    assert calls == [1, 2, 3, 2]
    assert sq.stats().hits == 2
