import pytest

from tmto import (
    DistinguishedPointAttack,
    HellmanAttack,
    HellmanTable,
    OneWayFunction,
    PasswordSpace,
    RainbowAttack,
    RainbowTable,
    hellman_coverage,
    measure_success,
    rainbow_coverage,
)

BITS = 14
N = 1 << BITS


@pytest.fixture
def f():
    return OneWayFunction(BITS)


def test_oneway_deterministic_in_range(f):
    assert f(123) == f(123)
    assert all(0 <= f(x) < N for x in range(500))


def test_oneway_counter(f):
    for x in range(10):
        f(x)
    assert f.reset_counter() == 10
    assert f.evaluations == 0


def test_oneway_rejects_bad_bits():
    with pytest.raises(ValueError):
        OneWayFunction(2)


def test_password_roundtrip():
    ps = PasswordSpace(4)
    for i in (0, 1, 35, 36, 123456, ps.size - 1):
        assert ps.decode(ps.encode(i)) == i
    with pytest.raises(ValueError):
        ps.encode(ps.size)


def test_hellman_table_chain_end_consistent(f):
    tb = HellmanTable(f, m=20, t=15, variant=3, seed=2).build()
    for end, starts in tb.ends.items():
        for s in starts:
            x = s
            for _ in range(15):
                x = tb.step(x)
            assert x == end


def test_hellman_inverts_with_good_coverage(f):
    # m t^2 = N rule: t = 32 -> m = 16; use many tables
    atk = HellmanAttack(f, m=64, t=16, tables=16, seed=1)
    atk.precompute()
    stats = measure_success(atk, f, trials=150, seed=5)
    assert stats.success_rate > 0.6


def test_hellman_returned_preimage_is_valid(f):
    atk = HellmanAttack(f, m=64, t=16, tables=16, seed=1)
    atk.precompute()
    hits = 0
    for x in range(0, N, 97):
        y = f(x)
        r = atk.invert(y)
        if r is not None:
            assert f(r) == y
            hits += 1
    assert hits > 0


def test_hellman_unreachable_returns_none(f):
    atk = HellmanAttack(f, m=1, t=2, tables=1, seed=1)
    atk.precompute()
    misses = sum(atk.invert(f(x)) is None for x in range(200))
    assert misses > 150


def test_distinguished_points_work(f):
    atk = DistinguishedPointAttack(f, m=300, d=4, tables=12, seed=3)
    atk.precompute()
    stats = measure_success(atk, f, trials=150, seed=7)
    assert stats.success_rate > 0.5
    assert atk.memory > 0


def test_dp_chains_end_in_distinguished_point(f):
    atk = DistinguishedPointAttack(f, m=100, d=3, tables=1, seed=4)
    atk.precompute()
    tb = atk.tables[0]
    assert all(tb.is_dp(e) for e in tb.ends)


def test_rainbow_chain_ends_consistent(f):
    tb = RainbowTable(f, m=30, t=20, seed=1).build()
    for end, starts in tb.ends.items():
        for s in starts:
            assert tb.chain_end(s, 0, 20) == end


def test_rainbow_inverts(f):
    atk = RainbowAttack(f, m=1200, t=30, tables=2, seed=2)
    atk.precompute()
    stats = measure_success(atk, f, trials=200, seed=9)
    assert stats.success_rate > 0.7


def test_rainbow_matches_theory(f):
    m, t = 1500, 40
    atk = RainbowAttack(f, m=m, t=t, tables=1, seed=6)
    atk.precompute()
    stats = measure_success(atk, f, trials=400, seed=11)
    theory = rainbow_coverage(N, m, t)
    assert abs(stats.success_rate - theory) < 0.12


def test_rainbow_beats_hellman_at_equal_memory(f):
    # ~1000 stored pairs each, same chain length and one table vs one table
    rb = RainbowAttack(f, m=1000, t=30, tables=1, seed=1)
    rb.precompute()
    hm = HellmanAttack(f, m=1000, t=30, tables=1, seed=1)
    hm.precompute()
    a = measure_success(rb, f, trials=200, seed=3).success_rate
    b = measure_success(hm, f, trials=200, seed=3).success_rate
    assert a > b


def test_coverage_formulas_monotone():
    assert rainbow_coverage(N, 100, 10) < rainbow_coverage(N, 1000, 10)
    assert rainbow_coverage(N, 500, 10) < rainbow_coverage(N, 500, 50)
    assert hellman_coverage(N, 50, 10, 1) < hellman_coverage(N, 50, 10, 4)
    assert 0 <= hellman_coverage(N, 100, 20) <= 1


def test_deterministic_tables(f):
    a = RainbowTable(f, 50, 10, seed=4).build()
    b = RainbowTable(OneWayFunction(BITS), 50, 10, seed=4).build()
    assert a.ends == b.ends
