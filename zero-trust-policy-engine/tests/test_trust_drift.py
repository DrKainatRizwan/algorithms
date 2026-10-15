import math

import numpy as np
import pytest

from zero_trust import (CUSUM, PageHinkley, RunningStats, Signal, TrustScorer,
                        haversine_km, ks_statistic, psi)


def test_haversine_known_distances():
    assert haversine_km(51.5, -0.1, 51.5, -0.1) == 0.0
    d = haversine_km(51.5074, -0.1278, 40.7128, -74.0060)  # London - New York
    assert 5500 < d < 5620


def test_trust_prior_and_signals():
    ts = TrustScorer(prior=0.8, half_life=3600)
    assert ts.score("a") == pytest.approx(0.8)
    bad = ts.update("a", 0.0, [Signal("x", -2.0)])
    good = ts.update("b", 0.0, [Signal("y", 2.0)])
    assert bad < 0.8 < good
    assert ts.history("a")[0][0] == 0.0


def test_trust_decays_towards_prior_with_half_life():
    ts = TrustScorer(prior=0.5, half_life=100.0)
    ts.update("a", 0.0, [Signal("x", -4.0)])
    l0 = math.log(ts.score("a", 0.0) / (1 - ts.score("a", 0.0)))
    p = ts.score("a", 100.0)
    l1 = math.log(p / (1 - p))
    assert l1 == pytest.approx(l0 / 2, rel=1e-6)
    assert ts.score("a", 1e6) == pytest.approx(0.5, abs=1e-6)


def test_trust_is_bounded_and_reset():
    ts = TrustScorer(min_score=0.01, max_score=0.99)
    for i in range(50):
        ts.update("a", float(i), [Signal("x", -5)])
    assert ts.score("a") >= 0.01 - 1e-12
    ts.reset("a")
    assert ts.score("a") == pytest.approx(ts.prior)
    with pytest.raises(ValueError):
        TrustScorer(prior=1.0)
    with pytest.raises(ValueError):
        TrustScorer(half_life=0)


def test_running_stats_match_numpy():
    x = np.random.default_rng(1).normal(3, 2, 500)
    rs = RunningStats()
    for v in x:
        rs.push(float(v))
    assert rs.mean == pytest.approx(x.mean())
    assert rs.std == pytest.approx(x.std(ddof=1))
    assert rs.zscore(rs.mean + rs.std) == pytest.approx(1.0)


def test_page_hinkley_detects_shift_but_not_stationary():
    rng = np.random.default_rng(2)
    ph = PageHinkley(delta=0.3, threshold=8.0)
    assert not any(ph.update(float(v)) for v in rng.normal(0, 1, 100))
    ph = PageHinkley(delta=0.3, threshold=8.0)
    stream = np.concatenate([rng.normal(0, 1, 100), rng.normal(2.5, 1, 100)])
    hits = [i for i, v in enumerate(stream) if ph.update(float(v))]
    assert hits and 100 <= hits[0] < 130


def test_cusum_two_sided():
    c = CUSUM(k=0.5, h=4.0)
    assert [c.update(0.0) for _ in range(20)] == [0] * 20
    assert any(c.update(2.0) == 1 for _ in range(10))
    assert any(c.update(-2.0) == -1 for _ in range(10))


def test_psi_and_ks_separate_distributions():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 1000)
    same = rng.normal(0, 1, 1000)
    shifted = rng.normal(1.5, 1, 1000)
    assert psi(a, same) < 0.1 < 0.25 < psi(a, shifted)
    d0, p0 = ks_statistic(a, same)
    d1, p1 = ks_statistic(a, shifted)
    assert p0 > 0.01 and p1 < 1e-6 and d1 > d0


def test_ks_identical_samples_and_empty():
    x = [1.0, 2.0, 3.0]
    assert ks_statistic(x, x) == (0.0, 1.0)
    with pytest.raises(ValueError):
        ks_statistic([], x)
    with pytest.raises(ValueError):
        psi(x, [])
