import numpy as np
import pytest

from ldp_frequency import (PrefixTreeHeavyHitters, Rappor, audit,
                           longitudinal_attack_accuracy, sample_users, true_frequencies,
                           zipf_distribution)


def test_rappor_privacy_parameters():
    r = Rappor(5, f=0.5, p=0.75, q=0.25)
    assert r.epsilon_permanent == pytest.approx(2 * np.log(0.75 / 0.25))
    assert r.epsilon < r.epsilon_permanent  # a single report leaks less
    assert audit(r) == pytest.approx(r.epsilon, abs=1e-9)


def test_rappor_from_permanent_epsilon_roundtrip():
    r = Rappor.from_permanent_epsilon(8, 3.0)
    assert r.epsilon_permanent == pytest.approx(3.0)
    with pytest.raises(ValueError):
        Rappor(4, f=1.5)


def test_rappor_estimator_unbiased():
    d, n = 10, 80_000
    rng = np.random.default_rng(6)
    values = sample_users(zipf_distribution(d, 1.0), n, rng)
    r = Rappor(d, f=0.4, p=0.8, q=0.2)
    est = r.run(values, rng)
    assert np.abs(est - true_frequencies(values, d)).max() < 5 * np.sqrt(r.variance(n, 0.3))


def test_memoisation_blocks_longitudinal_averaging():
    rng = np.random.default_rng(7)
    r = Rappor(6, f=0.6, p=0.7, q=0.3)
    vals = rng.integers(0, 6, size=3000)
    acc_fresh = longitudinal_attack_accuracy(r, vals, 400, memoize=False, rng=rng)
    acc_memo = longitudinal_attack_accuracy(r, vals, 400, memoize=True, rng=rng)
    assert acc_fresh > 0.97
    assert acc_memo < 0.8


def test_heavy_hitters_recovers_planted_items():
    bits, n = 16, 150_000
    rng = np.random.default_rng(21)
    planted = np.array([0xBEEF, 0x1234, 0xA5A5])
    probs = [0.25, 0.15, 0.10]
    u = rng.random(n)
    values = rng.integers(0, 1 << bits, size=n)
    values[u < 0.25] = planted[0]
    values[(u >= 0.25) & (u < 0.40)] = planted[1]
    values[(u >= 0.40) & (u < 0.50)] = planted[2]
    hh = PrefixTreeHeavyHitters(bits, epsilon=4.0, k=5, step=4)
    res = hh.discover(values, rng)
    found = dict(res.items)
    assert set(planted.tolist()) <= set(found)
    assert [it for it, _ in res.items][:3] == planted.tolist()
    for item, p in zip(planted, probs):
        assert found[int(item)] == pytest.approx(p, abs=0.05)
    assert len(res.levels) == 4 and res.levels[-1].prefix_bits == 16
    assert sum(l.group_size for l in res.levels) == n


def test_heavy_hitters_validation_and_levels():
    hh = PrefixTreeHeavyHitters(10, 1.0, 3, step=4)
    assert hh.lengths == [4, 8, 10]
    with pytest.raises(ValueError):
        hh.discover(np.array([1 << 10]), np.random.default_rng(0))
    with pytest.raises(ValueError):
        PrefixTreeHeavyHitters(0, 1.0, 3)
