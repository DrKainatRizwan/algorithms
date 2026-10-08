import math
import numpy as np
import pytest

from dp_toolkit import *
from dp_toolkit.mechanisms import (exponential_probabilities,
                                   debias_randomized_response)


def rng(s=0):
    return np.random.default_rng(s)


def test_laplace_unbiased_and_scale():
    r = rng(1)
    xs = np.array([laplace_mechanism(10.0, 1.0, 0.5, r) for _ in range(20000)])
    assert abs(xs.mean() - 10) < 0.1
    assert abs(xs.std() - math.sqrt(2) * 2.0) < 0.15


def test_laplace_vector_shape_and_validation():
    out = laplace_mechanism(np.zeros(5), 1, 1, rng())
    assert out.shape == (5,)
    with pytest.raises(ValueError):
        laplace_mechanism(1, 1, 0, rng())


def test_laplace_pure_dp_density_ratio():
    # density ratio at neighbouring outputs is bounded by e^eps
    eps, scale = 0.7, 1 / 0.7
    for y in np.linspace(-5, 5, 21):
        ratio = math.exp(-abs(y - 0) / scale) / math.exp(-abs(y - 1) / scale)
        assert ratio <= math.exp(eps) + 1e-9


def test_analytic_sigma_meets_delta_and_beats_classical():
    s = analytic_gaussian_sigma(1.0, 0.5, 1e-5)
    assert gaussian_delta(s, 1.0, 0.5) <= 1e-5 * (1 + 1e-6)
    assert gaussian_delta(s * 0.98, 1.0, 0.5) > 1e-5
    assert s < classical_gaussian_sigma(1.0, 0.5, 1e-5)


def test_gaussian_delta_monotone_in_sigma():
    d = [gaussian_delta(s, 1.0, 1.0) for s in (0.5, 1, 2, 4)]
    assert all(a > b for a, b in zip(d, d[1:]))


def test_gaussian_mechanism_noise_level():
    r = rng(3)
    sig = analytic_gaussian_sigma(2.0, 1.0, 1e-6)
    xs = np.array([gaussian_mechanism(0.0, 2.0, 1.0, 1e-6, r) for _ in range(5000)])
    assert abs(xs.std() / sig - 1) < 0.05


def test_geometric_mechanism_symmetric_integer():
    r = rng(4)
    xs = np.array([geometric_mechanism(5, 1, 1.0, r) for _ in range(20000)])
    assert abs(xs.mean() - 5) < 0.1
    # P(noise=0) = (1-a)/(1+a)
    a = math.exp(-1)
    assert abs((xs == 5).mean() - (1 - a) / (1 + a)) < 0.01


def test_exponential_mechanism_probabilities_and_dp_ratio():
    s1 = np.array([1.0, 3.0, 2.0])
    s2 = s1 + np.array([1, -1, 0])        # neighbour: scores shift by <= sens
    eps = 1.0
    p1 = exponential_probabilities(s1, 1.0, eps)
    p2 = exponential_probabilities(s2, 1.0, eps)
    assert np.all(p1 / p2 <= math.exp(eps) + 1e-9)
    assert np.isclose(p1.sum(), 1) and p1.argmax() == 1


def test_exponential_mechanism_prefers_best():
    r = rng(5)
    picks = [exponential_mechanism([0, 0, 50, 0], 1.0, 2.0, r) for _ in range(200)]
    assert picks.count(2) > 190


def test_randomized_response_debias():
    r = rng(6)
    bits = (r.random(50000) < 0.3).astype(int)
    noisy = randomized_response(bits, 1.0, r)
    assert abs(debias_randomized_response(noisy, 1.0) - bits.mean()) < 0.02


def test_basic_accountant_budget():
    a = BasicAccountant(1.0, 1e-5)
    a.spend(0.4, 5e-6)
    a.spend(0.5, 4e-6)
    assert math.isclose(a.remaining()[0], 0.1)
    with pytest.raises(PrivacyBudgetExceeded):
        a.spend(0.2)


def test_advanced_beats_basic_for_many_small_queries():
    e, d = AdvancedAccountant.compose(0.01, 0.0, 1000, 1e-6)
    assert e < 10.0 and d == 1e-6
    assert AdvancedAccountant.best(1.0, 0.0, 3, 1e-6)[0] == 3.0


def test_gaussian_rdp_closed_form_and_conversion():
    acc = RDPAccountant()
    acc.compose_gaussian(sigma=2.0, steps=10)
    expected = 10 * np.array(acc.orders) / 8.0
    assert np.allclose(acc.rdp, expected)
    eps = acc.epsilon(1e-5)
    # never worse than the textbook conversion eps = rdp(a) + log(1/delta)/(a-1)
    a = 4
    assert 0 < eps <= 10 * a / 8.0 + math.log(1e5) / (a - 1)


def test_rdp_beats_basic_composition_gaussian():
    sigma, k, delta = 5.0, 100, 1e-5
    acc = RDPAccountant()
    acc.compose_gaussian(sigma, steps=k)
    eps_single = analytic_gaussian_sigma  # noqa
    # per-step (eps, delta/k) via analytic calibration, basic composition
    lo, hi = 0.0, 50.0
    for _ in range(60):
        m = (lo + hi) / 2
        if gaussian_delta(sigma, 1.0, m) > delta / k:
            lo = m
        else:
            hi = m
    assert acc.epsilon(delta) < k * hi


def test_subsampling_amplifies_privacy():
    full = RDPAccountant(); full.compose_gaussian(1.0, steps=50)
    sub = RDPAccountant(); sub.compose_subsampled_gaussian(1.0, 0.01, steps=50)
    assert sub.epsilon(1e-5) < full.epsilon(1e-5) / 5


def test_subsampled_q1_equals_gaussian_and_q0_zero():
    o = (2, 4, 8)
    assert np.allclose(subsampled_gaussian_rdp(o, 1.5, 1.0), gaussian_rdp(o, 1.5))
    assert np.allclose(subsampled_gaussian_rdp(o, 1.5, 0.0), 0)


def test_subsampled_rdp_monotone_in_q():
    o = (2, 4, 8, 16)
    a = subsampled_gaussian_rdp(o, 1.0, 0.01)
    b = subsampled_gaussian_rdp(o, 1.0, 0.1)
    assert np.all(a <= b)


def test_noise_for_target_hits_epsilon():
    acc = RDPAccountant()
    s = acc.noise_for_target(2.0, 1e-5, q=0.01, steps=1000)
    chk = RDPAccountant(); chk.compose_subsampled_gaussian(s, 0.01, 1000)
    assert chk.epsilon(1e-5) <= 2.0 + 1e-9
    assert chk.epsilon(1e-5) > 1.9


def test_laplace_rdp_capped_by_pure_eps():
    acc = RDPAccountant(); acc.compose_laplace(0.5, steps=1)
    assert np.all(acc.rdp <= 0.5 + 1e-12)


def test_private_count_sum_mean_accuracy():
    r = rng(7)
    data = r.uniform(0, 10, 5000)
    assert abs(private_count(data, lambda x: x > 5, 1.0, r) - (data > 5).sum()) < 30
    assert abs(private_sum(data, 0, 10, 1.0, r) - data.sum()) < 150
    assert abs(private_mean(data, 0, 10, 1.0, r) - data.mean()) < 0.1


def test_private_histogram_nonnegative_and_close():
    r = rng(8)
    data = r.normal(0, 1, 10000)
    bins = np.linspace(-4, 4, 17)
    h = private_histogram(data, bins, 1.0, r)
    true, _ = np.histogram(data, bins)
    assert (h >= 0).all() and np.abs(h - true).max() < 30


def test_private_quantile_median():
    r = rng(9)
    data = r.normal(5, 1, 2000)
    m = private_quantile(data, 0.5, 0, 10, 1.0, r)
    assert abs(m - np.median(data)) < 0.3


def test_above_threshold_finds_big_query():
    r = rng(10)
    qs = [0] * 20 + [100] + [0] * 5
    hits = [above_threshold(qs, 50, 1.0, rng(s)) for s in range(50)]
    assert sum(h == 20 for h in hits) >= 45


def test_dp_logistic_regression_learns_and_reports_eps():
    r = rng(11)
    n = 4000
    X = r.normal(size=(n, 2))
    y = (X @ np.array([2.0, -1.0]) > 0).astype(float)
    w, eps, _ = dp_logistic_regression(X, y, 5, 200, 1.0, 1.0, 1.0, 1e-5, r)
    pred = (np.hstack([X, np.ones((n, 1))]) @ w > 0)
    assert (pred == y).mean() > 0.9
    assert 0 < eps < 20
