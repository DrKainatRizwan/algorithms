import numpy as np
import pytest

from ldp_frequency import (GeneralizedRandomizedResponse, HadamardResponse,
                           OptimizedLocalHashing, UnaryEncoding, audit, fwht,
                           make_oracle, privacy_loss, sample_users, true_frequencies,
                           zipf_distribution)
from ldp_frequency.oracles import parity

EPS = 1.0


def test_fwht_matches_hadamard_matrix():
    k = 16
    idx = np.arange(k)
    H = (-1.0) ** parity(idx[:, None] & idx[None, :])
    v = np.random.default_rng(0).normal(size=k)
    assert np.allclose(fwht(v), H @ v)
    with pytest.raises(ValueError):
        fwht(np.ones(6))


def test_parity_matches_bin_count():
    xs = np.arange(0, 5000, 7)
    assert [int(p) for p in parity(xs)] == [bin(int(x)).count("1") & 1 for x in xs]


@pytest.mark.parametrize("name,d", [("GRR", 6), ("SUE", 6), ("OUE", 6), ("OLH", 6), ("HR", 6)])
def test_exact_privacy_budget_is_met(name, d):
    o = make_oracle(name, d, 1.3)
    loss = audit(o)
    assert loss <= 1.3 + 1e-9
    if name != "OLH":  # OLH with a coarse hash may collide, so only an upper bound
        assert loss == pytest.approx(1.3, abs=1e-9)


def test_channel_rows_are_distributions():
    for name in ("GRR", "OUE", "HR", "OLH"):
        chan = make_oracle(name, 5, 0.7).channel_matrix()
        assert np.allclose(chan.sum(axis=1), 1.0)
        assert (chan >= 0).all()


def test_privacy_loss_detects_violation():
    chan = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert privacy_loss(chan) == pytest.approx(np.log(9))
    with pytest.raises(ValueError):
        privacy_loss(np.array([[0.5, 0.2], [0.5, 0.5]]))


@pytest.mark.parametrize("name", ["GRR", "SUE", "OUE", "OLH", "HR"])
def test_estimators_are_unbiased(name):
    d, n = 12, 60_000
    rng = np.random.default_rng(11)
    probs = zipf_distribution(d, 1.0)
    values = sample_users(probs, n, rng)
    truth = true_frequencies(values, d)
    o = make_oracle(name, d, EPS)
    est = o.run(values, rng)
    tol = 5 * np.sqrt(o.variance(n, truth.max()))
    assert np.abs(est - truth).max() < tol


@pytest.mark.parametrize("name", ["GRR", "OUE", "OLH", "HR"])
def test_empirical_variance_matches_formula(name):
    d, n, trials = 8, 2000, 150
    rng = np.random.default_rng(5)
    values = np.zeros(n, dtype=np.int64)  # all mass on item 0, item 5 has f = 0
    o = make_oracle(name, d, EPS)
    ests = np.array([o.run(values, rng)[5] for _ in range(trials)])
    ratio = ests.var(ddof=1) / o.variance(n, 0.0)
    assert 0.65 < ratio < 1.5
    assert abs(ests.mean()) < 5 * np.sqrt(o.variance(n) / trials)


def test_grr_keeps_truth_with_probability_p():
    o = GeneralizedRandomizedResponse(10, 2.0)
    rng = np.random.default_rng(1)
    vals = np.full(50_000, 3)
    rep = o.perturb(vals, rng)
    assert np.mean(rep == 3) == pytest.approx(o.p, abs=0.01)
    others = np.bincount(rep, minlength=10)[[i for i in range(10) if i != 3]] / vals.size
    assert np.allclose(others, o.q, atol=0.01)


def test_olh_default_bucket_count_and_report_shape():
    o = OptimizedLocalHashing(100, 2.0)
    assert o.g == round(np.exp(2.0) + 1)
    rep = o.perturb(np.arange(50), np.random.default_rng(0))
    assert rep.shape == (50, 3)
    assert (rep[:, 2] < o.g).all()


def test_hadamard_reports_fit_in_log_bits_and_stay_in_support_when_noiseless():
    o = HadamardResponse(20, 30.0)  # essentially noiseless
    assert o.k == 32 and o.report_bits == 5
    rng = np.random.default_rng(2)
    vals = rng.integers(0, 20, size=4000)
    rep = o.perturb(vals, rng)
    assert rep.max() < 32
    assert (parity((vals + 1) & rep) == 0).all()
    est = o.estimate(rep)
    assert np.abs(est - true_frequencies(vals, 20)).max() < 5 * np.sqrt(o.variance(4000))


def test_unary_variants_and_communication():
    s, o = UnaryEncoding(8, 1.0, "symmetric"), UnaryEncoding(8, 1.0)
    assert s.p == pytest.approx(1 - s.q)
    assert o.p == 0.5 and o.variance(1000) < s.variance(1000)
    with pytest.raises(ValueError):
        UnaryEncoding(8, 1.0, "bogus")


def test_input_validation():
    with pytest.raises(ValueError):
        GeneralizedRandomizedResponse(1, 1.0)
    with pytest.raises(ValueError):
        GeneralizedRandomizedResponse(5, 0.0)
    with pytest.raises(ValueError):
        GeneralizedRandomizedResponse(5, 1.0).perturb(np.array([7]), np.random.default_rng(0))
    with pytest.raises(ValueError):
        make_oracle("nope", 5, 1.0)


def test_olh_variance_is_independent_of_domain_size():
    a, b = OptimizedLocalHashing(10, 1.5), OptimizedLocalHashing(10_000, 1.5)
    assert a.variance(1000) == b.variance(1000)
    assert GeneralizedRandomizedResponse(10_000, 1.5).variance(1000) > b.variance(1000)


def test_determinism_with_seed():
    o = OptimizedLocalHashing(30, 1.0)
    v = np.arange(30) % 7
    r1 = o.perturb(v, np.random.default_rng(9))
    r2 = o.perturb(v, np.random.default_rng(9))
    assert np.array_equal(r1, r2)
