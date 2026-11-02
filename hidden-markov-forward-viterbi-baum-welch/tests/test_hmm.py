import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from hmm_toolkit import (HMM, random_hmm, forward, backward, likelihood,
                         posteriors, expected_transitions, viterbi,
                         posterior_decode, log_forward, log_viterbi, logsumexp,
                         baum_welch, supervised_fit, brute_force_likelihood,
                         brute_force_viterbi, match_states, state_accuracy,
                         bic, select_n_states)


def weather():
    return HMM([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]],
               [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])


def test_validation_rejects_bad_rows():
    with pytest.raises(ValueError):
        HMM([0.5, 0.5], [[0.9, 0.2], [0.5, 0.5]], [[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        HMM([0.5, 0.6], [[0.5, 0.5], [0.5, 0.5]], [[1, 0], [0, 1]])


def test_bad_observations_rejected():
    with pytest.raises(ValueError):
        forward(weather(), [0, 5])
    with pytest.raises(ValueError):
        forward(weather(), [])


def test_forward_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        h = random_hmm(3, 4, rng)
        x = rng.integers(0, 4, size=6)
        assert likelihood(h, x) == pytest.approx(brute_force_likelihood(h, x))


def test_log_forward_matches_scaled():
    rng = np.random.default_rng(1)
    h = random_hmm(4, 5, rng)
    x = h.sample(50, rng)[1]
    assert log_forward(h, x)[1] == pytest.approx(likelihood(h, x))


def test_long_sequence_no_underflow():
    rng = np.random.default_rng(2)
    h = weather()
    x = h.sample(5000, rng)[1]
    ll = likelihood(h, x)
    assert np.isfinite(ll) and ll < -1000
    assert log_forward(h, x)[1] == pytest.approx(ll, rel=1e-9)


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(5):
        h = random_hmm(3, 3, rng)
        x = rng.integers(0, 3, size=7)
        p, lp = viterbi(h, x)
        bp, blp = brute_force_viterbi(h, x)
        assert lp == pytest.approx(blp)
        from hmm_toolkit.analysis import _path_prob
        assert _path_prob(h, p, x) == pytest.approx(_path_prob(h, bp, x))


def test_log_viterbi_agrees():
    rng = np.random.default_rng(4)
    h = random_hmm(4, 6, rng)
    x = h.sample(80, rng)[1]
    p1, l1 = viterbi(h, x)
    p2, l2 = log_viterbi(h, x)
    assert l1 == pytest.approx(l2) and np.array_equal(p1, p2)


def test_posteriors_normalised_and_consistent():
    rng = np.random.default_rng(5)
    h = random_hmm(3, 4, rng)
    x = h.sample(30, rng)[1]
    g = posteriors(h, x)
    assert np.allclose(g.sum(axis=1), 1)
    xi = expected_transitions(h, x)
    assert xi.sum() == pytest.approx(len(x) - 1)
    assert np.allclose(xi.sum(axis=1), g[:-1].sum(axis=0))


def test_posteriors_match_enumeration():
    h = weather()
    x = np.array([0, 2, 1, 2])
    from itertools import product
    from hmm_toolkit.analysis import _path_prob
    g = np.zeros((4, 2))
    for z in product(range(2), repeat=4):
        p = _path_prob(h, z, x)
        for t, s in enumerate(z):
            g[t, s] += p
    g /= g.sum(axis=1, keepdims=True)
    assert np.allclose(posteriors(h, x), g)


def test_backward_forward_identity():
    rng = np.random.default_rng(6)
    h = random_hmm(3, 3, rng)
    x = h.sample(20, rng)[1]
    alpha, c = forward(h, x)
    beta = backward(h, x, c)
    # sum_i alpha_hat_t(i) beta_hat_t(i) == 1 for every t
    assert np.allclose((alpha * beta).sum(axis=1), 1)


def test_baum_welch_monotone():
    rng = np.random.default_rng(7)
    true = weather()
    x = true.sample(600, rng)[1]
    res = baum_welch(x, 2, np.random.default_rng(8), n_symbols=3, tol=0, max_iter=60)
    d = np.diff(res.log_likelihoods[:-1])
    assert np.all(d > -1e-7)
    assert res.final_ll > likelihood(random_hmm(2, 3, np.random.default_rng(9)), x)


def test_baum_welch_recovers_parameters():
    rng = np.random.default_rng(10)
    true = HMM([0.5, 0.5], [[0.9, 0.1], [0.15, 0.85]],
               [[0.8, 0.15, 0.05], [0.05, 0.25, 0.7]])
    seqs = [true.sample(400, rng)[1] for _ in range(5)]
    res = baum_welch(seqs, 2, np.random.default_rng(11), n_symbols=3,
                     n_restarts=4, max_iter=300)
    perm = np.argsort(res.hmm.B[:, 2])
    A, B = res.hmm.A[np.ix_(perm, perm)], res.hmm.B[perm]
    assert np.abs(A - true.A).max() < 0.06
    assert np.abs(B - true.B).max() < 0.06
    assert res.final_ll >= sum(likelihood(true, s) for s in seqs) - 15


def test_supervised_fit_close_to_truth():
    rng = np.random.default_rng(12)
    true = weather()
    data = [true.sample(500, rng) for _ in range(10)]
    h = supervised_fit([d[0] for d in data], [d[1] for d in data], 2, 3)
    assert np.abs(h.A - true.A).max() < 0.05
    assert np.abs(h.B - true.B).max() < 0.05


def test_match_states_and_accuracy():
    z = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])
    assert state_accuracy(z, pred, 3) == 1.0
    assert np.array_equal(match_states(z, pred, 3)[pred], z)


def test_viterbi_beats_chance_on_sticky_chain():
    rng = np.random.default_rng(13)
    h = HMM([0.5, 0.5], [[0.95, 0.05], [0.05, 0.95]], [[0.8, 0.2], [0.2, 0.8]])
    z, x = h.sample(2000, rng)
    assert np.mean(viterbi(h, x)[0] == z) > 0.85
    assert np.mean(posterior_decode(h, x) == z) >= np.mean(viterbi(h, x)[0] == z) - 0.01


def test_logsumexp_handles_neg_inf():
    assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf
    assert logsumexp(np.array([0.0, 0.0])) == pytest.approx(np.log(2))


def test_stationary_distribution():
    h = weather()
    s = h.stationary()
    assert np.allclose(s @ h.A, s) and s.sum() == pytest.approx(1)


def test_bic_selects_true_order():
    rng = np.random.default_rng(14)
    true = HMM([0.5, 0.5], [[0.95, 0.05], [0.05, 0.95]], [[0.9, 0.1], [0.1, 0.9]])
    seqs = [true.sample(600, rng)[1] for _ in range(3)]
    scores = select_n_states(seqs, [1, 2, 3], 2, np.random.default_rng(15))
    assert min(scores, key=scores.get) == 2
    assert bic(-100.0, 2, 3, 50) > 200
