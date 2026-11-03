import math

import numpy as np
import pytest

from smoothing_cert import (ABSTAIN, LinearSoftmax, MLPClassifier, SmoothedClassifier,
                            accuracy, binom_two_sided_p, certified_accuracy,
                            clopper_pearson_lower, clopper_pearson_upper, exact_linear_radius,
                            make_blobs, make_rings, minimal_l2_boundary_distance, norm_cdf,
                            norm_ppf, pgd_l2)
from smoothing_cert.evaluate import certify_dataset


@pytest.fixture(scope="module")
def linear_setup():
    X, y = make_blobs(600, dim=6, classes=2, sep=3.0, seed=1)
    m = LinearSoftmax(6, 2, seed=0)
    m.fit(X, y, epochs=40, lr=0.05, noise_sigma=0.25, seed=0)
    return m, X, y


def test_norm_ppf_inverts_cdf():
    for p in [1e-6, 0.01, 0.2, 0.5, 0.77, 0.99, 1 - 1e-6]:
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, rel=1e-8)
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)


def test_clopper_pearson_known_values():
    # all successes: lower bound is alpha^(1/n)
    assert clopper_pearson_lower(20, 20, 0.05) == pytest.approx(0.05 ** (1 / 20), abs=1e-8)
    assert clopper_pearson_lower(0, 20, 0.05) == 0.0
    assert clopper_pearson_upper(20, 20, 0.05) == 1.0
    lo, hi = clopper_pearson_lower(50, 100, 0.025), clopper_pearson_upper(50, 100, 0.025)
    assert lo == pytest.approx(0.3983, abs=1e-3) and hi == pytest.approx(0.6017, abs=1e-3)


def test_binomial_test():
    assert binom_two_sided_p(5, 10) == pytest.approx(1.0)
    assert binom_two_sided_p(10, 10) == pytest.approx(2 / 1024)


def test_training_learns_blobs_and_rings():
    X, y = make_blobs(500, dim=8, classes=3, seed=2)
    m = LinearSoftmax(8, 3)
    losses = m.fit(X, y, epochs=30)
    assert losses[-1] < losses[0] and accuracy(m, X, y) > 0.9
    Xr, yr = make_rings(600, seed=3)
    mlp = MLPClassifier(2, 32, 2, seed=1)
    mlp.fit(Xr, yr, epochs=80, lr=0.02)
    assert accuracy(mlp, Xr, yr) > 0.95


@pytest.mark.parametrize("cls", ["lin", "mlp"])
def test_input_gradient_matches_finite_difference(cls):
    rng = np.random.default_rng(0)
    m = LinearSoftmax(5, 3, seed=1) if cls == "lin" else MLPClassifier(5, 7, 3, seed=1)
    if cls == "lin":
        m.W = rng.normal(size=m.W.shape)
    x = rng.normal(size=(1, 5)); y = np.array([1])
    def loss(v):
        return -np.log(m.proba(v)[0, 1])
    g = m.input_gradient(x, y)[0]
    for i in range(5):
        e = np.zeros((1, 5)); e[0, i] = 1e-6
        assert g[i] == pytest.approx((loss(x + e) - loss(x - e)) / 2e-6, abs=1e-5)


def test_certify_radius_matches_closed_form_for_linear_binary(linear_setup):
    m, X, y = linear_setup
    sm = SmoothedClassifier(m, 2, sigma=0.5, seed=0)
    x = X[0]
    res = sm.certify(x, n0=200, n=20000, alpha=0.001)
    # Binary linear: pA = Phi(margin / (sigma ||w||)), so radius = exact distance to boundary.
    exact = exact_linear_radius(m, x)
    assert res.label == m.predict(x)[0]
    assert 0 < res.radius <= exact + 1e-9
    assert res.radius > 0.6 * exact or exact < 0.3


def test_radius_from_pa_formula():
    assert SmoothedClassifier.radius_from_pa(0.5, 1.0) == pytest.approx(0.0, abs=1e-9)
    assert SmoothedClassifier.radius_from_pa(0.8413447, 1.0) == pytest.approx(1.0, abs=1e-4)
    r1 = SmoothedClassifier.radius_from_pa(0.9, 0.5)
    r2 = SmoothedClassifier.radius_from_pa(0.9, 1.0)
    assert r2 == pytest.approx(2 * r1)


def test_certificate_is_sound_against_pgd_on_linear(linear_setup):
    m, X, y = linear_setup
    sm = SmoothedClassifier(m, 2, sigma=0.5, seed=1)
    checked = 0
    for x, t in zip(X[:25], y[:25]):
        r = sm.certify(x, n0=100, n=2000)
        if r.label == ABSTAIN:
            continue
        xa = pgd_l2(m, x, int(t), eps=0.95 * r.radius, steps=30)
        # base model within radius may still flip, but the smoothed vote must not
        assert sm.predict(xa, n=2000, alpha=0.01) in (r.label, ABSTAIN) or r.label != t
        checked += 1
    assert checked > 10


def test_abstain_on_ambiguous_point():
    X, y = make_blobs(400, dim=4, classes=2, seed=5)
    m = LinearSoftmax(4, 2); m.fit(X, y, epochs=20)
    # point exactly on the decision boundary
    w = m.W[:, 0] - m.W[:, 1]; b = m.b[0] - m.b[1]
    x = -b * w / (w @ w)
    sm = SmoothedClassifier(m, 2, sigma=1.0, seed=0)
    res = sm.certify(x, n0=100, n=1000)
    assert res.label == ABSTAIN and res.radius == 0.0
    assert sm.predict(x, n=1000, alpha=1e-4) == ABSTAIN


def test_radius_grows_with_sigma_for_confident_point(linear_setup):
    m, X, y = linear_setup
    x = X[np.argmax(np.abs(m.logits(X)[:, 0] - m.logits(X)[:, 1]))]
    radii = [SmoothedClassifier(m, 2, s, seed=3).certify(x, n0=100, n=3000).radius
             for s in (0.1, 0.4)]
    assert radii[1] > radii[0] > 0


def test_certified_accuracy_monotone_and_bounded():
    X, y = make_blobs(300, dim=8, classes=3, sep=4.0, seed=7)
    m = MLPClassifier(8, 24, 3, seed=0); m.fit(X, y, epochs=40, noise_sigma=0.5)
    sm = SmoothedClassifier(m, 3, 0.5, seed=0)
    res = certify_dataset(sm, X[:40], n0=30, n=300)
    accs = [certified_accuracy(res, y[:40], r) for r in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert all(a >= b for a, b in zip(accs, accs[1:]))
    assert accs[0] > 0.7 and accs[-1] <= accs[0]
    # max radius achievable with n=300 and alpha=.001
    cap = 0.5 * norm_ppf(0.001 ** (1 / 300))  # sigma * Phi^-1(alpha^(1/n))
    assert all(r.radius <= cap + 1e-9 for r in res)


def test_determinism_with_seed(linear_setup):
    m, X, _ = linear_setup
    a = SmoothedClassifier(m, 2, 0.5, seed=9).certify(X[3])
    b = SmoothedClassifier(m, 2, 0.5, seed=9).certify(X[3])
    assert a == b


def test_boundary_distance_upper_bounds_exact_radius(linear_setup):
    m, X, y = linear_setup
    x = X[1]
    exact = exact_linear_radius(m, x)
    found = minimal_l2_boundary_distance(m, x, int(m.predict(x)[0]), hi=30.0)
    assert found >= exact - 1e-6
    assert found == pytest.approx(exact, rel=0.02)


def test_invalid_sigma():
    with pytest.raises(ValueError):
        SmoothedClassifier(LinearSoftmax(2, 2), 2, sigma=0.0)
