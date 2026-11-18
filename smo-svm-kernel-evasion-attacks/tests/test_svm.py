import numpy as np
import pytest

from smo_svm import (SVC, OneVsOneSVC, OneVsRestSVC, make_kernel, make_blobs, make_moons,
                     make_circles, make_xor, train_test_split, deepfool, pgd_linf,
                     robust_accuracy, minimum_distance, kfold_indices, cross_val_score,
                     grid_search)


@pytest.mark.parametrize("name", ["linear", "poly", "rbf"])
def test_gram_symmetric_psd(name):
    X = np.random.default_rng(0).standard_normal((30, 3))
    K = make_kernel(name, gamma=0.5, degree=2)(X, X)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8


@pytest.mark.parametrize("name", ["linear", "poly", "rbf"])
def test_kernel_gradient_finite_difference(name):
    rng = np.random.default_rng(1)
    k = make_kernel(name, gamma=0.3, degree=3)
    Z, x = rng.standard_normal((5, 3)), rng.standard_normal(3)
    g = k.grad(x, Z)
    h = 1e-6
    for d in range(3):
        e = np.zeros(3); e[d] = h
        num = (k(x + e, Z) - k(x - e, Z))[0] / (2 * h)
        assert np.allclose(g[:, d], num, rtol=1e-4, atol=1e-6)


def test_unknown_kernel():
    with pytest.raises(ValueError):
        make_kernel("nope")(np.zeros((1, 2)), np.zeros((1, 2)))


def test_blobs_linear_accuracy():
    X, y = make_blobs(200, seed=1)
    Xa, Xb, ya, yb = train_test_split(X, y, seed=1)
    assert SVC(C=1, kernel="linear").fit(Xa, ya).score(Xb, yb) > 0.95


def test_dual_constraints_and_kkt():
    X, y = make_moons(120, noise=0.2, seed=2)
    m = SVC(C=2.0, kernel="rbf", gamma=1.0).fit(X, y)
    assert abs(np.sum(m._alpha * m._t)) < 1e-6
    assert m._alpha.min() >= 0 and m._alpha.max() <= 2.0 + 1e-12
    assert m.kkt_violation() < 0.05


def test_duality_gap_small():
    X, y = make_blobs(100, std=1.5, seed=3)
    m = SVC(C=1.0, kernel="linear", tol=1e-4).fit(X, y)
    p, d = m.primal_objective(), m.dual_objective()
    assert p >= d - 1e-6
    assert (p - d) / max(1.0, abs(p)) < 0.02


def test_rbf_beats_linear_on_circles():
    X, y = make_circles(200, seed=4)
    lin = SVC(kernel="linear").fit(X, y).score(X, y)
    rbf = SVC(kernel="rbf", gamma=2.0, C=5).fit(X, y).score(X, y)
    assert lin < 0.75 and rbf > 0.97


def test_poly_solves_xor():
    X, y = make_xor(200, seed=5)
    assert SVC(kernel="poly", degree=2, C=5).fit(X, y).score(X, y) > 0.95


def test_deterministic_training():
    X, y = make_moons(100, seed=6)
    a = SVC(seed=3).fit(X, y)
    b = SVC(seed=3).fit(X, y)
    assert np.allclose(a._alpha, b._alpha) and a.intercept_ == b.intercept_


def test_arbitrary_labels_and_errors():
    X, y = make_blobs(60, seed=7)
    m = SVC(kernel="linear").fit(X, np.where(y == 1, 7, 3))
    assert set(m.predict(X)) <= {3, 7}
    with pytest.raises(ValueError):
        SVC().fit(X, np.arange(60) % 3)
    with pytest.raises(ValueError):
        SVC(C=0)


def test_decision_gradient_matches_numeric():
    X, y = make_moons(80, seed=8)
    m = SVC(kernel="rbf", gamma=1.0).fit(X, y)
    x = np.array([0.3, 0.2]); h = 1e-6
    num = [(m.decision_function(x + h * e)[0] - m.decision_function(x - h * e)[0]) / (2 * h)
           for e in np.eye(2)]
    assert np.allclose(m.gradient(x), num, atol=1e-5)


def test_multiclass_ovr_ovo():
    centers = np.array([[0, 0], [5, 0], [0, 5]], float)
    X, y = make_blobs(180, centers, std=0.7, seed=9)
    assert OneVsRestSVC(kernel="rbf", gamma=0.2).fit(X, y).score(X, y) > 0.95
    assert OneVsOneSVC(kernel="linear").fit(X, y).score(X, y) > 0.95


def test_deepfool_linear_is_exact_distance():
    X, y = make_blobs(100, seed=10)
    m = SVC(kernel="linear", C=1).fit(X, y)
    w = m.dual_coef_ @ m.support_vectors_
    x = X[0]
    adv = deepfool(m, x, overshoot=0.0)
    expected = abs(m.decision_function(x)[0]) / np.linalg.norm(w)
    assert np.linalg.norm(adv - x) == pytest.approx(expected, rel=1e-6)


def test_deepfool_flips_prediction_rbf():
    X, y = make_moons(100, seed=11)
    m = SVC(kernel="rbf", gamma=1.0).fit(X, y)
    for x in X[:10]:
        adv = deepfool(m, x, overshoot=0.05)
        assert m.predict(adv)[0] != m.predict(x)[0]
    assert (minimum_distance(m, X[:5]) > 0).all()


def test_pgd_respects_budget_and_hurts_accuracy():
    X, y = make_blobs(100, std=1.0, seed=12)
    m = SVC(kernel="linear").fit(X, y)
    adv = pgd_linf(m, X[0], y[0], eps=0.5)
    assert np.abs(adv - X[0]).max() <= 0.5 + 1e-12
    clean = m.score(X, y)
    assert robust_accuracy(m, X, y, 0.0) == pytest.approx(clean)
    assert robust_accuracy(m, X, y, 2.0) < clean


def test_cv_and_grid_search():
    X, y = make_moons(90, seed=13)
    folds = kfold_indices(90, 5, seed=0)
    allval = np.sort(np.concatenate([v for _, v in folds]))
    assert np.array_equal(allval, np.arange(90))
    s = cross_val_score(lambda: SVC(gamma=1.0), X, y, k=3)
    assert len(s) == 3 and s.mean() > 0.8
    best, score, res = grid_search(X, y, {"C": [0.1, 10], "gamma": [0.01, 2.0]}, k=3)
    assert len(res) == 4 and best["gamma"] == 2.0 and score >= max(r[1] for r in res)
