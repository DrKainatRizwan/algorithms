import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poison_robust import (LogisticRegression, boundary_flip, evaluate_grid, flip_influence,
                           format_table, gradient_poisoning, influence_label_flip, knn_sanitize,
                           poison_input_gradient, random_label_flip, removal_influence, sigmoid,
                           sphere_slab_filter, train_val_test_split, trim_fit,
                           trimmed_gradient_fit, validation_gradient)


@pytest.fixture(scope="module")
def ds():
    return train_val_test_split(n_train=200, n_val=200, n_test=600, d=2, separation=2.5, seed=3)


def test_sigmoid_stable_and_symmetric():
    z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    assert s[0] == 0.0 and s[-1] == 1.0 and s[2] == 0.5
    assert np.allclose(s + sigmoid(-z), 1.0)


def test_gradient_matches_finite_difference(ds):
    m = LogisticRegression(0.7)
    rng = np.random.default_rng(0)
    w = rng.normal(size=3)
    g = m.grad(ds.X_train, ds.y_train, w)
    num = np.zeros(3)
    for i in range(3):
        e = np.zeros(3); e[i] = 1e-6
        num[i] = (m.loss(ds.X_train, ds.y_train, w + e) - m.loss(ds.X_train, ds.y_train, w - e)) / 2e-6
    assert np.allclose(g, num, rtol=1e-5, atol=1e-5)


def test_hessian_matches_finite_difference(ds):
    m = LogisticRegression(0.7)
    w = np.array([0.3, -0.2, 0.1])
    H = m.hessian(ds.X_train, w)
    num = np.zeros((3, 3))
    for i in range(3):
        e = np.zeros(3); e[i] = 1e-6
        num[:, i] = (m.grad(ds.X_train, ds.y_train, w + e) - m.grad(ds.X_train, ds.y_train, w - e)) / 2e-6
    assert np.allclose(H, num, rtol=1e-5, atol=1e-5)


def test_fit_reaches_stationary_point_and_good_accuracy(ds):
    m = LogisticRegression(1.0).fit(ds.X_train, ds.y_train)
    assert np.linalg.norm(m.grad(ds.X_train, ds.y_train)) < 1e-7
    assert m.accuracy(ds.X_test, ds.y_test) > 0.8


def test_sample_weights_equal_duplication(ds):
    X, y = ds.X_train[:50], ds.y_train[:50]
    wts = np.ones(50); wts[:10] = 2.0
    a = LogisticRegression(1.0).fit(X, y, sample_weight=wts)
    b = LogisticRegression(1.0).fit(np.vstack([X, X[:10]]), np.concatenate([y, y[:10]]))
    assert np.allclose(a.w, b.w, atol=1e-8)


def test_removal_influence_predicts_retraining(ds):
    X, y = ds.X_train, ds.y_train
    m = LogisticRegression(1.0).fit(X, y)
    pred = removal_influence(m, X, y, ds.X_val, ds.y_val)
    base = m.loss(ds.X_val, ds.y_val, reg=False)
    actual = []
    for i in range(40):
        keep = np.arange(len(y)) != i
        mi = LogisticRegression(1.0).fit(X[keep], y[keep])
        actual.append(mi.loss(ds.X_val, ds.y_val, reg=False) - base)
    assert np.corrcoef(pred[:40], actual)[0, 1] > 0.95


def test_flip_influence_predicts_retraining(ds):
    X, y = ds.X_train, ds.y_train
    m = LogisticRegression(1.0).fit(X, y)
    pred = flip_influence(m, X, y, ds.X_val, ds.y_val)
    base = m.loss(ds.X_val, ds.y_val, reg=False)
    actual = []
    for i in range(40):
        y2 = y.copy(); y2[i] = 1 - y2[i]
        actual.append(LogisticRegression(1.0).fit(X, y2).loss(ds.X_val, ds.y_val, reg=False) - base)
    assert np.corrcoef(pred[:40], actual)[0, 1] > 0.9


def test_poison_input_gradient_matches_finite_difference(ds):
    X, y = ds.X_train[:100], ds.y_train[:100]
    x_c, y_c = np.array([0.8, -0.4]), 1.0
    Xa, ya = np.vstack([X, x_c]), np.append(y, y_c)
    m = LogisticRegression(1.0).fit(Xa, ya)
    g = poison_input_gradient(m, Xa, x_c, y_c, ds.X_val, ds.y_val)

    def val(x):
        mm = LogisticRegression(1.0, tol=1e-13).fit(np.vstack([X, x]), np.append(y, y_c))
        return mm.loss(ds.X_val, ds.y_val, reg=False)

    num = np.zeros(2)
    for j in range(2):
        e = np.zeros(2); e[j] = 1e-5
        num[j] = (val(x_c + e) - val(x_c - e)) / 2e-5
    assert np.allclose(g, num, rtol=1e-3, atol=1e-4)


def test_random_flip_budget_and_mask(ds):
    r = random_label_flip(ds.X_train, ds.y_train, 0.15, np.random.default_rng(0))
    assert r.n_poisoned == 30
    assert np.all((r.y != ds.y_train) == r.poisoned)
    assert np.array_equal(r.X, ds.X_train)


def test_influence_flip_stronger_than_random(ds):
    rnd = random_label_flip(ds.X_train, ds.y_train, 0.2, np.random.default_rng(0))
    inf = influence_label_flip(ds.X_train, ds.y_train, ds.X_val, ds.y_val, 0.2)
    assert inf.n_poisoned == 40
    a_rnd = LogisticRegression(1.0).fit(rnd.X, rnd.y).accuracy(ds.X_test, ds.y_test)
    a_inf = LogisticRegression(1.0).fit(inf.X, inf.y).accuracy(ds.X_test, ds.y_test)
    assert a_inf < a_rnd
    assert inf.history[-1] > inf.history[0]


def test_boundary_flip_selects_small_margin(ds):
    r = boundary_flip(ds.X_train, ds.y_train, 0.1)
    m = LogisticRegression(1.0).fit(ds.X_train, ds.y_train)
    margin = np.abs(m.decision_function(ds.X_train))
    assert margin[r.poisoned].max() <= np.sort(margin)[r.n_poisoned - 1] + 1e-12


def test_gradient_poisoning_increases_validation_loss(ds):
    r = gradient_poisoning(ds.X_train, ds.y_train, ds.X_val, ds.y_val, 0.1,
                           np.random.default_rng(0), steps=15)
    assert r.X.shape[0] == 220 and r.poisoned.sum() == 20
    clean = LogisticRegression(1.0).fit(ds.X_train, ds.y_train).loss(ds.X_val, ds.y_val, reg=False)
    assert max(r.history) > r.history[0] > clean * 0.99
    m = LogisticRegression(1.0).fit(r.X, r.y)
    assert m.loss(ds.X_val, ds.y_val, reg=False) > clean * 1.2


def test_gradient_poisoning_zero_budget(ds):
    r = gradient_poisoning(ds.X_train, ds.y_train, ds.X_val, ds.y_val, 0.0, np.random.default_rng(0))
    assert r.n_poisoned == 0 and r.X.shape == ds.X_train.shape


def test_trim_recovers_from_influence_flips(ds):
    inf = influence_label_flip(ds.X_train, ds.y_train, ds.X_val, ds.y_val, 0.2)
    plain = LogisticRegression(1.0).fit(inf.X, inf.y).accuracy(ds.X_test, ds.y_test)
    out = trim_fit(inf.X, inf.y, alpha=0.2)
    assert out.model.accuracy(ds.X_test, ds.y_test) > plain + 0.02
    assert out.kept.sum() == 160


def test_sphere_slab_removes_far_outliers(ds):
    X = np.vstack([ds.X_train, np.full((10, 2), 12.0)])
    y = np.concatenate([ds.y_train, np.zeros(10, dtype=int)])
    out = sphere_slab_filter(X, y, remove_frac=0.1)
    assert not out.kept[-10:].any()
    assert out.kept[:-10].mean() > 0.85


def test_knn_sanitize_catches_isolated_flips(ds):
    r = random_label_flip(ds.X_train, ds.y_train, 0.1, np.random.default_rng(1))
    out = knn_sanitize(r.X, r.y, k=9)
    dropped = ~out.kept
    assert (dropped & r.poisoned).sum() / r.n_poisoned > 0.6


def test_trimmed_gradient_resists_extreme_points(ds):
    X = np.vstack([ds.X_train, np.full((20, 2), -30.0)])
    y = np.concatenate([ds.y_train, np.ones(20, dtype=int)])
    plain = LogisticRegression(1.0).fit(X, y).accuracy(ds.X_test, ds.y_test)
    robust = trimmed_gradient_fit(X, y, beta=0.12).model.accuracy(ds.X_test, ds.y_test)
    assert robust > plain


def test_validation_gradient_zero_when_val_equals_train(ds):
    m = LogisticRegression(0.0 + 1e-9).fit(ds.X_train, ds.y_train)
    g = validation_gradient(m, ds.X_train, ds.y_train)
    assert np.linalg.norm(g) < 1e-3


def test_evaluate_grid_shape_and_determinism(ds):
    kw = dict(attacks=("random-flip", "influence-flip"), defenses=("none", "trim"), fracs=(0.1,))
    a = evaluate_grid(ds, **kw)
    b = evaluate_grid(ds, **kw)
    assert len(a) == 4
    assert a == b
    assert all(0.0 <= r["test_acc"] <= 1.0 for r in a)
    table = format_table(a)
    assert "influence-flip" in table and "trim" in table
