import numpy as np
import pytest
from xai_toolkit import *


@pytest.fixture(scope="module")
def setup():
    X, y, w = make_tabular(500, 6, 3, seed=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, seed=1)
    lg = LogisticModel().fit(Xtr, ytr)
    return Xtr, Xte, ytr, yte, lg


def test_data_shapes_and_determinism():
    a = make_tabular(50, 5, 2, seed=3); b = make_tabular(50, 5, 2, seed=3)
    assert a[0].shape == (50, 5) and np.array_equal(a[1], b[1])
    assert np.count_nonzero(a[2]) == 2


def test_logistic_learns(setup):
    _, Xte, _, yte, lg = setup
    assert np.mean((lg.predict(Xte) > 0.5) == yte) > 0.7


def test_mlp_and_gradient_fd():
    X, y, _ = make_tabular(300, 4, 3, seed=2)
    m = MLPClassifier(hidden=8, epochs=40, seed=0).fit(X, y)
    g = m.gradient(X[:3]); eps = 1e-6
    for j in range(4):
        e = np.zeros(4); e[j] = eps
        fd = (m.predict(X[:3] + e) - m.predict(X[:3] - e)) / (2 * eps)
        assert np.allclose(g[:, j], fd, atol=1e-6)


def test_tree_and_forest_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)); y = (X[:, 0] > 0).astype(float)
    t = RegressionTree(3).fit(X, y)
    assert np.mean((t.predict(X) > 0.5) == y) > 0.95
    rf = RandomForest(10, 4, seed=0).fit(X, y)
    assert np.mean((rf.predict(X) > 0.5) == y) > 0.9


def test_exact_shapley_linear_closed_form(setup):
    Xtr, Xte, _, _, lg = setup
    bg = Xtr[:40]; x = Xte[0]
    vf = ValueFunction(lg.decision, x, bg)
    assert np.allclose(exact_shapley(vf), linear_shap(lg.w, x, bg), atol=1e-9)


def test_exact_shapley_efficiency_nonlinear(setup):
    Xtr, Xte, _, _, lg = setup
    vf = ValueFunction(lg.predict, Xte[1], Xtr[:30])
    phi = exact_shapley(vf)
    assert efficiency_gap(phi, vf.base, vf.full) < 1e-10


def test_shapley_symmetry_and_null_player():
    f = lambda Z: Z[:, 0] + Z[:, 1] + 0 * Z[:, 2]
    bg = np.zeros((1, 3)); x = np.array([2.0, 2.0, 5.0])
    phi = exact_shapley(ValueFunction(f, x, bg))
    assert np.isclose(phi[0], phi[1]) and abs(phi[2]) < 1e-12


def test_interaction_split_equally():
    f = lambda Z: Z[:, 0] * Z[:, 1]
    phi = exact_shapley(ValueFunction(f, np.array([3.0, 2.0]), np.zeros((1, 2))))
    assert np.allclose(phi, [3.0, 3.0])


def test_kernel_shap_full_enumeration_matches_exact(setup):
    Xtr, Xte, _, _, lg = setup
    vf = ValueFunction(lg.predict, Xte[2], Xtr[:30])
    assert np.allclose(kernel_shap(vf), exact_shapley(vf), atol=1e-8)


def test_kernel_shap_sampled_close_and_efficient(setup):
    Xtr, Xte, _, _, lg = setup
    vf = ValueFunction(lg.predict, Xte[3], Xtr[:30])
    ex = exact_shapley(vf)
    ks = kernel_shap(vf, n_samples=60, seed=0)
    assert efficiency_gap(ks, vf.base, vf.full) < 1e-9
    assert np.max(np.abs(ks - ex)) < 0.05


def test_permutation_shapley_efficiency_and_accuracy(setup):
    Xtr, Xte, _, _, lg = setup
    vf = ValueFunction(lg.predict, Xte[4], Xtr[:30])
    ps = permutation_shapley(vf, 200, seed=1)
    assert efficiency_gap(ps, vf.base, vf.full) < 1e-9
    assert np.max(np.abs(ps - exact_shapley(vf))) < 0.03


def test_kernel_weight_symmetry():
    assert shapley_kernel_weight(6, 2) == pytest.approx(shapley_kernel_weight(6, 4))


def test_lime_recovers_linear_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)); w = np.array([2.0, -1.0, 0.0, 0.0, 0.5])
    lime = LimeTabular(X, n_samples=1500, seed=0)
    ex = lime.explain(X[0], lambda Z: Z @ w + 1.0)
    assert np.allclose(ex.coef, w, atol=0.02) and ex.r2 > 0.999


def test_lime_sparse_selection_picks_important():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6))
    f = lambda Z: 3 * Z[:, 2] - 2 * Z[:, 4]
    ex = LimeTabular(X, seed=0).explain(X[0], f, n_features=2)
    assert set(ex.selected.tolist()) == {2, 4}
    assert ex.top(1)[0][0] == 2


def test_lime_deterministic(setup):
    Xtr, Xte, _, _, lg = setup
    a = LimeTabular(Xtr, seed=5).explain(Xte[0], lg.predict).coef
    b = LimeTabular(Xtr, seed=5).explain(Xte[0], lg.predict).coef
    assert np.array_equal(a, b)


def test_integrated_gradients_completeness():
    X, y, _ = make_tabular(300, 4, 3, seed=2)
    m = MLPClassifier(hidden=8, epochs=30, seed=0).fit(X, y)
    x, b = X[0], X.mean(axis=0)
    ig = integrated_gradients(m.gradient, x, b, steps=200)
    assert abs(ig.sum() - (m.predict(x[None])[0] - m.predict(b[None])[0])) < 1e-3


def test_deletion_prefers_good_attribution(setup):
    Xtr, Xte, _, _, lg = setup
    x = Xte[0]; bg = Xtr[:50]
    phi = exact_shapley(ValueFunction(lg.decision, x, bg))
    sign = 1 if lg.decision(x[None])[0] > 0 else -1
    good = deletion_curve(lambda Z: sign * lg.decision(Z), x, phi * sign, bg.mean(0))
    rnd = deletion_curve(lambda Z: sign * lg.decision(Z), x, -phi * sign, bg.mean(0))
    assert deletion_auc(good) <= deletion_auc(rnd) + 1e-12
    assert len(good) == 7


def test_rank_correlation_and_overlap():
    a = np.array([1., 2, 3, 4])
    assert rank_correlation(a, a) == pytest.approx(1.0)
    assert rank_correlation(a, -a) == pytest.approx(-1.0)
    assert topk_overlap(a, a[::-1], 2) == 0.0


def test_tree_model_shap_efficiency():
    X, y, _ = make_tabular(300, 5, 3, seed=4)
    rf = RandomForest(8, 4, seed=0).fit(X, y)
    vf = ValueFunction(rf.predict, X[0], X[:20])
    phi = exact_shapley(vf)
    assert efficiency_gap(phi, vf.base, vf.full) < 1e-10


def test_stability_metric_range(setup):
    Xtr, Xte, _, _, lg = setup
    s = explanation_stability(lambda sd: LimeTabular(Xtr, seed=sd).explain(Xte[0], lg.predict).coef, [0, 1, 2])
    assert 0.5 < s <= 1.0
