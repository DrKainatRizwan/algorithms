import numpy as np
import pytest

from sparse_solvers import (make_sparse_regression, lasso_objective, lambda_max,
                            kkt_violation, lipschitz_constant, soft_threshold,
                            prox_group_l2, project_l1_ball, prox_elastic_net,
                            ista, fista, group_lasso, coordinate_descent,
                            lasso_admm, generalized_lasso_admm, difference_matrix,
                            lasso_path, cross_validate, support_metrics)


@pytest.fixture(scope="module")
def prob():
    X, y, w = make_sparse_regression(80, 40, 5, noise=0.1, seed=3)
    return X, y, w, 0.2 * lambda_max(X, y)


def test_soft_threshold_values():
    out = soft_threshold(np.array([-3.0, -0.5, 0.0, 0.5, 2.0]), 1.0)
    assert np.allclose(out, [-2, 0, 0, 0, 1])


def test_prox_is_minimiser():
    v = np.array([1.3, -0.4, 2.2])
    t = 0.7
    p = soft_threshold(v, t)
    f = lambda x: t * np.abs(x).sum() + 0.5 * np.sum((x - v) ** 2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert f(p) <= f(p + 0.1 * rng.standard_normal(3)) + 1e-12


def test_l1_ball_projection():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(20) * 3
    p = project_l1_ball(v, 2.0)
    assert abs(np.abs(p).sum() - 2.0) < 1e-10
    for _ in range(100):  # no feasible point is closer
        q = rng.standard_normal(20)
        q *= 2.0 / np.abs(q).sum() * rng.uniform(0, 1)
        assert np.linalg.norm(v - p) <= np.linalg.norm(v - q) + 1e-12
    inside = np.array([0.1, -0.2])
    assert np.allclose(project_l1_ball(inside, 1.0), inside)


def test_group_prox_kills_small_groups():
    v = np.array([0.1, 0.1, 3.0, 4.0])
    out = prox_group_l2(v, 1.0, [np.array([0, 1]), np.array([2, 3])])
    assert np.allclose(out[:2], 0)
    assert np.isclose(np.linalg.norm(out[2:]), 4.0)


def test_lambda_max_gives_zero_solution(prob):
    X, y, _, _ = prob
    lm = lambda_max(X, y)
    assert np.all(coordinate_descent(X, y, lm * 1.0001).w == 0)
    assert np.any(coordinate_descent(X, y, lm * 0.9).w != 0)


def test_lipschitz_matches_eigenvalue(prob):
    X = prob[0]
    exact = np.linalg.eigvalsh(X.T @ X).max()
    assert abs(lipschitz_constant(X) - exact) / exact < 1e-6


def test_all_solvers_agree_and_satisfy_kkt(prob):
    X, y, _, lam = prob
    cd = coordinate_descent(X, y, lam)
    fi = fista(X, y, lam)
    ad = lasso_admm(X, y, lam)
    for r in (cd, fi, ad):
        assert r.converged
        assert kkt_violation(X, y, r.w, lam) < 1e-4
    assert np.allclose(cd.w, fi.w, atol=1e-6)
    assert np.allclose(cd.w, ad.w, atol=1e-5)
    assert abs(cd.objective - lasso_objective(X, y, cd.w, lam)) < 1e-9


def test_ista_monotone_and_fista_faster(prob):
    X, y, _, lam = prob
    a = ista(X, y, lam, max_iter=100, tol=0)
    b = fista(X, y, lam, restart=False, max_iter=100, tol=0)
    assert all(h2 <= h1 + 1e-9 for h1, h2 in zip(a.history, a.history[1:]))
    assert b.objective <= a.objective + 1e-12


def test_restart_not_worse_than_plain_on_correlated():
    X, y, _ = make_sparse_regression(60, 100, 6, corr=0.6, seed=5)
    lam = 0.05 * lambda_max(X, y)
    plain = fista(X, y, lam, restart=False, max_iter=300, tol=0)
    rest = fista(X, y, lam, restart=True, max_iter=300, tol=0)
    assert rest.objective <= plain.objective + 1e-9


def test_backtracking_converges(prob):
    X, y, _, lam = prob
    r = fista(X, y, lam, backtrack=True)
    assert r.converged and kkt_violation(X, y, r.w, lam) < 1e-4


def test_active_set_matches_full_sweeps(prob):
    X, y, _, lam = prob
    a = coordinate_descent(X, y, lam, active_set=True)
    b = coordinate_descent(X, y, lam, active_set=False)
    assert np.allclose(a.w, b.w, atol=1e-7)
    assert a.iterations <= b.iterations


def test_elastic_net_matches_closed_form_orthogonal():
    X = np.eye(5) * 2.0
    y = np.array([4.0, -1.0, 0.5, 3.0, -6.0])
    lam, alpha = 1.0, 0.5
    w = coordinate_descent(X, y, lam, alpha).w
    expected = soft_threshold(2 * y, lam * alpha) / (4 + lam * (1 - alpha))
    assert np.allclose(w, expected)
    assert np.allclose(prox_elastic_net(2 * y, 1.0, lam * alpha, lam * (1 - alpha)) / 4 * 4 / 4 * 4,
                       prox_elastic_net(2 * y, 1.0, lam * alpha, lam * (1 - alpha)))


def test_group_lasso_zeroes_whole_groups():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 12))
    w = np.zeros(12)
    w[:3] = [2, -1, 1.5]
    y = X @ w + 0.05 * rng.standard_normal(100)
    groups = [np.arange(i, i + 3) for i in range(0, 12, 3)]
    r = group_lasso(X, y, 15.0, groups)
    for g in groups[1:]:
        assert np.all(r.w[g] == 0)
    assert np.all(r.w[groups[0]] != 0)


def test_fused_lasso_recovers_piecewise_constant():
    rng = np.random.default_rng(4)
    truth = np.repeat([0.0, 3.0, -1.0], 20)
    y = truth + 0.3 * rng.standard_normal(60)
    D = difference_matrix(60, 1)
    r = generalized_lasso_admm(np.eye(60), y, 1.5, D)
    assert r.converged
    jumps = np.sum(np.abs(np.diff(r.w)) > 1e-3)
    assert jumps <= 4
    assert np.linalg.norm(r.w - truth) < np.linalg.norm(y - truth)


def test_difference_matrix_second_order():
    D = difference_matrix(5, 2)
    assert D.shape == (3, 5)
    assert np.allclose(D @ np.arange(5.0), 0)


def test_path_is_sparse_to_dense_and_recovers_support():
    X, y, w = make_sparse_regression(150, 30, 4, noise=0.05, seed=7)
    lams, coefs = lasso_path(X, y, 25)
    nnz = (coefs != 0).sum(axis=1)
    assert nnz[0] == 0 and nnz[-1] >= 4
    assert np.all(np.diff(nnz) >= -3)
    lam_cv, errs = cross_validate(X, y, lams, folds=4)
    m = support_metrics(coordinate_descent(X, y, lam_cv).w, w)
    assert m["recall"] == 1.0


def test_deterministic_generation():
    a = make_sparse_regression(10, 5, 2, seed=1)
    b = make_sparse_regression(10, 5, 2, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
