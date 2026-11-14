import itertools

import numpy as np
import pytest

from interior_point import (GeneralProblem, chebyshev_regression, l1_regression,
                            min_variance_portfolio, random_convex_qp,
                            random_feasible_lp, solve, solve_barrier, solve_lp,
                            solve_qp, svm_dual, to_standard_form)


def brute_force_lp(c, A, b):
    """Enumerate basic feasible solutions of min c'x, Ax=b, x>=0."""
    m, n = A.shape
    best = np.inf
    for cols in itertools.combinations(range(n), m):
        B = A[:, cols]
        if abs(np.linalg.det(B)) < 1e-10:
            continue
        xb = np.linalg.solve(B, b)
        if np.all(xb >= -1e-9):
            best = min(best, float(c[list(cols)] @ xb))
    return best


def test_textbook_lp():
    # max 3x+5y: x<=4, 2y<=12, 3x+2y<=18  -> optimum 36 at (2, 6)
    r = solve_lp([-3, -5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18])
    assert r.success
    assert np.allclose(r.x, [2, 6], atol=1e-6)
    assert r.objective == pytest.approx(-36, abs=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_random_lp_matches_vertex_enumeration(seed):
    rng = np.random.default_rng(seed)
    m, n = 3, 7
    A = rng.standard_normal((m, n))
    b = A @ rng.uniform(0.5, 2, n)
    c = rng.uniform(0.1, 2, n)  # c>0 and x>=0 -> bounded
    r = solve_lp(c, A_eq=A, b_eq=b)
    assert r.success
    assert r.objective == pytest.approx(brute_force_lp(c, A, b), abs=1e-6)


def test_lp_duality_gap_and_complementarity():
    p = random_feasible_lp(10, 25, seed=3)
    r = solve(p).raw
    assert r.converged
    assert abs(r.x @ r.s) < 1e-6
    assert np.all(r.x >= 0) and np.all(r.s >= 0)
    assert np.linalg.norm(p.A_eq @ r.x - p.b_eq) < 1e-6


def test_qp_kkt_conditions():
    p = random_convex_qp(4, 12, seed=1)
    sol = solve(p)
    assert sol.success
    raw = sol.raw
    grad = p.Q @ raw.x + p.c - p.A_eq.T @ raw.y
    assert np.allclose(grad, raw.s, atol=1e-6)
    assert np.all(raw.s > -1e-8) and abs(raw.x @ raw.s) < 1e-6


def test_qp_simplex_matches_projected_gradient():
    rng = np.random.default_rng(0)
    n = 6
    M = rng.standard_normal((n, n))
    Q = M @ M.T + 0.5 * np.eye(n)
    c = rng.standard_normal(n)
    r = solve_qp(Q, c, A_eq=np.ones((1, n)), b_eq=[1.0])

    def proj(v):  # Euclidean projection onto the simplex
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1
        k = np.nonzero(u - css / (np.arange(n) + 1) > 0)[0][-1]
        return np.maximum(v - css[k] / (k + 1), 0)

    x = np.ones(n) / n
    L = np.linalg.eigvalsh(Q).max()
    for _ in range(20000):
        x = proj(x - (Q @ x + c) / L)
    assert np.allclose(r.x, x, atol=1e-4)


def test_free_and_bounded_variables():
    # min x + y, -2 <= x <= 3, y free, y >= |x| - 1: x + |x| - 1 >= -1 (attained for x <= 0)
    r = solve_lp([1, 1], A_ub=[[1, -1], [-1, -1]], b_ub=[1, 1],
                 lb=[-2, -np.inf], ub=[3, np.inf])
    assert r.success
    assert r.objective == pytest.approx(-1.0, abs=1e-6)  # any x in [-2, 0] with y = -x - 1
    assert r.max_violation < 1e-7


def test_upper_bound_only_variable():
    r = solve_lp([-1.0], lb=[-np.inf], ub=[4.0])
    assert r.success and r.x[0] == pytest.approx(4.0, abs=1e-6)


def test_standard_form_recover_is_consistent():
    p = random_convex_qp(2, 5, seed=2)
    p.ub[:] = 5.0
    red = to_standard_form(p)
    sol = solve(p)
    z = sol.raw.x
    assert np.allclose(red.std.A @ z, red.std.b, atol=1e-6)
    assert red.std.objective(z) + 0.5 * red.x0 @ p.Q @ red.x0 + p.c @ red.x0 == pytest.approx(
        p.objective(red.recover(z)), abs=1e-7)


def test_barrier_agrees_with_primal_dual():
    rng = np.random.default_rng(4)
    n = 5
    G = np.vstack([rng.standard_normal((8, n)), np.eye(n), -np.eye(n)])
    h = np.concatenate([rng.uniform(1, 2, 8), np.full(2 * n, 3.0)])
    c = rng.standard_normal(n)
    b = solve_barrier(None, c, G, h, tol=1e-9)
    pd = solve_lp(c, A_ub=G, b_ub=h, lb=np.full(n, -np.inf))
    assert pd.success
    assert b.objective == pytest.approx(pd.objective, abs=1e-6)


def test_barrier_qp_and_infeasible_detection():
    Q = np.eye(2)
    sol = solve_barrier(Q, [-2, -2], np.array([[1.0, 1.0]]), np.array([1.0]))
    assert np.allclose(sol.x, [0.5, 0.5], atol=1e-5)
    with pytest.raises(ValueError):
        solve_barrier(None, [1.0], np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))


def test_infeasible_and_unbounded_not_reported_optimal():
    inf = solve_lp([1, 1], A_eq=[[1, 1]], b_eq=[-1.0])  # x+y=-1, x,y>=0
    assert not inf.success
    unb = solve_lp([-1.0, 0.0], A_ub=[[-1.0, 1.0]], b_ub=[1.0])
    assert not unb.success


def test_l1_regression_resists_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 40)
    X = np.column_stack([np.ones(40), x])
    y = 1 + 2 * x + 0.01 * rng.standard_normal(40)
    y[::8] += 20
    w1 = l1_regression(X, y)
    w2 = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.linalg.norm(w1 - [1, 2]) < 0.1
    assert np.linalg.norm(w2 - [1, 2]) > 1.0


def test_chebyshev_regression_equioscillation():
    x = np.linspace(-1, 1, 30)
    X = np.column_stack([np.ones(30), x])
    y = x ** 2
    w, rmax = chebyshev_regression(X, y)
    res = y - X @ w
    assert np.max(np.abs(res)) == pytest.approx(rmax, abs=1e-6)
    assert rmax == pytest.approx(0.5, abs=0.01)  # best linear fit to x^2 on [-1,1]


def test_portfolio_constraints():
    rng = np.random.default_rng(2)
    F = rng.standard_normal((5, 5))
    cov = F @ F.T / 5 + 0.05 * np.eye(5)
    mu = rng.uniform(0.02, 0.12, 5)
    w = min_variance_portfolio(cov, mu, target_return=0.08, max_weight=0.6)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert mu @ w >= 0.08 - 1e-6
    assert w.min() >= -1e-7 and w.max() <= 0.6 + 1e-7


def test_svm_separates_blobs():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-2, 0.6, (25, 2)), rng.normal(2, 0.6, (25, 2))])
    y = np.array([-1] * 25 + [1] * 25)
    w, b = svm_dual(X, y, C=10.0)
    assert np.all(np.sign(X @ w + b) == y)


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError):
        GeneralProblem(c=[1, 2], A_eq=[[1, 2, 3]], b_eq=[1])
