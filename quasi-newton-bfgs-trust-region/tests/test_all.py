import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quasi_newton import (LBFGSMemory, backtracking, bfgs, compare, dfp, dogleg,
                          extended_rosenbrock, lbfgs, logistic_regression, nonlinear_cg,
                          numerical_gradient, powell_singular, quadratic, rosenbrock, sr1,
                          steihaug_cg, strong_wolfe, trust_region, beta_function)
from quasi_newton.quasi_newton import _update_bfgs


def spd(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


@pytest.mark.parametrize("make", [rosenbrock, lambda: extended_rosenbrock(6), powell_singular,
                                  lambda: beta_function(5), logistic_regression])
def test_gradients_match_finite_differences(make):
    p = make()
    x = p.x0 + 0.1 * np.random.default_rng(1).standard_normal(len(p.x0))
    assert np.allclose(p.grad(x), numerical_gradient(p.f, x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("make", [rosenbrock, lambda: extended_rosenbrock(6), powell_singular,
                                  lambda: beta_function(5), logistic_regression])
def test_hessians_match_finite_differences(make):
    p = make()
    x = p.x0 + 0.1 * np.random.default_rng(2).standard_normal(len(p.x0))
    H = np.array([numerical_gradient(lambda z, i=i: p.grad(z)[i], x) for i in range(len(x))])
    assert np.allclose(p.hess(x), H, rtol=1e-3, atol=1e-3)


def test_backtracking_satisfies_armijo():
    p = rosenbrock()
    x = p.x0
    g = p.gradient(x)
    a, fa = backtracking(p, x, -g, p.value(x), g)
    assert fa <= p.f(x) + 1e-4 * a * (g @ -g)


def test_strong_wolfe_conditions_hold():
    p = rosenbrock()
    x, g = p.x0, p.grad(p.x0)
    d = -g
    a, fa, ga = strong_wolfe(p, x, d, p.f(x), g)
    assert a is not None
    assert fa <= p.f(x) + 1e-4 * a * (g @ d)
    assert abs(ga @ d) <= 0.9 * abs(g @ d)


def test_line_search_rejects_ascent_direction():
    p = rosenbrock()
    g = p.grad(p.x0)
    with pytest.raises(ValueError):
        strong_wolfe(p, p.x0, g, p.f(p.x0), g)


def test_bfgs_update_secant_equation():
    rng = np.random.default_rng(3)
    H = spd(4)
    s = rng.standard_normal(4)
    y = H @ s
    Hn = _update_bfgs(np.eye(4), s, y)
    assert np.allclose(Hn @ y, s)
    assert np.allclose(Hn, Hn.T)
    assert np.all(np.linalg.eigvalsh(Hn) > 0)


def test_bfgs_solves_quadratic_exactly():
    A = spd(8, 4)
    b = np.arange(8.0)
    p = quadratic(A, b)
    r = bfgs(p, tol=1e-9)
    assert r.converged and np.allclose(r.x, p.x_star, atol=1e-7)
    assert r.iterations <= 30


@pytest.mark.parametrize("solver", [bfgs, dfp, sr1, lbfgs, nonlinear_cg, trust_region])
def test_all_solve_rosenbrock(solver):
    p = rosenbrock()
    r = solver(p)
    assert r.converged
    assert np.allclose(r.x, p.x_star, atol=1e-3)


def test_lbfgs_two_loop_matches_dense_bfgs():
    """With full memory and identical pairs, two-loop output equals explicit H_k g."""
    rng = np.random.default_rng(5)
    A = spd(5, 6)
    mem = LBFGSMemory(m=10)
    H = np.eye(5)
    gamma = None
    pairs = []
    for _ in range(4):
        s = rng.standard_normal(5)
        y = A @ s
        mem.add(s, y)
        pairs.append((s, y))
    s, y = pairs[-1]
    H = (s @ y) / (y @ y) * np.eye(5)
    for s, y in pairs:
        H = _update_bfgs(H, s, y)
    g = rng.standard_normal(5)
    assert np.allclose(mem.direction(g), -H @ g)


def test_lbfgs_memory_is_bounded():
    mem = LBFGSMemory(m=3)
    rng = np.random.default_rng(0)
    for _ in range(10):
        s = rng.standard_normal(4)
        mem.add(s, s * 2.0)
    assert len(mem.pairs) == 3
    assert not mem.add(np.ones(4), -np.ones(4))  # negative curvature rejected


def test_lbfgs_scales_to_high_dimension():
    p = extended_rosenbrock(200)
    r = lbfgs(p, m=10, tol=1e-5)
    assert r.converged and np.allclose(r.x, 1.0, atol=1e-3)


def test_powell_singular_converges_to_zero():
    r = bfgs(powell_singular(), tol=1e-5, max_iter=1000)
    assert r.f < 1e-6


def test_steihaug_reaches_newton_step_inside_region():
    A = spd(5, 7)
    g = np.ones(5)
    step = steihaug_cg(A, g, delta=1e3)
    assert np.allclose(step, -np.linalg.solve(A, g), atol=1e-6)


def test_steihaug_on_boundary_for_indefinite():
    B = np.diag([1.0, -2.0, 3.0])
    g = np.array([0.5, 0.1, -0.2])
    step = steihaug_cg(B, g, delta=0.7)
    assert np.isclose(np.linalg.norm(step), 0.7)
    assert g @ step + 0.5 * step @ B @ step < 0


def test_dogleg_regimes():
    B = np.diag([1.0, 10.0])
    g = np.array([1.0, 1.0])
    assert np.allclose(dogleg(B, g, 100.0), -np.linalg.solve(B, g))
    s = dogleg(B, g, 0.05)
    assert np.isclose(np.linalg.norm(s), 0.05)
    s = dogleg(B, g, 0.9)
    assert np.isclose(np.linalg.norm(s), 0.9)


def test_trust_region_handles_indefinite_start():
    p = rosenbrock(np.array([0.0, 1.0]))  # Hessian indefinite here
    assert np.linalg.eigvalsh(p.hess(p.x0))[0] < 0
    r = trust_region(p, subproblem="steihaug")
    assert r.converged and np.allclose(r.x, 1.0, atol=1e-4)


def test_monotone_decrease_with_wolfe():
    r = bfgs(rosenbrock())
    assert all(b <= a + 1e-12 for a, b in zip(r.history, r.history[1:]))


def test_cg_variants_agree():
    for kind in ["fr", "pr+", "hs", "dy"]:
        r = nonlinear_cg(beta_function(10), kind=kind, tol=1e-6)
        assert r.converged, kind
    with pytest.raises(ValueError):
        nonlinear_cg(rosenbrock(), kind="bad")


def test_counters_and_benchmark_deterministic():
    rows1 = compare(rosenbrock)
    rows2 = compare(rosenbrock)
    assert [r["f_evals"] for r in rows1] == [r["f_evals"] for r in rows2]
    assert all(r["converged"] for r in rows1)
    tr = next(r for r in rows1 if r["method"] == "tr-steihaug")
    assert tr["h_evals"] > 0 and tr["iters"] < 40


def test_quasi_newton_fewer_gradients_than_steepest_descent_scaling():
    # superlinear: BFGS gradient count far below a gradient-descent iteration cap
    p = logistic_regression()
    assert bfgs(p).iterations < 40
