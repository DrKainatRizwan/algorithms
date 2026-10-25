import numpy as np
import pytest

from simplex_lp import (INF, LinearProgram, build_dual, check_farkas, check_optimality,
                        check_unbounded_ray, cost_ranging, revised_simplex, rhs_ranging,
                        simplex_standard, solve)
from simplex_lp.problems import (assignment_lp, beale_cycling_example, chebyshev_fit,
                                 diet_problem, klee_minty, l1_regression, max_flow_lp,
                                 production_planning, random_feasible_lp,
                                 solve_matrix_game, transportation)


def test_product_mix_optimum_and_duals():
    lp = production_planning()
    sol = solve(lp)
    assert sol.status == "optimal"
    assert sol.objective == pytest.approx(36.0)
    np.testing.assert_allclose(sol.x, [2, 6], atol=1e-9)
    np.testing.assert_allclose(sol.duals, [0, 1.5, 1.0], atol=1e-9)
    assert check_optimality(lp, sol.x, sol.duals).ok()


def test_strong_duality_via_explicit_dual_lp():
    for seed in range(5):
        lp = random_feasible_lp(8, 6, seed)
        p = solve(lp)
        d = solve(build_dual(lp.with_explicit_bounds()))
        assert p.status == d.status == "optimal"
        assert p.objective == pytest.approx(d.objective, abs=1e-7)


def test_dual_with_free_variables_and_equalities():
    lp = LinearProgram([1, 2, 0], A_ub=[[-1, -1, 0]], b_ub=[-2], A_eq=[[1, 0, 1]], b_eq=[3],
                       bounds=[(0, INF), (0, INF), (-INF, INF)])
    p, d = solve(lp), solve(build_dual(lp))
    assert p.objective == pytest.approx(d.objective)


def test_build_dual_rejects_general_bounds():
    with pytest.raises(ValueError):
        build_dual(LinearProgram([1.0], bounds=[(0, 5)]))


def test_bland_terminates_where_dantzig_cycles():
    c, A, b, basis = beale_cycling_example()
    assert simplex_standard(c, A, b, rule="dantzig", basis=basis).status == "cycling"
    res = simplex_standard(c, A, b, rule="bland", basis=basis)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-1.25)
    hyb = simplex_standard(c, A, b, rule="hybrid", basis=basis)
    assert hyb.status == "optimal" and hyb.objective == pytest.approx(-1.25)


def test_klee_minty_dantzig_visits_all_vertices():
    for n in (3, 4, 5):
        sol = solve(klee_minty(n), rule="dantzig")
        assert sol.iterations == 2 ** n - 1
        assert sol.objective == pytest.approx(100.0 ** (n - 1))
        assert solve(klee_minty(n), rule="bland").objective == pytest.approx(100.0 ** (n - 1))


def test_infeasible_with_farkas_certificate():
    lp = LinearProgram([1, 1], A_ub=[[1, 1], [-1, -1]], b_ub=[1, -3])
    sol = solve(lp)
    assert sol.status == "infeasible"
    assert check_farkas(sol)


def test_unbounded_with_ray_certificate():
    lp = LinearProgram([-1, -1], A_ub=[[1, -1]], b_ub=[2])
    sol = solve(lp)
    assert sol.status == "unbounded"
    assert check_unbounded_ray(lp, sol.ray)
    assert not check_unbounded_ray(lp, np.array([0.0, 0.0]))


def test_tableau_and_revised_agree_on_random_lps():
    for seed in range(12):
        lp = random_feasible_lp(10, 7, seed, equality_fraction=0.2)
        a = solve(lp, method="tableau")
        b = solve(lp, method="revised", rule="bland")
        assert a.status == b.status == "optimal"
        assert a.objective == pytest.approx(b.objective, abs=1e-7)
        assert lp.is_feasible(a.x) and lp.is_feasible(b.x)
        assert check_optimality(lp, a.x, a.duals).ok(1e-5)


def test_bounds_variants_and_free_variables():
    # min x + y with x in [2,5], y free, x + y >= 1  ->  x=2, y=-1
    lp = LinearProgram([1, 1], A_ub=[[-1, -1]], b_ub=[-1], bounds=[(2, 5), (-INF, INF)])
    sol = solve(lp)
    assert sol.objective == pytest.approx(1.0)
    assert sol.x[0] + sol.x[1] == pytest.approx(1.0)
    # upper-bounded-only variable: max x s.t. x <= 3, x <= 7 bound
    lp2 = LinearProgram([1], bounds=[(-INF, 3)], maximize=True)
    assert solve(lp2).objective == pytest.approx(3.0)


def test_redundant_equalities_are_handled():
    lp = LinearProgram([1, 1], A_eq=[[1, 1], [2, 2]], b_eq=[2, 4])
    sol = solve(lp)
    assert sol.status == "optimal" and sol.objective == pytest.approx(2.0)
    rev = solve(lp, method="revised")
    assert rev.status == "optimal" and rev.objective == pytest.approx(2.0)


def test_rhs_ranging_predicts_objective_change():
    lp = production_planning()
    sol = solve(lp)
    ranges = rhs_ranging(sol)
    lo, hi = ranges[2]
    assert (lo, hi) == (pytest.approx(-6.0), pytest.approx(6.0))
    delta = 3.0
    lp2 = LinearProgram(lp.c, lp.A_ub, lp.b_ub + [0, 0, delta], maximize=True)
    assert solve(lp2).objective == pytest.approx(sol.objective + sol.duals[2] * delta)
    # outside the range the linear prediction fails
    lp3 = LinearProgram(lp.c, lp.A_ub, lp.b_ub + [0, 0, 9.0], maximize=True)
    assert solve(lp3).objective < sol.objective + sol.duals[2] * 9.0 - 1e-6


def test_cost_ranging_keeps_basis_optimal():
    lp = production_planning()
    sol = solve(lp)
    rng = cost_ranging(sol, lp)
    lo, hi = rng[0]
    assert lo == pytest.approx(0.0) and hi == pytest.approx(7.5)
    inside = LinearProgram([7.0, 5.0], lp.A_ub, lp.b_ub, maximize=True)
    np.testing.assert_allclose(solve(inside).x, sol.x, atol=1e-9)
    outside = LinearProgram([8.0, 5.0], lp.A_ub, lp.b_ub, maximize=True)
    assert not np.allclose(solve(outside).x, sol.x)


def test_transportation_and_assignment():
    lp = transportation([20, 30], [10, 25, 15], [[8, 6, 10], [9, 12, 13]])
    sol = solve(lp)
    assert sol.status == "optimal"
    assert lp.is_feasible(sol.x)
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], float)
    a = solve(assignment_lp(cost))
    assert a.objective == pytest.approx(5.0)
    assert np.allclose(np.round(a.x), a.x, atol=1e-8)


def test_max_flow_lp_matches_known_value():
    edges = [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12), (3, 2, 9),
             (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    sol = solve(max_flow_lp(6, edges, 0, 5))
    assert sol.objective == pytest.approx(23.0)


def test_matrix_game_values():
    v, p, q = solve_matrix_game([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    assert v == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(p, 1 / 3, atol=1e-9)
    v, p, q = solve_matrix_game([[3, -1], [-2, 4]])
    assert v == pytest.approx(1.0)
    P = np.array([[3, -1], [-2, 4]])
    assert (p @ P).min() == pytest.approx(v) and (P @ q).max() == pytest.approx(v)


def test_l1_and_chebyshev_fits():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(30), np.linspace(0, 1, 30)])
    y = 2.0 + 3.0 * X[:, 1]
    y_out = y.copy()
    y_out[[4, 17]] += 50.0
    w = l1_regression(X, y_out)
    np.testing.assert_allclose(w, [2.0, 3.0], atol=1e-7)
    noisy = y + rng.uniform(-0.1, 0.1, 30)
    w2, err = chebyshev_fit(X, noisy)
    assert np.max(np.abs(noisy - X @ w2)) == pytest.approx(err, abs=1e-8)
    assert err <= 0.1 + 1e-9


def test_diet_problem_meets_requirements():
    lp = diet_problem()
    sol = solve(lp)
    assert sol.status == "optimal" and lp.is_feasible(sol.x)
    assert check_optimality(lp, sol.x, sol.duals).ok(1e-6)


def test_input_validation():
    with pytest.raises(ValueError):
        LinearProgram([1, 2], A_ub=[[1, 2, 3]], b_ub=[1])
    with pytest.raises(ValueError):
        LinearProgram([1], bounds=[(3, 1)])
    with pytest.raises(ValueError):
        simplex_standard(np.ones(2), np.eye(2), np.ones(2), rule="nope")
    c, A, b, _ = beale_cycling_example()
    with pytest.raises(ValueError):
        simplex_standard(c, A, b, basis=[0, 1, 2], rule="bland")


def _brute_force_min(lp):
    """Enumerate vertices of {A x <= b, x >= 0} (2-3 variables) and take the best."""
    from itertools import combinations
    n = lp.n_vars
    A = np.vstack([lp.A_ub, -np.eye(n)])
    b = np.concatenate([lp.b_ub, np.zeros(n)])
    best = INF
    for idx in combinations(range(A.shape[0]), n):
        M = A[list(idx)]
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        x = np.linalg.solve(M, b[list(idx)])
        if np.all(A @ x <= b + 1e-9):
            best = min(best, float(lp.c @ x))
    return best


def test_matches_vertex_enumeration_on_small_random_lps():
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(40):
        A = rng.uniform(0.1, 3.0, size=(4, 3))  # positive rows keep the region bounded
        b = rng.uniform(1.0, 5.0, size=4)
        c = rng.normal(size=3)
        lp = LinearProgram(c, A, b)
        sol = solve(lp)
        assert sol.status == "optimal"
        assert sol.objective == pytest.approx(_brute_force_min(lp), abs=1e-8)
        checked += 1
    assert checked == 40
