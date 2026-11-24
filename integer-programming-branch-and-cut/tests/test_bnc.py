import itertools

import numpy as np
import pytest

from branch_and_cut import (MIP, SolverOptions, assignment_side, brute_force, cover_cuts,
                            facility_location, gomory_cuts, knapsack, random_ip, set_cover,
                            solve_lp, solve_mip)


def test_lp_basic_optimum():
    # max x+y  s.t. x+2y<=4, 3x+y<=6
    r = solve_lp([-1, -1], [[1, 2], [3, 1]], [4, 6])
    assert r.status == "optimal"
    assert r.obj == pytest.approx(-2.8)
    assert r.x == pytest.approx([1.6, 1.2])


def test_lp_infeasible_and_unbounded():
    assert solve_lp([1, 1], [[1, 1]], [-1]).status == "infeasible"
    assert solve_lp([-1, 0], [[0, 1]], [1]).status == "unbounded"


def test_lp_equalities_and_bounds():
    r = solve_lp([1, 2, 3], A_eq=[[1, 1, 1]], b_eq=[10], lb=[1, 1, 1], ub=[5, 5, 5])
    assert r.status == "optimal"
    assert r.obj == pytest.approx(1 * 5 + 2 * 4 + 3 * 1)
    assert r.x.sum() == pytest.approx(10)


def test_lp_matches_vertex_enumeration():
    rng = np.random.default_rng(3)
    for _ in range(10):
        A = rng.uniform(0.1, 3, (3, 2))
        b = rng.uniform(2, 6, 3)
        c = -rng.uniform(0.5, 2, 2)
        r = solve_lp(c, A, b)
        full = np.vstack([A, -np.eye(2)])
        rhs = np.concatenate([b, np.zeros(2)])
        best = 0.0
        for i, j in itertools.combinations(range(5), 2):
            M = full[[i, j]]
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            p = np.linalg.solve(M, rhs[[i, j]])
            if np.all(full @ p <= rhs + 1e-9):
                best = min(best, c @ p)
        assert r.obj == pytest.approx(best, abs=1e-7)


def test_redundant_equalities_handled():
    r = solve_lp([1, 1], A_eq=[[1, 1], [2, 2]], b_eq=[3, 6])
    assert r.status == "optimal" and r.obj == pytest.approx(3)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("opts", [SolverOptions(), SolverOptions(gomory_rounds=3, cover_rounds=2)])
def test_matches_brute_force(seed, opts):
    mip = random_ip(5, 3, seed)
    ref, _ = brute_force(mip)
    res = solve_mip(mip, opts)
    assert res.status == "optimal"
    assert res.obj == pytest.approx(ref)
    assert mip.is_feasible(res.x)


@pytest.mark.parametrize("rule", ["first_fractional", "most_fractional", "pseudocost", "strong"])
@pytest.mark.parametrize("sel", ["best_first", "depth_first"])
def test_branching_rules_agree(rule, sel):
    mip = knapsack(18, seed=4, multi=2)
    ref = solve_mip(mip, SolverOptions(branching="most_fractional"))
    res = solve_mip(mip, SolverOptions(branching=rule, selection=sel))
    assert res.status == "optimal"
    assert res.obj == pytest.approx(ref.obj)


def test_knapsack_against_dp():
    mip = knapsack(25, seed=7)
    w = mip.A_ub[0].astype(int)
    cap = int(mip.b_ub[0])
    v = (-mip.c).astype(int)
    dp = np.zeros(cap + 1, int)
    for wi, vi in zip(w, v):
        dp[wi:] = np.maximum(dp[wi:], dp[:cap + 1 - wi] + vi)
    res = solve_mip(mip, SolverOptions(gomory_rounds=2, cover_rounds=2))
    assert -res.obj == pytest.approx(dp[cap])


def test_infeasible_integer_problem():
    # 2x = 1 has no integer solution although the LP relaxation is feasible
    mip = MIP([1.0], A_eq=[[2.0]], b_eq=[1.0], ub=[5.0])
    assert solve_mip(mip).status == "infeasible"


def test_unbounded_detected():
    mip = MIP([-1.0, 0.0], A_ub=[[0.0, 1.0]], b_ub=[3.0])
    assert solve_mip(mip).status == "unbounded"


def test_gomory_cuts_valid_for_all_integer_points():
    mip = random_ip(4, 3, seed=2)
    lp = solve_lp(mip.c, mip.A_ub, mip.b_ub, lb=mip.lb, ub=mip.ub, want_tableau=True)
    cuts = gomory_cuts(mip, lp)
    assert cuts, "expected at least one violated cut"
    for a, b in cuts:
        assert a @ lp.x > b + 1e-9          # cuts off the LP optimum
    for pt in itertools.product(range(5), repeat=4):
        x = np.array(pt, float)
        if np.all(mip.A_ub @ x <= mip.b_ub + 1e-9):
            for a, b in cuts:
                assert a @ x <= b + 1e-6


def test_cover_cuts_valid_and_violated():
    mip = knapsack(10, seed=1, tightness=0.3)
    lp = solve_lp(mip.c, mip.A_ub, mip.b_ub, lb=mip.lb, ub=mip.ub)
    cuts = cover_cuts(mip, lp.x)
    assert cuts
    for a, b in cuts:
        assert a @ lp.x > b
    for pt in itertools.product(range(2), repeat=10):
        x = np.array(pt, float)
        if mip.A_ub[0] @ x <= mip.b_ub[0]:
            assert all(a @ x <= b + 1e-9 for a, b in cuts)


def test_cuts_tighten_root_bound():
    mip = knapsack(20, seed=3, multi=3, tightness=0.4)
    plain = solve_mip(mip)
    cut = solve_mip(mip, SolverOptions(gomory_rounds=4, cover_rounds=3))
    assert cut.root_bound_after >= plain.root_bound_after - 1e-9
    assert cut.obj == pytest.approx(plain.obj)
    assert cut.cuts_added > 0


def test_set_cover_and_assignment_feasible():
    sc = solve_mip(set_cover(14, 18, seed=2))
    assert sc.status == "optimal" and set_cover(14, 18, seed=2).is_feasible(sc.x)
    asg = assignment_side(5, seed=1)
    r = solve_mip(asg)
    assert r.status == "optimal" and asg.is_feasible(r.x)
    perms = [p for p in itertools.permutations(range(5))]
    best = min(sum(asg.c[i * 5 + p[i]] for i in range(5)) for p in perms
               if sum(asg.A_ub[0][i * 5 + p[i]] for i in range(5)) <= asg.b_ub[0])
    assert r.obj == pytest.approx(best)


def test_facility_location_mixed_integer():
    mip = facility_location(4, 6, seed=5)
    res = solve_mip(mip, SolverOptions(gomory_rounds=2))
    assert res.status == "optimal" and mip.is_feasible(res.x)
    nf = 4 * 6
    # enumerate all open sets: each customer picks the nearest open facility
    c = mip.c[:nf].reshape(4, 6)
    fixed = mip.c[nf:]
    best = min(fixed[list(s)].sum() + c[list(s)].min(axis=0).sum()
               for k in range(1, 5) for s in itertools.combinations(range(4), k))
    assert res.obj == pytest.approx(best, abs=1e-6)


def test_node_limit_reports_bound():
    mip = knapsack(40, seed=2, multi=3)
    res = solve_mip(mip, SolverOptions(node_limit=3, rounding_heuristic=False))
    assert res.status == "node_limit"
    assert res.bound <= res.obj + 1e-9 or res.x is None
