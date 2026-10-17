import numpy as np
import pytest

from assignment_solvers import (
    InfeasibleError,
    auction_assignment,
    bottleneck_assignment,
    brute_force,
    forced_pair_cost,
    gated_assignment,
    hungarian,
    random_cost_matrix,
    reduced_costs,
    validate_cost_matrix,
    verify_duals,
)


def test_known_3x3():
    c = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    r = hungarian(c)
    assert r.cost == 5.0
    assert r.as_dict() == {0: 1, 1: 0, 2: 2}


@pytest.mark.parametrize("seed", range(8))
def test_hungarian_matches_brute_force_square(seed):
    c = random_cost_matrix(6, seed=seed, high=30)
    assert hungarian(c).cost == brute_force(c)


@pytest.mark.parametrize("shape", [(3, 6), (6, 3), (1, 5), (5, 1)])
def test_hungarian_rectangular(shape):
    c = random_cost_matrix(*shape, seed=3, high=50)
    r = hungarian(c)
    assert len(r.rows) == min(shape)
    assert len(set(r.cols)) == len(r.cols) and len(set(r.rows)) == len(r.rows)
    assert r.cost == brute_force(c)


def test_maximize():
    c = random_cost_matrix(6, seed=11)
    assert hungarian(c, maximize=True).cost == brute_force(c, maximize=True)


def test_negative_and_float_costs():
    rng = np.random.default_rng(5)
    c = rng.normal(size=(6, 6))
    assert hungarian(c).cost == pytest.approx(brute_force(c))


def test_forbidden_entries_and_infeasible():
    c = np.array([[np.inf, 1.0], [2.0, np.inf]])
    assert hungarian(c).cost == 3.0
    with pytest.raises(InfeasibleError):
        hungarian(np.array([[np.inf, np.inf], [1.0, 2.0]]))


def test_dual_certificate_hungarian():
    for seed in range(5):
        c = random_cost_matrix(12, 15, seed=seed)
        r = hungarian(c)
        assert verify_duals(c, r)
        assert (reduced_costs(c, r.u, r.v) >= -1e-9).all()


def test_dual_certificate_detects_bad_matching():
    c = random_cost_matrix(5, seed=2)
    r = hungarian(c)
    r.cols = np.roll(r.cols, 1)
    assert not verify_duals(c, r)


@pytest.mark.parametrize("seed", range(6))
def test_auction_exact_on_integers(seed):
    c = random_cost_matrix(30, seed=seed, high=1000)
    assert auction_assignment(c).cost == hungarian(c).cost


@pytest.mark.parametrize("shape", [(5, 9), (9, 5)])
def test_auction_rectangular(shape):
    c = random_cost_matrix(*shape, seed=4)
    r = auction_assignment(c)
    assert r.cost == hungarian(c).cost
    assert len(r.rows) == min(shape)


def test_auction_float_within_bound():
    rng = np.random.default_rng(9)
    c = rng.random((25, 25))
    eps = 1e-3
    r = auction_assignment(c, eps_final=eps)
    assert r.cost - hungarian(c).cost <= 25 * eps + 1e-12


def test_auction_maximize_and_duals():
    c = random_cost_matrix(15, seed=21, high=200)
    r = auction_assignment(c, maximize=True)
    assert r.cost == hungarian(c, maximize=True).cost
    assert verify_duals(c, r, tol=1.0, maximize=True)


def test_auction_scaling_reduces_bids():
    c = random_cost_matrix(40, seed=1, high=10000)
    _, scaled = auction_assignment(c, scale=4.0, return_stats=True)
    _, plain = auction_assignment(c, scale=1e12, return_stats=True)
    assert scaled.bids < plain.bids


def test_auction_rejects_infinite():
    with pytest.raises(ValueError):
        auction_assignment([[1.0, np.inf], [2.0, 3.0]])


def test_validation_errors():
    for bad in ([1, 2, 3], [[np.nan]], np.zeros((0, 3)), [[-np.inf, 1.0]]):
        with pytest.raises(ValueError):
            validate_cost_matrix(bad)


def test_bottleneck_minimises_maximum():
    c = random_cost_matrix(6, seed=7, high=40)
    from itertools import permutations

    best = min(max(c[i, p[i]] for i in range(6)) for p in permutations(range(6)))
    r = bottleneck_assignment(c)
    assert c[r.rows, r.cols].max() == best


def test_bottleneck_tie_break_total_cost():
    c = np.array([[1.0, 5.0, 5.0], [5.0, 1.0, 5.0], [5.0, 5.0, 5.0]])
    r = bottleneck_assignment(c)
    assert r.cost == 7.0 and c[r.rows, r.cols].max() == 5.0


def test_gated_assignment():
    c = np.array([[1.0, 9.0, 9.0], [9.0, 2.0, 9.0], [9.0, 9.0, 9.0], [0.5, 9.0, 9.0]])
    matches, ur, uc = gated_assignment(c, gate=3.0)
    assert sorted(matches) == [(1, 1), (3, 0)]
    assert ur == [0, 2] and uc == [2]


def test_gated_maximises_matches():
    c = np.array([[1.0, 2.0], [2.0, 100.0]])
    matches, ur, uc = gated_assignment(c, gate=5.0)
    assert sorted(matches) == [(0, 1), (1, 0)] and not ur and not uc


def test_forced_pair_regret():
    c = random_cost_matrix(6, seed=13)
    opt = hungarian(c)
    for i, j in zip(opt.rows, opt.cols):
        assert forced_pair_cost(c, i, j) == opt.cost
    worst = forced_pair_cost(c, 0, int(np.argmax(c[0])))
    assert worst >= opt.cost
    assert worst == min(
        sum(c[k, p[k]] for k in range(6))
        for p in __import__("itertools").permutations(range(6))
        if p[0] == int(np.argmax(c[0]))
    )
