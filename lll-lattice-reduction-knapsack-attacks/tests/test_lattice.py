import itertools
import random
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_reduction import (TruncatedLCG, babai_nearest_plane, bkz_reduce, cvp,
                               gram_schmidt, hermite_factor, integer_relation,
                               is_lll_reduced, knapsack_density, lll_reduce,
                               minimal_polynomial, orthogonality_defect, predict_next,
                               random_knapsack, recover_lcg_states, solve_knapsack, svp)
from lattice_reduction.linalg import lattice_det_sq, norm_sq, same_lattice


def random_basis(n, bits, seed):
    rng = random.Random(seed)
    while True:
        b = [[rng.randrange(-(2 ** bits), 2 ** bits) for _ in range(n)] for _ in range(n)]
        try:
            gram_schmidt(b)
            return b
        except ValueError:
            continue


def test_two_dim_matches_gauss():
    red = lll_reduce([[1, 0], [1000, 1]], Fraction(99, 100))
    assert norm_sq(red[0]) == 1
    assert abs(lattice_det_sq(red)) == 1


@pytest.mark.parametrize("seed", range(4))
def test_lll_preserves_lattice_and_is_reduced(seed):
    b = random_basis(6, 12, seed)
    red = lll_reduce(b)
    assert is_lll_reduced(red, Fraction(3, 4))
    assert lattice_det_sq(red) == lattice_det_sq(b)
    assert same_lattice(b, red)


def test_lll_first_vector_bound():
    for seed in range(5):
        b = random_basis(7, 10, seed + 10)
        red = lll_reduce(b)
        det = float(lattice_det_sq(b)) ** 0.5
        assert norm_sq(red[0]) ** 0.5 <= 2 ** ((7 - 1) / 4) * det ** (1 / 7) * 1.0001


def test_lll_improves_orthogonality():
    b = random_basis(8, 14, 3)
    assert orthogonality_defect(lll_reduce(b)) < orthogonality_defect(b)


def test_lll_rejects_dependent_vectors():
    with pytest.raises(ValueError):
        lll_reduce([[1, 2, 3], [2, 4, 6], [0, 0, 1]])


def test_is_lll_reduced_detects_bad_basis():
    assert not is_lll_reduced([[1, 0], [1000, 1]])


def test_svp_matches_bruteforce():
    b = random_basis(3, 4, 5)
    best = min(norm_sq([sum(c * r[t] for c, r in zip(co, b)) for t in range(3)])
               for co in itertools.product(range(-12, 13), repeat=3) if any(co))
    assert norm_sq(svp(b)) == best


def test_cvp_not_worse_than_babai_and_matches_bruteforce():
    rng = random.Random(7)
    b = lll_reduce(random_basis(3, 4, 8))
    for _ in range(5):
        t = [rng.randrange(-60, 60) for _ in range(3)]
        v = cvp(b, t)
        d = sum((x - y) ** 2 for x, y in zip(v, t))
        d_babai = sum((x - y) ** 2 for x, y in zip(babai_nearest_plane(b, t), t))
        brute = min(sum((sum(c * r[i] for c, r in zip(co, b)) - t[i]) ** 2 for i in range(3))
                    for co in itertools.product(range(-15, 16), repeat=3))
        assert d <= d_babai
        assert d == brute


def test_bkz_reduces_and_not_worse_than_lll():
    b = random_basis(10, 16, 21)
    l = lll_reduce(b, Fraction(99, 100))
    z = bkz_reduce(b, 6)
    assert is_lll_reduced(z, Fraction(99, 100))
    assert same_lattice(b, z)
    assert norm_sq(z[0]) <= norm_sq(l[0])
    assert hermite_factor(z) <= hermite_factor(l) + 1e-12


def test_knapsack_low_density_recovered():
    for seed in range(5):
        w, secret, s = random_knapsack(18, 0.5, random.Random(seed))
        x = solve_knapsack(w, s)
        assert x is not None and sum(a * b for a, b in zip(w, x)) == s


def test_knapsack_high_density_fails_gracefully():
    w = [3, 5, 7, 11, 13, 17, 19, 23]
    assert knapsack_density(w) > 1
    # target not reachable -> no solution can be returned
    assert solve_knapsack(w, 10 ** 6) is None


def test_integer_relation_golden_ratio():
    getcontext().prec = 80
    phi = (1 + Decimal(5).sqrt()) / 2
    assert minimal_polynomial(phi, 2, 40) == [-1, -1, 1]


def test_minimal_polynomial_sqrt2_plus_sqrt3():
    getcontext().prec = 120
    alpha = Decimal(2).sqrt() + Decimal(3).sqrt()
    assert minimal_polynomial(alpha, 4, 60) == [1, 0, -10, 0, 1]


def test_integer_relation_none_for_independent_values():
    getcontext().prec = 40
    vals = [Decimal(1), Decimal(2).sqrt(), Decimal(3).sqrt()]
    assert integer_relation(vals, 20) is None


def test_lcg_state_recovery_and_prediction():
    a, c, m = 1103515245, 12345, 2 ** 31
    for seed in (1, 2, 3):
        g = TruncatedLCG(a, c, m, 14, seed * 7919)
        pairs = [g.next() for _ in range(8)]
        full = [p[0] for p in pairs]
        out = [p[1] for p in pairs]
        states = recover_lcg_states(out, a, c, m, 14)
        assert states == full
        future = predict_next(states, a, c, m, 5)
        assert future == [g.next()[0] for _ in range(5)]


def test_lcg_recovery_fails_when_too_many_bits_hidden():
    a, c, m = 1103515245, 12345, 2 ** 31
    g = TruncatedLCG(a, c, m, 30, 99)
    pairs = [g.next() for _ in range(6)]
    truth = [p[0] for p in pairs]
    out = [p[1] for p in pairs]
    states = recover_lcg_states(out, a, c, m, 30)
    # with a single visible bit the state is not determined by the outputs
    assert states is None or states != [x for x in truth]
