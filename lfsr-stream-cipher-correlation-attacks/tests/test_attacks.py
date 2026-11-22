import numpy as np
import pytest

from stream_cryptanalysis import (
    FilterGenerator, GeffeGenerator, LFSR, ThresholdGenerator, add_noise,
    algebraic_attack, berlekamp_massey, correlation_attack, fast_correlation_attack,
    geffe_attack, parity_checks, required_length, unknown_count, walsh_hadamard,
)
from stream_cryptanalysis.correlation import brute_force_scores, correlation_scores
from stream_cryptanalysis.fast_correlation import check_satisfaction


def test_wht_matches_definition():
    rng = np.random.default_rng(0)
    a = rng.integers(-5, 5, 16)
    h = np.array([[(-1) ** bin(i & j).count("1") for j in range(16)] for i in range(16)])
    assert np.array_equal(walsh_hadamard(a), h @ a)
    with pytest.raises(ValueError):
        walsh_hadamard(np.zeros(6))


def test_wht_scores_equal_brute_force():
    r = LFSR.primitive(8)
    rng = np.random.default_rng(1)
    z = add_noise(r.stream(77, 120), 0.2, rng)
    assert np.array_equal(correlation_scores(r, z), brute_force_scores(r, z))


def test_correlation_attack_noisy():
    r = LFSR.primitive(16)
    rng = np.random.default_rng(2)
    z = add_noise(r.stream(0xBEEF, 600), 0.25, rng)
    res = correlation_attack(r, z)
    assert res.state == 0xBEEF
    assert res.score > res.runner_up_score


def test_required_length_monotone():
    assert required_length(16, 0.1) < required_length(16, 0.3) < required_length(16, 0.4)
    assert required_length(20, 0.2) > required_length(10, 0.2)


def test_geffe_correlation_is_three_quarters():
    g = GeffeGenerator(LFSR.primitive(7), LFSR.primitive(8), LFSR.primitive(9))
    z = g.keystream((5, 77, 301), 20000)
    for reg, st in ((g.r2, 77), (g.r3, 301)):
        assert abs(np.mean(reg.stream(st, 20000) == z) - 0.75) < 0.02
    assert abs(np.mean(g.r1.stream(5, 20000) == z) - 0.5) < 0.02


def test_geffe_divide_and_conquer():
    g = GeffeGenerator(LFSR.primitive(11), LFSR.primitive(12), LFSR.primitive(13))
    states = (1234, 3001, 6000)
    z = g.keystream(states, 400)
    assert geffe_attack(g, z) == states


def test_berlekamp_massey_breaks_geffe_within_bound():
    g = GeffeGenerator(LFSR.primitive(3), LFSR.primitive(4), LFSR.primitive(5))
    z = g.keystream((5, 9, 17), 2 * g.linear_complexity_bound + 40)
    L, _ = berlekamp_massey(list(z))
    assert 0 < L <= g.linear_complexity_bound


def test_threshold_generator_correlation():
    regs = (LFSR.primitive(7), LFSR.primitive(8), LFSR.primitive(9))
    g = ThresholdGenerator(regs)
    sts = (3, 55, 200)
    z = g.keystream(sts, 20000)
    for r, s in zip(regs, sts):
        assert abs(np.mean(r.stream(s, 20000) == z) - 0.75) < 0.02


def test_parity_checks_hold_on_clean_sequence():
    r = LFSR.primitive(31)
    s = r.stream(0xACE1, 3000)
    checks = parity_checks(r, 3000)
    assert checks.shape[1] == 3
    sat, tot = check_satisfaction(s, checks)
    assert np.array_equal(sat, tot) and tot.max() > 0


def test_fast_correlation_attack_recovers_31_bit_state():
    r = LFSR.primitive(31)
    st = 0x2468ACE
    rng = np.random.default_rng(3)
    z = add_noise(r.stream(st, 20000), 0.3, rng)
    res = fast_correlation_attack(r, z, 0.3)
    assert res.state == st and res.residual_violations == 0
    assert np.array_equal(res.corrected, r.stream(st, 20000))


def test_fast_correlation_fails_gracefully_on_pure_noise():
    r = LFSR.primitive(31)
    rng = np.random.default_rng(4)
    z = rng.integers(0, 2, 3000).astype(np.uint8)
    assert fast_correlation_attack(r, z, 0.45).state is None


def test_algebraic_attack_degree_two():
    f = FilterGenerator(LFSR.primitive(12), (0, 2, 5, 7, 11), ((0,), (1, 2), (3, 4), (2, 3)))
    st = 0xABC
    z = f.keystream(st, 150)
    assert f.degree == 2
    assert algebraic_attack(f, z) == st


def test_algebraic_attack_degree_three():
    f = FilterGenerator(LFSR.primitive(9), (0, 1, 4, 6, 8), ((1,), (0, 2, 3), (2, 4), (3,)))
    st = 0x155
    assert algebraic_attack(f, f.keystream(st, 300)) == st


def test_algebraic_attack_rejects_inconsistent_data():
    f = FilterGenerator(LFSR.primitive(8), (0, 1, 3, 6), ((0,), (1, 2), (3,)))
    z = f.keystream(0x5B, 100).copy()
    assert algebraic_attack(f, z) == 0x5B
    rng = np.random.default_rng(5)
    noisy = add_noise(z, 0.3, rng)
    assert algebraic_attack(f, noisy) is None


def test_filter_generator_validation_and_unknowns():
    with pytest.raises(ValueError):
        FilterGenerator(LFSR.primitive(8), (0, 9), ((0,),))
    with pytest.raises(ValueError):
        FilterGenerator(LFSR.primitive(8), (0, 1), ((2,),))
    assert unknown_count(12, 2) == 12 + 66
