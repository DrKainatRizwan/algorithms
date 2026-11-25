import math

import numpy as np

from prng_attacks import (
    LCG, MT19937, Xorshift128Plus, berlekamp_massey, bit_matrix, lsb_stream,
    monobit_test, run_battery, runs_test, serial_test, spectral_test,
    autocorrelation_test, linear_complexity_test,
)
from prng_attacks.stats import chi2_sf, gammaincc


def test_gammaincc_known_values():
    assert math.isclose(gammaincc(1.0, 2.0), math.exp(-2.0), rel_tol=1e-9)
    assert math.isclose(chi2_sf(3.841458820694124, 1), 0.05, rel_tol=1e-6)
    assert math.isclose(chi2_sf(18.307038053275146, 10), 0.05, rel_tol=1e-6)


def test_berlekamp_massey_known_lfsr():
    # x^4 + x + 1 LFSR sequence has linear complexity 4
    s = [1, 0, 0, 0]
    for i in range(4, 100):
        s.append(s[i - 3] ^ s[i - 4])
    assert berlekamp_massey(s) == 4
    assert berlekamp_massey([0] * 10) == 0
    assert berlekamp_massey([0, 0, 0, 1]) == 4


def test_mersenne_twister_passes_battery():
    bits = bit_matrix(MT19937(2024).outputs(8000), 32)
    results = run_battery(bits)
    assert all(r.passed() for r in results), [(r.name, r.p_value) for r in results]


def test_constant_and_biased_streams_fail():
    assert not monobit_test(np.ones(10000, dtype=np.uint8)).passed()
    rng = np.random.default_rng(0)
    biased = (rng.random(20000) < 0.55).astype(np.uint8)
    assert not monobit_test(biased).passed()


def test_alternating_stream_fails_runs_and_serial():
    alt = np.tile(np.array([0, 1], dtype=np.uint8), 5000)
    assert not runs_test(alt).passed()
    assert not serial_test(alt).passed()
    assert not autocorrelation_test(alt, 1).passed()
    assert not spectral_test(alt).passed()


def test_lcg_power_of_two_low_bit_is_detected():
    g = LCG(1, 1664525, 1013904223, 2 ** 32)
    bits = lsb_stream(g.outputs(4000))
    assert berlekamp_massey(bits) <= 2
    assert not linear_complexity_test(bits).passed()


def test_xorshift_lsb_has_linear_complexity_128():
    g = Xorshift128Plus(88172645463325252, 1234567)
    bits = lsb_stream(g.outputs(1500))
    assert berlekamp_massey(bits) == 128
    assert not linear_complexity_test(bits).passed()


def test_xorshift_full_words_pass_frequency_tests():
    g = Xorshift128Plus(88172645463325252, 1234567)
    bits = bit_matrix(g.outputs(6000), 64)
    assert monobit_test(bits).passed()
    assert runs_test(bits).passed()
    assert serial_test(bits).passed()
