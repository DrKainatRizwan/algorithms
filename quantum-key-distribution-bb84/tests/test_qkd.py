import math

import numpy as np
import pytest

from qkd_bb84 import (
    BB84Config, Channel, InterceptResend, run_bb84, encode, measure, cascade,
    parity, binary_entropy, secret_fraction, secure_key_length, toeplitz_hash,
    privacy_amplify, detection_probability, expected_qber_intercept,
    qber_threshold, hoeffding_margin, sweep_eavesdropping,
)


def test_same_basis_measurement_is_deterministic():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 500)
    bases = rng.integers(0, 2, 500)
    out, _ = measure(encode(bits, bases), bases, rng)
    assert np.array_equal(out, bits)


def test_wrong_basis_measurement_is_unbiased():
    rng = np.random.default_rng(2)
    n = 20000
    bits = rng.integers(0, 2, n)
    out, _ = measure(encode(bits, np.zeros(n, int)), np.ones(n, int), rng)
    assert abs(out.mean() - 0.5) < 0.02
    assert abs(np.mean(out == bits) - 0.5) < 0.02


def test_measurement_collapses_state():
    rng = np.random.default_rng(3)
    n = 1000
    bits = rng.integers(0, 2, n)
    ang = encode(bits, np.zeros(n, int))
    out, post = measure(ang, np.ones(n, int), rng)
    again, _ = measure(post, np.ones(n, int), rng)
    assert np.array_equal(out, again)


def test_noiseless_channel_zero_qber_and_matching_keys():
    r = run_bb84(BB84Config(n_photons=3000, seed=4), Channel())
    assert r.true_qber == 0.0 and not r.aborted
    assert r.keys_match and r.final_key_alice.size > 0


def test_sifting_keeps_about_half():
    r = run_bb84(BB84Config(n_photons=8000, seed=5), Channel())
    assert abs(r.n_sifted / r.n_sent - 0.5) < 0.03


def test_loss_reduces_detections():
    r = run_bb84(BB84Config(n_photons=8000, seed=6), Channel(loss=0.6))
    assert abs(r.n_detected / 8000 - 0.4) < 0.03
    assert r.true_qber == 0.0


def test_full_intercept_resend_gives_25_percent_qber():
    cfg = BB84Config(n_photons=20000, seed=7)
    r = run_bb84(cfg, Channel(eve=InterceptResend(1.0)))
    assert abs(r.true_qber - 0.25) < 0.02
    assert r.aborted


def test_partial_intercept_matches_theory():
    r = run_bb84(BB84Config(n_photons=30000, seed=8),
                 Channel(eve=InterceptResend(0.4)))
    assert abs(r.true_qber - expected_qber_intercept(0.4)) < 0.012


def test_eve_knowledge_about_half_of_sifted_bits():
    r = run_bb84(BB84Config(n_photons=20000, seed=9),
                 Channel(eve=InterceptResend(1.0)))
    assert abs(r.eve_known_fraction - 0.5) < 0.03


def test_misalignment_qber_matches_sin_squared():
    th = 0.15
    r = run_bb84(BB84Config(n_photons=30000, seed=10, reconcile=False),
                 Channel(misalignment=th))
    assert abs(r.true_qber - math.sin(th) ** 2) < 0.01


def test_noisy_channel_still_yields_identical_keys_after_cascade():
    r = run_bb84(BB84Config(n_photons=8000, seed=11),
                 Channel(loss=0.2, misalignment=0.12))
    assert not r.aborted
    assert r.errors_corrected > 0
    assert r.keys_match
    assert r.final_key_alice.size > 100


def test_cascade_corrects_all_errors():
    rng = np.random.default_rng(12)
    a = rng.integers(0, 2, 2000)
    b = a.copy()
    flips = rng.choice(2000, 60, replace=False)
    b[flips] ^= 1
    fixed, nfix, leaked = cascade(a, b, 0.03, rng)
    assert np.array_equal(fixed, a)
    assert nfix >= 60 and leaked > 0
    # leakage should be well under the number of bits
    assert leaked < 0.6 * 2000


def test_cascade_no_errors_is_identity():
    rng = np.random.default_rng(13)
    a = rng.integers(0, 2, 500)
    fixed, nfix, _ = cascade(a, a.copy(), 0.01, rng)
    assert nfix == 0 and np.array_equal(fixed, a)


def test_parity():
    assert parity(np.array([1, 1, 1, 0]), 0, 4) == 1
    assert parity(np.array([1, 1, 1, 0]), 0, 2) == 0


def test_binary_entropy_and_threshold():
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.0) == 0.0
    assert qber_threshold() == pytest.approx(0.110, abs=0.002)
    assert secret_fraction(0.0) == 1.0 and secret_fraction(0.2) == 0.0


def test_secure_key_length_monotone_in_qber():
    ls = [secure_key_length(5000, q, 300) for q in (0.0, 0.02, 0.05, 0.08)]
    assert ls == sorted(ls, reverse=True) and ls[0] > ls[-1]


def test_toeplitz_hash_is_linear_and_deterministic():
    rng = np.random.default_rng(14)
    x, y = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    hx, hy = toeplitz_hash(x, 50, 3), toeplitz_hash(y, 50, 3)
    assert np.array_equal(toeplitz_hash(x ^ y, 50, 3), hx ^ hy)
    assert np.array_equal(hx, toeplitz_hash(x, 50, 3))
    assert not np.array_equal(hx, toeplitz_hash(x, 50, 4))


def test_privacy_amplification_output_is_balanced():
    rng = np.random.default_rng(15)
    out = privacy_amplify(rng.integers(0, 2, 1000), 400, 5)
    assert out.size == 400 and abs(out.mean() - 0.5) < 0.07


def test_detection_probability_grows_with_check_bits():
    assert detection_probability(1.0, 10) == pytest.approx(1 - 0.75**10)
    assert detection_probability(0.5, 50) > detection_probability(0.5, 5)
    assert detection_probability(0.0, 100) == 0.0


def test_hoeffding_margin_shrinks():
    assert hoeffding_margin(10000, 1e-6) < hoeffding_margin(100, 1e-6)


def test_sweep_aborts_under_strong_attack():
    rows = sweep_eavesdropping([0.0, 1.0], n_photons=3000, trials=3, seed=1)
    assert rows[0]["abort_rate"] == 0.0 and rows[0]["key_bits"] > 0
    assert rows[1]["abort_rate"] == 1.0


def test_reproducible_with_seed():
    c = lambda: Channel(loss=0.3, misalignment=0.1)
    a = run_bb84(BB84Config(n_photons=2000, seed=21), c())
    b = run_bb84(BB84Config(n_photons=2000, seed=21), c())
    assert np.array_equal(a.final_key_alice, b.final_key_alice)
