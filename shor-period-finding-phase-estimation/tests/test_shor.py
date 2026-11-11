import numpy as np
import pytest

from shor_sim import (continued_fraction, convergents, factors_from_order, gcd,
                      modinv, modpow, multiplicative_order, order_recovery_probability,
                      outcome_distribution, peak_mass, phase_estimation_distribution,
                      qft_fft, qft_gates, qft_matrix, recover_denominator,
                      resource_estimate, sample_order, shor_factor, split_probability,
                      theoretical_success_floor)
from shor_sim.modular import euler_phi, is_probable_prime, perfect_power
from shor_sim.period import combine_orders, counting_qubits
from shor_sim.qft import fft_recursive
from shor_sim.qpe import (eigenvectors_of_modmul, estimate_phase,
                          modular_multiplication_unitary)


def rand_state(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return v / np.linalg.norm(v)


def test_number_theory_basics():
    assert gcd(84, 36) == 12
    assert modpow(7, 128, 15) == pow(7, 128, 15)
    assert (modinv(7, 40) * 7) % 40 == 1
    with pytest.raises(ValueError):
        modinv(6, 9)
    assert is_probable_prime(104729) and not is_probable_prime(104729 * 3)
    assert perfect_power(3**7) == (3, 7) and perfect_power(15) is None
    assert euler_phi(36) == 12
    assert multiplicative_order(7, 15) == 4


def test_continued_fractions():
    assert continued_fraction(415, 93) == [4, 2, 6, 7]
    assert convergents(415, 93)[-1] == (415, 93)
    assert recover_denominator(192, 256, 15) == 4
    assert recover_denominator(0, 256, 15) == 0


def test_qft_unitary_and_matrix_agree_with_fft():
    f = qft_matrix(4)
    assert np.allclose(f @ f.conj().T, np.eye(16))
    v = rand_state(4)
    assert np.allclose(f @ v, qft_fft(v))
    assert np.allclose(fft_recursive(v), np.fft.ifft(v) * 16)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_gate_level_qft_matches_matrix(n):
    v = rand_state(n, seed=n)
    assert np.allclose(qft_gates(v, n), qft_matrix(n) @ v)
    assert np.allclose(qft_gates(v, n, inverse=True), qft_matrix(n, True) @ v)


def test_qft_inverse_roundtrip():
    v = rand_state(6, 3)
    assert np.allclose(qft_gates(qft_gates(v, 6), 6, inverse=True), v)


def test_qpe_exact_phase_is_deterministic():
    t = 5
    phi = 11 / 32
    u = np.array([[1, 0], [0, np.exp(2j * np.pi * phi)]])
    dist = phase_estimation_distribution(u, np.array([0, 1]), t)
    assert dist.argmax() == 11 and dist[11] == pytest.approx(1.0)


def test_qpe_inexact_phase_peaks_near_truth():
    t, phi = 6, 0.3
    u = np.diag([1, np.exp(2j * np.pi * phi)])
    dist = phase_estimation_distribution(u, np.array([0, 1]), t)
    assert abs(dist.argmax() / 64 - phi) <= 1 / 64
    assert dist.sum() == pytest.approx(1.0)
    assert dist[round(phi * 64)] > 4 / np.pi**2 - 1e-9
    est = estimate_phase(u, np.array([0, 1]), t, np.random.default_rng(1))
    assert abs(est - phi) < 0.1


def test_modmul_eigenvectors():
    a, n = 7, 15
    r = multiplicative_order(a, n)
    u = modular_multiplication_unitary(a, n)
    for phase, v in eigenvectors_of_modmul(a, n, r):
        assert np.allclose(u @ v, np.exp(2j * np.pi * phase) * v)
        assert np.linalg.norm(v) == pytest.approx(1.0)


def test_qpe_on_modmul_matches_shor_distribution():
    a, n, t = 7, 15, 6
    u = modular_multiplication_unitary(a, n)
    one = np.zeros(n)
    one[1] = 1
    qpe = phase_estimation_distribution(u, one, t)
    assert np.allclose(qpe, outcome_distribution(a, n, t), atol=1e-12)


def test_backends_agree_and_gates_agree():
    a, n, t = 2, 15, 7
    comp = outcome_distribution(a, n, t, "compressed")
    full = outcome_distribution(a, n, t, "full")
    gates = outcome_distribution(a, n, t, "compressed", use_gates=True)
    assert np.allclose(comp, full, atol=1e-12)
    assert np.allclose(comp, gates, atol=1e-12)
    assert comp.sum() == pytest.approx(1.0)


def test_distribution_peaks_at_multiples_of_q_over_r():
    a, n, t = 7, 15, 8
    d = outcome_distribution(a, n, t)
    for y in (0, 64, 128, 192):
        assert d[y] == pytest.approx(0.25)
    assert peak_mass(a, n, t) == pytest.approx(1.0)


def test_order_recovery_probability_reasonable():
    assert order_recovery_probability(7, 15, t=8) == pytest.approx(0.5)
    assert order_recovery_probability(2, 21) >= theoretical_success_floor(6) - 1e-9


def test_sample_order_valid_for_both_backends():
    for backend in ("compressed", "full"):
        rng = np.random.default_rng(5)
        orders = {sample_order(7, 15, rng, backend).order for _ in range(12)}
        assert orders <= {0, 1, 2, 4} and 4 in orders
    assert combine_orders(7, 15, [2, 4]) == 4
    assert combine_orders(7, 15, [0, 0]) == 0
    assert counting_qubits(15) == 8


def test_factors_from_order():
    assert factors_from_order(7, 4, 15) == (3, 5)
    assert factors_from_order(14, 2, 15) is None
    assert factors_from_order(2, 3, 21) is None


@pytest.mark.parametrize("n", [15, 21, 33, 35, 77, 143, 221])
def test_shor_factors_semiprimes(n):
    res = shor_factor(n, np.random.default_rng(n))
    assert res.success
    p, q = res.factors
    assert p * q == n and 1 < p <= q < n


def test_shor_trivial_cases_and_full_backend():
    rng = np.random.default_rng(0)
    assert shor_factor(14, rng).method == "even"
    assert not shor_factor(97, rng).success
    assert shor_factor(27, rng).factors == (3, 9)
    res = shor_factor(15, np.random.default_rng(2), backend="full")
    assert res.factors == (3, 5)
    with pytest.raises(ValueError):
        shor_factor(3, rng)


def test_split_probability_and_resources():
    assert split_probability(15) == pytest.approx(6 / 7)
    assert 0.5 <= split_probability(77) <= 1
    r = resource_estimate(15)
    assert r["total_qubits"] == 12 and r["qft_controlled_phases"] == 28
