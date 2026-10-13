import numpy as np
import pytest

from vqe_simulator import (VQE, PauliHamiltonian, adam, adapt_vqe,
                           apply_pauli_string, finite_difference_gradient,
                           h2_hamiltonian, h2_uccsd, hamiltonian_variational,
                           hardware_efficient, heisenberg_chain, nelder_mead,
                           parameter_shift_gradient, pauli_matrix, rotosolve,
                           shot_energy, spsa, transverse_field_ising)


def rand_state(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("label", ["XYZ", "YYI", "ZIX", "IYI", "YZY"])
def test_pauli_apply_matches_matrix(label):
    s = rand_state(3, 1)
    assert np.allclose(apply_pauli_string(label, s), pauli_matrix(label) @ s)


def test_hamiltonian_expectation_matches_dense():
    h = heisenberg_chain(3) + transverse_field_ising(3)
    s = rand_state(3, 2)
    assert h.expectation(s) == pytest.approx(np.real(s.conj() @ h.matrix() @ s))


def test_h2_ground_energy_reference():
    assert h2_hamiltonian().ground_energy() == pytest.approx(-1.857275, abs=1e-5)


def test_variance_zero_on_eigenstate():
    h = transverse_field_ising(3)
    assert h.variance(h.ground_state()) == pytest.approx(0.0, abs=1e-10)


def test_ansatz_state_is_normalised():
    a = hardware_efficient(3, 2)
    x = np.random.default_rng(0).normal(size=a.num_params)
    assert np.linalg.norm(a.run(x)) == pytest.approx(1.0)


def test_cnot_creates_bell_state():
    from vqe_simulator import Ansatz
    a = Ansatz(2)
    a.single("Y", 0, 0).cx(0, 1)
    s = a.run(np.array([np.pi / 2]))
    assert abs(s[0]) ** 2 == pytest.approx(0.5)
    assert abs(s[3]) ** 2 == pytest.approx(0.5)


def test_parameter_shift_matches_finite_difference():
    h = transverse_field_ising(3, 1.0, 0.7)
    a = hardware_efficient(3, 2)
    x = np.random.default_rng(3).normal(size=a.num_params)
    g1 = parameter_shift_gradient(a, h, x)
    g2 = finite_difference_gradient(lambda p: a.energy(p, h), x)
    assert np.allclose(g1, g2, atol=1e-6)


def test_shared_parameter_gradient():
    h = transverse_field_ising(3, 1.0, 0.5)
    a = hamiltonian_variational(h, 2)
    x = np.random.default_rng(4).normal(size=a.num_params)
    g1 = parameter_shift_gradient(a, h, x)
    g2 = finite_difference_gradient(lambda p: a.energy(p, h), x)
    assert a.num_params == 4
    assert np.allclose(g1, g2, atol=1e-6)


def test_h2_uccsd_reaches_exact_energy():
    h = h2_hamiltonian()
    r = VQE(h, h2_uccsd()).run("adam", x0=np.array([0.1]), iters=150)
    assert abs(r.error) < 1e-5


def test_variational_principle():
    h = heisenberg_chain(4)
    v = VQE(h, hardware_efficient(4, 2))
    for seed in range(5):
        assert v.exact_energy(v.initial_params(1.0, seed)) >= h.ground_energy() - 1e-9


def test_hartree_fock_energy_above_ground():
    h = h2_hamiltonian()
    a = h2_uccsd()
    e_hf = a.energy(np.array([0.0]), h)
    assert e_hf > h.ground_energy() + 0.01


def test_tfim_vqe_close_to_exact():
    h = transverse_field_ising(4, 1.0, 1.0)
    v = VQE(h, hardware_efficient(4, 3))
    r = v.run("adam", iters=300)
    assert r.error < 5e-3 and r.error > -1e-9


def test_rotosolve_on_h2_hea():
    h = h2_hamiltonian()
    v = VQE(h, hardware_efficient(2, 2))
    r = v.run("rotosolve", iters=8, x0=v.initial_params(0.5))
    assert abs(r.error) < 1e-3


def test_nelder_mead_quadratic():
    res = nelder_mead(lambda x: float((x[0] - 1) ** 2 + (x[1] + 2) ** 2),
                      np.zeros(2), iters=500)
    assert np.allclose(res.x, [1, -2], atol=1e-4)


def test_spsa_reduces_energy_with_shot_noise():
    h = h2_hamiltonian()
    v = VQE(h, h2_uccsd(), shots=2000, seed=5)
    r = v.run("spsa", x0=np.array([0.0]), iters=200)
    assert r.error < 0.01


def test_shot_energy_unbiased():
    h = h2_hamiltonian()
    s = h2_uccsd().run(np.array([0.3]))
    rng = np.random.default_rng(0)
    est = np.mean([shot_energy(h, s, 500, rng) for _ in range(300)])
    assert est == pytest.approx(h.expectation(s), abs=0.01)


def test_adapt_vqe_converges_on_h2():
    h = h2_hamiltonian()
    ref = h2_uccsd()
    ref.ops = ref.ops[:1]  # keep only the HF preparation
    ref.num_params = 0
    ans, x, es = adapt_vqe(h, ["YX", "XY", "ZZ", "XX", "YY"], ref, max_ops=4)
    assert es[-1] == pytest.approx(h.ground_energy(), abs=1e-4)
    assert len(ans.ops) >= 2 and all(e >= h.ground_energy() - 1e-9 for e in es)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        PauliHamiltonian({"XX": 1.0, "X": 1.0})
    with pytest.raises(ValueError):
        PauliHamiltonian({"XQ": 1.0})
    with pytest.raises(ValueError):
        VQE(h2_hamiltonian(), hardware_efficient(3, 1))
    with pytest.raises(ValueError):
        VQE(h2_hamiltonian(), h2_uccsd()).run("bogus")
