import math

import numpy as np
import pytest

from qsim_core import (
    Circuit, StateVector, deutsch_jozsa, dft_matrix, estimate_phase, ghz_circuit,
    grover_circuit, grover_search, grover_success_curve, inverse_qft_circuit,
    optimal_grover_iterations, qft_circuit,
)
from qsim_core import gates


def test_hadamard_superposition():
    st = Circuit(1).h(0).run()
    assert np.allclose(st.probabilities(), [0.5, 0.5])


def test_big_endian_ordering():
    st = Circuit(3).x(0).run()
    assert np.argmax(st.probabilities()) == 0b100


def test_bell_state_amplitudes():
    st = Circuit(2).h(0).cx(0, 1).run()
    assert np.allclose(st.vector, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])


def test_ghz_sampling_only_extremes():
    st = ghz_circuit(4).run()
    counts = st.sample(500, np.random.default_rng(0))
    assert set(counts) <= {"0000", "1111"}
    assert 180 < counts["0000"] < 320


def test_apply_matches_dense_kron():
    rng = np.random.default_rng(1)
    v = rng.normal(size=8) + 1j * rng.normal(size=8)
    v /= np.linalg.norm(v)
    st = StateVector.from_vector(v)
    st.apply(gates.H, [1])
    expected = gates.kron_all(gates.I2, gates.H, gates.I2) @ v
    assert np.allclose(st.vector, expected)


def test_controlled_matches_dense_with_noncontiguous_qubits():
    rng = np.random.default_rng(2)
    v = rng.normal(size=8) + 1j * rng.normal(size=8)
    v /= np.linalg.norm(v)
    st = StateVector.from_vector(v)
    st.apply_controlled(gates.rx(0.7), [2], [0])
    # dense: control qubit 2 (LSB), target qubit 0 (MSB)
    expected = v.copy().reshape(2, 2, 2)
    u = gates.rx(0.7)
    expected[:, :, 1] = np.tensordot(u, expected[:, :, 1], axes=(1, 0))
    assert np.allclose(st.vector, expected.reshape(-1))


def test_circuit_unitary_is_unitary_and_inverse_undoes():
    c = Circuit(3).h(0).cx(0, 1).t(2).cp(0.3, 2, 0).swap(0, 2).ry(1.1, 1)
    u = c.unitary()
    assert gates.is_unitary(u)
    full = Circuit(3).compose(c).compose(c.inverse())
    assert np.allclose(full.unitary(), np.eye(8))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_qft_matches_dft_matrix(n):
    assert np.allclose(qft_circuit(n).unitary(), dft_matrix(n))


def test_qft_inverse_roundtrip():
    c = Circuit(4).compose(qft_circuit(4)).compose(inverse_qft_circuit(4))
    assert np.allclose(c.unitary(), np.eye(16))


def test_qft_gate_counts():
    n = 6
    counts = qft_circuit(n).gate_counts()
    assert counts["H"] == n and counts["SWAP"] == n // 2
    assert sum(v for k, v in counts.items() if k.startswith("CP")) == n * (n - 1) // 2


@pytest.mark.parametrize("n,marked", [(3, [5]), (5, [17]), (6, [0]), (6, [63])])
def test_grover_finds_single_marked(n, marked):
    st = grover_circuit(n, marked).run()
    assert st.probabilities()[marked[0]] > 0.94  # n=3 peaks at 0.945 analytically


def test_grover_multiple_marked():
    n, marked = 6, [3, 10, 40, 41]
    st = grover_circuit(n, marked).run()
    assert st.probabilities()[marked].sum() > 0.94  # n=3 peaks at 0.945 analytically


def test_grover_iteration_formula_and_curve_peak():
    n = 8
    k = optimal_grover_iterations(n, 1)
    assert k == 12
    curve = grover_success_curve(n, [77], 20)
    assert int(np.argmax(curve)) == k
    assert curve[0] == pytest.approx(1 / 256)


def test_grover_search_sampling():
    counts = grover_search(5, [9], np.random.default_rng(3), shots=100)
    assert counts.get("01001", 0) >= 95


def test_measure_collapses_state():
    st = Circuit(2).h(0).cx(0, 1).run()
    out = st.measure(0, np.random.default_rng(5))
    assert np.allclose(st.probabilities(), [1, 0, 0, 0] if out == 0 else [0, 0, 0, 1])
    assert st.measure(1, np.random.default_rng(6)) == out


def test_phase_estimation_exact_and_approximate():
    est, p = estimate_phase(4, 0.3125)  # 5/16, exactly representable
    assert est == 0.3125 and p == pytest.approx(1.0)
    est, p = estimate_phase(6, 1 / 3)
    assert abs(est - 1 / 3) < 1 / 64 and p > 0.4


def test_deutsch_jozsa():
    assert deutsch_jozsa(3, lambda x: 0) == "constant"
    assert deutsch_jozsa(3, lambda x: 1) == "constant"
    assert deutsch_jozsa(3, lambda x: x & 1) == "balanced"
    assert deutsch_jozsa(3, lambda x: bin(x).count("1") & 1) == "balanced"


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Circuit(2).cx(0, 0)
    with pytest.raises(ValueError):
        Circuit(2).h(2)
    with pytest.raises(ValueError):
        Circuit(1).add("bad", np.array([[1, 1], [0, 1]], dtype=complex), [0])


def test_depth_and_norm_preserved():
    c = Circuit(3).h(0).h(1).h(2).cx(0, 1).cx(1, 2)
    assert c.depth() == 3
    assert c.run().is_normalised()
