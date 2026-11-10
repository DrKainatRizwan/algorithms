import numpy as np
import pytest

from stabilizer_sim import (Circuit, Pauli, Tableau, ghz_circuit, random_clifford_circuit,
                            tableau_to_statevector, fidelity, steane_code, five_qubit_code,
                            bit_flip_code, memory_experiment)


def test_pauli_multiplication_matches_matrices():
    rng = np.random.default_rng(1)
    for _ in range(30):
        a = Pauli.from_label("".join(rng.choice(list("IXYZ"), 3)))
        b = Pauli.from_label("".join(rng.choice(list("IXYZ"), 3)))
        k, r = a.multiply(b)
        assert np.allclose(a.to_matrix() @ b.to_matrix(), (1j ** k) * r.to_matrix())


def test_pauli_commutation_matches_matrices():
    a, b = Pauli.from_label("XZ"), Pauli.from_label("ZX")
    assert a.commutes(b)
    c = Pauli.from_label("XI")
    assert not c.commutes(Pauli.from_label("ZI"))
    am, bm = c.to_matrix(), Pauli.from_label("ZI").to_matrix()
    assert np.allclose(am @ bm, -bm @ am)


def test_initial_state_stabilizers():
    t = Tableau(3)
    assert [s.label for s in t.stabilizers()] == ["+ZII", "+IZI", "+IIZ"]
    assert t.is_valid()


def test_bell_state_stabilizers():
    t = Circuit(2).h(0).cx(0, 1).run_tableau()
    assert t.expectation(Pauli.from_label("XX")) == 1
    assert t.expectation(Pauli.from_label("ZZ")) == 1
    assert t.expectation(Pauli.from_label("YY")) == -1
    assert t.expectation(Pauli.from_label("ZI")) == 0


@pytest.mark.parametrize("seed", range(8))
def test_random_clifford_matches_statevector(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    c = random_clifford_circuit(n, 6, rng)
    t = c.run_tableau()
    assert t.is_valid()
    sv = c.run_statevector().psi
    assert fidelity(sv, tableau_to_statevector(t)) == pytest.approx(1.0, abs=1e-9)


def test_all_gate_types_match_statevector():
    c = Circuit(3)
    c.h(0).s(1).add("sdg", 2).add("y", 0).x(1).z(2).cx(0, 1).cz(1, 2).add("swap", 0, 2).h(2)
    t = c.run_tableau()
    assert fidelity(c.run_statevector().psi, tableau_to_statevector(t)) == pytest.approx(1.0, abs=1e-9)


def test_circuit_followed_by_inverse_is_identity():
    rng = np.random.default_rng(5)
    c = random_clifford_circuit(5, 8, rng)
    t = (c + c.inverse()).run_tableau()
    assert [s.label for s in t.stabilizers()] == ["+ZIIII", "+IZIII", "+IIZII", "+IIIZI", "+IIIIZ"]


def test_ghz_measurements_are_correlated_and_balanced():
    rng = np.random.default_rng(2)
    counts = ghz_circuit(6).sample(400, rng)
    assert set(counts) <= {"000000", "111111"}
    assert 140 < counts.get("000000", 0) < 260


def test_deterministic_measurement_and_post_measurement_collapse():
    rng = np.random.default_rng(3)
    t = Circuit(2).x(0).run_tableau()
    assert t.measure(0, rng) == 1 and t.measure(1, rng) == 0
    b = Circuit(2).h(0).cx(0, 1).run_tableau()
    first = b.measure(0, rng)
    assert b.measure(1, rng) == first
    assert b.expectation(Pauli.single(2, 0, "Z")) == (1 if first == 0 else -1)


def test_measure_pauli_matches_statevector_statistics():
    # |+> measured in X is deterministic; measured in Y is a fair coin
    rng = np.random.default_rng(4)
    t = Circuit(1).h(0).run_tableau()
    assert t.measure_pauli(Pauli.from_label("X"), rng) == 0
    outs = [Circuit(1).h(0).run_tableau().measure_pauli(Pauli.from_label("Y"), rng) for _ in range(200)]
    assert 70 < sum(outs) < 130


def test_measure_pauli_projects_state():
    rng = np.random.default_rng(6)
    t = Tableau(2)
    out = t.measure_pauli(Pauli.from_label("XX"), rng)
    assert t.expectation(Pauli.from_label("XX")) == (1 if out == 0 else -1)
    assert t.is_valid()


def test_entanglement_entropy():
    assert ghz_circuit(5).run_tableau().entanglement_entropy([0, 1]) == 1
    assert Tableau(4).entanglement_entropy([0, 1]) == 0
    bell2 = Circuit(4).h(0).cx(0, 2).h(1).cx(1, 3).run_tableau()
    assert bell2.entanglement_entropy([0, 1]) == 2
    assert bell2.entanglement_entropy([0, 2]) == 0


def test_codes_have_valid_structure():
    for code, d in [(steane_code(), 3), (five_qubit_code(), 3)]:
        assert code.distance_upper_bound() == d
        assert len(code.table) == 2 ** len(code.generators)


def test_single_qubit_errors_corrected():
    for code in (steane_code(), five_qubit_code()):
        for q in range(code.n):
            for k in "XYZ":
                t = code.encode_zero()
                err = Pauli.single(code.n, q, k)
                t.apply_pauli(err)
                t.apply_pauli(code.decode(code.measure_syndrome(t)))
                assert code.logical_ok(t, "z")
                assert all(t.expectation(g) == 1 for g in code.generators)


def test_bit_flip_code_fails_on_phase_errors():
    code = bit_flip_code()
    t = code.encode_plus()
    t.apply_pauli(Pauli.from_label("ZII"))
    t.apply_pauli(code.decode(code.measure_syndrome(t)) or Pauli.from_label("III"))
    assert not code.logical_ok(t, "x")


def test_logical_error_rate_suppressed_below_threshold():
    code = steane_code()
    rng = np.random.default_rng(7)
    p = 0.01
    rate = memory_experiment(code, p, 600, rng)
    assert rate < p


def test_invalid_code_rejected():
    with pytest.raises(ValueError):
        from stabilizer_sim import StabilizerCode
        StabilizerCode("bad", ["XI", "ZI"], "XX", "ZZ")


def test_circuit_validation():
    with pytest.raises(ValueError):
        Circuit(2).cx(0, 0)
    with pytest.raises(ValueError):
        Circuit(2).add("t", 0)
