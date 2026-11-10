"""Demo: Clifford simulation at scale, entanglement scaling and error correction."""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stabilizer_sim import (Circuit, Pauli, ghz_circuit, random_clifford_circuit,
                            tableau_to_statevector, fidelity, steane_code,
                            five_qubit_code, bit_flip_code, memory_experiment)


def main() -> None:
    rng = np.random.default_rng(2024)

    print("== Bell state stabilizers ==")
    t = Circuit(2).h(0).cx(0, 1).run_tableau()
    print("stabilizers:", [s.label for s in t.stabilizers()])

    print("\n== Cross-check against dense simulation (random Clifford circuits) ==")
    for n in (4, 6, 8):
        c = random_clifford_circuit(n, 10, rng)
        f = fidelity(c.run_statevector().psi, tableau_to_statevector(c.run_tableau()))
        print(f"n={n}  gates={len(c):4d}  fidelity={f:.12f}")

    print("\n== Runtime scaling: tableau vs state vector (GHZ + random Clifford, depth 6) ==")
    print(f"{'n':>5} {'tableau (s)':>12} {'statevector (s)':>16}")
    for n in (8, 12, 14, 50, 100, 200):
        c = random_clifford_circuit(n, 6, np.random.default_rng(n))
        t0 = time.perf_counter(); c.run_tableau(); tt = time.perf_counter() - t0
        if n <= 14:
            t0 = time.perf_counter(); c.run_statevector(); ts = f"{time.perf_counter() - t0:16.4f}"
        else:
            ts = f"{'infeasible':>16}"
        print(f"{n:>5} {tt:12.4f} {ts}")

    print("\n== GHZ-100: sampling and entanglement ==")
    g = ghz_circuit(100)
    counts = g.sample(20, rng)
    print("distinct outcomes:", len(counts), "all-equal:", all(len(set(k)) == 1 for k in counts))
    print("entropy of first 50 qubits:", g.run_tableau().entanglement_entropy(list(range(50))), "bit")

    print("\n== Entanglement of random Clifford circuits (n=24, half cut) ==")
    for depth in (1, 2, 4, 8, 16):
        vals = [random_clifford_circuit(24, depth, np.random.default_rng(s)).run_tableau()
                .entanglement_entropy(list(range(12))) for s in range(10)]
        print(f"depth={depth:2d}  mean S = {np.mean(vals):5.2f} bits (max 12)")

    print("\n== Logical error rate under depolarising noise (200 trials) ==")
    codes = [bit_flip_code(), steane_code(), five_qubit_code()]
    print(f"{'p':>7} " + " ".join(f"{c.name:>24}" for c in codes))
    for p in (0.01, 0.05, 0.10):
        rates = [memory_experiment(c, p, 200, np.random.default_rng(11)) for c in codes]
        print(f"{p:7.3f} " + " ".join(f"{r:24.4f}" for r in rates))


if __name__ == "__main__":
    main()
