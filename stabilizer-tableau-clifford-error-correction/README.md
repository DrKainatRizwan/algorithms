# Stabilizer Tableau Simulator: Clifford Circuits and Quantum Error-Correcting Codes

## Overview

A from-scratch implementation of the Gottesman-Knill theorem: circuits built from Hadamard, phase, CNOT (and Pauli) gates, with Pauli measurements, can be simulated in polynomial time on a classical computer. The package implements the Aaronson-Gottesman tableau (CHP) simulator, a dense state-vector reference used for cross-validation, and stabilizer codes (3-qubit bit-flip, Steane [[7,1,3]], five-qubit [[5,1,3]]) with lookup-table decoding and a Monte-Carlo memory experiment.

Modules (`stabilizer_sim/`):

- `pauli.py` – signed Pauli strings in symplectic form, exact phase tracking, dense matrices.
- `tableau.py` – destabilizer/stabilizer tableau, gates, Z and Pauli measurement, expectation values, entanglement entropy via GF(2) rank.
- `circuit.py` – circuit container, inverse, sampling, GHZ and random Clifford generators.
- `statevector.py` – 2^n reference simulator and tableau-to-state reconstruction.
- `codes.py` – stabilizer codes, syndrome decoder, noisy memory experiment.

## How it works

A stabilizer state on `n` qubits is the unique +1 eigenstate of `n` commuting Pauli operators. Each is stored as a row of binary vectors `(x | z)` plus a sign bit, alongside `n` destabilizer rows that make the basis symplectic. Clifford gates update every row column-wise (e.g. H swaps x and z and flips the sign where both are set; CNOT adds x_a to x_b and z_b to z_a). Measuring qubit `a` in Z is random if some stabilizer has x_a = 1 (the pivot is replaced by ±Z_a and other rows are multiplied by it), otherwise deterministic and read off a product of stabilizers computed in a scratch row. General Pauli measurement is reduced to Z measurement by a basis change and CNOT fold. For a region A the entanglement entropy is `rank_GF(2)(G_A) - |A|`.

For codes, logical states are prepared by post-selected stabilizer measurements, errors are applied as Pauli gates, the syndrome is computed exactly from expectation values, and a minimum-weight table gives the correction.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Clifford gate | O(n) | O(n²) tableau |
| Z measurement | O(n²) | O(n²) |
| Pauli expectation | O(n²) | O(n²) |
| Entropy of region | O(n³) worst case | O(n²) |
| State-vector gate (reference) | O(2ⁿ) | O(2ⁿ) |

## Usage

```python
import numpy as np
from stabilizer_sim import Circuit, Pauli, ghz_circuit, steane_code, memory_experiment

t = Circuit(2).h(0).cx(0, 1).run_tableau()
print([s.label for s in t.stabilizers()])      # ['+XX', '+ZZ']
print(ghz_circuit(100).run_tableau().entanglement_entropy(list(range(50))))  # 1

rate = memory_experiment(steane_code(), 0.01, 200, np.random.default_rng(0))
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests (25 tests).

## Results

Output of `examples/demo.py` (seeded):

```
== Bell state stabilizers ==
stabilizers: ['+XX', '+ZZ']

== Cross-check against dense simulation (random Clifford circuits) ==
n=4  gates=  49  fidelity=1.000000000000
n=6  gates=  88  fidelity=1.000000000000
n=8  gates= 129  fidelity=1.000000000000

== Runtime scaling: tableau vs state vector (GHZ + random Clifford, depth 6) ==
    n  tableau (s)  statevector (s)
    8       0.0005           0.0010
   12       0.0008           0.0034
   14       0.0008           0.0091
   50       0.0034       infeasible
  100       0.0075       infeasible
  200       0.0166       infeasible

== GHZ-100: sampling and entanglement ==
distinct outcomes: 2 all-equal: True
entropy of first 50 qubits: 1 bit

== Entanglement of random Clifford circuits (n=24, half cut) ==
depth= 1  mean S =  1.20 bits (max 12)
depth= 2  mean S =  2.20 bits (max 12)
depth= 4  mean S =  4.60 bits (max 12)
depth= 8  mean S =  9.80 bits (max 12)
depth=16  mean S = 10.90 bits (max 12)

== Logical error rate under depolarising noise (200 trials) ==
      p       bit-flip [[3,1,1]]         Steane [[7,1,3]]     Five-qubit [[5,1,3]]
  0.010                   0.0050                   0.0000                   0.0000
  0.050                   0.0450                   0.0250                   0.0100
  0.100                   0.1050                   0.1000                   0.0650
```

The tableau agrees with the dense simulator to fidelity 1 on random circuits, handles 200 qubits in about 0.02 s where a state vector is infeasible, and random Clifford circuits approach the volume-law half-cut entropy (about 11 of 12 bits at depth 16). Both distance-3 codes suppress the logical error rate below the bare 3-qubit repetition code, with the five-qubit code best at p = 0.10.

## References

- D. Gottesman, "The Heisenberg Representation of Quantum Computers", 1998.
- S. Aaronson and D. Gottesman, "Improved Simulation of Stabilizer Circuits", Phys. Rev. A 70, 052328 (2004).
- D. Gottesman, "Stabilizer Codes and Quantum Error Correction", PhD thesis, Caltech, 1997.
- A. Steane, "Error Correcting Codes in Quantum Theory", Phys. Rev. Lett. 77, 793 (1996).
- R. Laflamme et al., "Perfect Quantum Error Correcting Code", Phys. Rev. Lett. 77, 198 (1996).
- M. Nielsen and I. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press.

---

**Author:** Dr. Kainat Rizwan
