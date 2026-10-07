# Quantum Circuit State-Vector Simulator with Grover's Search and QFT

## Overview
A small, dependency-light (NumPy only) state-vector simulator written from scratch. It supports arbitrary k-qubit gates, multi-controlled gates, projective measurement and sampling, and ships implementations of the Quantum Fourier Transform, Grover's search (single and multiple marked items), quantum phase estimation, GHZ preparation and Deutsch–Jozsa.

## How it works
- **State representation.** `n` qubits are stored as a tensor of shape `(2,)*n`; qubit 0 is the most significant bit. A k-qubit gate is applied with a single `tensordot` over the relevant axes followed by `moveaxis`, so no `2^n x 2^n` matrix is ever built.
- **Controlled gates.** Controls are handled by slicing the tensor at control = 1 and applying the gate to the slice, which costs `O(2^(n-c))` for `c` controls.
- **QFT.** `H` on each qubit followed by controlled phase rotations `P(pi/2^(j-i))` and a final qubit reversal; it is verified against the dense DFT matrix.
- **Grover.** Uniform superposition, then `floor(pi/4 * sqrt(N/M))` rounds of a phase oracle (X-conjugated multi-controlled Z per marked state) and the diffusion operator `H^n X^n MCZ X^n H^n`.
- **Phase estimation.** Counting register with controlled-`U^(2^k)`, followed by the inverse QFT.

## Complexity
| Operation | Time | Space |
|---|---|---|
| k-qubit gate | O(2^n · 2^k) | O(2^n) |
| controlled gate (c controls) | O(2^(n-c) · 2^k) | O(2^n) |
| QFT circuit | O(n^2) gates | O(2^n) simulation |
| Grover | O(sqrt(N/M)) iterations, each O(M·n) gates | O(2^n) |

## Usage
```python
import numpy as np
from qsim_core import grover_search, qft_circuit, Circuit

print(grover_search(5, [9], np.random.default_rng(0), shots=20))   # {'01001': 20}
state = Circuit(2).h(0).cx(0, 1).run()
print(state.vector)                                                  # Bell state
```

Run `python examples/demo.py` for the demo and `python -m pytest -q` for the tests (26 tests).

## Results
Output of `examples/demo.py` (single marked item, one core):

| n | N | iterations | P(success) | time (s) |
|--:|--:|--:|--:|--:|
| 4 | 16 | 3 | 0.9613 | 0.004 |
| 6 | 64 | 6 | 0.9966 | 0.009 |
| 8 | 256 | 12 | 0.9999 | 0.017 |
| 10 | 1024 | 25 | 0.9995 | 0.046 |
| 12 | 4096 | 50 | 0.9999 | 0.122 |

The QFT circuit matches the DFT matrix to within 5e-15 for up to 6 qubits. Phase estimation of phase 1/3 gives errors of 0.0417, 0.0104, 0.0026 and 0.00065 for 3, 5, 7 and 9 counting qubits (each step of two qubits cuts the error by about 4x).

## References
- L. K. Grover, "A fast quantum mechanical algorithm for database search", STOC 1996.
- D. Coppersmith, "An approximate Fourier transform useful in quantum factoring", IBM Research Report RC19642, 1994.
- A. Kitaev, "Quantum measurements and the Abelian stabilizer problem", 1995.
- M. Nielsen and I. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press, 2000.

---

**Author:** Dr. Kainat Rizwan
