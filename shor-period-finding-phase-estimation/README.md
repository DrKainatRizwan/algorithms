# Shor's Algorithm Simulator: Quantum Period Finding, Phase Estimation and Continued Fractions

## Overview

A from-scratch NumPy simulation of Shor's factoring algorithm. The quantum part
(order finding) is simulated on a state vector, including the modular
exponentiation oracle, a gate-level quantum Fourier transform (QFT) and a
projective measurement of the work register. The classical part (continued
fractions, gcd post-processing, prime and perfect-power screening) is written
on plain Python integers. A generic quantum phase estimation (QPE) routine is
included and shown to reproduce the same outcome distribution.

Modules (`shor_sim/`):

| Module | Contents |
|---|---|
| `modular.py` | Euclid, extended gcd, modular inverse/power, Miller-Rabin, perfect powers, continued fractions |
| `qft.py` | Dense QFT, gate-level QFT circuit (H + controlled phases + swaps), recursive FFT |
| `qpe.py` | Phase estimation for any unitary, modular-multiplication unitary and its eigenvectors |
| `period.py` | Full and compressed order-finding back-ends, sampling, order reduction |
| `shor.py` | End-to-end factoring with an attempt trace |
| `analysis.py` | Exact success probabilities, peak mass, resource estimates |

## How it works

1. Reject even `N`, primes and perfect powers classically.
2. Pick a random base `a`; if `gcd(a, N) > 1` a factor is found immediately.
3. Prepare `Q^-1/2 sum_x |x>|a^x mod N>` with `Q = 2^t >= N^2`.
4. Measure the work register. The counting register collapses to a uniform comb
   over `x0, x0 + r, x0 + 2r, ...` where `r` is the order of `a`.
5. Apply the QFT. Outcomes concentrate near `s Q / r`, so `y/Q ~ s/r`.
6. Expand `y/Q` as a continued fraction; a convergent denominator below `N`
   is a candidate for `r` (verified by `a^r = 1 mod N`; small multiples are tried).
7. If `r` is even and `a^(r/2) != -1`, then `gcd(a^(r/2) +- 1, N)` splits `N`.

Because shifting the comb only changes phases, the outcome distribution is the
same for every work-register value. The `compressed` back-end exploits this and
stores only the counting register; the `full` back-end keeps the `(Q, N)` joint
state and is used to validate it. QPE on the multiplication-by-`a` unitary
with input `|1>` (an equal superposition of its eigenvectors with phases
`s/r`) gives the identical distribution, which is tested.

## Complexity

| Step | Time | Space |
|---|---|---|
| Gate-level QFT, `n` qubits | `O(n^2)` gates (simulated: `O(n^2 2^n)`) | `O(2^n)` |
| FFT-based QFT | `O(Q log Q)` | `O(Q)` |
| Compressed order finding | `O(Q log Q)`, `Q ~ N^2` | `O(Q)` |
| Full joint simulation | `O(Q N)` | `O(Q N)` |
| Continued fractions | `O(log N)` | `O(log N)` |

On real hardware the textbook circuit needs `3 log2 N` qubits and `O((log N)^3)` gates.

## Usage

```python
import numpy as np
from shor_sim import shor_factor, outcome_distribution, order_recovery_probability

res = shor_factor(221, np.random.default_rng(221))
print(res.factors, res.quantum_runs)       # (13, 17) 2

dist = outcome_distribution(7, 15, t=8)    # peaks at 0, 64, 128, 192
print(order_recovery_probability(7, 15))   # 0.5
```

Run the demo with `python examples/demo.py` and the tests with `python -m pytest -q`.

## Results

From `examples/demo.py` (t = ceil(2 log2 N), single run ~5 s):

| a | N | order r | P(recover r in one run) | mass within 1/2 of a peak |
|---|---|---|---|---|
| 7 | 15 | 4 | 0.500 | 1.000 |
| 2 | 21 | 6 | 0.319 | 0.793 |
| 2 | 35 | 12 | 0.321 | 0.790 |
| 3 | 55 | 20 | 0.384 | 0.780 |
| 2 | 77 | 30 | 0.259 | 0.776 |
| 5 | 91 | 12 | 0.330 | 0.789 |

Factoring (seeded): 15 = 3x5, 77 = 7x11 (2 runs), 221 = 13x17 (2 runs),
667 = 23x29 (1 run, 3.5 s dominated by the `Q = 2^20` oracle table). The
gate-level QFT agrees with the dense matrix to 6e-15. Single-run order recovery
exceeds the `phi(r)/r * 4/pi^2` bound, as expected.

## References

- P. W. Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete
  Logarithms on a Quantum Computer", SIAM J. Comput. 26(5), 1997.
- A. Kitaev, "Quantum measurements and the Abelian Stabilizer Problem", 1995.
- R. Cleve, A. Ekert, C. Macchiavello, M. Mosca, "Quantum algorithms revisited", Proc. R. Soc. A, 1998.
- M. Nielsen, I. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press, 2000, ch. 5.
- D. Coppersmith, "An approximate Fourier transform useful in quantum factoring", 1994.

---

**Author:** Dr. Kainat Rizwan
