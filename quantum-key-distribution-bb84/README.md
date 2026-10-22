# Quantum Key Distribution: BB84 Simulator with Eavesdropper Detection

## Overview
A from-scratch simulator of the BB84 protocol (Bennett & Brassard, 1984) using only NumPy. Photons are modelled as polarisation angles, so measurement statistics follow Malus' law and measurement collapses the state. The package covers the full pipeline: state preparation, a lossy/noisy channel, an intercept-resend eavesdropper, sifting, QBER estimation, Cascade error reconciliation and Toeplitz-hash privacy amplification with a finite-size key-length bound.

## How it works
1. **Preparation** – Alice draws random bits and bases (rectilinear 0/90°, diagonal 45/135°) and sends one photon per bit.
2. **Channel** – photons may be lost, rotated (misalignment adds `sin²θ` errors) or replaced by dark counts. Eve can measure a chosen fraction of photons in random bases and resend them, inducing a 25% error rate on attacked photons.
3. **Sifting** – Bob measures in random bases; positions with detection and matching bases are kept (~50% of detected photons).
4. **QBER estimation** – a random sample is publicly compared; if the error rate exceeds 11% (the Shor–Preskill threshold where `1 − 2h(e) = 0`) the session aborts.
5. **Reconciliation** – Cascade runs 4 passes with doubling block sizes, binary-search error location and back-tracking; every disclosed parity is counted as leakage.
6. **Privacy amplification** – a random Toeplitz matrix over GF(2) (2-universal) compresses the reconciled key to `n(1 − h(e + δ)) − leak − 2 log₂(1/ε)` bits, where δ is a Hoeffding margin.

## Complexity
- Simulation: O(n) time and memory for n photons (vectorised).
- Cascade: O(p·n) parity computations plus O(log k) per located error; leakage about n·h(e)·(1+small factor).
- Toeplitz hashing: O(n·m) as implemented (direct matrix–vector product), O(n log n) possible with FFT.

## Usage
```python
from qkd_bb84 import BB84Config, Channel, InterceptResend, run_bb84

r = run_bb84(BB84Config(n_photons=6000, seed=1), Channel(loss=0.1, misalignment=0.1))
print(r.true_qber, r.keys_match, r.final_key_alice.size)

attacked = run_bb84(BB84Config(n_photons=6000, seed=1), Channel(eve=InterceptResend(1.0)))
print(attacked.aborted)   # True
```
Run `python -m pytest -q` for the tests and `python examples/demo.py` for the demo.

## Results (from `examples/demo.py`, seed 1, 6000 photons)
| Scenario | Sifted | QBER | Final key | Outcome |
|---|---|---|---|---|
| Ideal channel | 2959 | 0.000 | 1575 | keys match |
| 10% loss, 0.10 rad misalignment | 2653 | 0.010 | 1137 | keys match |
| Full intercept-resend | 2959 | 0.253 | 0 | aborted |
| 20% intercept-resend | 2959 | 0.056 | 227 | keys match (key heavily reduced) |

Sweep over attack fraction (4000 photons, 8 trials, 10% loss):

| Fraction | QBER | Theory | Abort rate | Key bits |
|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.00 | 849 |
| 0.1 | 0.027 | 0.025 | 0.00 | 414 |
| 0.2 | 0.054 | 0.050 | 0.25 | 67 |
| 0.3 | 0.079 | 0.075 | 1.00 | 0 |
| 0.5 | 0.127 | 0.125 | 1.00 | 0 |
| 1.0 | 0.252 | 0.250 | 1.00 | 0 |

Comparing n sifted bits detects a full attack with probability `1 − (3/4)^n` (0.944 at n=10, 0.997 at n=20).

## References
- C. H. Bennett, G. Brassard, "Quantum cryptography: Public key distribution and coin tossing", Proc. IEEE ICCSSP, 1984.
- G. Brassard, L. Salvail, "Secret-key reconciliation by public discussion", EUROCRYPT 1993.
- P. Shor, J. Preskill, "Simple proof of security of the BB84 quantum key distribution protocol", Phys. Rev. Lett. 85, 2000.
- C. H. Bennett, G. Brassard, C. Crépeau, U. Maurer, "Generalized privacy amplification", IEEE Trans. Inf. Theory 41, 1995.
- M. Tomamichel, C. C. W. Lim, N. Gisin, R. Renner, "Tight finite-key analysis for quantum cryptography", Nat. Commun. 3, 2012.

---

**Author:** Dr. Kainat Rizwan
