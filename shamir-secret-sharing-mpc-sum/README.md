# Shamir Secret Sharing and Secure Multi-Party Computation Sum Protocol

## Overview

A from-scratch implementation (standard library only) of Shamir's threshold secret sharing and the protocols built on top of it: Feldman verifiable sharing, Berlekamp-Welch error-correcting reconstruction, proactive share refresh, dropout-tolerant secure sums and a small BGW engine for private addition and multiplication.

Modules in `secret_sharing/`:

- `field.py` – Miller-Rabin, modular inverse, Horner evaluation, Lagrange interpolation, polynomial division, safe-prime search.
- `shamir.py` – `(t, n)` sharing, reconstruction, consistency checks, homomorphic share arithmetic, refresh, byte-string sharing.
- `vss.py` – Feldman commitments in a Schnorr group, share verification, cheater identification.
- `robust.py` – Berlekamp-Welch decoding of corrupted shares.
- `mpc.py` – additive and Shamir secure sums, BGW engine with degree reduction, private mean/variance and equality test.

## How it works

**Sharing.** To share `s` with threshold `t`, pick a random polynomial `f` of degree `t-1` over GF(p) with `f(0)=s`, and give party `i` the point `(i, f(i))`. Any `t` points determine `f` by Lagrange interpolation, `s = Σ w_i f(x_i)`. Any `t-1` points are consistent with every candidate secret, giving information-theoretic secrecy.

**Linearity.** Adding shares point-wise shares the sum of the secrets, so a secure sum needs no interaction beyond distributing shares and opening one aggregate. Because only `t` evaluators are needed to open, up to `n-t` parties can drop out after the dealing phase.

**Verifiability.** The dealer publishes `C_j = g^{a_j}` for every coefficient. A share `(x, y)` is valid iff `g^y = Π C_j^{x^j}`.

**Error correction.** With `n ≥ t + 2e` shares, Berlekamp-Welch solves a linear system for `Q = P·E` where `E` locates the `e` corrupted points, recovering the secret even when shares are wrong.

**Multiplication (BGW).** Local products of two degree `t-1` sharings lie on a degree `2t-2` polynomial. Each party reshares its product; the Lagrange recombination of those resharings is a fresh degree `t-1` sharing of the product. This needs `n ≥ 2t-1` points (the engine enforces `n ≥ 2t+1` for a corruption threshold `t`).

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Split (Horner) | O(n·t) | O(n) |
| Reconstruct at 0 | O(t²) | O(t) |
| Feldman verify | O(t) exponentiations | O(t) |
| Berlekamp-Welch | O(n³) Gaussian elimination | O(n²) |
| Secure sum, m inputs | O(m·n·t) | O(m·n) |
| BGW multiplication | O(n²·t) | O(n²) messages |

## Usage

```python
import random
from secret_sharing import make_shares, reconstruct, shamir_secure_sum

shares = make_shares(123456, threshold=3, n=5, rng=random.Random(0))
assert reconstruct(shares[1:4]) == 123456

res = shamir_secure_sum([10, 20, 30, 40, 50], threshold=3, dropped_after=[0, 4])
print(res.total)  # 150
```

## Results

Output of `python examples/demo.py` (127-bit Mersenne prime field):

- Secure sum of 7 salaries (true sum 427000): additive protocol gives 427000; Shamir with threshold 4 and 3 evaluators dropped still gives 427000 using 54 messages.
- Excluding a party that never dealt gives 379000, as expected.
- Berlekamp-Welch with `(t=3, n=11)` recovers the secret with 4 corrupted shares and pinpoints them.
- Feldman VSS with `(4, 7)` accepts all 7 honest shares and flags the forged share.
- BGW mean/variance of 7 private values: 420 messages in 7 rounds; `1234 × 5678 = 7006652` computed on shares.

Timing (single thread, mean of 20 splits):

| n | threshold | split | reconstruct |
|---|-----------|-------|-------------|
| 10 | 4 | 0.02 ms | 0.03 ms |
| 50 | 11 | 0.12 ms | 0.07 ms |
| 200 | 51 | 1.63 ms | 1.56 ms |

Run the test suite with `python -m pytest -q` (22 tests).

## References

- A. Shamir, "How to Share a Secret", Communications of the ACM 22(11), 1979.
- P. Feldman, "A Practical Scheme for Non-interactive Verifiable Secret Sharing", FOCS 1987.
- M. Ben-Or, S. Goldwasser, A. Wigderson, "Completeness Theorems for Non-Cryptographic Fault-Tolerant Distributed Computation", STOC 1988.
- L. Welch, E. Berlekamp, "Error correction for algebraic block codes", US Patent 4,633,470, 1986.
- A. Herzberg et al., "Proactive Secret Sharing", CRYPTO 1995.

---

**Author:** Dr. Kainat Rizwan
