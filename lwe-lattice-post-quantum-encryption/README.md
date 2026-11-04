# Post-Quantum Lattice Cryptography Demo: Learning With Errors Encryption

## Overview
A from-scratch implementation (Python 3.12, NumPy only) of lattice-based public-key cryptography built on the Learning With Errors (LWE) problem, together with the attacks that show why parameters matter:

- **Regev LWE** bitwise public-key encryption with an analytic decryption-failure model.
- **Ring-LWE** (LPR-style) encryption over `Z_q[x]/(x^n+1)` with a negacyclic NTT.
- **KEM** using a Fujisaki-Okamoto transform with re-encryption and implicit rejection (toy parameters).
- **Cryptanalysis**: LLL reduction, the primal (Kannan embedding) attack, brute force and a heuristic BKZ block-size estimate.

## How it works
- **LWE**: the public key is `(A, b = A s + e mod q)`. To encrypt a bit, sum a random subset `r` of the samples, `c1 = rᵀA`, `c2 = rᵀb + bit·⌊q/2⌋`. Decrypting computes `c2 − ⟨c1, s⟩ = ⟨r, e⟩ + bit·⌊q/2⌋`, which is decoded by distance to 0 or q/2. Correctness needs `|⟨r, e⟩| < q/4`; with Gaussian error the failure probability is about `erfc((q/4)/(√2·σ·√(m/2)))`.
- **Ring-LWE**: `b = a·s + e` in `R_q`, `u = a·r + e1`, `v = b·r + e2 + ⌊q/2⌋·m` encrypts n bits at once. Products use a twisted NTT (multiply by powers of a primitive 2n-th root, then a cyclic radix-2 NTT).
- **KEM**: the message `m` is random, coins and key are derived by hashing `m` with the public key; decapsulation re-encrypts and compares, returning a hash of a secret seed on mismatch.
- **Primal attack**: put `A` in a q-ary lattice basis, append `(b, M)` (Kannan embedding), and run LLL. The unusually short vector `(e, M)` reveals the error, then `s` follows by linear algebra mod q.
- **Estimate**: finds the smallest BKZ block size β with `√(β/d)·σ ≤ δ(β)^(2β−d−1)·q^((m−n)/d)`, costed as `2^(0.292β)`. This is only a rough heuristic.

## Complexity
| Operation | Time | Space |
|---|---|---|
| LWE keygen / encrypt bit / decrypt bit | O(mn) / O(mn) / O(n) | O(mn) |
| Ring multiplication, schoolbook / NTT | O(n²) / O(n log n) | O(n) |
| RLWE encrypt / decrypt | O(n log n) | O(n) |
| LLL (dimension d, entries of size B) | polynomial, ~O(d⁴ log B) swaps × GSO update | O(d²) |
| Brute force | O(qⁿ · m n) | O(m) |

## Usage
```python
import numpy as np
from lattice_crypto import RLWEParams, kem_keygen, kem_encaps, kem_decaps

rng = np.random.default_rng(1)
params = RLWEParams(n=256, q=7681)
pub, sk = kem_keygen(params, rng)
ct, key = kem_encaps(pub, params, rng)
assert kem_decaps(sk, params, ct) == key
```
Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests (19 tests).

## Results
From `examples/demo.py` (seeded):

| Experiment | Result |
|---|---|
| LWE n=64, q=3329, m=160, σ=1 | 256/256 bits decrypted |
| Error rate σ=40 / 70 / 100 (observed vs Gaussian model) | 0.028 / 0.205 / 0.340 vs 0.020 / 0.184 / 0.352 |
| NTT vs schoolbook multiplication (n=256, q=7681) | 0.55 ms vs 1.06 ms |
| KEM key agreement, tampered ciphertext | agrees; tampering gives an unrelated key |
| Primal attack, q=97, σ=1, m=2n+8 | n=6: 0.7 s, n=10: 3.0 s, n=14: 8.8 s, secret recovered each time |
| Estimated β / classical cost (σ=1) | n=384: 97 / 2^28, n=512: 149 / 2^44, n=768: 256 / 2^75, n=1024 (q=12289): 335 / 2^98 |

The attack recovers secrets at toy sizes only; the estimates are rough and the parameters here are not secure or intended for real use.

## References
- O. Regev, "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography", STOC 2005 / J. ACM 2009.
- V. Lyubashevsky, C. Peikert, O. Regev, "On Ideal Lattices and Learning with Errors over Rings", EUROCRYPT 2010.
- A. K. Lenstra, H. W. Lenstra, L. Lovász, "Factoring Polynomials with Rational Coefficients", Math. Ann. 1982.
- E. Alkim, L. Ducas, T. Pöppelmann, P. Schwabe, "Post-quantum key exchange - a new hope", USENIX Security 2016.
- R. Avanzi et al., "CRYSTALS-Kyber", NIST PQC submission.
- E. Fujisaki, T. Okamoto, "Secure Integration of Asymmetric and Symmetric Encryption Schemes", CRYPTO 1999.

---

**Author:** Dr. Kainat Rizwan
