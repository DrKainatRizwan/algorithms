# Elliptic Curve Cryptography from Scratch: ECDH and ECDSA over Finite Fields

## Overview

A from-scratch implementation of elliptic-curve cryptography over prime fields using only the Python standard library. It includes the group law in affine and Jacobian coordinates, three scalar-multiplication strategies, SEC1 point encoding, ECDH key agreement with HKDF, an ECIES-style hybrid scheme, ECDSA with RFC 6979 deterministic nonces and public-key recovery, and attacks on weak parameters (baby-step giant-step, Pollard's rho, nonce reuse). Curves included: secp256k1, NIST P-256, and a textbook toy curve over F_17.

Package layout (`elliptic_curves/`):

| Module | Contents |
|--------|----------|
| `field.py` | extended-Euclid inverse, Miller-Rabin, Tonelli-Shanks square roots |
| `curves.py` | `Curve` class, affine/Jacobian arithmetic, double-and-add, Montgomery ladder, width-4 wNAF, Shamir's trick, SEC1 encoding |
| `ecdh.py` | key generation, public-key validation, ECDH, HKDF-SHA256, ECIES-style encrypt-then-MAC |
| `ecdsa.py` | RFC 6979 nonces, sign, verify, public-key recovery |
| `attacks.py` | BSGS, Pollard's rho, nonce-reuse key recovery, search for prime-order toy curves |

## How it works

Points on `y^2 = x^3 + ax + b (mod p)` form an abelian group. Affine addition needs a modular inversion per operation, so scalar multiplication works in Jacobian coordinates `(X, Y, Z) ~ (X/Z^2, Y/Z^3)` and inverts once at the end. Three multiplication methods are provided: plain double-and-add, a Montgomery ladder (same operation pattern for every bit), and width-4 wNAF, which reduces additions to about `n/5`. `u1*G + u2*Q` in ECDSA verification uses Shamir's simultaneous trick.

**ECDH.** Each party computes `d*Q_peer`; the shared secret is the x-coordinate, fed through HKDF. Peer keys are validated (on curve, not infinity) to block invalid-curve attacks.

**ECDSA.** Signing: `r = (kG).x mod n`, `s = k^-1 (z + r d) mod n`, where the nonce `k` comes from the HMAC-DRBG construction of RFC 6979 so no randomness is needed. Verification checks `(u1 G + u2 Q).x == r` with `u1 = z/s`, `u2 = r/s`. Recovery solves `Q = r^-1 (sR - zG)` for each candidate `R`.

**Attacks.** If two signatures share a nonce, `k = (z1 - z2)/(s1 - s2)` and `d = (s1 k - z1)/r`. On curves with small group order, BSGS (`O(sqrt n)` time and memory) and Pollard's rho with Floyd cycle detection (`O(sqrt n)` time, `O(1)` memory) recover `d` from `Q = dG`.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Scalar mult (k of b bits) | O(b) field mults per bit, ~b doublings + b/5 additions (wNAF) | O(1) (O(2^(w-2)) table) |
| ECDSA sign / verify | 1 / ~1.3 scalar mults | O(1) |
| Modular inverse | O(log^2 p) | O(1) |
| BSGS | O(sqrt n) | O(sqrt n) |
| Pollard's rho | O(sqrt n) expected | O(1) |

## Usage

```python
import hashlib
from elliptic_curves import P256, generate_keypair, ecdh_shared_secret, sign, verify

alice, bob = generate_keypair(P256), generate_keypair(P256)
assert ecdh_shared_secret(P256, alice.private, bob.public) == \
       ecdh_shared_secret(P256, bob.private, alice.public)

digest = hashlib.sha256(b"hello").digest()
sig = sign(P256, alice.private, digest)
assert verify(P256, alice.public, digest, sig)
```

Run the demo with `python examples/demo.py` and the tests with `python -m pytest -q`.

## Results

Output of `examples/demo.py` (single core, pure Python):

| Scalar multiplication, secp256k1 | ms |
|----------------------------------|----|
| Affine double-and-add | 14.22 |
| Jacobian double-and-add | 1.92 |
| Montgomery ladder | 3.30 |
| Jacobian wNAF (w=4) | 1.64 |

| Curve | Sign | Verify |
|-------|------|--------|
| secp256k1 | 1.8 ms | 2.7 ms |
| P-256 | 1.7 ms | 2.6 ms |

| Group order bits | Order | BSGS | Pollard's rho |
|------------------|-------|------|---------------|
| 11 | 1031 | 0.11 ms | 0.77 ms |
| 14 | 9857 | 0.31 ms | 1.50 ms |
| 17 | 100313 | 1.00 ms | 1.96 ms |

The test suite (21 tests) checks the RFC 6979 P-256 vector, the RFC 5869 HKDF vector, known secp256k1 multiples, group laws, agreement of all multiplication methods with an affine reference, encoding round trips, tamper rejection, key recovery and both discrete-log solvers.

This code is for study: it is not constant-time and must not be used to protect real secrets.

## References

- N. Koblitz, "Elliptic curve cryptosystems", Mathematics of Computation 48, 1987.
- V. Miller, "Use of elliptic curves in cryptography", CRYPTO 1985.
- SEC 1 / SEC 2: Standards for Efficient Cryptography Group, 2009/2010.
- T. Pornin, RFC 6979: Deterministic Usage of DSA and ECDSA, 2013.
- H. Krawczyk, P. Eronen, RFC 5869: HKDF, 2010.
- D. Hankerson, A. Menezes, S. Vanstone, *Guide to Elliptic Curve Cryptography*, Springer, 2004.
- J. Pollard, "Monte Carlo methods for index computation mod p", Math. Comp. 32, 1978.

---

**Author:** Dr. Kainat Rizwan
