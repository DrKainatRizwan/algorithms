# SHA-2, SHA-1 and MD5 from Scratch with Length-Extension Attacks, Collision Search and Rainbow Tables

## Overview

`hash_lab` implements MD5, SHA-1, SHA-256 and SHA-512 on top of a single generic Merkle-Damgård core, then uses that core to demonstrate the structural weaknesses of the construction: length-extension forgery against the secret-prefix MAC, birthday and rho collision search on truncated digests, and rainbow-table inversion. HMAC, HKDF and PBKDF2 are included as the standard remedies. Everything uses only the standard library (`hashlib` appears only in tests and the demo as a reference oracle).

## How it works

- **Merkle-Damgård core (`mdcore.py`)**: the message is padded (`0x80`, zeros, bit length) and fed block by block through a compression function; the digest is the final chaining state. The class accepts an explicit `state` and byte `count`, so hashing can resume from any digest.
- **Hashes (`hashes.py`)**: SHA-2 round constants are derived by exact integer square and cube roots of primes instead of being tabulated.
- **Length extension (`attacks.py`)**: from `H(secret‖m)` the attacker unpacks the state, computes the glue padding for a guessed secret length, and continues hashing an extension, producing a valid tag for `m‖glue‖ext`. `recover_secret_length` tries lengths against a verification oracle. HMAC's nested keyed hashing blocks this.
- **Collisions**: `birthday_collision` stores digests in a table; `rho_collision` iterates `x → f(x)` and uses Floyd cycle detection followed by locating the cycle entry, returning the two distinct predecessors with O(1) memory.
- **Rainbow tables (`rainbow.py`)**: chains alternate hash and a column-dependent reduction; only start and end points are stored. Lookup tries each possible column position and regenerates the chain on an endpoint match (Oechslin).

## Complexity

| Operation | Time | Space |
|---|---|---|
| Hash n bytes | O(n) | O(1) |
| Length extension | O(len(ext)) | O(1) |
| Secret length search | O(L) oracle queries | O(1) |
| Birthday collision, b-bit | O(2^(b/2)) | O(2^(b/2)) |
| Rho collision, b-bit | O(2^(b/2)) | O(1) |
| Rainbow build (m chains, length t) | O(m·t) | O(m) |
| Rainbow lookup | O(t²) | O(1) extra |
| PBKDF2, c iterations | O(c) HMACs | O(1) |

## Usage

```python
from hash_lab import SHA256, prefix_mac, length_extend

secret, msg = b"unknown-key", b"user=alice&amount=10"
tag = prefix_mac(SHA256, secret, msg)
forged_msg, forged_tag = length_extend(SHA256, tag, len(secret), msg, b"&amount=999999")
assert prefix_mac(SHA256, secret, forged_msg) == forged_tag
```

Run the tests and demo:

```
python -m pytest -q
python examples/demo.py
```

## Results

Output of `examples/demo.py` (pure Python, single core):

| Hash | µs per 380 bytes |
|---|---|
| MD5 | 379 |
| SHA-1 | 747 |
| SHA-256 | 1871 |
| SHA-512 | 1295 |

All four match `hashlib` on lengths straddling every padding boundary. The length-extension attack recovered the true secret length (15) and forged a valid tag; the same attack on HMAC failed.

Collision search on truncated SHA-256 (evaluations until the first collision):

| Bits | Birthday | Rho | √(π/2·2^b) |
|---|---|---|---|
| 16 | 311 | 1359 | 321 |
| 20 | 1972 | 1802 | 1283 |
| 22 | 7113 | 13458 | 2567 |

Rainbow table over 20,736 four-letter passwords (a–l) with SHA-1: 503 distinct chains (length 25) built in 1.8 s, recovering 22 of 40 sampled passwords in 1.0 s. Chain merges limit coverage, which is the known motivation for many shorter tables.

## References

- R. Merkle, "A Certified Digital Signature", CRYPTO 1989; I. Damgård, "A Design Principle for Hash Functions", CRYPTO 1989.
- NIST FIPS 180-4, *Secure Hash Standard*.
- R. Rivest, RFC 1321, *The MD5 Message-Digest Algorithm*.
- M. Bellare, R. Canetti, H. Krawczyk, "Keying Hash Functions for Message Authentication", CRYPTO 1996; RFC 2104 (HMAC).
- H. Krawczyk, P. Eronen, RFC 5869 (HKDF); RFC 8018 (PBKDF2).
- P. Oechslin, "Making a Faster Cryptanalytic Time-Memory Trade-Off", CRYPTO 2003.
- P. van Oorschot, M. Wiener, "Parallel Collision Search with Cryptanalytic Applications", J. Cryptology 1999.

---

**Author:** Dr. Kainat Rizwan
