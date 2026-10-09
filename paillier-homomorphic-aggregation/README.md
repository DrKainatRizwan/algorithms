# Homomorphic Encryption Playground: Paillier Cryptosystem and Private Aggregation

## Overview
A from-scratch implementation of the Paillier additively homomorphic cryptosystem in pure Python (standard library only), with number-theory primitives, fixed-point encoding of signed real numbers, and several privacy-preserving aggregation protocols: secure sums, federated gradient averaging, encrypted dot products with plaintext weights, private voting and an additive-masking protocol.

## How it works
- **Keys.** Pick primes `p, q`; `n = pq`, `λ = lcm(p-1, q-1)`, generator `g = n + 1`.
- **Encryption.** `c = (1 + m·n) · r^n mod n²` with random `r ∈ Z_n*`. Encryption is probabilistic.
- **Decryption.** `m = L(c^λ mod n²) · λ⁻¹ mod n` where `L(x) = (x-1)/n`. A CRT variant decrypts modulo `p²` and `q²` separately and recombines.
- **Homomorphism.** `Enc(a)·Enc(b) = Enc(a+b)` and `Enc(a)^k = Enc(k·a)`; negative scalars use a modular inverse.
- **Real numbers.** Values are encoded as signed fixed-point integers (scale 10⁶). `EncryptedNumber` tracks an exponent so that multiplying by a float scalar and then adding stays consistent.
- **Protocols.** The server only ever adds ciphertexts; the key holder decrypts the aggregate. `blinded_sum_protocol` additionally adds a random mask so the decryptor sees nothing about the true total.

Paillier supports ciphertext+ciphertext and ciphertext×plaintext only; ciphertext×ciphertext is rejected.

## Complexity
For an `k`-bit modulus, encryption and decryption are dominated by modular exponentiation: O(k³) bit operations with schoolbook multiplication. Homomorphic addition is one multiplication mod n² (O(k²)). Ciphertexts occupy 2k bits. Aggregating `N` clients with `d` coordinates costs O(N·d) encryptions and additions plus `d` decryptions.

## Usage
```python
import random
from paillier_playground import generate_keypair, private_mean_vectors

rng = random.Random(0)
pub, priv = generate_keypair(512, seed=1)
print(private_mean_vectors(priv, [[1.0, 2.0], [3.0, 4.0]], rng))  # [2.0, 3.0]
```

Run `python examples/demo.py` for the demo and `python -m pytest -q` for the tests (17 tests).

## Results
Measured with `examples/demo.py` (single core, pure Python):

| Modulus | Keygen | Decrypt (textbook) | Decrypt (CRT) | Speedup |
|--------:|-------:|-------------------:|--------------:|--------:|
| 256 bit | 2.6 ms | 0.37 ms | 0.32 ms | 1.2x |
| 512 bit | 8.2 ms | 1.92 ms | 1.49 ms | 1.3x |
| 1024 bit | 139.5 ms | 12.60 ms | 7.56 ms | 1.7x |

Secure mean of 10 client gradient vectors (dimension 5, 512-bit key) matches the plaintext mean to 2.3e-07 (fixed-point rounding) in 0.12 s. The private vote tally over 30 ballots and 4 options matched the ground truth exactly.

## References
- P. Paillier, "Public-Key Cryptosystems Based on Composite Degree Residuosity Classes", EUROCRYPT 1999.
- I. Damgård, M. Jurik, "A Generalisation, a Simplification and Some Applications of Paillier's Probabilistic Public-Key System", PKC 2001.
- J. Katz, Y. Lindell, *Introduction to Modern Cryptography*, 3rd ed., CRC Press, 2020.
- K. Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning", CCS 2017.

---

**Author:** Dr. Kainat Rizwan
