# Number-Theoretic Cryptanalysis Toolkit: Factoring, Discrete Logarithms and RSA Attacks

## Overview

A from-scratch library (standard library only) of the classical algorithms used to attack
integer-factorisation and discrete-logarithm problems, together with textbook-RSA
weaknesses that follow from them. It is meant for studying the cost of these attacks at
toy sizes, not for breaking real keys.

| Module | Contents |
|---|---|
| `primes.py` | Sieve, Miller-Rabin, extended gcd, CRT, Jacobi symbol, Tonelli-Shanks |
| `factoring.py` | Trial division, Fermat, Pollard rho (Brent), Pollard p-1, full `factorize` |
| `quadratic_sieve.py` | Quadratic sieve with GF(2) elimination on bitmasks |
| `discrete_log.py` | Baby-step giant-step, Pollard rho for DLP, Pohlig-Hellman |
| `index_calculus.py` | Index calculus mod p with linear algebra over Z/q^e |
| `rsa_attacks.py` | Wiener, Fermat on close primes, shared primes, common modulus, Hastad broadcast |

## How it works

* **Pollard rho (Brent).** Iterates x -> x^2 + c mod n; a cycle modulo a hidden prime p
  appears after about sqrt(p) steps, detected by a gcd. Brent's cycle finding batches gcds.
* **Pollard p-1.** If p-1 divides B!, then a^(B!) = 1 mod p, so gcd(a^(B!) - 1, n) exposes p.
* **Quadratic sieve.** Sieve x^2 - n for values smooth over a factor base of primes with
  (n/p) = 1, record exponent parities as bit vectors, and find subsets summing to zero
  mod 2. Each gives X^2 = Y^2 mod n; gcd(X - Y, n) is a factor with probability about 1/2.
* **BSGS / rho for DLP.** Meet-in-the-middle with a table of g^j, or a pseudo-random walk
  tracking (a, b) with x = g^a h^b and solving at a collision.
* **Pohlig-Hellman.** For smooth group order, solve in each prime-power subgroup digit by
  digit and combine with CRT; cost is governed by the largest prime factor.
* **Index calculus.** Collect relations g^k = prod q_i^e_i, solve for log q_i modulo each
  prime power of p-1, then find s with h*g^s smooth to read off log h.
* **RSA.** Wiener uses continued-fraction convergents of e/n to find a small d; Fermat
  finds p, q when they are close; gcd scans reveal shared primes; common-modulus uses
  Bezout coefficients; Hastad combines e ciphertexts with CRT and takes an integer root.

## Complexity

| Algorithm | Time | Space |
|---|---|---|
| Miller-Rabin | O(k log^3 n) | O(1) |
| Pollard rho | O(n^(1/4)) expected | O(1) |
| Pollard p-1 | O(B log B log^2 n) | O(1) |
| Fermat | O(\|p-q\|^2 / sqrt(n)) | O(1) |
| Quadratic sieve | L_n[1/2, 1] | O(factor base + relations) |
| BSGS | O(sqrt(r)) | O(sqrt(r)) |
| Rho for DLP | O(sqrt(r)) | O(1) |
| Pohlig-Hellman | O(sum e_i (log n + sqrt(q_i))) | O(sqrt(q_max)) |
| Index calculus | L_p[1/2, c] | O(factor base^2) |
| Wiener | O(log n) | O(1) |

## Usage

```python
import random
from number_theory_attacks import factorize, quadratic_sieve, pohlig_hellman, generate_rsa, wiener_attack

print(factorize(600851475143))                  # {71: 1, 839: 1, 1471: 1, 6857: 1}
print(quadratic_sieve(10007 * 10009))
key = generate_rsa(256, random.Random(1), small_d=True)
d, p, q = wiener_attack(key.e, key.n)
```

Run `python examples/demo.py` for the benchmarks and `python -m pytest -q` for the tests
(47 tests, all deterministic).

## Results

Output of `examples/demo.py` (milliseconds, single core):

```
 bits  rho-Brent  quad.sieve  factor
   32        0.1         5.5  47417
   47        1.8        14.7  9641399
   64       33.4        86.7  2438202817
   79      580.6      1017.7  654953915797

 bits     BSGS      rho  index-calc   (safe primes)
   20      0.1      0.8         4.0
   24      0.6      2.5         7.8
   28      1.9      5.1        15.1
```

Pollard rho wins at these sizes because the quadratic sieve here is pure Python with a
single polynomial; its sub-exponential growth only pays off at larger moduli. Pohlig-Hellman
recovers an exponent in a 48-bit group with a 37-smooth order in under a millisecond, and
the demo also recovers a 125-bit Wiener key, a close-prime 512-bit modulus, a shared prime
and a Hastad broadcast message.

## References

* J. M. Pollard, "A Monte Carlo method for factorization", BIT 15, 1975.
* R. P. Brent, "An improved Monte Carlo factorization algorithm", BIT 20, 1980.
* C. Pomerance, "The quadratic sieve factoring algorithm", EUROCRYPT 1984.
* S. Pohlig and M. Hellman, "An improved algorithm for computing logarithms over GF(p)", 1978.
* M. Wiener, "Cryptanalysis of short RSA secret exponents", IEEE Trans. IT 36, 1990.
* D. Boneh, "Twenty years of attacks on the RSA cryptosystem", Notices of the AMS, 1999.
* A. Menezes, P. van Oorschot, S. Vanstone, *Handbook of Applied Cryptography*, 1996.

---

**Author:** Dr. Kainat Rizwan
