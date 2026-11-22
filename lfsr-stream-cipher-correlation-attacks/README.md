# LFSR Stream Cipher Cryptanalysis: Correlation, Fast Correlation and Algebraic Attacks

## Overview

From-scratch implementations of the classical attacks on keystream generators built from linear feedback shift registers (LFSRs), using only Python and NumPy:

- LFSR arithmetic, Berlekamp-Massey and linear-complexity analysis
- Geffe, majority (threshold) and non-linear filter generators
- Siegenthaler correlation attack, with all `2^n` candidate states scored at once by a Walsh-Hadamard transform, and the Geffe divide-and-conquer attack
- Meier-Staffelbach style fast correlation attack using sparse parity checks from powers of the feedback polynomial, which works on a 31-bit register where exhaustive search is impractical
- Algebraic (linearisation) attack on filter generators, solved by GF(2) Gaussian elimination

## How the algorithms work

**LFSR and Berlekamp-Massey.** An LFSR of length `n` satisfies `s_{t+n} = XOR_{i in taps} s_{t+i}`. Every output bit is a known linear form of the initial state, so `n` clean bits determine the state by solving a GF(2) system. Berlekamp-Massey finds the shortest LFSR generating any sequence from `2L` bits, which breaks generators of low linear complexity (the Geffe generator has `L <= n1 n2 + n1 n3 + n3`).

**Correlation attack.** The Geffe output equals each of two registers with probability 3/4. For a noisy view `z = s(x) + e` the score of candidate `x` is `sum_t (-1)^(z_t + <mask_t, x>)`. Grouping by `mask_t` turns this into a Walsh-Hadamard transform, giving every score in `O(n 2^n)`. Registers are attacked independently; the selector register is then solved linearly from the positions where the other two disagree.

**Fast correlation attack.** If `f(x)` is sparse then `f(x)^(2^k) = f(x^(2^k))` is equally sparse, so each `k` gives a low-weight parity check `s_t + s_{t+a 2^k} + s_{t+n 2^k} = 0`. A noisy bit lying in mostly violated checks is probably wrong. Flipping such bits iteratively converges to a codeword; the state is then read off the first `n` bits and validated against the keystream.

**Algebraic attack.** For a filter of degree `d`, each keystream bit gives a degree-`d` equation in the initial-state bits. Replacing every monomial of degree up to `d` by a fresh unknown yields a linear system with `sum_{k<=d} C(n,k)` unknowns; the degree-1 unknowns are the key, accepted only if the null space does not touch them.

## Complexity

| Attack | Time | Space |
|---|---|---|
| Berlekamp-Massey | `O(L^2)` from `2L` bits | `O(L)` |
| Correlation scoring (WHT) | `O(N n + n 2^n)` | `O(2^n)` |
| Naive correlation scoring | `O(N 2^n)` | `O(2^n)` |
| Geffe divide and conquer | `O(2^n2 n2 + 2^n3 n3 + n1^3)` instead of `2^(n1+n2+n3)` | `O(2^max n)` |
| Fast correlation (per round) | `O(N log N)` for `log N` checks of weight `w` | `O(N)` |
| Algebraic (degree `d`) | `O(U^3 / 64)` with `U = sum C(n,k)`, word-parallel bit rows | `O(U^2 / 64)` |

## Usage

```python
import numpy as np
from stream_cryptanalysis import LFSR, add_noise, fast_correlation_attack

reg = LFSR.primitive(31)                      # x^31 + x^3 + 1
rng = np.random.default_rng(0)
z = add_noise(reg.stream(0x1234567, 20000), 0.3, rng)
print(hex(fast_correlation_attack(reg, z, 0.3).state))   # 0x1234567
```

Run the tests with `python -m pytest -q` and the demo with `python examples/demo.py`.

## Results (from `examples/demo.py`)

- Geffe generator with registers of 11, 12 and 13 bits: all three states recovered in about 2 ms (joint search would be `2^36`). Berlekamp-Massey measures linear complexity 288, equal to the bound, and needs only 576 bits.
- Walsh-Hadamard scoring of all 4096 states of a 12-bit register (300 bits, noise 0.2) takes 0.3 ms against 168 ms by brute force, about 500x faster, with identical scores.
- Fast correlation attack on `x^31 + x^3 + 1`, successes out of 5 seeds:

| Noise p | N = 5000 | N = 20000 | N = 100000 |
|---|---|---|---|
| 0.20 | 5/5 | 5/5 | 5/5 |
| 0.30 | 5/5 | 5/5 | 5/5 |
| 0.40 | 0/5 | 3/5 | 5/5 |
| 0.45 | 0/5 | 0/5 | 0/5 |

- Algebraic attack on degree-2 filters: 78 unknowns (n=12) and 105 unknowns (n=14) both solved in well under 0.1 s.

## References

- T. Siegenthaler, "Decrypting a class of stream ciphers using ciphertext only", IEEE Trans. Computers, 1985.
- W. Meier and O. Staffelbach, "Fast correlation attacks on certain stream ciphers", Journal of Cryptology, 1989.
- J. L. Massey, "Shift-register synthesis and BCH decoding", IEEE Trans. Information Theory, 1969.
- N. Courtois and W. Meier, "Algebraic attacks on stream ciphers with linear feedback", EUROCRYPT 2003.
- A. Menezes, P. van Oorschot and S. Vanstone, *Handbook of Applied Cryptography*, Chapter 6.

---

**Author:** Dr. Kainat Rizwan
