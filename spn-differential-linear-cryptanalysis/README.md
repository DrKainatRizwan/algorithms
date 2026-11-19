# Differential and Linear Cryptanalysis of a Substitution-Permutation Network

## Overview

A from-scratch toolkit that builds a small 16-bit SPN block cipher and breaks reduced-round versions of it with the two classical statistical attacks: differential cryptanalysis (Biham–Shamir) and linear cryptanalysis (Matsui). It includes S-box analysis (difference distribution table, linear approximation table, Walsh spectrum), a branch-and-bound search for the best characteristic and linear trail, and last-round subkey recovery together with success-rate experiments. Only the standard library and numpy are used.

Package layout (`spn_cryptanalysis/`):

| Module | Contents |
|---|---|
| `sbox.py` | DDT, LAT, Walsh spectrum, differential uniformity, linearity, nonlinearity, three example S-boxes |
| `spn.py` | 16-bit SPN with vectorised encrypt/decrypt over numpy arrays |
| `differential.py` | characteristic search, empirical probability, chosen-plaintext key recovery |
| `linear.py` | piling-up lemma, trail search, Matsui Algorithm 2 style key recovery |
| `analysis.py` | S-box reports, success-rate curves, avalanche test |

## How it works

**Cipher.** `R` rounds of key XOR, four parallel 4-bit S-boxes and a bit permutation; the last round replaces the permutation with a final key XOR (`R+1` independent round keys).

**Differential attack.** The DDT entry `ddt[a][b]` counts inputs `x` with `S(x)^S(x^a)=b`. A characteristic chains transitions through the first `R-1` rounds; under the Markov-cipher assumption its probability is the product of the S-box transition probabilities. The search is depth-first with the bound `p * q_max^(remaining rounds)`. The attacker encrypts plaintext pairs with the input difference, discards pairs whose ciphertext difference is non-zero on S-boxes that are inactive in the last round (the filter), and for each guess of the active last-round key nibbles peels the final layer and counts how often the predicted difference appears. The right key collects about `p` votes per pair, wrong keys about `1/16` per active nibble.

**Linear attack.** `lat[a][b] = #{x : a·x = b·S(x)} - 8` gives a bias `lat/16`. Approximations are chained across rounds with the piling-up lemma, `eps = 2^(m-1) * prod(eps_i)` over `m` active S-boxes. The attacker partially decrypts the last layer under every guess of the active nibbles and measures `|T - N/2|` where `T` counts how often `α·P ⊕ β·x` is zero. Ciphertexts are first aggregated into `16^k` classes so the cost per guess does not depend on `N`. About `c/eps²` known plaintexts suffice.

## Complexity

* DDT / LAT construction: `O(16²)` and `O(16³)` for the 4-bit S-box.
* Trail search: worst case exponential in rounds times active S-boxes, heavily pruned by the bound; the demo searches in well under a second.
* Differential attack: `O(n · 16^k)` time for `n` pairs and `k` active key nibbles, `O(n)` memory (chosen plaintexts needed ≈ `c / p`).
* Linear attack: `O(n + 16^(2k))` time and memory, known plaintexts needed ≈ `c / eps²`.

## Usage

```python
import numpy as np
from spn_cryptanalysis import (SPN, SBOX_HEYS, best_characteristic,
                               differential_attack, best_trail, linear_attack)

rng = np.random.default_rng(2024)
cipher = SPN.random(4, rng)

ch = best_characteristic(SBOX_HEYS, 3)            # p = 27/1024
res = differential_attack(cipher, ch, 400, rng)
print(res.nibble_positions, res.best_guess)

tr = best_trail(SBOX_HEYS, 3)                     # bias ~ 0.059
res = linear_attack(cipher, tr, 20000, rng)
print(res.nibble_positions, res.best_guess)
```

Run `python examples/demo.py` for the full demonstration and `python -m pytest -q` for the tests.

## Results

Output of `examples/demo.py` (seed 2024, about 6 s):

| S-box | Diff. uniformity | Linearity | Nonlinearity | Best 3-round char. | Best 3-round bias |
|---|---|---|---|---|---|
| Heys | 8 | 6 | 2 | 0.02637 (27/1024) | 0.05933 |
| PRESENT | 4 | 4 | 4 | 0.00391 | 0.03125 |
| affine | 16 | 8 | 0 | 1.0 | 0.5 |

Differential attack on 4 rounds (characteristic `00b0 → 0505`; measured probability 0.02647 against 0.02637 predicted); success rate over 40 random keys:

| Chosen pairs | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| Success | 0.03 | 0.38 | 0.72 | 0.93 | 1.00 |

Linear attack on 4 rounds (`α = 0006`, `β = 7007`, piled-up bias 0.0593):

| Known plaintexts | 500 | 1000 | 2000 | 4000 | 8000 | 16000 |
|---|---|---|---|---|---|---|
| Success | 0.28 | 0.65 | 0.88 | 0.93 | 1.00 | 0.97 |

Best characteristic probability for the PRESENT S-box through this permutation: 2^-2, 2^-4, 2^-8, 2^-12 for 1–4 rounds. The bias of a trail varies with the actual key (the piling-up value holds on average), which is why the empirical bias for a single key can differ from the prediction.

## References

* E. Biham and A. Shamir, *Differential Cryptanalysis of DES-like Cryptosystems*, Journal of Cryptology 4(1), 1991.
* M. Matsui, *Linear Cryptanalysis Method for DES Cipher*, EUROCRYPT 1993.
* H. M. Heys, *A Tutorial on Linear and Differential Cryptanalysis*, Cryptologia 26(3), 2002.
* A. Bogdanov et al., *PRESENT: An Ultra-Lightweight Block Cipher*, CHES 2007.
* X. Lai, J. Massey and S. Murphy, *Markov Ciphers and Differential Cryptanalysis*, EUROCRYPT 1991.

---

**Author:** Dr. Kainat Rizwan
