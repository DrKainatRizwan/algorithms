# Side-Channel Power Analysis of AES: DPA, CPA and Boolean Masking

## Overview
A from-scratch AES-128 (FIPS-197) together with a simulator of Hamming-weight power leakage and the classic power-analysis attacks against it: difference-of-means DPA, correlation power analysis (CPA), and second-order CPA with a centred-product combiner against a first-order Boolean-masked implementation. Evaluation metrics (key rank, success rate, guessing entropy, traces-to-success) are included.

## How it works
- **Target (`aes.py`)**: S-box derived from GF(2^8) inversion plus the affine map, key schedule and its inverse, full block encryption.
- **Leakage (`leakage.py`)**: per key byte, samples `HW(p^k)` and `HW(S[p^k])` plus Gaussian noise. The masked variant leaks `HW(m)` and `HW(S[p^k]^m)` with a fresh random mask `m` per trace.
- **DPA**: for each key guess, partition traces by one predicted S-box output bit and take the maximal absolute difference of means. The correct guess produces the largest peak.
- **CPA**: Pearson correlation between `HW(S[p^guess])` and every sample, computed for all 256 guesses with one matrix product.
- **Masking and second-order CPA**: with a uniform mask each sample is independent of the key, so first-order attacks fail. The centred product `(a-mean a)(b-mean b)` of the mask and masked-output samples has expectation `2 - HW(S)/2` given the key-dependent S-box value, so correlating it with `HW(S[p^guess])` recovers the key.

## Complexity
For N traces, S samples and 256 guesses: CPA/DPA cost O(256·N·S) time and O(256·N) memory. Second-order combining adds O(N) per byte. Key schedule inversion is O(1).

## Usage
```python
from power_analysis import simulate_traces, recover_key, cpa_attack
key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
ts = simulate_traces(key, 300, noise_sigma=1.0, seed=1)
assert recover_key(ts, cpa_attack) == key
```
Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests.

## Results (from `examples/demo.py`)
Success rate of byte-0 recovery, 30 experiments each:

| Attack | sigma | Traces for 90% success |
|---|---|---|
| CPA | 1 | 25 |
| CPA | 3 | 200 |
| DPA (8 bits summed) | 1 | 200 |
| DPA (8 bits summed) | 3 | 800 |

Masked implementation (sigma = 0.5, 5000 traces): first-order CPA ranks the true byte 34th; second-order CPA ranks it first and recovers all 16 key bytes. Second-order CPA reaches 90% success at 250 traces over 20 experiments.

## References
- Kocher, Jaffe, Jun. *Differential Power Analysis*, CRYPTO 1999.
- Brier, Clavier, Olivier. *Correlation Power Analysis with a Leakage Model*, CHES 2004.
- Messerges. *Using Second-Order Power Analysis to Attack DPA Resistant Software*, CHES 2000.
- Mangard, Oswald, Popp. *Power Analysis Attacks: Revealing the Secrets of Smart Cards*, Springer 2007.
- NIST FIPS-197, *Advanced Encryption Standard*, 2001.

---

**Author:** Dr. Kainat Rizwan
