# Reed-Solomon and BCH Codes with Berlekamp-Massey Decoding

## Overview

From-scratch implementations of two classic algebraic error-correcting code families, using only the standard library:

- **Reed-Solomon (RS)** codes over GF(2^m) with systematic encoding and a complete errors-and-erasures decoder (syndromes, Forney syndromes, Berlekamp-Massey, Chien search, Forney's magnitude formula).
- **Binary primitive BCH** codes with bit-packed carry-less encoding and BM/Chien decoding.
- A second locator solver (**Peterson-Gorenstein-Zierler**) used to cross-check Berlekamp-Massey.
- Channel models (symbol/bit errors, bursts), a block interleaver and seeded Monte-Carlo experiments.

## How it works

1. **Field arithmetic.** GF(2^m) is built from a primitive polynomial; log/antilog tables make multiplication a table lookup.
2. **Encoding.** The generator is `g(x) = prod (x - alpha^(fcr+i))`. A message `m(x)` becomes `c(x) = m(x) x^nsym + (m(x) x^nsym mod g(x))`. For BCH, `g` is the lcm of the minimal polynomials of `alpha..alpha^(2t)`.
3. **Syndromes.** `S_j = r(alpha^(fcr+j))`; all zero iff `r` is a codeword.
4. **Erasures.** Known positions give `Gamma(x) = prod (1 + alpha^p x)`. The Forney syndromes `S(x)Gamma(x) mod x^nsym` hide the erasures from the error search.
5. **Berlekamp-Massey** finds the shortest LFSR (the error locator `sigma(x)`) generating the syndrome sequence.
6. **Chien search** evaluates `Psi = sigma * Gamma` at every `alpha^-p` to find error positions.
7. **Forney's formula** gives magnitudes `e_p = X^(1-fcr) Omega(X^-1) / Psi'(X^-1)`, with `Omega = S Psi mod x^nsym`.
8. The corrected word is re-checked: a non-zero residual syndrome or a locator that does not split raises `DecodingFailure`, so most over-capacity patterns are detected rather than miscorrected.

An RS code corrects any pattern with `2*errors + erasures <= n - k`. A BCH code with designed distance `2t+1` corrects `t` bit errors.

## Complexity

| Step | Time | Space |
|------|------|-------|
| RS encode | O(n * nsym) | O(n) |
| Syndromes | O(n * nsym) | O(nsym) |
| Berlekamp-Massey | O(nsym^2) | O(nsym) |
| PGZ | O(t^4) naive elimination | O(t^2) |
| Chien search | O(n * t) | O(t) |
| Forney | O(t^2) | O(t) |
| Tables | O(2^m) setup | O(2^m) |

## Usage

```python
import random
from reed_solomon_bch import ReedSolomon, BCH, add_symbol_errors

rs = ReedSolomon(255, 223)            # corrects 16 symbol errors
msg = list(range(223))
noisy, _ = add_symbol_errors(rs.encode(msg), 16, random.Random(0))
assert rs.decode(noisy) == msg

# with erasures (known bad positions)
rs.decode(noisy, erasures=[3, 9])

bch = BCH(m=6, t=3)                   # BCH(63, 45)
cw = bch.encode([1, 0] * 22 + [1])
cw[5] ^= 1; cw[40] ^= 1
assert bch.decode(cw) == bch.extract(bch.encode([1, 0] * 22 + [1]))
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests (23 tests).

## Results

Output of `examples/demo.py` (seeded):

| Experiment | Result |
|------------|--------|
| RS(255,223), 16 errors | all located and corrected |
| RS(63,47), (errors, erasures) = (8,0), (4,8), (0,16), (5,6) | all corrected (2e+s = 16) |
| RS(31,21), 5 errors (t) | 100.0% corrected |
| RS(31,21), 6 / 8 / 12 errors | 99.9% / 99.6% / 99.7% detected, 0.1% / 0.4% / 0.4% miscorrected |
| BCH(63,45), 0-3 bit errors | 100% corrected |
| BCH(63,45), 4+ bit errors | 0% corrected (detected or rejected) |
| BCH(15,7) minimum distance | 5 (equals designed distance) |
| 40-symbol burst, depth-8 interleaver, RS(31,21) | 8/8 codewords recovered |
| 10-symbol burst, no interleaver | decoding failure |
| RS(255,223) decode time, 16 errors | BM 1.2 ms, PGZ 1.5 ms |

## References

- I. S. Reed and G. Solomon, "Polynomial Codes over Certain Finite Fields", *J. SIAM* 8(2), 1960.
- R. C. Bose and D. K. Ray-Chaudhuri, "On a Class of Error Correcting Binary Group Codes", *Information and Control* 3, 1960.
- E. R. Berlekamp, *Algebraic Coding Theory*, McGraw-Hill, 1968.
- J. L. Massey, "Shift-Register Synthesis and BCH Decoding", *IEEE Trans. Inf. Theory* 15(1), 1969.
- G. D. Forney, "On Decoding BCH Codes", *IEEE Trans. Inf. Theory* 11(4), 1965.
- W. W. Peterson, "Encoding and Error-Correction Procedures for the Bose-Chaudhuri Codes", 1960.
- S. Lin and D. Costello, *Error Control Coding*, 2nd ed., Pearson, 2004.

---

**Author:** Dr. Kainat Rizwan
