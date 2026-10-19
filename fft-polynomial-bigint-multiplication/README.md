# FFT Library with Polynomial Multiplication and Big-Integer Arithmetic

## Overview

A from-scratch Fast Fourier Transform toolkit in Python 3.12 and NumPy (no `numpy.fft`). It contains:

- `fft.py` – naive DFT, recursive and iterative radix-2 FFT, Bluestein chirp-z transform for arbitrary lengths, 2-D FFT, real-input helpers.
- `ntt.py` – number-theoretic transform over three NTT-friendly primes, with Garner/CRT reconstruction for exact integer convolution and convolution modulo arbitrary moduli.
- `convolution.py` – linear, circular and 2-D convolution, cross-correlation, and wildcard string matching.
- `polynomial.py` – a `Polynomial` class with automatic choice of schoolbook / FFT / NTT multiplication, product trees, Newton series inversion and fast division.
- `bigint.py` – `BigNum` limb arithmetic with schoolbook, FFT and NTT multiplication, Karatsuba baseline, fast-doubling Fibonacci and product-tree factorial.

## How it works

**Iterative FFT.** The input is permuted into bit-reversed order, then `log2 N` stages combine pairs of half-size transforms with the butterfly `(u, v) -> (u + w v, u - w v)`. Each stage is vectorised across all blocks, and the same code works on batched input.

**Bluestein.** Writing `nk = (n² + k² − (k−n)²)/2` turns a length-`N` DFT into a convolution with a chirp `exp(−iπn²/N)`, evaluated with power-of-two FFTs. Chirp angles use `n² mod 2N` to stay accurate for large `N`.

**NTT and exact convolution.** The FFT works unchanged in `Z_p` when `p = c·2^k + 1`. Convolving modulo 998244353, 167772161 and 469762049 and combining with Garner's algorithm gives exact results for coefficients up to about 3.9·10²⁵ in magnitude; larger inputs fall back to exact Python-integer schoolbook multiplication.

**Polynomials.** Products pick a method by operand type and size. Division uses `rev(q) = rev(a)·rev(b)⁻¹ mod x^(deg a − deg b + 1)` with a Newton inverse `g ← g(2 − f g)` that doubles precision per step. `from_roots` builds a balanced product tree.

**Big integers.** Numbers are split into base-10⁴ limbs; multiplication is a convolution of limb vectors followed by carry propagation. With the float FFT the result is rounded, so `fft_rounding_error` reports how close the coefficients were to integers.

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Naive DFT | O(N²) | O(N²) matrix |
| Radix-2 FFT / NTT | O(N log N) | O(N) |
| Bluestein (any N) | O(N log N), ~3 FFTs of size ≥ 2N | O(N) |
| Exact convolution (3 primes) | O(N log N) | O(N) |
| Series inverse / poly division | O(M(n)) | O(n) |
| Product tree from roots | O(M(n) log n) | O(n log n) |
| Big-int multiply (n limbs) | O(n log n) | O(n) |

## Usage

```python
from fft_toolkit import BigNum, Polynomial, fft, convolve_exact, fibonacci, wildcard_match

X = fft([1, 2, 3, 4, 5])                      # any length
p = Polynomial.from_roots([1, 2, 3]) ** 2     # exact integer coefficients
c = convolve_exact([1, 2, 3], [4, 5, 6])      # [4, 13, 28, 27, 18]
a = BigNum.from_str("9" * 5000)
print(len(str(a * a)))                        # 10000
print(str(fibonacci(100000))[:10])            # 2597406934
print(wildcard_match("abracadabra", "a?a"))   # [3, 5]
```

Run `python examples/demo.py` for the demo and `python -m pytest -q` for the tests (44 tests).

## Results

Output of `examples/demo.py` (single run, milliseconds):

| Digits | Schoolbook limbs | Karatsuba (CPython ints) | FFT | NTT | Max fractional error |
|-------:|-----------------:|-------------------------:|----:|----:|---------------------:|
| 1,000 | 4.6 | 0.1 | 0.7 | 3.4 | 1.9e-06 |
| 4,000 | 74.8 | 0.6 | 1.0 | 7.0 | 7.6e-06 |
| 16,000 | 852.9 | 5.2 | 2.9 | 26.5 | 3.8e-05 |
| 64,000 | – | – | 90.7 | 100.6 | 1.8e-04 |

- FFT of length 4096: naive DFT 332 ms, recursive 23.7 ms, iterative 0.47 ms; error against the DFT is about 1e-13.
- F(200000) (41,798 digits) by fast doubling: 43 ms; 5000! (16,326 digits): 153 ms, both verified against Python's built-in integers.
- The Karatsuba baseline bottoms out in CPython's native multiplication, so it is fast at small sizes; the FFT overtakes it near 16,000 digits and the gap widens with size.

## References

- J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series", *Math. Comp.* 19 (1965).
- L. Bluestein, "A linear filtering approach to the computation of the discrete Fourier transform", *IEEE Trans. Audio Electroacoust.* 18 (1970).
- A. Schönhage and V. Strassen, "Schnelle Multiplikation großer Zahlen", *Computing* 7 (1971).
- A. Karatsuba and Yu. Ofman, "Multiplication of multidigit numbers on automata", *Doklady Akad. Nauk SSSR* 145 (1962).
- H. L. Garner, "The residue number system", *IRE Trans. Electronic Computers* (1959).
- T. Cormen, C. Leiserson, R. Rivest, C. Stein, *Introduction to Algorithms*, ch. 30.
- J. von zur Gathen and J. Gerhard, *Modern Computer Algebra*, ch. 8-9.
- P. Clifford and R. Clifford, "Simple deterministic wildcard matching", *Inf. Process. Lett.* 101 (2007).

---

**Author:** Dr. Kainat Rizwan
