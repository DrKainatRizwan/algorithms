"""Demo: FFT accuracy, polynomial arithmetic, big-integer products and benchmarks."""
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.set_int_max_str_digits(0)

from fft_toolkit import (BigNum, Polynomial, convolve_exact, convolve_fft, fft, fibonacci,
                         factorial, karatsuba_int, wildcard_match)
from fft_toolkit.bigint import fft_rounding_error, ntt_limbs, fft_limbs, schoolbook_limbs
from fft_toolkit.fft import dft_naive, fft_iterative, fft_recursive


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t) * 1000


def main() -> None:
    rng = np.random.default_rng(0)
    print("== FFT accuracy vs naive DFT (max abs error) ==")
    for n in (256, 1000):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        print(f"N={n:5d}  fft={np.max(np.abs(fft(x) - dft_naive(x))):.2e}")

    print("\n== FFT timing, N=4096 (ms) ==")
    x = rng.normal(size=4096) + 0j
    for name, fn in (("naive DFT", dft_naive), ("recursive", fft_recursive), ("iterative", fft_iterative)):
        _, ms = timed(fn, x)
        print(f"{name:10s} {ms:8.2f}")

    print("\n== Polynomial arithmetic ==")
    p = Polynomial.from_roots(list(range(1, 201)))
    print(f"degree {p.degree}, p(1)=0: {p(1) == 0}, leading coeff {p.coeffs[-1]}, p(0)=200!: {p(0) == math.factorial(200)}")
    q = Polynomial([1, 1]) ** 100
    print("(1+x)^100 middle coefficient == C(100,50):", q.coeffs[50] == math.comb(100, 50))

    print("\n== Wildcard matching ==")
    print("text 'abracadabra', pattern 'a?a' ->", wildcard_match("abracadabra", "a?a"))

    print("\n== Big-integer multiplication of two n-digit numbers (ms) ==")
    print(f"{'digits':>8} {'school':>9} {'karatsuba':>10} {'fft':>8} {'ntt':>8} {'max frac err':>13}")
    r = random.Random(1)
    for digits in (1000, 4000, 16000, 64000):
        a = r.randrange(10 ** (digits - 1), 10 ** digits)
        b = r.randrange(10 ** (digits - 1), 10 ** digits)
        la, lb = BigNum.from_int(a).limbs, BigNum.from_int(b).limbs
        school = timed(schoolbook_limbs, la, lb)[1] if digits <= 16000 else float("nan")
        kara = timed(karatsuba_int, a, b)[1] if digits <= 16000 else float("nan")
        res_f, tf = timed(fft_limbs, la, lb)
        res_n, tn = timed(ntt_limbs, la, lb)
        assert BigNum(res_f).to_int() == a * b == BigNum(res_n).to_int()
        err = fft_rounding_error(la, lb)
        print(f"{digits:8d} {school:9.1f} {kara:10.1f} {tf:8.1f} {tn:8.1f} {err:13.2e}")

    print("\n== Big constants ==")
    f, ms = timed(fibonacci, 200000)
    s = str(f)
    print(f"F(200000): {len(s)} digits, starts {s[:12]}..., {ms:.0f} ms")
    f2, ms = timed(factorial, 5000)
    print(f"5000!: {len(str(f2))} digits, correct: {f2.to_int() == math.factorial(5000)}, {ms:.0f} ms")


if __name__ == "__main__":
    main()
