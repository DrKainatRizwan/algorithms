import random
import sys

import pytest

from fft_toolkit import BigNum, factorial, fibonacci, karatsuba_int
from fft_toolkit.bigint import fft_limbs, fft_rounding_error, schoolbook_limbs

sys.set_int_max_str_digits(0)


def rnd(digits, seed):
    r = random.Random(seed)
    return r.randrange(10 ** (digits - 1), 10 ** digits)


def test_string_roundtrip():
    s = "123456789012345678901234567890"
    assert str(BigNum.from_str(s)) == s
    assert str(BigNum.from_str("0000")) == "0"
    with pytest.raises(ValueError):
        BigNum.from_str("12a")


@pytest.mark.parametrize("method", ["school", "fft", "ntt", "auto"])
def test_multiply_methods_agree_with_python(method):
    a, b = rnd(500, 1), rnd(430, 2)
    got = BigNum.from_int(a).multiply(BigNum.from_int(b), method)
    assert got.to_int() == a * b


def test_large_fft_multiplication():
    a, b = rnd(20000, 3), rnd(18000, 4)
    got = BigNum.from_str(str(a)) * BigNum.from_str(str(b))
    assert str(got) == str(a * b)


def test_edge_cases_zero_and_carry():
    nines = BigNum.from_str("9" * 300)
    assert (nines * nines).to_int() == (10 ** 300 - 1) ** 2
    assert str(nines * BigNum([0])) == "0"
    assert (nines.multiply(nines, "ntt")).to_int() == (10 ** 300 - 1) ** 2


def test_add_sub_compare():
    a, b = rnd(100, 5), rnd(90, 6)
    A, B = BigNum.from_int(a), BigNum.from_int(b)
    assert (A + B).to_int() == a + b
    assert (A - B).to_int() == a - b
    assert B < A
    with pytest.raises(ValueError):
        B - A


def test_pow_and_divmod_small():
    assert (BigNum.from_int(7) ** 200).to_int() == 7 ** 200
    q, r = BigNum.from_int(10 ** 50 + 123).divmod_small(97)
    assert (q.to_int(), r) == divmod(10 ** 50 + 123, 97)


def test_factorial_and_fibonacci():
    import math
    assert factorial(500).to_int() == math.factorial(500)
    assert factorial(1).to_int() == 1
    a, b = 0, 1
    for _ in range(3000):
        a, b = b, a + b
    assert fibonacci(3000).to_int() == a


def test_karatsuba_matches_builtin():
    a, b = rnd(1500, 7), rnd(1200, 8)
    assert karatsuba_int(a, b) == a * b
    assert karatsuba_int(0, b) == 0


def test_rounding_error_small_and_limb_methods_equal():
    r = random.Random(9)
    a = [r.randrange(10 ** 4) for _ in range(2000)]
    b = [r.randrange(10 ** 4) for _ in range(2000)]
    assert fft_rounding_error(a, b) < 0.1
    assert fft_limbs(a[:60], b[:60]) == schoolbook_limbs(a[:60], b[:60])
