import random
from math import prod

import pytest

from number_theory_attacks import (crt, egcd, is_prime, jacobi, modinv, next_prime,
                                   random_prime, sieve, tonelli_shanks)


def test_sieve_counts():
    assert len(sieve(100)) == 25
    assert len(sieve(10_000)) == 1229
    assert sieve(1) == []


def test_miller_rabin_matches_sieve():
    ps = set(sieve(5000))
    assert all(is_prime(n) == (n in ps) for n in range(5000))


def test_miller_rabin_carmichael_and_large():
    for c in (561, 1105, 1729, 2465, 6601, 3215031751):
        assert not is_prime(c)
    assert is_prime(2 ** 61 - 1)
    assert is_prime(2 ** 89 - 1)
    assert not is_prime(2 ** 67 - 1)


def test_next_and_random_prime():
    assert next_prime(13) == 17 and next_prime(1) == 2
    p = random_prime(48, random.Random(1))
    assert p.bit_length() == 48 and is_prime(p)


def test_egcd_modinv_crt():
    g, x, y = egcd(240, 46)
    assert g == 2 and 240 * x + 46 * y == 2
    assert modinv(7, 40) * 7 % 40 == 1
    with pytest.raises(ValueError):
        modinv(6, 9)
    x, M = crt([2, 3, 2], [3, 5, 7])
    assert (x, M) == (23, 105)


def test_jacobi_matches_euler_criterion():
    p = 101
    for a in range(1, p):
        assert jacobi(a, p) == (1 if pow(a, (p - 1) // 2, p) == 1 else -1)


@pytest.mark.parametrize("p", [7, 13, 17, 41, 10007, 998244353])
def test_tonelli_shanks(p):
    rng = random.Random(p)
    for _ in range(20):
        a = rng.randrange(1, p)
        s = a * a % p
        r = tonelli_shanks(s, p)
        assert r * r % p == s
