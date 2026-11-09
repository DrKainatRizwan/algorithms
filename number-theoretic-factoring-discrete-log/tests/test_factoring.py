import random
from math import prod

import pytest

from number_theory_attacks import (factorize, fermat_factor, is_prime, next_prime,
                                   pollard_p_minus_1, pollard_rho_brent, quadratic_sieve,
                                   random_prime, trial_division)


def test_trial_division():
    f, rest = trial_division(2 ** 5 * 3 * 7 * 10007 * 10009, 100)
    assert f == [2] * 5 + [3, 7] and rest == 10007 * 10009


def test_fermat_close_primes():
    p = next_prime(10 ** 9)
    q = next_prime(p + 100)
    assert fermat_factor(p * q) == (p, q)


def test_fermat_gives_up_on_unbalanced():
    assert fermat_factor(10007 * 1000003, max_steps=50) is None


def test_pollard_rho_semiprime():
    rng = random.Random(3)
    p, q = random_prime(30, rng), random_prime(30, rng)
    d = pollard_rho_brent(p * q, random.Random(5))
    assert d in (p, q)


def test_pollard_rho_even_and_prime_power():
    assert pollard_rho_brent(1000) == 2
    d = pollard_rho_brent(10007 ** 2)
    assert d is None or d == 10007


def test_pollard_p_minus_1_smooth():
    m = 2 * 3 * 5 * 7 * 11 * 13 * 17
    # prime p with p - 1 = m * k, k small => p - 1 is 200-smooth
    p = next(m * k + 1 for k in range(2, 200) if is_prime(m * k + 1) and k % 17)
    q = 1000003  # q - 1 = 2 * 3 * 166667: not smooth
    d = pollard_p_minus_1(p * q, 200)
    assert d == p


def test_pollard_p_minus_1_fails_when_not_smooth():
    assert pollard_p_minus_1(1000003 * 1000033, 100) is None


def test_quadratic_sieve_known_semiprimes():
    for n in (10007 * 10009, 1000003 * 1000033, 1299709 * 15485863):
        f = quadratic_sieve(n)
        assert f is not None and 1 < f < n and n % f == 0


def test_quadratic_sieve_larger():
    rng = random.Random(11)
    p, q = random_prime(24, rng), random_prime(24, rng)
    f = quadratic_sieve(p * q)
    assert f in (p, q)


@pytest.mark.parametrize("n", [1, 2, 97, 360, 2 ** 40, 600851475143, 10 ** 12 + 39,
                               999983 ** 2, 1000003 * 1000033 * 7])
def test_factorize_roundtrip(n):
    f = factorize(n)
    assert prod(p ** e for p, e in f.items()) == n
    assert all(is_prime(p) for p in f)
