import random

import pytest

from number_theory_attacks import (baby_step_giant_step, factorize, index_calculus,
                                   is_prime, next_prime, pohlig_hellman, pollard_rho_dlog)


def primitive_root(p):
    fs = factorize(p - 1)
    g = 2
    while any(pow(g, (p - 1) // q, p) == 1 for q in fs):
        g += 1
    return g


def test_bsgs_small():
    p, g = 1019, primitive_root(1019)
    for x in (0, 1, 5, 500, 1017):
        assert baby_step_giant_step(g, pow(g, x, p), p) == x


def test_bsgs_subgroup():
    p = 2063  # 2063 = 2*1031+1
    assert is_prime(p) and is_prime(1031)
    g = pow(primitive_root(p), 2, p)  # order 1031
    h = pow(g, 777, p)
    assert baby_step_giant_step(g, h, p, 1031) == 777


def test_bsgs_no_solution():
    p = 23
    assert baby_step_giant_step(4, 5, p, 11) is None or pow(4, baby_step_giant_step(4, 5, p, 11), p) == 5


def test_rho_dlog_prime_order_subgroup():
    q = 100003 if is_prime(100003) else next_prime(100003)
    p = 2 * q + 1
    while not is_prime(p):
        q = next_prime(q)
        p = 2 * q + 1
    g = pow(primitive_root(p), 2, p)
    rng = random.Random(4)
    for _ in range(3):
        x = rng.randrange(1, q)
        got = pollard_rho_dlog(g, pow(g, x, p), p, q, random.Random(9))
        assert got is not None and pow(g, got, p) == pow(g, x, p)


def test_pohlig_hellman_smooth_order():
    # p-1 = 2^4 * 3^3 * 5^2 * 7 * 11 * 13 smooth
    k = 2 ** 4 * 3 ** 3 * 5 ** 2 * 7 * 11 * 13
    p = next(c for c in range(k + 1, 40 * k, k) if is_prime(c))
    g = primitive_root(p)
    for x in (1, 12345, p - 2):
        assert pohlig_hellman(g, pow(g, x, p), p) == x


def test_pohlig_hellman_large_smooth_prime():
    k = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31
    p = next(c for c in range(k + 1, 400 * k, 2 * k) if is_prime(c))
    g = primitive_root(p)
    x = 987654321 % (p - 1)
    assert pohlig_hellman(g, pow(g, x, p), p) == x


def test_pohlig_hellman_rejects_non_member():
    p = 1019
    g = pow(primitive_root(p), 2, p)  # order (p-1)/2
    nonres = next(a for a in range(2, p) if pow(a, (p - 1) // 2, p) != 1)
    assert pohlig_hellman(g, nonres, p, (p - 1) // 2) is None


@pytest.mark.parametrize("seed", [1, 2])
def test_index_calculus(seed):
    rng = random.Random(seed)
    p = next_prime(rng.getrandbits(22) | 1 << 21)
    g = primitive_root(p)
    x = rng.randrange(1, p - 1)
    h = pow(g, x, p)
    got = index_calculus(g, h, p, random.Random(seed))
    assert got is not None and pow(g, got, p) == h
