import random

import pytest

from number_theory_attacks import (common_modulus_attack, generate_rsa, hastad_broadcast,
                                   integer_root, shared_prime_attack, wiener_attack,
                                   fermat_factor, is_prime, random_prime)


def test_keygen_roundtrip():
    k = generate_rsa(128, random.Random(1))
    m = 123456789
    assert pow(pow(m, k.e, k.n), k.d, k.n) == m


def test_wiener_recovers_small_d():
    k = generate_rsa(256, random.Random(2), small_d=True)
    d, p, q = wiener_attack(k.e, k.n)
    assert d == k.d and {p, q} == {k.p, k.q}


def test_wiener_fails_on_standard_key():
    k = generate_rsa(256, random.Random(3))
    assert wiener_attack(k.e, k.n) is None


def test_fermat_on_close_prime_key():
    k = generate_rsa(256, random.Random(4), close_primes=True)
    p, q = fermat_factor(k.n)
    assert {p, q} == {k.p, k.q}


def test_shared_prime_scan():
    rng = random.Random(5)
    p, q, r = (random_prime(64, rng) for _ in range(3))
    s = random_prime(64, rng)
    out = shared_prime_attack([p * q, p * r, s * random_prime(64, rng)])
    assert set(out) == {p * q, p * r}
    assert out[p * q][0] == p


def test_common_modulus():
    k = generate_rsa(256, random.Random(6), e=3)
    n, m = k.n, 0xDEADBEEFCAFE
    e1, e2 = 3, 65537
    assert common_modulus_attack(n, e1, e2, pow(m, e1, n), pow(m, e2, n)) == m


def test_integer_root():
    for x in (0, 1, 2, 26, 27, 28, 10 ** 60, 12345 ** 7 + 5):
        for k in (2, 3, 7):
            r = integer_root(x, k)
            assert r ** k <= x < (r + 1) ** k


def test_hastad_broadcast():
    rng = random.Random(7)
    ns = []
    while len(ns) < 3:
        p, q = random_prime(96, rng), random_prime(96, rng)
        if p != q and (p - 1) % 3 and (q - 1) % 3:
            ns.append(p * q)
    m = 2 ** 100 + 12345
    cs = [pow(m, 3, n) for n in ns]
    assert hastad_broadcast(cs, ns, 3) == m
