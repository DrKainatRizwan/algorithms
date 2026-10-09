import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paillier_playground import *  # noqa: F401,F403
from paillier_playground.numtheory import lcm, random_coprime


@pytest.fixture(scope="module")
def keys():
    return generate_keypair(256, seed=1)


@pytest.fixture
def rng():
    return random.Random(7)


def test_egcd_and_modinv():
    g, x, y = egcd(240, 46)
    assert g == 2 and 240 * x + 46 * y == 2
    assert modinv(3, 11) == 4
    with pytest.raises(ValueError):
        modinv(6, 9)


def test_crt():
    x = crt(2, 3, 3, 5)
    assert x % 3 == 2 and x % 5 == 3 and 0 <= x < 15


def test_primality():
    primes = [2, 3, 97, 7919, 2**61 - 1]
    comps = [1, 4, 561, 1105, 7917, (2**31 - 1) * (2**61 - 1)]
    assert all(is_probable_prime(p) for p in primes)
    assert not any(is_probable_prime(c) for c in comps)


def test_random_prime_bits(rng):
    p = random_prime(64, rng)
    assert p.bit_length() == 64 and is_probable_prime(p)


def test_keygen_deterministic_and_sized():
    a, _ = generate_keypair(128, seed=5)
    b, _ = generate_keypair(128, seed=5)
    assert a == b and a.n.bit_length() == 128


def test_roundtrip_both_decrypts(keys, rng):
    pub, priv = keys
    for m in [0, 1, 12345, pub.n - 1]:
        c = pub.raw_encrypt(m, rng)
        assert priv.decrypt(c) == m
        assert priv.decrypt_textbook(c) == m


def test_probabilistic_encryption(keys, rng):
    pub, _ = keys
    assert pub.raw_encrypt(5, rng) != pub.raw_encrypt(5, rng)


def test_additive_homomorphism(keys, rng):
    pub, priv = keys
    c = pub.add(pub.raw_encrypt(1000, rng), pub.raw_encrypt(234, rng))
    assert priv.decrypt(c) == 1234


def test_scalar_mul_and_negative(keys, rng):
    pub, priv = keys
    c = pub.raw_encrypt(21, rng)
    assert priv.decrypt(pub.mul_plain(c, 2)) == 42
    assert priv.decrypt_signed(pub.mul_plain(c, -3)) == -63


def test_rerandomize(keys, rng):
    pub, priv = keys
    c = pub.raw_encrypt(99, rng)
    c2 = pub.rerandomize(c, rng)
    assert c != c2 and priv.decrypt(c2) == 99


def test_float_arithmetic(keys, rng):
    pub, priv = keys
    a = EncryptedNumber.encrypt(pub, 3.25, rng)
    b = EncryptedNumber.encrypt(pub, -1.5, rng)
    assert (a + b).decrypt(priv) == pytest.approx(1.75)
    assert (a - b).decrypt(priv) == pytest.approx(4.75)
    assert (a * 2.5).decrypt(priv) == pytest.approx(8.125)
    assert (a * -2).decrypt(priv) == pytest.approx(-6.5)
    assert (a + 10).decrypt(priv) == pytest.approx(13.25)
    assert ((a * 0.5) + b).decrypt(priv) == pytest.approx(0.125)


def test_ciphertext_product_rejected(keys, rng):
    pub, _ = keys
    a = EncryptedNumber.encrypt(pub, 1, rng)
    with pytest.raises(TypeError):
        a * a


def test_overflow(keys, rng):
    pub, _ = keys
    with pytest.raises(OverflowError):
        EncryptedNumber.encrypt(pub, float(pub.n), rng)


def test_private_sum_and_mean(keys, rng):
    _, priv = keys
    vals = [1.5, 2.25, -0.75, 10.0]
    assert private_sum(priv, vals, rng) == pytest.approx(sum(vals))
    vecs = [[1.0, 2.0], [3.0, -4.0], [5.0, 6.0]]
    assert private_mean_vectors(priv, vecs, rng) == pytest.approx([3.0, 4 / 3])


def test_weighted_sum_and_dot(keys, rng):
    pub, priv = keys
    assert weighted_sum(priv, [1, 2, 3], [0.5, 0.25, 2.0], rng) == pytest.approx(7.0)
    x = [0.5, -1.0, 2.0]
    w = [2.0, 0.5, -0.25]
    d = encrypted_dot(encrypt_vector(pub, x, rng), w)
    assert d.decrypt(priv) == pytest.approx(0.0)


def test_vote(keys, rng):
    _, priv = keys
    ballots = [0, 2, 2, 1, 2, 0]
    assert private_vote(priv, ballots, 3, rng) == [2, 1, 3]
    with pytest.raises(ValueError):
        private_vote(priv, [5], 3, rng)


def test_blinded_sum_hides_total(keys, rng):
    pub, priv = keys
    vals = [10, -4, 7]
    total, seen = blinded_sum_protocol(priv, vals, rng)
    assert total == 13 and seen[0] != 13
