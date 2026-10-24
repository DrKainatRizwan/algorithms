import random

import numpy as np
import pytest

from secure_aggregation.crypto import (AuthenticationError, decrypt, dh_agree,
                                       dh_keygen, encrypt, kdf, prg_vector)
from secure_aggregation.field import P, shamir_reconstruct, shamir_share


def test_shamir_roundtrip_any_subset():
    rng = random.Random(1)
    secret = rng.randrange(P)
    sh = shamir_share(secret, list(range(1, 9)), 4, rng)
    for subset in ([1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 8, 2]):
        assert shamir_reconstruct([(i, sh[i]) for i in subset]) == secret


def test_shamir_below_threshold_is_wrong():
    rng = random.Random(2)
    secret = 123456789
    sh = shamir_share(secret, list(range(1, 7)), 4, rng)
    assert shamir_reconstruct([(i, sh[i]) for i in (1, 2, 3)]) != secret


def test_shamir_validation():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        shamir_share(5, [1, 2], 3, rng)
    with pytest.raises(ValueError):
        shamir_share(5, [0, 1], 2, rng)
    with pytest.raises(ValueError):
        shamir_reconstruct([(1, 2), (1, 3)])


def test_dh_agreement_symmetric():
    rng = random.Random(3)
    a, A = dh_keygen(rng)
    b, B = dh_keygen(rng)
    assert dh_agree(a, B) == dh_agree(b, A)
    with pytest.raises(ValueError):
        dh_agree(a, 1)


def test_encrypt_decrypt_and_tamper():
    key = kdf(42, b"enc")
    blob = encrypt(key, b"n1", b"hello world" * 10)
    assert decrypt(key, b"n1", blob) == b"hello world" * 10
    bad = bytes([blob[0] ^ 1]) + blob[1:]
    with pytest.raises(AuthenticationError):
        decrypt(key, b"n1", bad)
    with pytest.raises(AuthenticationError):
        decrypt(key, b"n2", blob)


def test_prg_deterministic_and_in_range():
    a = prg_vector(b"seed", 100, 32)
    b = prg_vector(b"seed", 100, 32)
    c = prg_vector(b"seee", 100, 32)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.max() < 2**32
