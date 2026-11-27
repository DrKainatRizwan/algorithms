import random

import numpy as np
import pytest

from private_retrieval import CubePIR, Database, PaillierPIR, TrivialPIR, XORPIR, is_private
from private_retrieval.audit import (cube_cost, paillier_cost, total_variation,
                                     trivial_cost, view_distribution, xor_cost)
from private_retrieval.paillier import keygen


def test_database_roundtrip():
    db = Database.from_records([b"ab", b"c", b"def"])
    assert db.n == 3 and db.record_len == 3
    assert db.record(1) == b"c\x00\x00"
    with pytest.raises(IndexError):
        db.record(3)
    assert db.padded(5).shape == (5, 3)


def test_paillier_homomorphism():
    rng = random.Random(1)
    sk = keygen(128, rng)
    pk = sk.public
    a, b = 12345, 6789
    ca, cb = pk.encrypt(a, rng), pk.encrypt(b, rng)
    assert sk.decrypt(ca) == a
    assert sk.decrypt(pk.add(ca, cb)) == a + b
    assert sk.decrypt(pk.scalar_mul(ca, 17)) == 17 * a
    assert pk.encrypt(5, rng) != pk.encrypt(5, rng)  # probabilistic
    with pytest.raises(ValueError):
        pk.encrypt(pk.n, rng)


def test_trivial_pir():
    db = Database.random(10, 4)
    res = TrivialPIR().retrieve(db, 7)
    assert res.record == db.record(7) and res.download_bits == 8 * 40


@pytest.mark.parametrize("k", [2, 3, 5])
def test_xor_pir_correct(k):
    db = Database.random(37, 9, seed=k)
    s = XORPIR(k)
    rng = random.Random(k)
    for i in (0, 1, 18, 36):
        assert s.retrieve(db, i, rng).record == db.record(i)


def test_xor_pir_requires_two_servers():
    with pytest.raises(ValueError):
        XORPIR(1)


def test_xor_pir_privacy_exact():
    s = XORPIR(3)
    assert is_private(s, 4, [0])
    assert is_private(s, 4, [0, 2])      # any k-1 colluders learn nothing
    assert is_private(s, 4, [1, 2])
    assert not is_private(s, 4, [0, 1, 2])  # all servers together learn the index


def test_full_collusion_reveals_index():
    s = XORPIR(2)
    d0 = view_distribution(s, 3, 0, [0, 1])
    d5 = view_distribution(s, 3, 2, [0, 1])
    assert total_variation(d0, d5) == 1.0


@pytest.mark.parametrize("d,n", [(1, 7), (2, 25), (2, 30), (3, 60)])
def test_cube_pir_correct(d, n):
    db = Database.random(n, 5, seed=n)
    s = CubePIR(d)
    rng = random.Random(d)
    for i in range(n):
        assert s.retrieve(db, i, rng).record == db.record(i)


def test_cube_side_and_servers():
    assert CubePIR(3).side(27) == 3 and CubePIR(3).side(28) == 4
    assert CubePIR(2).servers == 4 and CubePIR(3).servers == 8


def test_cube_pir_privacy_exact():
    s = CubePIR(2)
    assert is_private(s, 4, [0])
    assert is_private(s, 4, [3])
    assert not is_private(s, 4, [0, 1, 2, 3])


def test_cube_upload_beats_xor():
    n, L = 4096, 16
    assert cube_cost(n, L, 2)[0] < xor_cost(n, L, 2)[0] / 10
    assert cube_cost(n, L, 3)[0] < cube_cost(n, L, 2)[0] * 2
    assert trivial_cost(n, L)[1] > xor_cost(n, L)[1]


@pytest.mark.parametrize("rows", [1, 0, 3])
def test_paillier_pir_correct(rows):
    db = Database.random(20, 21, seed=3)  # multi-chunk records for 128-bit key
    s = PaillierPIR(key_bits=128, seed=9, rows=rows)
    for i in (0, 6, 19):
        res = s.retrieve(db, i)
        assert res.record == db.record(i)


def test_paillier_pir_queries_look_random():
    s = PaillierPIR(key_bits=128, seed=2)
    q1, q2 = s.query(16, 3), s.query(16, 3)
    assert q1 != q2 and len(set(q1)) == 4  # sqrt layout: 4 columns


def test_paillier_cost_model_matches_run():
    db = Database.random(16, 8, seed=1)
    s = PaillierPIR(key_bits=128, seed=1)
    res = s.retrieve(db, 5)
    up, down = paillier_cost(16, 8, 128)
    assert res.upload_bits == up and res.download_bits == down
    assert paillier_cost(10000, 8, 128)[0] < paillier_cost(10000, 8, 128, rows=1)[0] / 50
