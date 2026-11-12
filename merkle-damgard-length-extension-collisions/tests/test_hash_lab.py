import hashlib
import hmac as std_hmac
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hash_lab import (ALGORITHMS, MD5, SHA1, SHA256, SHA512, RainbowTable,
                      birthday_collision, constant_time_equal, hkdf, hmac,
                      length_extend, md_padding, pbkdf2, prefix_mac,
                      recover_secret_length, rho_collision, truncated_hash)
from hash_lab.mdcore import first_primes, iroot

LENGTHS = [0, 1, 3, 55, 56, 63, 64, 65, 111, 112, 119, 127, 128, 129, 300, 1000]


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_matches_hashlib_on_many_lengths(name):
    cls = ALGORITHMS[name]
    for n in LENGTHS:
        data = bytes((i * 7 + 3) % 256 for i in range(n))
        assert cls.hash(data) == hashlib.new(name, data).digest(), (name, n)


def test_known_vectors():
    assert SHA256.hash(b"abc").hex() == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert MD5.hash(b"").hex() == "d41d8cd98f00b204e9800998ecf8427e"
    assert SHA1.hash(b"abc").hex() == "a9993e364706816aba3e25717850c26c9cd0d89d"


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_incremental_update_and_copy(name):
    cls = ALGORITHMS[name]
    data = bytes(range(256)) * 3
    h = cls()
    for i in range(0, len(data), 37):
        h.update(data[i:i + 37])
    fork = h.copy()
    fork.update(b"more")
    assert h.digest() == hashlib.new(name, data).digest()
    assert fork.digest() == hashlib.new(name, data + b"more").digest()


def test_derived_constants_and_helpers():
    assert first_primes(5) == [2, 3, 5, 7, 11]
    assert iroot(10**30, 3) == 10**10
    assert iroot(26, 3) == 2
    pad = md_padding(3, 64, 8, "big")
    assert (3 + len(pad)) % 64 == 0 and pad[0] == 0x80


def test_state_requires_aligned_count():
    with pytest.raises(ValueError):
        SHA256(state=SHA256.initial_state(), count=5)
    with pytest.raises(ValueError):
        SHA256(count=64)


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_length_extension_forges_valid_tag(name):
    cls = ALGORITHMS[name]
    secret = b"k3y-unknown-to-attacker"
    msg = b"user=alice&amount=10"
    tag = prefix_mac(cls, secret, msg)
    ext = b"&amount=1000000&admin=true"
    forged_msg, forged_tag = length_extend(cls, tag, len(secret), msg, ext)
    assert forged_msg.startswith(msg) and forged_msg.endswith(ext)
    assert forged_tag == hashlib.new(name, secret + forged_msg).digest()


def test_secret_length_recovery_via_oracle():
    secret = b"s" * 19
    msg = b"cmd=list"
    tag = prefix_mac(SHA256, secret, msg)
    oracle = lambda m, t: prefix_mac(SHA256, secret, m) == t
    res = recover_secret_length(SHA256, tag, msg, b"&cmd=delete", oracle)
    assert res is not None and res[0] == 19
    assert recover_secret_length(SHA256, tag, msg, b"x", oracle, max_len=10) is None


def test_length_extension_fails_against_hmac():
    key, msg = b"k" * 16, b"hello"
    tag = hmac(SHA256, key, msg)
    forged_msg, forged_tag = length_extend(SHA256, tag, len(key), msg, b"!")
    assert hmac(SHA256, key, forged_msg) != forged_tag
    with pytest.raises(ValueError):
        length_extend(SHA256, tag[:16], 16, msg, b"!")


@pytest.mark.parametrize("name", ["md5", "sha1", "sha256", "sha512"])
@pytest.mark.parametrize("keylen", [0, 10, 64, 200])
def test_hmac_matches_stdlib(name, keylen):
    key = bytes(range(keylen % 256)) * (keylen // 256 + 1)
    key = key[:keylen]
    assert hmac(ALGORITHMS[name], key, b"message") == std_hmac.new(
        key, b"message", name).digest()


def test_hkdf_rfc5869_case1():
    ikm = bytes([0x0B] * 22)
    salt = bytes(range(13))
    info = bytes(range(0xF0, 0xFA))
    okm = hkdf(SHA256, ikm, 42, salt, info)
    assert okm.hex() == ("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56"
                         "ecc4c5bf34007208d5b887185865")


def test_pbkdf2_matches_stdlib():
    out = pbkdf2(SHA1, b"password", b"salt", 50, 30)
    assert out == hashlib.pbkdf2_hmac("sha1", b"password", b"salt", 50, 30)
    with pytest.raises(ValueError):
        pbkdf2(SHA1, b"p", b"s", 0, 10)


def test_constant_time_equal():
    assert constant_time_equal(b"abc", b"abc")
    assert not constant_time_equal(b"abc", b"abd")
    assert not constant_time_equal(b"abc", b"ab")


def test_truncated_hash_shape():
    f = truncated_hash(SHA256, 12)
    out = f(b"x")
    assert len(out) == 2 and out[1] & 0x0F == 0
    with pytest.raises(ValueError):
        truncated_hash(SHA256, 0)


def test_birthday_collision_is_real_and_near_expectation():
    f = truncated_hash(SHA256, 24)
    m1, m2, n = birthday_collision(f)
    assert m1 != m2 and f(m1) == f(m2)
    assert 50 < n < 20000  # expectation is about 2.5k for 24 bits


def test_rho_collision_constant_memory():
    f = truncated_hash(SHA256, 20)
    m1, m2, evals = rho_collision(f, seed=1)
    assert m1 != m2 and f(m1) == f(m2)
    assert evals < 100000


def test_rainbow_table_recovers_passwords():
    f = lambda pw: SHA1.hash(pw)
    rt = RainbowTable(f, "abcdefghij", 3, chains=250, chain_len=20, seed=5)
    assert len(rt) <= 250
    found = 0
    for idx in range(0, 1000, 17):
        pw = rt.index_to_pw(idx)
        got = rt.lookup(f(pw))
        if got is not None:
            assert f(got) == f(pw)
            found += 1
    assert found / len(range(0, 1000, 17)) > 0.5
    assert rt.lookup(f(b"zzz")) is None or f(rt.lookup(f(b"zzz"))) == f(b"zzz")
