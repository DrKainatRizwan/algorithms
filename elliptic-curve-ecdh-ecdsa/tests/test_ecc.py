import hashlib
import random

import pytest

from elliptic_curves import (
    P256, SECP256K1, TOY17, baby_step_giant_step, ecdh_shared_secret, ecies_decrypt,
    ecies_encrypt, find_prime_order_curve, generate_keypair, hkdf_sha256, inv_mod,
    is_probable_prime, pollard_rho, recover_key_from_nonce_reuse, recover_public_keys,
    rfc6979_nonce, sign, sqrt_mod, verify,
)


def h(msg: bytes) -> bytes:
    return hashlib.sha256(msg).digest()


def test_modular_helpers():
    assert inv_mod(3, 17) * 3 % 17 == 1
    assert is_probable_prime(2 ** 127 - 1) and not is_probable_prime(2 ** 127 + 1)
    for p in (17, 13, 10007, SECP256K1.p):
        for a in (2, 5, 123456789):
            r = sqrt_mod(a * a, p)
            assert r is not None and r * r % p == a * a % p


def test_generators_on_curve_with_correct_order():
    for c in (SECP256K1, P256, TOY17):
        assert c.is_on_curve(c.G)
        assert c.mul(c.n) is None
        assert c.mul(c.n - 1) == c.neg(c.G)


def test_secp256k1_known_double():
    assert SECP256K1.mul(2) == (
        0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5,
        0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A,
    )


def test_toy_curve_textbook_multiples():
    assert TOY17.mul(2) == (6, 3)
    assert TOY17.mul(3) == (10, 6)
    assert TOY17.mul(19) is None


@pytest.mark.parametrize("method", ["double_and_add", "ladder", "wnaf"])
def test_scalar_mul_methods_agree_with_affine_reference(method):
    rng = random.Random(1)
    for c in (TOY17, SECP256K1, P256):
        for _ in range(6):
            k = rng.randrange(0, c.n)
            assert c.mul(k, method=method) == c.mul_naive(k, c.G)


def test_group_laws():
    c = SECP256K1
    rng = random.Random(2)
    P, Q, R = (c.mul(rng.randrange(1, c.n)) for _ in range(3))
    assert c.add(c.add(P, Q), R) == c.add(P, c.add(Q, R))
    assert c.add(P, Q) == c.add(Q, P)
    assert c.add(P, c.neg(P)) is None
    assert c.mul(5, P) == c.mul_add(2, P, 3, P)


def test_mul_add_matches_separate_products():
    c = P256
    rng = random.Random(3)
    a, b = rng.randrange(c.n), rng.randrange(c.n)
    Q = c.mul(rng.randrange(1, c.n))
    assert c.mul_add(a, c.G, b, Q) == c.add(c.mul(a), c.mul(b, Q))


def test_wnaf_digits_reconstruct_scalar():
    for k in (1, 7, 255, 123456789, 2 ** 100 + 12345):
        d = SECP256K1.wnaf_digits(k, 4)
        assert sum(x << i for i, x in enumerate(d)) == k
        assert all(x == 0 or x % 2 for x in d)
        nz = [i for i, x in enumerate(d) if x]
        assert all(b - a >= 4 for a, b in zip(nz, nz[1:]))


def test_point_encoding_roundtrip():
    for c in (TOY17, SECP256K1, P256):
        for k in (1, 2, 3, 5):
            P = c.mul(k % c.n)
            for comp in (True, False):
                assert c.decode_point(c.encode_point(P, comp)) == P
    with pytest.raises(ValueError):
        SECP256K1.decode_point(b"\x04" + b"\x01" * 64)


def test_ecdh_agreement_and_invalid_key_rejected():
    rng = random.Random(4)
    for c in (SECP256K1, P256):
        a, b = generate_keypair(c, rng), generate_keypair(c, rng)
        assert ecdh_shared_secret(c, a.private, b.public) == ecdh_shared_secret(c, b.private, a.public)
        with pytest.raises(ValueError):
            ecdh_shared_secret(c, a.private, (b.public[0], (b.public[1] + 1) % c.p))


def test_hkdf_rfc5869_vector():
    okm = hkdf_sha256(bytes([0x0B] * 22), 42, bytes(range(13)), bytes(range(0xF0, 0xFA)))
    assert okm.hex() == ("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
                         "34007208d5b887185865")


def test_ecies_roundtrip_and_tamper_detection():
    rng = random.Random(5)
    c = P256
    kp = generate_keypair(c, rng)
    blob = ecies_encrypt(c, kp.public, b"attack at dawn", rng)
    assert ecies_decrypt(c, kp.private, blob) == b"attack at dawn"
    bad = blob[:-40] + bytes([blob[-40] ^ 1]) + blob[-39:]
    with pytest.raises(ValueError):
        ecies_decrypt(c, kp.private, bad)


def test_rfc6979_p256_vector():
    d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
    digest = h(b"sample")
    assert rfc6979_nonce(P256, d, digest) == \
        0xA6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60
    r, s = sign(P256, d, digest)
    assert r == 0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716
    assert s == 0xF7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8


def test_sign_verify_and_rejections():
    rng = random.Random(6)
    for c in (SECP256K1, P256):
        kp = generate_keypair(c, rng)
        sig = sign(c, kp.private, h(b"msg"))
        assert verify(c, kp.public, h(b"msg"), sig)
        assert not verify(c, kp.public, h(b"other"), sig)
        assert not verify(c, kp.public, h(b"msg"), (sig[0], (sig[1] + 1) % c.n))
        assert not verify(c, kp.public, h(b"msg"), (0, sig[1]))
        other = generate_keypair(c, rng)
        assert not verify(c, other.public, h(b"msg"), sig)


def test_low_s_normalisation():
    rng = random.Random(7)
    kp = generate_keypair(SECP256K1, rng)
    for i in range(5):
        r, s = sign(SECP256K1, kp.private, h(bytes([i])), low_s=True)
        assert s <= SECP256K1.n // 2
        assert verify(SECP256K1, kp.public, h(bytes([i])), (r, s))


def test_public_key_recovery():
    rng = random.Random(8)
    for c in (SECP256K1, P256):
        kp = generate_keypair(c, rng)
        sig = sign(c, kp.private, h(b"recover me"))
        assert kp.public in recover_public_keys(c, h(b"recover me"), sig)


def test_nonce_reuse_reveals_private_key():
    rng = random.Random(9)
    c = SECP256K1
    kp = generate_keypair(c, rng)
    k = rng.randrange(1, c.n)
    s1 = sign(c, kp.private, h(b"one"), k=k)
    s2 = sign(c, kp.private, h(b"two"), k=k)
    assert recover_key_from_nonce_reuse(c, h(b"one"), s1, h(b"two"), s2) == kp.private
    with pytest.raises(ValueError):
        recover_key_from_nonce_reuse(c, h(b"one"), s1, h(b"two"), sign(c, kp.private, h(b"two")))


def test_discrete_log_solvers_on_generated_curve():
    c = find_prime_order_curve(2000, seed=1)
    assert is_probable_prime(c.n) and c.mul(c.n) is None
    rng = random.Random(10)
    for _ in range(5):
        k = rng.randrange(1, c.n)
        Q = c.mul(k)
        assert baby_step_giant_step(c, c.G, Q) == k
        assert c.mul(pollard_rho(c, c.G, Q, seed=3)) == Q


def test_toy_curve_full_ecdsa_with_exhaustive_check():
    d = 7
    Q = TOY17.mul(d)
    digest = h(b"toy")
    sig = sign(TOY17, d, digest)
    assert verify(TOY17, Q, digest, sig)
    assert baby_step_giant_step(TOY17, TOY17.G, Q) == d
