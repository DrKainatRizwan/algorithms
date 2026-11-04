import numpy as np
import pytest

from lattice_crypto import *
from lattice_crypto.attacks import embedding_basis, mod_inverse_matrix
from lattice_crypto.sampling import center


def rng(seed=0):
    return np.random.default_rng(seed)


def test_cbd_moments():
    x = centered_binomial(rng(1), 2, 200000)
    assert abs(x.mean()) < 0.01
    assert abs(x.var() - 1.0) < 0.02  # eta/2 = 1


def test_discrete_gaussian_std_and_range():
    x = discrete_gaussian(rng(2), 3.0, 100000)
    assert abs(x.std() - 3.0) < 0.05
    assert np.abs(x).max() <= 18


def test_lwe_roundtrip_bits():
    p = LWEParams(n=32, q=3329, m=96, sigma=1.0)
    r = rng(3)
    kp = lwe_keygen(p, r)
    bits = r.integers(0, 2, 64)
    cts = lwe_encrypt_bits(kp.public, p, bits, r)
    assert lwe_decrypt_bits(kp.s, p, cts) == bits.tolist()


def test_lwe_wrong_key_fails_often():
    p = LWEParams()
    r = rng(4)
    kp = lwe_keygen(p, r)
    bits = r.integers(0, 2, 200)
    cts = lwe_encrypt_bits(kp.public, p, bits, r)
    wrong = (kp.s + 1 + r.integers(0, p.q, p.n)) % p.q
    acc = np.mean(np.array(lwe_decrypt_bits(wrong, p, cts)) == bits)
    assert 0.35 < acc < 0.65


def test_failure_probability_monotone_in_sigma():
    lo = decryption_failure_probability(LWEParams(sigma=1.0))
    hi = decryption_failure_probability(LWEParams(sigma=80.0))
    assert lo < 1e-6 < hi


def test_high_noise_actually_fails():
    p = LWEParams(n=16, q=3329, m=64, sigma=120.0)
    r = rng(5)
    kp = lwe_keygen(p, r)
    bits = r.integers(0, 2, 300)
    out = np.array(lwe_decrypt_bits(kp.s, p, lwe_encrypt_bits(kp.public, p, bits, r)))
    assert (out != bits).mean() > 0.1


def test_ntt_roundtrip_and_matches_schoolbook():
    ring = NegacyclicRing(256, 7681)
    r = rng(6)
    a, b = r.integers(0, 7681, 256), r.integers(0, 7681, 256)
    assert np.array_equal(ring.intt(ring.ntt(a)), a)
    assert np.array_equal(ring.mul(a, b), ring.mul_schoolbook(a, b))


def test_negacyclic_wraparound():
    ring = NegacyclicRing(8, 17)
    x = np.zeros(8, dtype=np.int64); x[1] = 1
    x7 = np.zeros(8, dtype=np.int64); x7[7] = 1
    prod = ring.mul(x, x7)  # x^8 = -1
    assert prod[0] == 16 and prod[1:].sum() == 0


def test_ring_without_ntt_falls_back():
    ring = NegacyclicRing(8, 11)
    assert not ring.has_ntt
    a = np.arange(8); b = np.arange(8)[::-1]
    assert np.array_equal(ring.mul(a, b), ring.mul_schoolbook(a, b))


def test_ring_distributivity():
    ring = NegacyclicRing(64, 7681)
    r = rng(7)
    a, b, c = (r.integers(0, 7681, 64) for _ in range(3))
    assert np.array_equal(ring.mul(a, ring.add(b, c)), ring.add(ring.mul(a, b), ring.mul(a, c)))


def test_rlwe_roundtrip():
    p = RLWEParams()
    r = rng(8)
    pub, s = rlwe_keygen(p, r)
    for _ in range(5):
        bits = r.integers(0, 2, p.n)
        assert np.array_equal(rlwe_decrypt(s, p, rlwe_encrypt(pub, p, bits, r)), bits)


def test_kem_agreement_and_determinism():
    p = RLWEParams()
    pub, sk = kem_keygen(p, rng(9))
    ct, k1 = kem_encaps(pub, p, rng(10))
    assert kem_decaps(sk, p, ct) == k1
    ct2, k2 = kem_encaps(pub, p, rng(10))
    assert k1 == k2 and np.array_equal(ct[1], ct2[1])


def test_kem_implicit_rejection_on_tampering():
    p = RLWEParams()
    pub, sk = kem_keygen(p, rng(11))
    (u, v), k = kem_encaps(pub, p, rng(12))
    v2 = v.copy(); v2[0] = (v2[0] + p.q // 2) % p.q  # flip one message bit
    k_bad = kem_decaps(sk, p, (u, v2))
    assert k_bad != k and len(k_bad) == 32
    assert kem_decaps(sk, p, (u, v2)) == k_bad  # deterministic rejection


def test_lll_reduces_and_preserves_lattice():
    B = np.array([[1, 1, 1], [-1, 0, 2], [3, 5, 6]])
    R = lll_reduce(B)
    assert abs(round(np.linalg.det(B))) == abs(round(np.linalg.det(R)))
    assert np.linalg.norm(R[0]) <= np.linalg.norm(B[0]) + 1e-9
    assert sorted(map(tuple, np.abs(R[0:1]))) == [(0, 1, 0)]


def test_lll_finds_short_vector_in_qary_lattice():
    q = 101
    B = np.array([[q, 0], [37, 1]])
    R = lll_reduce(B)
    assert np.linalg.norm(R[0]) < 12


def test_mod_inverse():
    r = rng(13)
    while True:
        M = r.integers(0, 97, (5, 5))
        inv = mod_inverse_matrix(M, 97)
        if inv is not None:
            break
    assert np.array_equal((M @ inv) % 97, np.eye(5, dtype=np.int64))


def test_primal_attack_recovers_secret():
    p = LWEParams(n=8, q=97, m=24, sigma=1.0)
    kp = lwe_keygen(p, rng(14))
    s = primal_attack(kp.A, kp.b, p.q, p.sigma)
    assert s is not None and np.array_equal(s, kp.s)


def test_brute_force_matches_secret():
    p = LWEParams(n=3, q=13, m=12, sigma=0.7)
    kp = lwe_keygen(p, rng(15))
    s = brute_force_lwe(kp.A, kp.b, p.q, bound=int(np.abs(kp.e).max()))
    assert s is not None
    assert np.abs(center((kp.b - kp.A @ s) % p.q, p.q)).max() <= np.abs(kp.e).max()


def test_estimate_beta_grows_with_dimension():
    b1, _ = estimate_beta(256, 3329, 1.0)
    b2, _ = estimate_beta(512, 3329, 1.0)
    assert b2 > b1 >= 40
