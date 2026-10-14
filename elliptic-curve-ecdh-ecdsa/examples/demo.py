"""Demo: scalar multiplication benchmark, ECDH, ECDSA, key recovery and attacks on weak curves."""

import hashlib
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from elliptic_curves import (  # noqa: E402
    P256, SECP256K1, baby_step_giant_step, ecdh_shared_secret, ecies_decrypt, ecies_encrypt,
    find_prime_order_curve, generate_keypair, pollard_rho, recover_key_from_nonce_reuse,
    recover_public_keys, sign, verify,
)


def h(m: bytes) -> bytes:
    return hashlib.sha256(m).digest()


def timeit(fn, reps):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000


def main() -> None:
    rng = random.Random(2024)

    print("== Scalar multiplication on secp256k1 (ms per 256-bit multiplication) ==")
    c = SECP256K1
    ks = [rng.randrange(c.n) for _ in range(10)]
    for method in ("double_and_add", "ladder", "wnaf"):
        it = iter(ks * 3)
        print(f"  {method:15s} {timeit(lambda: c.mul(next(it), method=method), 10):7.2f}")
    it = iter(ks * 3)
    print(f"  {'affine naive':15s} {timeit(lambda: c.mul_naive(next(it), c.G), 10):7.2f}")

    print("\n== ECDH on P-256 ==")
    alice, bob = generate_keypair(P256, rng), generate_keypair(P256, rng)
    s1 = ecdh_shared_secret(P256, alice.private, bob.public)
    s2 = ecdh_shared_secret(P256, bob.private, alice.public)
    print("  shared secrets match:", s1 == s2, "|", s1.hex()[:32], "...")
    blob = ecies_encrypt(P256, bob.public, b"meet at the usual place", rng)
    print("  ECIES round trip    :", ecies_decrypt(P256, bob.private, blob))

    print("\n== ECDSA (RFC 6979 deterministic nonces) ==")
    for curve in (SECP256K1, P256):
        kp = generate_keypair(curve, rng)
        msg = h(b"transfer 10 coins")
        sig = sign(curve, kp.private, msg)
        t_s = timeit(lambda: sign(curve, kp.private, msg), 5)
        t_v = timeit(lambda: verify(curve, kp.public, msg, sig), 5)
        ok = verify(curve, kp.public, msg, sig)
        bad = verify(curve, kp.public, h(b"transfer 1000 coins"), sig)
        rec = kp.public in recover_public_keys(curve, msg, sig)
        print(f"  {curve.name:10s} valid={ok} tampered={bad} recovered_key={rec} "
              f"sign={t_s:.1f}ms verify={t_v:.1f}ms")

    print("\n== Nonce reuse attack on secp256k1 ==")
    kp = generate_keypair(SECP256K1, rng)
    k = rng.randrange(1, SECP256K1.n)
    sa = sign(SECP256K1, kp.private, h(b"tx A"), k=k)
    sb = sign(SECP256K1, kp.private, h(b"tx B"), k=k)
    d = recover_key_from_nonce_reuse(SECP256K1, h(b"tx A"), sa, h(b"tx B"), sb)
    print("  private key recovered:", d == kp.private)

    print("\n== Discrete log on small prime-order curves ==")
    print(f"  {'bits':>5} {'order':>9} {'BSGS ms':>9} {'rho ms':>9}")
    for p_min in (10 ** 3, 10 ** 4, 10 ** 5):
        cv = find_prime_order_curve(p_min, seed=1)
        d = rng.randrange(1, cv.n)
        Q = cv.mul(d)
        t = time.perf_counter()
        r1 = baby_step_giant_step(cv, cv.G, Q)
        t1 = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        r2 = pollard_rho(cv, cv.G, Q, seed=1)
        t2 = (time.perf_counter() - t) * 1000
        assert r1 == d and cv.mul(r2) == Q
        print(f"  {cv.n.bit_length():5d} {cv.n:9d} {t1:9.2f} {t2:9.2f}")


if __name__ == "__main__":
    main()
