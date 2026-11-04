"""Demo: LWE / Ring-LWE encryption, a KEM, and breaking toy LWE with LLL."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lattice_crypto import *  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(2024)

    print("== Regev LWE encryption ==")
    p = LWEParams(n=64, q=3329, m=160, sigma=1.0)
    kp = lwe_keygen(p, rng)
    bits = rng.integers(0, 2, 256)
    out = np.array(lwe_decrypt_bits(kp.s, p, lwe_encrypt_bits(kp.public, p, bits, rng)))
    print(f"n={p.n} q={p.q} m={p.m}: {int((out == bits).sum())}/256 bits correct, "
          f"predicted failure prob {decryption_failure_probability(p):.2e}")
    for sigma in (40.0, 70.0, 100.0):
        ps = LWEParams(64, 3329, 160, sigma)
        errs = []
        for _ in range(10):  # average over keys: the noise mean depends on the key's error vector
            k2 = lwe_keygen(ps, rng)
            o = np.array(lwe_decrypt_bits(k2.s, ps, lwe_encrypt_bits(k2.public, ps, bits, rng)))
            errs.append((o != bits).mean())
        print(f"  sigma={sigma:>5}: error rate {np.mean(errs):.3f} "
              f"(predicted {decryption_failure_probability(ps):.3f})")

    print("\n== Ring-LWE: NTT vs schoolbook ==")
    ring = NegacyclicRing(256, 7681)
    a, b = rng.integers(0, 7681, 256), rng.integers(0, 7681, 256)
    t = time.perf_counter(); [ring.mul(a, b) for _ in range(20)]; t_ntt = (time.perf_counter() - t) / 20
    t = time.perf_counter(); [ring.mul_schoolbook(a, b) for _ in range(20)]; t_sb = (time.perf_counter() - t) / 20
    print(f"equal: {np.array_equal(ring.mul(a, b), ring.mul_schoolbook(a, b))}; "
          f"NTT {t_ntt*1e3:.2f} ms, schoolbook {t_sb*1e3:.2f} ms")
    rp = RLWEParams()
    pub, s = rlwe_keygen(rp, rng)
    msg = rng.integers(0, 2, rp.n)
    ok = np.array_equal(rlwe_decrypt(s, rp, rlwe_encrypt(pub, rp, msg, rng)), msg)
    print(f"RLWE n={rp.n} q={rp.q}: 256-bit message roundtrip ok={ok}")

    print("\n== KEM (FO transform) ==")
    kpub, ksk = kem_keygen(rp, rng)
    ct, key = kem_encaps(kpub, rp, rng)
    print("keys agree:", kem_decaps(ksk, rp, ct) == key, "| key:", key.hex()[:32], "...")
    u, v = ct
    v2 = v.copy(); v2[3] = (v2[3] + rp.q // 2) % rp.q
    print("tampered ct yields different key:", kem_decaps(ksk, rp, (u, v2)) != key)

    print("\n== Primal lattice attack (LLL + Kannan embedding) ==")
    for n in (6, 10, 14):
        pp = LWEParams(n=n, q=97, m=2 * n + 8, sigma=1.0)
        k = lwe_keygen(pp, rng)
        t = time.perf_counter()
        s_hat = primal_attack(k.A, k.b, pp.q, pp.sigma)
        dt = time.perf_counter() - t
        print(f"  n={n:>2} m={pp.m:>2}: recovered={s_hat is not None and np.array_equal(s_hat, k.s)} in {dt:.2f}s")

    print("\n== Heuristic BKZ block size (core-SVP style) ==")
    for n, q in ((384, 3329), (512, 3329), (768, 3329), (1024, 12289)):
        beta, bits_ = estimate_beta(n, q, 1.0)
        print(f"  n={n:>4} q={q}: beta~{beta}, classical cost ~2^{bits_:.0f}")


if __name__ == "__main__":
    main()
