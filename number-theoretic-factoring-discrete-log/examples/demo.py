"""Demo: factoring, discrete-log and RSA-attack benchmarks."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from number_theory_attacks import (baby_step_giant_step, factorize, fermat_factor,
                                   generate_rsa, hastad_broadcast, index_calculus,
                                   next_prime, pohlig_hellman, pollard_rho_brent,
                                   pollard_rho_dlog, quadratic_sieve, random_prime,
                                   shared_prime_attack, wiener_attack, is_prime)


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, (time.perf_counter() - t) * 1000


def primitive_root(p):
    fs = factorize(p - 1)
    g = 2
    while any(pow(g, (p - 1) // q, p) == 1 for q in fs):
        g += 1
    return g


def main() -> None:
    rng = random.Random(2024)
    print("== Factoring semiprimes (time in ms) ==")
    print(f"{'bits':>5} {'rho-Brent':>10} {'quad.sieve':>11}  factor")
    for bits in (32, 48, 64, 80):
        p, q = random_prime(bits // 2, rng), random_prime(bits - bits // 2, rng)
        n = p * q
        f1, t1 = timed(pollard_rho_brent, n, random.Random(1))
        f2, t2 = timed(quadratic_sieve, n)
        assert f1 in (p, q) and f2 in (p, q)
        print(f"{n.bit_length():>5} {t1:>10.1f} {t2:>11.1f}  {min(p, q)}")

    print("\n== Discrete logarithms mod a safe prime ==")
    print(f"{'bits':>5} {'BSGS':>8} {'rho':>8} {'index-calc':>11}")
    for bits in (20, 24, 28):
        while True:
            q = random_prime(bits - 1, rng)
            if is_prime(2 * q + 1):
                break
        p = 2 * q + 1
        g = primitive_root(p)
        x = rng.randrange(2, p - 1)
        h = pow(g, x, p)
        gs = pow(g, 2, p)
        xs = x % q
        hs = pow(gs, xs, p)
        _, tb = timed(baby_step_giant_step, gs, hs, p, q)
        r, tr = timed(pollard_rho_dlog, gs, hs, p, q, random.Random(3))
        assert pow(gs, r, p) == hs
        ic, ti = timed(index_calculus, g, h, p, random.Random(4))
        assert pow(g, ic, p) == h
        print(f"{p.bit_length():>5} {tb:>8.1f} {tr:>8.1f} {ti:>11.1f}")

    print("\n== Pohlig-Hellman on a smooth group order ==")
    k = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    p = next(c for c in range(k + 1, 2000 * k, 2 * k) if is_prime(c))
    g = primitive_root(p)
    x = rng.randrange(p - 1)
    got, t = timed(pohlig_hellman, g, pow(g, x, p), p)
    print(f"p has {p.bit_length()} bits, p-1 = {factorize(p - 1)}")
    print(f"recovered exponent correctly: {got == x} in {t:.1f} ms")

    print("\n== RSA attacks (512-bit modulus) ==")
    key = generate_rsa(512, rng, small_d=True)
    d, _, _ = wiener_attack(key.e, key.n)
    print(f"Wiener: recovered {d.bit_length()}-bit d correctly: {d == key.d}")
    key = generate_rsa(512, rng, close_primes=True)
    p, q = fermat_factor(key.n)
    print(f"Fermat: |p-q| = {abs(key.p - key.q)}, factored: {{p, q}} == key: {({p, q} == {key.p, key.q})}")
    shared = random_prime(256, rng)
    mods = [shared * random_prime(256, rng), shared * random_prime(256, rng),
            random_prime(256, rng) * random_prime(256, rng)]
    print(f"Shared prime scan: {len(shared_prime_attack(mods))} of 3 moduli broken")
    ns = []
    while len(ns) < 3:
        p, q = random_prime(128, rng), random_prime(128, rng)
        if (p - 1) % 3 and (q - 1) % 3:
            ns.append(p * q)
    m = int.from_bytes(b"attack at dawn", "big")
    rec = hastad_broadcast([pow(m, 3, n) for n in ns], ns, 3)
    print(f"Hastad broadcast: {rec.to_bytes(14, 'big').decode()!r}")


if __name__ == "__main__":
    main()
