"""Demo: hashes, length extension, collisions and rainbow tables."""
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hash_lab import (ALGORITHMS, SHA1, SHA256, RainbowTable, birthday_collision,
                      expected_birthday_queries, hmac, length_extend, prefix_mac,
                      recover_secret_length, rho_collision, truncated_hash)


def main() -> None:
    print("== 1. Correctness vs hashlib ==")
    data = b"The quick brown fox" * 20
    for name, cls in ALGORITHMS.items():
        t = time.perf_counter()
        for _ in range(5):
            d = cls.hash(data)
        us = (time.perf_counter() - t) / 5 * 1e6
        ok = d == hashlib.new(name, data).digest()
        print(f"{name:7s} match={ok}  {us:8.1f} us per {len(data)} bytes")

    print("\n== 2. Length extension against H(secret || msg) ==")
    secret = b"hunter2-secret!"
    msg = b"user=alice&amount=10"
    tag = prefix_mac(SHA256, secret, msg)
    oracle = lambda m, t: prefix_mac(SHA256, secret, m) == t
    n, fm, ft = recover_secret_length(SHA256, tag, msg, b"&amount=999999", oracle)
    print(f"recovered secret length: {n} (true {len(secret)})")
    print(f"forged message: {fm!r}")
    print(f"forged tag valid: {oracle(fm, ft)}")
    fm2, ft2 = length_extend(SHA256, hmac(SHA256, secret, msg), len(secret), msg, b"x")
    print(f"same attack on HMAC valid: {hmac(SHA256, secret, fm2) == ft2}")

    print("\n== 3. Collision search on truncated SHA-256 ==")
    print(f"{'bits':>5} {'birthday evals':>15} {'rho evals':>10} {'expected':>9}")
    for bits in (16, 20, 22):
        f = truncated_hash(SHA256, bits)
        _, _, nb = birthday_collision(f)
        _, _, nr = rho_collision(f, seed=1)
        print(f"{bits:5d} {nb:15d} {nr:10d} {expected_birthday_queries(bits):9.0f}")

    print("\n== 4. Rainbow table on 4-char passwords (a-l), SHA-1 ==")
    alpha = "abcdefghijkl"
    t = time.perf_counter()
    rt = RainbowTable(SHA1.hash, alpha, 4, chains=700, chain_len=25, seed=3)
    print(f"built {len(rt)} chains in {time.perf_counter() - t:.2f}s "
          f"for space of {rt.space}")
    hits, total = 0, 40
    t = time.perf_counter()
    for i in range(total):
        pw = rt.index_to_pw((i * 1091) % rt.space)
        if rt.lookup(SHA1.hash(pw)) is not None:
            hits += 1
    print(f"recovered {hits}/{total} in {time.perf_counter() - t:.2f}s")


if __name__ == "__main__":
    main()
