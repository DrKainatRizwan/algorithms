"""Demo: DH-PSI, the hash-PSI dictionary attack, and several PIR schemes."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from private_retrieval import (CubePIR, Database, PaillierPIR, SchnorrGroup, XORPIR, bloom_psi,
                               dh_psi, dh_psi_cardinality, dictionary_attack, is_private,
                               naive_hash_psi)
from private_retrieval.audit import cube_cost, paillier_cost, trivial_cost, xor_cost


def main() -> None:
    group = SchnorrGroup.generate(128, seed=7)
    print(f"Group: {group.p.bit_length()}-bit safe prime")
    universe = [f"user{i:04d}" for i in range(2000)]
    rng = random.Random(0)
    server = rng.sample(universe, 200)
    client = rng.sample(universe, 150)
    truth = set(client) & set(server)

    print("\n== PSI ==  |client|=150 |server|=200, true overlap =", len(truth))
    for name, fn in [("naive-hash", lambda: naive_hash_psi(client, server)),
                     ("bloom(1%)", lambda: bloom_psi(client, server, 0.01)),
                     ("dh-psi", lambda: dh_psi(client, server, group, 1)),
                     ("dh-card", lambda: dh_psi_cardinality(client, server, group, 1))]:
        t = time.perf_counter()
        r = fn()
        dt = (time.perf_counter() - t) * 1000
        ok = r.intersection == {x.encode() for x in truth} if r.intersection else None
        print(f"{name:11s} card={r.cardinality:4d} exact={ok} "
              f"up={r.bytes_client_to_server:6d}B down={r.bytes_server_to_client:6d}B {dt:7.1f} ms")
    leaked = dictionary_attack(naive_hash_psi([], server).extra["hashes"], universe)
    print(f"dictionary attack on naive hashes recovers {len(leaked)}/{len(server)} server items")

    print("\n== Information-theoretic PIR (n=4096 records of 16 B) ==")
    n, L = 4096, 16
    db = Database.random(n, L, seed=1)
    i = 1234
    for name, s in [("xor-2", XORPIR(2)), ("xor-4", XORPIR(4)), ("cube-2", CubePIR(2)), ("cube-3", CubePIR(3))]:
        t = time.perf_counter()
        r = s.retrieve(db, i, random.Random(5))
        dt = (time.perf_counter() - t) * 1000
        print(f"{name:7s} correct={r.record == db.record(i)} up={r.upload_bits:7d}b "
              f"down={r.download_bits:5d}b {dt:7.1f} ms")
    print("trivial  up=0 down =", trivial_cost(n, L)[1], "bits")
    print("exact privacy audit (n=4): xor-3 single:", is_private(XORPIR(3), 4, [0]),
          "| all three colluding:", is_private(XORPIR(3), 4, [0, 1, 2]))

    print("\n== Single-server Paillier PIR (n=256, 16 B, 256-bit key) ==")
    db2 = Database.random(256, 16, seed=2)
    for label, rows in [("linear", 1), ("sqrt", 0)]:
        s = PaillierPIR(256, seed=3, rows=rows)
        t = time.perf_counter()
        r = s.retrieve(db2, 77)
        dt = time.perf_counter() - t
        print(f"{label:7s} correct={r.record == db2.record(77)} up={r.upload_bits:7d}b "
              f"down={r.download_bits:6d}b {dt:5.2f} s")
    for n_ in (10 ** 4, 10 ** 6):
        print(f"model n={n_}: linear up/down", paillier_cost(n_, 16, 256, 1),
              "sqrt", paillier_cost(n_, 16, 256), "cube-3 xor-style", cube_cost(n_, 16, 3),
              "xor-2", xor_cost(n_, 16))


if __name__ == "__main__":
    main()
