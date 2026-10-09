"""Demo: private federated gradient averaging, voting, and decryption benchmark."""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paillier_playground import (
    generate_keypair, private_mean_vectors, private_vote, encrypted_dot, encrypt_vector,
    blinded_sum_protocol,
)

rng = random.Random(2024)
for bits in (256, 512, 1024):
    t = time.perf_counter()
    pub, priv = generate_keypair(bits, seed=bits)
    kg = time.perf_counter() - t
    c = pub.raw_encrypt(123456789, rng)
    t = time.perf_counter()
    for _ in range(20):
        priv.decrypt_textbook(c)
    t_text = (time.perf_counter() - t) / 20
    t = time.perf_counter()
    for _ in range(20):
        priv.decrypt(c)
    t_crt = (time.perf_counter() - t) / 20
    print(f"{bits:5d}-bit: keygen {kg*1000:7.1f} ms | decrypt textbook {t_text*1000:6.2f} ms | "
          f"CRT {t_crt*1000:6.2f} ms | speedup {t_text/t_crt:.1f}x")

pub, priv = generate_keypair(512, seed=1)
clients = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(10)]
t = time.perf_counter()
agg = private_mean_vectors(priv, clients, rng)
dt = time.perf_counter() - t
plain = [sum(v[j] for v in clients) / 10 for j in range(5)]
print("\nSecure gradient mean:", [round(a, 6) for a in agg])
print("Plain gradient mean :", [round(a, 6) for a in plain])
print(f"Max error {max(abs(a-b) for a, b in zip(agg, plain)):.2e} in {dt:.2f}s")

ballots = [rng.randrange(4) for _ in range(30)]
print("\nVote tally:", private_vote(priv, ballots, 4, rng), "truth:", [ballots.count(i) for i in range(4)])

x, w = [0.2, -1.3, 0.7], [1.5, 0.4, -2.0]
print("Encrypted dot:", round(encrypted_dot(encrypt_vector(pub, x, rng), w).decrypt(priv), 6),
      "plain:", round(sum(a*b for a, b in zip(x, w)), 6))
total, seen = blinded_sum_protocol(priv, [5, 10, -3], rng)
print("Blinded sum:", total, "| key holder saw a masked value of", seen[0].bit_length(), "bits")
