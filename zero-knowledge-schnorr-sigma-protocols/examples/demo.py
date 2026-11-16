"""Demo: sigma protocols, range proofs, signatures and Fiat-Shamir pitfalls."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zk_sigma import *  # noqa: E402
from zk_sigma import analysis  # noqa: E402


def timed(f, n=1):
    t = time.perf_counter()
    for _ in range(n):
        out = f()
    return out, (time.perf_counter() - t) / n * 1000


def main() -> None:
    rng = random.Random(2026)
    G = standard_group(128)
    print(f"Group: {G.p.bit_length()}-bit safe prime, q has {G.q.bit_length()} bits\n")

    print("== Schnorr proof of knowledge ==")
    x = G.random_scalar(rng)
    rel = schnorr_relation(G, G.exp(G.g, x))
    pf, tp = timed(lambda: prove(rel, [x], rng), 50)
    ok, tv = timed(lambda: verify(rel, pf), 50)
    print(f"valid: {ok}  prove {tp:.3f} ms  verify {tv:.3f} ms")

    print("\n== Special soundness: extract witness from two transcripts ==")
    pv = Prover(rel, [x], rng)
    com = pv.commit(); k = pv._k[0]
    p1 = Proof(com, 17, ((k + 17 * x) % G.q,)); p2 = Proof(com, 99, ((k + 99 * x) % G.q,))
    print("extracted == witness:", extract(rel, p1, p2) == [x])

    print("\n== HVZK on the toy group (exact enumeration) ==")
    T = toy_group()
    trel = schnorr_relation(T, T.exp(T.g, 7))
    real, sim = analysis.transcript_distributions(trel, [7], challenge=5)
    print("total-variation distance real vs simulated:", analysis.total_variation(real, sim))
    rate = analysis.cheating_prover_success_rate(trel, 50000, rng)
    print(f"cheater success {rate:.4f} vs 1/q = {1 / T.q:.4f}")

    print("\n== Chaum-Pedersen / DLEQ ==")
    g2 = G.random_element(rng)
    d = dleq_relation(G, G.g, G.exp(G.g, x), g2, G.exp(g2, x))
    print("same-exponent proof:", verify(d, prove(d, [x], rng)))

    print("\n== Pedersen range proofs ==")
    ped = Pedersen(G)
    for n in (8, 16, 32):
        rp = RangeProver(ped, n)
        m = (1 << n) - 3
        (c, rpf), tp = timed(lambda: rp.prove(m, G.random_scalar(rng), rng))
        ok, tv = timed(lambda: rp.verify(c, rpf))
        print(f"{n:2d}-bit: valid={ok}  prove {tp:6.1f} ms  verify {tv:6.1f} ms  "
              f"proof ~ {rpf.size_in_group_elements()} elements")

    print("\n== Schnorr signatures ==")
    sk, pub = keygen(G, rng)
    sig = sign(G, sk, b"release v1.0", rng)
    print("verify:", verify_signature(G, pub, b"release v1.0", sig),
          "| tampered:", verify_signature(G, pub, b"release v1.1", sig))
    items = []
    for i in range(50):
        s_, p_ = keygen(G, rng); m = f"m{i}".encode()
        items.append((p_, m, sign(G, s_, m, rng)))
    _, t_ind = timed(lambda: all(verify_signature(G, p_, m, s_) for p_, m, s_ in items))
    ok, t_bat = timed(lambda: batch_verify(G, items, rng))
    print(f"50 signatures: individual {t_ind:.1f} ms, batch {t_bat:.1f} ms, batch ok={ok}")
    a = sign(G, sk, b"A", rng, nonce=424242); b = sign(G, sk, b"B", rng, nonce=424242)
    print("nonce reuse recovers key:", recover_key_from_nonce_reuse(G, b"A", a, b"B", b, pub) == sk)

    print("\n== Weak Fiat-Shamir (commitment not hashed) ==")
    victim = schnorr_relation(G, G.random_element(rng))
    print("forged proof accepted by weak verifier:",
          analysis.weak_verify(victim, analysis.forge_weak_proof(victim, rng)))
    print("same forgery strategy vs correct verifier, accepted of 200:",
          analysis.forge_against_strong(victim, rng, 200))


if __name__ == "__main__":
    main()
