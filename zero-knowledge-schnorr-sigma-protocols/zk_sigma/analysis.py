"""Empirical checks of soundness, zero knowledge and Fiat-Shamir pitfalls."""
from __future__ import annotations

import random
from collections import Counter

from .sigma import (LinearRelation, Proof, Prover, simulate, verify, verify_transcript,
                    _fs_challenge)
from .transcript import Transcript


def cheating_prover_success_rate(rel: LinearRelation, trials: int, rng: random.Random) -> float:
    """A prover without a witness guesses the challenge in advance.

    It commits using the simulator for the guessed challenge; it passes only
    when the verifier's random challenge equals the guess (probability 1/q).
    """
    q = rel.group.q
    wins = 0
    for _ in range(trials):
        guess = rng.randrange(q)
        fake = simulate(rel, guess, rng)
        c = rng.randrange(q)
        if c == guess and verify_transcript(rel, fake.commitment, c, fake.response):
            wins += 1
    return wins / trials


def transcript_distributions(rel: LinearRelation, witness: list[int], challenge: int):
    """Exact distributions of (commitment, response) for real vs simulated runs.

    Enumerates every nonce vector, so only suitable for toy groups. Returns
    two Counters of equal total mass.
    """
    g = rel.group
    n = rel.n_witness
    real: Counter = Counter()
    total = g.q ** n  # nonces uniform in 0..q-1
    for flat in range(total):
        v, ks = flat, []
        for _ in range(n):
            ks.append(v % g.q)
            v //= g.q
        com = tuple(rel.image(ks))
        resp = tuple((k + challenge * x) % g.q for k, x in zip(ks, witness))
        real[(com, resp)] += 1
    sim: Counter = Counter()
    for flat in range(total):
        v, ss = flat, []
        for _ in range(n):
            ss.append(v % g.q)
            v //= g.q
        lhs = rel.image(ss)
        com = tuple(l * pow(g.inv(y), challenge % g.q, g.p) % g.p for l, y in zip(lhs, rel.targets))
        sim[(com, tuple(ss))] += 1
    return real, sim


def total_variation(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def _weak_challenge(rel: LinearRelation) -> int:
    """Deliberately broken Fiat-Shamir: the commitment is *not* hashed."""
    t = Transcript("zk_sigma.weak_fs")
    rel.absorb(t)
    return t.challenge(rel.group.q)


def weak_prove(rel: LinearRelation, witness: list[int], rng: random.Random) -> Proof:
    pv = Prover(rel, witness, rng)
    com = pv.commit()
    c = _weak_challenge(rel)
    return Proof(com, c, pv.respond(c))


def weak_verify(rel: LinearRelation, proof: Proof) -> bool:
    return proof.challenge == _weak_challenge(rel) and verify_transcript(
        rel, proof.commitment, proof.challenge, proof.response)


def forge_weak_proof(rel: LinearRelation, rng: random.Random) -> Proof:
    """Forge without any witness: the challenge is known before committing."""
    return simulate(rel, _weak_challenge(rel), rng)


def forge_against_strong(rel: LinearRelation, rng: random.Random, attempts: int = 200) -> int:
    """Try the same trick against the real transcript; count accepted forgeries."""
    ok = 0
    for _ in range(attempts):
        guess = rng.randrange(rel.group.q)
        fake = simulate(rel, guess, rng)
        if verify(rel, Proof(fake.commitment, fake.challenge, fake.response)):
            ok += 1
    return ok
