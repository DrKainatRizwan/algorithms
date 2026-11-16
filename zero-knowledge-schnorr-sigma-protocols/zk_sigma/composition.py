"""OR-composition (Cramer-Damgard-Schoenmakers) of sigma protocols."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .sigma import LinearRelation, simulate, verify_transcript, Prover
from .transcript import Transcript


@dataclass(frozen=True)
class OrProof:
    commitments: tuple[tuple[int, ...], ...]
    challenges: tuple[int, ...]
    responses: tuple[tuple[int, ...], ...]


def _or_challenge(rels: list[LinearRelation], commitments, context: bytes) -> int:
    t = Transcript("zk_sigma.or")
    t.absorb_int("n", len(rels))
    for i, r in enumerate(rels):
        t.absorb_int(f"branch{i}.p", r.group.p)
        for j, row in enumerate(r.bases):
            t.absorb_ints(f"branch{i}.bases{j}", row)
        t.absorb_ints(f"branch{i}.targets", r.targets)
    t.absorb_int("ctx", int.from_bytes(context, "big") if context else 0)
    for i, c in enumerate(commitments):
        t.absorb_ints(f"commit{i}", c)
    return t.challenge(rels[0].group.q)


def prove_or(
    rels: list[LinearRelation],
    index: int,
    witness: list[int],
    rng: random.Random,
    context: bytes = b"",
) -> OrProof:
    """Prove that at least the statement ``rels[index]`` holds, hiding which."""
    q = rels[0].group.q
    n = len(rels)
    coms: list[tuple[int, ...] | None] = [None] * n
    chals = [0] * n
    resps: list[tuple[int, ...] | None] = [None] * n
    for i in range(n):
        if i == index:
            continue
        chals[i] = rng.randrange(q)
        sim = simulate(rels[i], chals[i], rng)
        coms[i], resps[i] = sim.commitment, sim.response
    pv = Prover(rels[index], witness, rng)
    coms[index] = pv.commit()
    total = _or_challenge(rels, coms, context)
    chals[index] = (total - sum(chals)) % q
    resps[index] = pv.respond(chals[index])
    return OrProof(tuple(coms), tuple(chals), tuple(resps))  # type: ignore[arg-type]


def verify_or(rels: list[LinearRelation], proof: OrProof, context: bytes = b"") -> bool:
    n = len(rels)
    if not (len(proof.commitments) == len(proof.challenges) == len(proof.responses) == n):
        return False
    q = rels[0].group.q
    total = _or_challenge(rels, list(proof.commitments), context)
    if sum(proof.challenges) % q != total:
        return False
    return all(
        verify_transcript(r, c, ch, s)
        for r, c, ch, s in zip(rels, proof.commitments, proof.challenges, proof.responses)
    )
