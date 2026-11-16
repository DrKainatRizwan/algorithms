"""Generic sigma protocol for linear relations of discrete logarithms.

A statement is a matrix of bases ``G[i][j]`` and targets ``Y[i]`` with
``Y[i] = prod_j G[i][j] ^ x[j]``. Schnorr, Chaum-Pedersen (DLEQ), Okamoto
and Pedersen-opening proofs are all special cases. The protocol is the usual
three-move commit / challenge / respond with special soundness and
honest-verifier zero knowledge.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .groups import SchnorrGroup
from .transcript import Transcript


@dataclass(frozen=True)
class LinearRelation:
    group: SchnorrGroup
    bases: tuple[tuple[int, ...], ...]
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.bases) != len(self.targets):
            raise ValueError("one target per row required")
        n = len(self.bases[0])
        if any(len(r) != n for r in self.bases):
            raise ValueError("ragged base matrix")

    @property
    def n_witness(self) -> int:
        return len(self.bases[0])

    def image(self, xs: list[int]) -> list[int]:
        g = self.group
        return [g.multiexp(list(row), xs) for row in self.bases]

    def holds(self, xs: list[int]) -> bool:
        return self.image(xs) == list(self.targets)

    def absorb(self, t: Transcript) -> None:
        t.absorb_int("p", self.group.p)
        for i, row in enumerate(self.bases):
            t.absorb_ints(f"bases{i}", row)
        t.absorb_ints("targets", self.targets)


@dataclass(frozen=True)
class Proof:
    commitment: tuple[int, ...]
    challenge: int
    response: tuple[int, ...]


# -- constructors -----------------------------------------------------------
def schnorr_relation(group: SchnorrGroup, y: int, base: int | None = None) -> LinearRelation:
    """Knowledge of x with y = base^x."""
    return LinearRelation(group, ((base or group.g,),), (y,))


def dleq_relation(group: SchnorrGroup, g1: int, y1: int, g2: int, y2: int) -> LinearRelation:
    """Chaum-Pedersen: same x with y1 = g1^x and y2 = g2^x."""
    return LinearRelation(group, ((g1,), (g2,)), (y1, y2))


def representation_relation(group: SchnorrGroup, bases: list[int], y: int) -> LinearRelation:
    """Okamoto: knowledge of (x_1..x_n) with y = prod bases_j^x_j."""
    return LinearRelation(group, (tuple(bases),), (y,))


def and_relation(rels: list[LinearRelation], shared: list[list[int]] | None = None) -> LinearRelation:
    """Conjunction of relations over a common group.

    Witness columns are concatenated; ``shared`` optionally lists groups of
    ``(relation_index * width + column)`` offsets that must hold the same
    witness value (handled by merging columns).
    """
    g = rels[0].group
    width = sum(r.n_witness for r in rels)
    rows: list[list[int]] = []
    off = 0
    for r in rels:
        for row in r.bases:
            full = [1] * width
            full[off:off + r.n_witness] = list(row)
            rows.append(full)
        off += r.n_witness
    targets = [y for r in rels for y in r.targets]
    if shared:
        drop: set[int] = set()
        for grp_cols in shared:
            keep, rest = grp_cols[0], grp_cols[1:]
            for c in rest:
                for row in rows:
                    row[keep] = row[keep] * row[c] % g.p
                    row[c] = 1
                drop.add(c)
        cols = [c for c in range(width) if c not in drop]
        rows = [[row[c] for c in cols] for row in rows]
    return LinearRelation(g, tuple(tuple(r) for r in rows), tuple(targets))


# -- interactive protocol -----------------------------------------------------
class Prover:
    """Stateful prover for one run of the three-move protocol."""

    def __init__(self, rel: LinearRelation, witness: list[int], rng: random.Random) -> None:
        if not rel.holds(witness):
            raise ValueError("witness does not satisfy the relation")
        self.rel, self.witness, self.rng = rel, [w % rel.group.q for w in witness], rng
        self._k: list[int] | None = None

    def commit(self) -> tuple[int, ...]:
        g = self.rel.group
        self._k = [g.random_nonce(self.rng) for _ in range(self.rel.n_witness)]
        return tuple(self.rel.image(self._k))

    def respond(self, challenge: int) -> tuple[int, ...]:
        if self._k is None:
            raise RuntimeError("commit first")
        q = self.rel.group.q
        out = tuple((k + challenge * x) % q for k, x in zip(self._k, self.witness))
        self._k = None  # a nonce must never be reused
        return out


def verify_transcript(rel: LinearRelation, commitment, challenge: int, response) -> bool:
    """Check ``prod G_ij^s_j == T_i * Y_i^c`` for every row."""
    g = rel.group
    if len(commitment) != len(rel.targets) or len(response) != rel.n_witness:
        return False
    if not all(g.is_member(t) for t in commitment):
        return False
    lhs = rel.image(list(response))
    return all(
        l == t * pow(y, challenge % g.q, g.p) % g.p
        for l, t, y in zip(lhs, commitment, rel.targets)
    )


def simulate(rel: LinearRelation, challenge: int, rng: random.Random) -> Proof:
    """HVZK simulator: pick the response first and solve for the commitment."""
    g = rel.group
    s = [g.random_nonce(rng) for _ in range(rel.n_witness)]
    lhs = rel.image(s)
    t = tuple(l * pow(g.inv(y), challenge % g.q, g.p) % g.p for l, y in zip(lhs, rel.targets))
    return Proof(t, challenge % g.q, tuple(s))


def extract(rel: LinearRelation, p1: Proof, p2: Proof) -> list[int]:
    """Special-soundness extractor from two accepting transcripts.

    Requires the same commitment and different challenges; for statements
    whose witness is not unique (n_witness > 1) the result is *a* valid
    witness, obtained from the response difference scaled by 1/(c1-c2).
    """
    q = rel.group.q
    if p1.commitment != p2.commitment or (p1.challenge - p2.challenge) % q == 0:
        raise ValueError("need same commitment and distinct challenges")
    inv = pow(p1.challenge - p2.challenge, -1, q)
    return [(a - b) * inv % q for a, b in zip(p1.response, p2.response)]


# -- non-interactive (Fiat-Shamir) ---------------------------------------------
def _fs_challenge(rel: LinearRelation, commitment, context: bytes) -> int:
    t = Transcript("zk_sigma.fiat_shamir")
    rel.absorb(t)
    t.absorb_int("ctx.len", len(context))
    t.absorb_int("ctx", int.from_bytes(context, "big") if context else 0)
    t.absorb_ints("commitment", commitment)
    return t.challenge(rel.group.q)


def prove(rel: LinearRelation, witness: list[int], rng: random.Random, context: bytes = b"") -> Proof:
    """Non-interactive proof bound to the statement and ``context``."""
    pv = Prover(rel, witness, rng)
    com = pv.commit()
    c = _fs_challenge(rel, com, context)
    return Proof(com, c, pv.respond(c))


def verify(rel: LinearRelation, proof: Proof, context: bytes = b"") -> bool:
    if proof.challenge != _fs_challenge(rel, proof.commitment, context):
        return False
    return verify_transcript(rel, proof.commitment, proof.challenge, proof.response)
