"""Pedersen commitments and zero-knowledge proofs about committed values."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .composition import OrProof, prove_or, verify_or
from .groups import SchnorrGroup
from .sigma import (LinearRelation, Proof, prove, representation_relation,
                    schnorr_relation, verify)


class Pedersen:
    """Perfectly hiding, computationally binding commitment ``g^m h^r``."""

    def __init__(self, group: SchnorrGroup) -> None:
        self.group = group

    def commit(self, m: int, r: int) -> int:
        g = self.group
        return g.mul(g.exp(g.g, m), g.exp(g.h, r))

    def open(self, c: int, m: int, r: int) -> bool:
        return self.commit(m, r) == c

    def add(self, c1: int, c2: int) -> int:
        """Homomorphic sum: commits to m1+m2 with randomness r1+r2."""
        return self.group.mul(c1, c2)

    def scale(self, c: int, k: int) -> int:
        return self.group.exp(c, k)

    # -- proofs ----------------------------------------------------------
    def opening_relation(self, c: int) -> LinearRelation:
        return representation_relation(self.group, [self.group.g, self.group.h], c)

    def prove_opening(self, c: int, m: int, r: int, rng: random.Random) -> Proof:
        return prove(self.opening_relation(c), [m, r], rng)

    def verify_opening(self, c: int, proof: Proof) -> bool:
        return verify(self.opening_relation(c), proof)

    def prove_equal(self, c1: int, c2: int, r1: int, r2: int, rng: random.Random) -> Proof:
        """Two commitments hide the same value: c1/c2 = h^(r1-r2)."""
        g = self.group
        rel = schnorr_relation(g, g.mul(c1, g.inv(c2)), g.h)
        return prove(rel, [r1 - r2], rng)

    def verify_equal(self, c1: int, c2: int, proof: Proof) -> bool:
        g = self.group
        return verify(schnorr_relation(g, g.mul(c1, g.inv(c2)), g.h), proof)

    # -- bit and range proofs ----------------------------------------------
    def _bit_relations(self, c: int) -> list[LinearRelation]:
        g = self.group
        return [
            schnorr_relation(g, c, g.h),                      # c = h^r   (m = 0)
            schnorr_relation(g, g.mul(c, g.inv(g.g)), g.h),   # c/g = h^r (m = 1)
        ]

    def prove_bit(self, c: int, bit: int, r: int, rng: random.Random) -> OrProof:
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        return prove_or(self._bit_relations(c), bit, [r], rng)

    def verify_bit(self, c: int, proof: OrProof) -> bool:
        return verify_or(self._bit_relations(c), proof)


@dataclass(frozen=True)
class RangeProof:
    bit_commitments: tuple[int, ...]
    bit_proofs: tuple[OrProof, ...]

    def size_in_group_elements(self) -> int:
        return len(self.bit_commitments) * (1 + 2 + 2 + 2)


class RangeProver:
    """Prove ``0 <= m < 2^n`` for a Pedersen commitment via bit decomposition."""

    def __init__(self, ped: Pedersen, n_bits: int) -> None:
        self.ped, self.n = ped, n_bits

    def prove(self, m: int, r: int, rng: random.Random) -> tuple[int, RangeProof]:
        """Return the commitment to ``m`` (randomness ``r``) and its range proof."""
        if not 0 <= m < (1 << self.n):
            raise ValueError("value out of range")
        g = self.ped.group
        q = g.q
        bits = [(m >> i) & 1 for i in range(self.n)]
        rs = [g.random_scalar(rng) for _ in range(self.n)]
        # fix the top randomness so that sum 2^i r_i = r (mod q)
        rest = sum((1 << i) * rs[i] for i in range(self.n - 1)) % q
        rs[-1] = (r - rest) * pow(1 << (self.n - 1), -1, q) % q
        coms = [self.ped.commit(b, ri) for b, ri in zip(bits, rs)]
        proofs = [self.ped.prove_bit(c, b, ri, rng) for c, b, ri in zip(coms, bits, rs)]
        return self.ped.commit(m, r), RangeProof(tuple(coms), tuple(proofs))

    def verify(self, c: int, proof: RangeProof) -> bool:
        g = self.ped.group
        if len(proof.bit_commitments) != self.n or len(proof.bit_proofs) != self.n:
            return False
        recomb = g.multiexp(list(proof.bit_commitments), [1 << i for i in range(self.n)])
        if recomb != c:
            return False
        return all(self.ped.verify_bit(ci, pi)
                   for ci, pi in zip(proof.bit_commitments, proof.bit_proofs))
