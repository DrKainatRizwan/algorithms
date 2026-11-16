"""Schnorr signatures (Fiat-Shamir of the Schnorr protocol) and batch verification."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .groups import SchnorrGroup
from .transcript import Transcript


@dataclass(frozen=True)
class Signature:
    r: int
    s: int


def _challenge(group: SchnorrGroup, r: int, pub: int, msg: bytes) -> int:
    t = Transcript("zk_sigma.schnorr_sig")
    t.absorb_int("R", r)
    t.absorb_int("pub", pub)
    t.absorb_int("msg.len", len(msg))
    t.absorb_int("msg", int.from_bytes(msg, "big") if msg else 0)
    return t.challenge(group.q)


def keygen(group: SchnorrGroup, rng: random.Random) -> tuple[int, int]:
    x = group.random_scalar(rng)
    return x, group.exp(group.g, x)


def sign(group: SchnorrGroup, x: int, msg: bytes, rng: random.Random,
         nonce: int | None = None) -> Signature:
    """Sign ``msg``; ``nonce`` is exposed only so attacks can be demonstrated."""
    k = nonce if nonce is not None else group.random_scalar(rng)
    r = group.exp(group.g, k)
    e = _challenge(group, r, group.exp(group.g, x), msg)
    return Signature(r, (k + e * x) % group.q)


def verify(group: SchnorrGroup, pub: int, msg: bytes, sig: Signature) -> bool:
    """Check ``g^s == R * pub^e`` with ``e = H(R, pub, msg)``."""
    if not (group.is_member(pub) and group.is_member(sig.r) and 0 <= sig.s < group.q):
        return False
    e = _challenge(group, sig.r, pub, msg)
    return group.exp(group.g, sig.s) == group.mul(sig.r, group.exp(pub, e))


def recover_key_from_nonce_reuse(group: SchnorrGroup, m1: bytes, s1: Signature,
                                 m2: bytes, s2: Signature, pub: int) -> int | None:
    """Two signatures sharing a nonce leak x = (s1-s2)/(e1-e2)."""
    if s1.r != s2.r:
        return None
    q = group.q
    e1, e2 = _challenge(group, s1.r, pub, m1), _challenge(group, s2.r, pub, m2)
    if (e1 - e2) % q == 0:
        return None
    x = (s1.s - s2.s) * pow(e1 - e2, -1, q) % q
    return x if group.exp(group.g, x) == pub else None


def batch_verify(group: SchnorrGroup, items: list[tuple[int, bytes, Signature]],
                 rng: random.Random) -> bool:
    """Small-exponent batch test: one aggregate equation with random weights.

    Checks ``g^(sum z_i s_i) == prod R_i^z_i * pub_i^(z_i e_i)``; a forged
    member survives only with probability about 2^-64.
    """
    q, p = group.q, group.p
    s_acc, rhs = 0, 1
    for pub, msg, sig in items:
        if not (group.is_member(pub) and group.is_member(sig.r) and 0 <= sig.s < q):
            return False
        z = rng.getrandbits(64) | 1
        e = _challenge(group, sig.r, pub, msg)
        s_acc = (s_acc + z * sig.s) % q
        rhs = rhs * pow(sig.r, z, p) % p * pow(pub, z * e % q, p) % p
    return group.exp(group.g, s_acc) == rhs
