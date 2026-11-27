"""Private set intersection protocols.

* :func:`naive_hash_psi` - exchange of plain hashes (fast, but broken for small
  universes; see :func:`dictionary_attack`).
* :func:`dh_psi` / :func:`dh_psi_cardinality` - semi-honest Diffie-Hellman PSI
  (Meadows 1986; Huberman-Franklin-Hogg 1999).
* :func:`bloom_psi` - one-round approximate PSI with a Bloom filter.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .numtheory import SchnorrGroup


def _b(x: str | bytes) -> bytes:
    return x if isinstance(x, bytes) else x.encode()


@dataclass
class PSIResult:
    """Outcome of a PSI run together with simple communication accounting."""

    intersection: set[bytes]
    cardinality: int
    bytes_client_to_server: int = 0
    bytes_server_to_client: int = 0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# naive hashing
# --------------------------------------------------------------------------
def _digest(x: bytes) -> bytes:
    return hashlib.sha256(x).digest()


def naive_hash_psi(client: Iterable[str | bytes], server: Iterable[str | bytes]) -> PSIResult:
    """Server sends ``SHA-256`` of each element; client intersects locally."""
    c = {_b(x) for x in client}
    s_hashes = {_digest(_b(y)) for y in server}
    inter = {x for x in c if _digest(x) in s_hashes}
    return PSIResult(inter, len(inter), 0, 32 * len(s_hashes), {"hashes": s_hashes})


def dictionary_attack(hashes: Iterable[bytes], universe: Iterable[str | bytes]) -> set[bytes]:
    """Recover set elements from plain hashes by enumerating a small universe."""
    hs = set(hashes)
    return {_b(u) for u in universe if _digest(_b(u)) in hs}


# --------------------------------------------------------------------------
# Diffie-Hellman PSI
# --------------------------------------------------------------------------
class DHParty:
    """One side of DH-PSI holding a secret exponent."""

    def __init__(self, group: SchnorrGroup, items: Sequence[str | bytes], rng: random.Random) -> None:
        self.group = group
        self.items = [_b(x) for x in dict.fromkeys(_b(i) for i in items)]
        self.rng = rng
        self.key = group.random_exponent(rng)

    def blind_own(self) -> list[int]:
        """``H(x)^k`` for each own item, in item order."""
        return [self.group.exp(self.group.hash_to_group(x), self.key) for x in self.items]

    def blind_foreign(self, values: Sequence[int]) -> list[int]:
        """Raise received group elements to the own key (order preserved)."""
        for v in values:
            if not self.group.contains(v):
                raise ValueError("received value is not a group element")
        return [self.group.exp(v, self.key) for v in values]

    def shuffled(self, values: Sequence[int]) -> list[int]:
        """Return a random permutation of ``values``."""
        out = list(values)
        self.rng.shuffle(out)
        return out


def dh_psi(client_items, server_items, group: SchnorrGroup, seed: int = 0) -> PSIResult:
    """Semi-honest DH-PSI revealing the intersection to the client.

    Round 1: client -> server ``H(x)^a``.  Round 2: server -> client the
    pairs ``(H(x)^{ab} in client order)`` and the shuffled set ``H(y)^b``.
    The client raises the latter to ``a`` and matches.
    """
    rng = random.Random(seed)
    client = DHParty(group, client_items, rng)
    server = DHParty(group, server_items, rng)
    elt = (group.p.bit_length() + 7) // 8
    msg1 = client.blind_own()
    double = server.blind_foreign(msg1)
    msg2 = server.shuffled(server.blind_own())
    server_double = {group.exp(v, client.key) for v in msg2}
    inter = {x for x, d in zip(client.items, double) if d in server_double}
    return PSIResult(inter, len(inter), elt * len(msg1), elt * (len(double) + len(msg2)))


def dh_psi_cardinality(client_items, server_items, group: SchnorrGroup, seed: int = 0) -> PSIResult:
    """DH-PSI variant revealing only ``|A ∩ B|``: the server shuffles the doubly blinded list."""
    rng = random.Random(seed)
    client = DHParty(group, client_items, rng)
    server = DHParty(group, server_items, rng)
    elt = (group.p.bit_length() + 7) // 8
    msg1 = client.blind_own()
    double = server.shuffled(server.blind_foreign(msg1))
    msg2 = server.shuffled(server.blind_own())
    server_double = {group.exp(v, client.key) for v in msg2}
    count = sum(1 for d in double if d in server_double)
    return PSIResult(set(), count, elt * len(msg1), elt * (len(double) + len(msg2)))


# --------------------------------------------------------------------------
# Bloom filter PSI
# --------------------------------------------------------------------------
class BloomFilter:
    """Standard Bloom filter built on double hashing of SHA-256."""

    def __init__(self, m_bits: int, k: int) -> None:
        if m_bits < 1 or k < 1:
            raise ValueError("m_bits and k must be positive")
        self.m, self.k = m_bits, k
        self.bits = bytearray((m_bits + 7) // 8)

    @classmethod
    def for_capacity(cls, n: int, fp_rate: float) -> "BloomFilter":
        """Size a filter for ``n`` items at target false-positive rate."""
        m = max(8, math.ceil(-n * math.log(fp_rate) / (math.log(2) ** 2)))
        k = max(1, round(m / max(n, 1) * math.log(2)))
        return cls(m, k)

    def _positions(self, x: bytes):
        h = hashlib.sha256(b"bf" + x).digest()
        h1 = int.from_bytes(h[:16], "big")
        h2 = int.from_bytes(h[16:], "big") | 1
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, x: bytes) -> None:
        for p in self._positions(x):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, x: bytes) -> bool:
        return all(self.bits[p >> 3] >> (p & 7) & 1 for p in self._positions(x))

    def expected_fp_rate(self, n: int) -> float:
        """Theoretical false positive rate after ``n`` insertions."""
        return (1 - math.exp(-self.k * n / self.m)) ** self.k


def bloom_psi(client_items, server_items, fp_rate: float = 0.01) -> PSIResult:
    """Server ships a Bloom filter; the client reports members (may include false positives)."""
    s = [_b(y) for y in server_items]
    bf = BloomFilter.for_capacity(len(s), fp_rate)
    for y in s:
        bf.add(y)
    c = {_b(x) for x in client_items}
    inter = {x for x in c if x in bf}
    return PSIResult(inter, len(inter), 0, len(bf.bits), {"filter": bf})
