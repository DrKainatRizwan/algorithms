"""Identifier space helpers for Kademlia: hashing, XOR metric and bucket indexing."""
from __future__ import annotations

import hashlib
import random

DEFAULT_BITS = 32


def hash_to_id(data: bytes | str, bits: int = DEFAULT_BITS) -> int:
    """Map arbitrary data to a ``bits``-bit identifier using SHA-256."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(data).digest(), "big")
    return digest >> (256 - bits) if bits <= 256 else digest


def xor_distance(a: int, b: int) -> int:
    """Kademlia distance: bitwise XOR interpreted as an integer."""
    return a ^ b


def bucket_index(own: int, other: int) -> int:
    """Index of the k-bucket holding ``other`` as seen from ``own``.

    Bucket ``i`` stores contacts whose distance lies in ``[2**i, 2**(i+1))``,
    i.e. contacts sharing exactly ``bits - 1 - i`` leading bits with ``own``.
    """
    d = own ^ other
    if d == 0:
        raise ValueError("a node does not store itself in its routing table")
    return d.bit_length() - 1


def random_unique_ids(n: int, bits: int, rng: random.Random) -> list[int]:
    """Draw ``n`` distinct random identifiers."""
    if n > (1 << bits):
        raise ValueError("identifier space too small")
    seen: set[int] = set()
    out: list[int] = []
    while len(out) < n:
        v = rng.getrandbits(bits)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def random_id_in_bucket(own: int, index: int, rng: random.Random) -> int:
    """A random identifier whose distance from ``own`` falls in bucket ``index``."""
    low = 1 << index
    d = low | rng.getrandbits(index) if index > 0 else 1
    return own ^ d
