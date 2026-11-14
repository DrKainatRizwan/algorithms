"""Hash functions used by the tables.

Every function maps a 64-bit integer to a slot index in ``[0, 2**bits)``.
Arbitrary Python keys are first reduced to 64-bit integers by
:func:`key_to_int`, which (unlike the built-in ``hash``) is stable across
processes so that all experiments are reproducible.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Protocol

MASK64 = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def splitmix64(x: int) -> int:
    """Finaliser of the SplitMix64 generator (a strong 64-bit bit mixer)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def fnv1a(data: bytes) -> int:
    """64-bit FNV-1a hash of a byte string."""
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & MASK64
    return h


def key_to_int(key: Any) -> int:
    """Deterministically reduce a hashable key to an unsigned 64-bit integer."""
    if isinstance(key, int):
        return key & MASK64
    if isinstance(key, str):
        return fnv1a(key.encode("utf-8"))
    if isinstance(key, (bytes, bytearray)):
        return fnv1a(bytes(key))
    if isinstance(key, tuple):
        acc = 0x345678
        for part in key:
            acc = splitmix64(acc ^ key_to_int(part))
        return acc
    return hash(key) & MASK64


class HashFunction(Protocol):
    """Callable-style protocol: map a 64-bit key to a ``bits``-bit index."""

    def index(self, x: int, bits: int) -> int: ...


class IdentityHash:
    """Take the low ``bits`` bits of the key.  Perfect for dense integer
    keys, catastrophic for keys sharing a power-of-two stride."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def index(self, x: int, bits: int) -> int:
        return x & ((1 << bits) - 1)


class MultiplyShift:
    """Dietzfelbinger multiply-shift hashing: ``(a*x + b mod 2^64) >> (64-bits)``.

    ``a`` is a random odd 64-bit multiplier.  Very cheap and universal in
    practice, though only weakly independent.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.a = rng.getrandbits(64) | 1
        self.b = rng.getrandbits(64)

    def index(self, x: int, bits: int) -> int:
        return ((self.a * x + self.b) & MASK64) >> (64 - bits)


class TabulationHash:
    """Simple tabulation hashing (Patrascu-Thorup).

    The key is split into eight bytes; each byte indexes a table of random
    64-bit words and the results are XORed.  3-independent, and provably
    good for linear probing and cuckoo hashing despite its simplicity.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.tables = [[rng.getrandbits(64) for _ in range(256)] for _ in range(8)]

    def index(self, x: int, bits: int) -> int:
        h = 0
        t = self.tables
        for i in range(8):
            h ^= t[i][(x >> (8 * i)) & 0xFF]
        return h >> (64 - bits)


class MixHash:
    """SplitMix64-based hash keyed by XOR with a random seed word."""

    def __init__(self, seed: int = 0) -> None:
        self.key = random.Random(seed).getrandbits(64)

    def index(self, x: int, bits: int) -> int:
        return splitmix64(x ^ self.key) >> (64 - bits)


HashFactory = Callable[[int], HashFunction]

HASH_FAMILIES: dict[str, HashFactory] = {
    "identity": IdentityHash,
    "multiply-shift": MultiplyShift,
    "tabulation": TabulationHash,
    "mix": MixHash,
}
