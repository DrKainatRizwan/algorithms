"""Cryptographic building blocks: label hashing and bit helpers.

Wire labels are 128-bit integers. The random oracle is SHA-256 truncated to
128 bits, domain-separated by a gate tweak. Randomness comes from a seeded
``random.Random`` so that experiments are reproducible; a production system
would draw labels from ``secrets``.
"""

from __future__ import annotations

import hashlib
import random

LABEL_BITS = 128
LABEL_BYTES = LABEL_BITS // 8
MASK = (1 << LABEL_BITS) - 1


def hash_labels(a: int, b: int, tweak: int) -> int:
    """Hash two labels and a tweak to a 128-bit integer."""
    data = a.to_bytes(LABEL_BYTES, "big") + b.to_bytes(LABEL_BYTES, "big") + tweak.to_bytes(8, "big")
    return int.from_bytes(hashlib.sha256(data).digest()[:LABEL_BYTES], "big")


def hash_single(a: int, tweak: int) -> int:
    """Tweakable hash of a single label (used by half-gates)."""
    return hash_labels(a, 0, tweak)


def lsb(x: int) -> int:
    """Least-significant bit, used as the point-and-permute colour bit."""
    return x & 1


def random_label(rng: random.Random) -> int:
    """Uniform random 128-bit label."""
    return rng.getrandbits(LABEL_BITS)


def random_offset(rng: random.Random) -> int:
    """Global free-XOR offset R; its colour bit is forced to 1."""
    return rng.getrandbits(LABEL_BITS) | 1


def int_to_bits(value: int, width: int) -> list[int]:
    """Little-endian bit decomposition."""
    if value < 0 or value >= 1 << width:
        raise ValueError(f"{value} does not fit in {width} bits")
    return [(value >> i) & 1 for i in range(width)]


def bits_to_int(bits: list[int]) -> int:
    """Inverse of :func:`int_to_bits`."""
    return sum(b << i for i, b in enumerate(bits))
