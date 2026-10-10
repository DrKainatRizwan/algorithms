"""Seeded 64-bit hashing built from FNV-1a and the splitmix64 finaliser."""

from __future__ import annotations

from typing import Hashable

MASK64 = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def to_bytes(item: Hashable) -> bytes:
    """Canonical byte encoding of str, bytes, int, or any other hashable via repr."""
    if isinstance(item, bytes):
        return b"b" + item
    if isinstance(item, str):
        return b"s" + item.encode("utf-8")
    if isinstance(item, int):
        n = (item.bit_length() + 8) // 8
        return b"i" + item.to_bytes(n, "little", signed=True)
    return b"r" + repr(item).encode("utf-8")


def _mix(z: int) -> int:
    """splitmix64 finaliser: a strong avalanche on 64-bit words."""
    z = (z + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def hash64(item: Hashable, seed: int = 0) -> int:
    """Deterministic seeded 64-bit hash (independent of PYTHONHASHSEED)."""
    h = _FNV_OFFSET
    for byte in to_bytes(item):
        h = ((h ^ byte) * _FNV_PRIME) & MASK64
    return _mix(h ^ _mix(seed & MASK64))


def hash_pair(item: Hashable, seed: int = 0) -> tuple[int, int]:
    """Two 32-bit halves used for Kirsch-Mitzenmacher double hashing."""
    h = hash64(item, seed)
    return h & 0xFFFFFFFF, (h >> 32) | 1  # odd second hash => full cycle for pow2 sizes


def double_hash_indices(item: Hashable, k: int, m: int, seed: int = 0) -> list[int]:
    """k indices in [0, m) via g_i = h1 + i*h2 (Kirsch & Mitzenmacher 2006)."""
    h1, h2 = hash_pair(item, seed)
    return [(h1 + i * h2) % m for i in range(k)]
