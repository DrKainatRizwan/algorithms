"""Deterministic, seedable 64-bit hash primitives (no reliance on Python's salted ``hash``)."""
from __future__ import annotations

MASK64 = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash of ``data``."""
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & MASK64
    return h


def mix64(x: int) -> int:
    """SplitMix64 finaliser: a strong bijective avalanche step on 64-bit integers."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def to_bytes(key: object) -> bytes:
    """Canonical byte encoding for str / bytes / int keys."""
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, int):
        return str(key).encode("ascii")
    raise TypeError(f"unsupported key type: {type(key).__name__}")


def hash_key(key: object, seed: int = 0) -> int:
    """Well-mixed 64-bit hash of ``key`` under ``seed``."""
    return mix64(fnv1a64(to_bytes(key)) ^ mix64(seed & MASK64))


def hash_pair(a: object, b: object, seed: int = 0) -> int:
    """Hash of an ordered pair, used for (key, node) scoring in rendezvous hashing."""
    return mix64(hash_key(a, seed) ^ mix64(hash_key(b, seed + 1)))


def unit_float(h: int) -> float:
    """Map a 64-bit hash to a float in the open interval (0, 1)."""
    return ((h >> 12) + 0.5) / float(1 << 52)
