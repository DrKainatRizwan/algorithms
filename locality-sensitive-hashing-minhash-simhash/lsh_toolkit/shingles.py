"""Text shingling and exact set similarity."""
from __future__ import annotations

from typing import Iterable

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


def fnv1a_64(data: str | bytes) -> int:
    """64-bit FNV-1a hash (deterministic across runs, unlike ``hash``)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = _FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK64
    return h


def char_shingles(text: str, k: int = 5) -> set[int]:
    """Hashed character k-shingles of ``text`` (whitespace-normalised, lower-cased)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    norm = " ".join(text.lower().split())
    if len(norm) <= k:
        return {fnv1a_64(norm)} if norm else set()
    return {fnv1a_64(norm[i:i + k]) for i in range(len(norm) - k + 1)}


def word_shingles(text: str, k: int = 3) -> set[int]:
    """Hashed word k-grams of ``text``."""
    if k < 1:
        raise ValueError("k must be >= 1")
    words = text.lower().split()
    if len(words) <= k:
        return {fnv1a_64(" ".join(words))} if words else set()
    return {fnv1a_64(" ".join(words[i:i + k])) for i in range(len(words) - k + 1)}


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Exact Jaccard similarity |A∩B| / |A∪B| (1.0 for two empty sets)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
