"""Brute-force baselines used for validation and benchmarking."""
from __future__ import annotations


def naive_locate(text: bytes, pattern: bytes) -> list[int]:
    """Sorted start offsets of all (overlapping) occurrences by direct scanning."""
    if not pattern:
        return list(range(len(text) + 1))
    out, i = [], text.find(pattern)
    while i != -1:
        out.append(i)
        i = text.find(pattern, i + 1)
    return out


def naive_count(text: bytes, pattern: bytes) -> int:
    """Number of overlapping occurrences."""
    return len(naive_locate(text, pattern))
