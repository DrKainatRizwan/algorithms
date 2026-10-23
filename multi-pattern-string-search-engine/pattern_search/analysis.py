"""String-structure helpers built on the matchers."""

from __future__ import annotations

from .single_pattern import failure_function, z_array


def border_array(s: str) -> list[int]:
    """Alias for the prefix function (longest proper border per prefix)."""
    return failure_function(s)


def shortest_period(s: str) -> int:
    """Smallest p such that s[i] == s[i+p] for all valid i (p = n - longest border)."""
    if not s:
        return 0
    return len(s) - failure_function(s)[-1]


def count_distinct_matches(text: str, patterns: list[str]) -> int:
    """Number of distinct patterns that occur at least once in ``text``."""
    from .aho_corasick import AhoCorasick
    uniq = [p for p in dict.fromkeys(patterns) if p]
    if not uniq:
        return 0
    ac = AhoCorasick(uniq)
    return sum(1 for c in ac.count(text).values() if c)


def longest_repeated_prefix_overlap(s: str) -> int:
    """Max Z-value over positions > 0: the longest prefix recurring in ``s``."""
    z = z_array(s)
    return max(z[1:], default=0)
