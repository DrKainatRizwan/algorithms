"""Multi-pattern string search: KMP, Z, Rabin-Karp and Aho-Corasick."""

from .single_pattern import kmp_search, failure_function, z_array, z_search, naive_search
from .rolling_hash import RollingHash, rabin_karp, rabin_karp_multi
from .aho_corasick import AhoCorasick, Match
from .engine import SearchEngine, resolve_overlaps
from .analysis import shortest_period, border_array, count_distinct_matches

__all__ = [
    "kmp_search", "failure_function", "z_array", "z_search", "naive_search",
    "RollingHash", "rabin_karp", "rabin_karp_multi",
    "AhoCorasick", "Match", "SearchEngine", "resolve_overlaps",
    "shortest_period", "border_array", "count_distinct_matches",
]
