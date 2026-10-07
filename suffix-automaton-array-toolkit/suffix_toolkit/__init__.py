"""Suffix array and suffix automaton toolkit for string analysis."""
from .suffix_array import build_suffix_array, build_suffix_array_naive, kasai_lcp, inverse_suffix_array
from .suffix_automaton import SuffixAutomaton
from .search import SuffixArrayIndex
from .analysis import (
    count_distinct_substrings,
    longest_repeated_substring,
    longest_common_substring,
    kth_substring,
    longest_common_substring_sa,
)
from .bwt import bwt_from_suffix_array, inverse_bwt

__all__ = [
    "build_suffix_array", "build_suffix_array_naive", "kasai_lcp", "inverse_suffix_array",
    "SuffixAutomaton", "SuffixArrayIndex", "count_distinct_substrings",
    "longest_repeated_substring", "longest_common_substring", "kth_substring",
    "longest_common_substring_sa", "bwt_from_suffix_array", "inverse_bwt",
]
