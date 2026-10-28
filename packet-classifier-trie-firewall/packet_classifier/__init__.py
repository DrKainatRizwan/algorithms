"""Packet classification with trie-based firewall rule matching."""

from .rules import Rule, Packet, parse_ip, format_ip, parse_prefix, parse_rule
from .linear import LinearClassifier
from .hier_trie import HierarchicalTrieClassifier, PrefixTrie
from .tuple_space import TupleSpaceClassifier
from .hicuts import HiCutsClassifier
from .generator import generate_rules, generate_packets
from .analysis import find_shadowed, find_conflicts

__all__ = [
    "Rule", "Packet", "parse_ip", "format_ip", "parse_prefix", "parse_rule",
    "LinearClassifier", "HierarchicalTrieClassifier", "PrefixTrie",
    "TupleSpaceClassifier", "HiCutsClassifier",
    "generate_rules", "generate_packets", "find_shadowed", "find_conflicts",
]
