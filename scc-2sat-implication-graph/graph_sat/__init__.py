"""Strongly connected components, 2-SAT and implication-graph tools."""

from .digraph import Digraph
from .scc import tarjan_scc, kosaraju_scc, gabow_scc, condensation
from .two_sat import TwoSat, SatResult
from .reachability import transitive_closure, transitive_reduction_dag, ReachabilityIndex
from .generators import random_digraph, random_2sat, planted_2sat, brute_force_2sat

__all__ = [
    "Digraph", "tarjan_scc", "kosaraju_scc", "gabow_scc", "condensation",
    "TwoSat", "SatResult", "transitive_closure", "transitive_reduction_dag",
    "ReachabilityIndex", "random_digraph", "random_2sat", "planted_2sat",
    "brute_force_2sat",
]
