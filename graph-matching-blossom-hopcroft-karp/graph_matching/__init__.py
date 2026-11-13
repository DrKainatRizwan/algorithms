"""Graph matching toolkit: Hopcroft-Karp, Edmonds' blossom and stable matching."""

from .bipartite import (
    hopcroft_karp,
    kuhn_matching,
    konig_vertex_cover,
    hall_violator,
)
from .blossom import edmonds_blossom, greedy_matching
from .stable import (
    gale_shapley,
    hospitals_residents,
    blocking_pairs,
    all_stable_matchings,
)
from .verify import (
    is_matching,
    brute_force_max_matching,
    is_vertex_cover,
)
from .generators import random_bipartite, random_graph, random_preferences

__all__ = [
    "hopcroft_karp", "kuhn_matching", "konig_vertex_cover", "hall_violator",
    "edmonds_blossom", "greedy_matching",
    "gale_shapley", "hospitals_residents", "blocking_pairs", "all_stable_matchings",
    "is_matching", "brute_force_max_matching", "is_vertex_cover",
    "random_bipartite", "random_graph", "random_preferences",
]
