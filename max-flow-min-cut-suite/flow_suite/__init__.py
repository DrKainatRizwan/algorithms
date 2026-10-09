"""Max-flow / min-cut suite: Edmonds-Karp, Dinic and push-relabel."""
from .graph import FlowNetwork
from .edmonds_karp import edmonds_karp
from .dinic import dinic
from .push_relabel import push_relabel
from .mincut import MinCut, min_cut, cut_capacity, verify_certificate, reachable_from
from .generators import (random_network, layered_network, bipartite_matching,
                         matching_from_flow, edge_disjoint_paths)

SOLVERS = {"edmonds_karp": edmonds_karp, "dinic": dinic, "push_relabel": push_relabel}

__all__ = ["FlowNetwork", "edmonds_karp", "dinic", "push_relabel", "MinCut", "min_cut",
           "cut_capacity", "verify_certificate", "reachable_from", "random_network",
           "layered_network", "bipartite_matching", "matching_from_flow",
           "edge_disjoint_paths", "SOLVERS"]
