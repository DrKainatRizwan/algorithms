"""gossip_sim: epidemic broadcast, rumor mongering, push-sum and anti-entropy."""
from .network import Graph, complete_graph, ring_lattice, erdos_renyi, watts_strogatz, barabasi_albert
from .protocols import SpreadResult, spread
from .rumor import RumorResult, rumor_mongering
from .aggregation import PushSumResult, push_sum
from .antientropy import Replica, anti_entropy_round, run_anti_entropy
from .theory import (push_rounds_estimate, mean_field_curve, mean_field_rounds,
                     rumor_residue)
from .simulate import TrialSummary, run_trials

__all__ = [
    "Graph", "complete_graph", "ring_lattice", "erdos_renyi", "watts_strogatz",
    "barabasi_albert", "SpreadResult", "spread", "RumorResult", "rumor_mongering",
    "PushSumResult", "push_sum", "Replica", "anti_entropy_round", "run_anti_entropy",
    "push_rounds_estimate", "mean_field_curve", "mean_field_rounds", "rumor_residue",
    "TrialSummary", "run_trials",
]
