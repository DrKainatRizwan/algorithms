"""Counterfactual regret minimisation for imperfect-information games."""
from .analysis import KUHN_VALUE, format_strategy, kuhn_alpha, kuhn_equilibrium, regret_bound
from .cfr import CFRConfig, CFRSolver, regret_matching
from .evaluation import (best_response, combine, expected_value, exploitability,
                         infoset_reach, nash_conv)
from .games import KuhnPoker, LeducHoldem
from .mccfr import MCCFRSolver
from .tree import GameTree, build_tree

__all__ = [
    "KUHN_VALUE", "CFRConfig", "CFRSolver", "GameTree", "KuhnPoker", "LeducHoldem",
    "MCCFRSolver", "best_response", "build_tree", "combine", "expected_value",
    "exploitability", "format_strategy", "infoset_reach", "kuhn_alpha",
    "kuhn_equilibrium", "nash_conv", "regret_bound", "regret_matching",
]
