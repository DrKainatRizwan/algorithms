"""Optimal assignment solvers: Hungarian (Kuhn-Munkres) and Bertsekas auction."""

from .analysis import forced_pair_cost, reduced_costs, verify_duals
from .applications import bottleneck_assignment, gated_assignment
from .auction import auction_assignment
from .hungarian import Assignment, InfeasibleError, hungarian
from .utils import brute_force, random_cost_matrix, validate_cost_matrix

__all__ = [
    "Assignment",
    "InfeasibleError",
    "hungarian",
    "auction_assignment",
    "bottleneck_assignment",
    "gated_assignment",
    "forced_pair_cost",
    "reduced_costs",
    "verify_duals",
    "brute_force",
    "random_cost_matrix",
    "validate_cost_matrix",
]
