"""Stable matching toolkit: Gale-Shapley, hospitals/residents, Irving's roommates, rotations."""
from .preferences import (random_marriage_instance, random_roommates_instance,
                          random_hr_instance, rank_table, validate_complete)
from .gale_shapley import (gale_shapley, blocking_pairs, is_stable, MatchingResult,
                           hospitals_residents, hr_blocking_pairs)
from .roommates import irving_roommates, roommates_blocking_pairs
from .lattice import (stable_matching_lattice, enumerate_stable_matchings,
                      find_rotations, find_all_rotations, egalitarian_matching, minimum_regret_matching,
                      sex_equal_matching)
from .analysis import (proposal_statistics, roommates_existence_rate,
                       count_stable_matchings, brute_force_marriage, brute_force_roommates)

__all__ = [n for n in dir() if not n.startswith("_")]
