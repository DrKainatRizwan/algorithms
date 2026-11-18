"""Brute-force oracles and statistical experiments."""
from __future__ import annotations

from itertools import permutations

import numpy as np

from .gale_shapley import blocking_pairs, gale_shapley
from .lattice import enumerate_stable_matchings
from .preferences import (Prefs, random_marriage_instance, random_roommates_instance)
from .roommates import irving_roommates, roommates_blocking_pairs


def brute_force_marriage(men: Prefs, women: Prefs) -> list[list[int]]:
    """All stable perfect matchings by checking every permutation (n <= 8)."""
    return [list(p) for p in permutations(range(len(men))) if not blocking_pairs(men, women, list(p))]


def _perfect_matchings(items: list[int]):
    if not items:
        yield []
        return
    a = items[0]
    for k in range(1, len(items)):
        b = items[k]
        rest = items[1:k] + items[k + 1:]
        for m in _perfect_matchings(rest):
            yield [(a, b)] + m


def brute_force_roommates(prefs: Prefs) -> list[list[int]]:
    """All stable perfect matchings of a roommates instance (n <= 10)."""
    out = []
    for pm in _perfect_matchings(list(range(len(prefs)))):
        match = [0] * len(prefs)
        for a, b in pm:
            match[a], match[b] = b, a
        if not roommates_blocking_pairs(prefs, match):
            out.append(match)
    return out


def proposal_statistics(sizes: list[int], trials: int = 20, seed: int = 0) -> dict[int, tuple[float, float]]:
    """Mean proposals by men-proposing and women-proposing runs for random instances."""
    res = {}
    for n in sizes:
        a, b = [], []
        for t in range(trials):
            men, women = random_marriage_instance(n, seed * 100003 + n * 1009 + t)
            a.append(gale_shapley(men, women).proposals)
            b.append(gale_shapley(women, men).proposals)
        res[n] = (float(np.mean(a)), float(np.mean(b)))
    return res


def roommates_existence_rate(n: int, trials: int = 200, seed: int = 0) -> float:
    """Fraction of random instances admitting a stable matching."""
    ok = sum(irving_roommates(random_roommates_instance(n, seed * 7919 + t)) is not None
             for t in range(trials))
    return ok / trials


def count_stable_matchings(n: int, trials: int = 20, seed: int = 0) -> float:
    """Average number of stable matchings in random marriage instances."""
    c = [len(enumerate_stable_matchings(*random_marriage_instance(n, seed * 31 + t)))
         for t in range(trials)]
    return float(np.mean(c))
