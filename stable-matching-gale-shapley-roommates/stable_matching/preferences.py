"""Preference-list generation and validation helpers."""
from __future__ import annotations

import numpy as np

Prefs = list[list[int]]


def random_marriage_instance(n: int, seed: int = 0) -> tuple[Prefs, Prefs]:
    """Uniformly random complete preference lists for ``n`` men and ``n`` women."""
    rng = np.random.default_rng(seed)
    men = [rng.permutation(n).tolist() for _ in range(n)]
    women = [rng.permutation(n).tolist() for _ in range(n)]
    return men, women


def random_roommates_instance(n: int, seed: int = 0) -> Prefs:
    """Random complete lists for ``n`` people (each ranks all others)."""
    rng = np.random.default_rng(seed)
    prefs = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        prefs.append([others[k] for k in rng.permutation(n - 1)])
    return prefs


def random_hr_instance(n_res: int, capacities: list[int], seed: int = 0,
                       accept_prob: float = 0.8) -> tuple[Prefs, Prefs]:
    """Random hospitals/residents instance with incomplete, mutually-consistent lists."""
    rng = np.random.default_rng(seed)
    h = len(capacities)
    res: Prefs = [[] for _ in range(n_res)]
    hosp: Prefs = [[] for _ in range(h)]
    for r in range(n_res):
        for j in rng.permutation(h):
            if rng.random() < accept_prob:
                res[r].append(int(j))
    for r in range(n_res):
        for j in res[r]:
            hosp[j].append(r)
    for j in range(h):
        hosp[j] = [hosp[j][k] for k in rng.permutation(len(hosp[j]))]
    return res, hosp


def rank_table(prefs: Prefs, universe: int | None = None) -> list[dict[int, int]]:
    """For each agent, map candidate -> rank (0 is best)."""
    return [{c: k for k, c in enumerate(p)} for p in prefs]


def validate_complete(men: Prefs, women: Prefs) -> None:
    """Raise ``ValueError`` unless both sides give complete strict lists."""
    n = len(men)
    if len(women) != n:
        raise ValueError("sides must have equal size")
    for side in (men, women):
        for p in side:
            if sorted(p) != list(range(n)):
                raise ValueError("preference list is not a permutation of 0..n-1")
