"""Rotations and the lattice of stable matchings (complete lists, equal sides)."""
from __future__ import annotations

from dataclasses import dataclass

from .gale_shapley import gale_shapley
from .preferences import Prefs, rank_table, validate_complete

Rotation = tuple[tuple[int, int], ...]  # ((man, woman), ...) pairs being shifted


@dataclass
class _State:
    match: tuple[int, ...]          # man -> woman
    lists: tuple[tuple[int, ...], ...]  # reduced man lists, first entry is current partner


def _initial_state(men: Prefs, women: Prefs) -> _State:
    validate_complete(men, women)
    wr = rank_table(women)
    m0 = gale_shapley(men, women).match
    inv = [0] * len(women)
    for m, w in enumerate(m0):
        inv[w] = m
    lists = []
    for m, p in enumerate(men):
        k = p.index(m0[m])
        lists.append(tuple(w for w in p[k:] if w == m0[m] or wr[w][m] < wr[w][inv[w]]))
    return _State(tuple(m0), tuple(lists))


def _exposed(state: _State) -> list[list[int]]:
    """Exposed rotations as cycles of men."""
    n = len(state.match)
    inv = [0] * n
    for m, w in enumerate(state.match):
        inv[w] = m

    def nxt(m: int) -> int | None:
        lst = state.lists[m]
        return inv[lst[1]] if len(lst) > 1 else None

    color = [0] * n
    cycles = []
    for s in range(n):
        if color[s]:
            continue
        path, m = [], s
        while m is not None and color[m] == 0:
            color[m] = 1
            path.append(m)
            m = nxt(m)
        if m is not None and color[m] == 1:
            cycles.append(path[path.index(m):])
        for x in path:
            color[x] = 2
    return cycles


def _rotation_of(state: _State, cycle: list[int]) -> Rotation:
    return tuple((m, state.match[m]) for m in cycle)


def _eliminate(state: _State, cycle: list[int], women: Prefs) -> _State:
    wr = rank_table(women)
    n = len(state.match)
    inv = [0] * n
    for m, w in enumerate(state.match):
        inv[w] = m
    match = list(state.match)
    for m in cycle:
        match[m] = state.lists[m][1]
    new_inv = [0] * n
    for m, w in enumerate(match):
        new_inv[w] = m
    lists = []
    for m in range(n):
        keep = []
        started = False
        for w in state.lists[m]:
            if w == match[m]:
                started = True
            if started and wr[w][m] <= wr[w][new_inv[w]]:
                keep.append(w)
        lists.append(tuple(keep))
    return _State(tuple(match), tuple(lists))


def find_rotations(men: Prefs, women: Prefs) -> list[Rotation]:
    """Rotations exposed in the man-optimal matching (the minimal elements of the poset)."""
    st = _initial_state(men, women)
    return [_rotation_of(st, c) for c in _exposed(st)]


def enumerate_stable_matchings(men: Prefs, women: Prefs, limit: int | None = None) -> list[list[int]]:
    """All stable matchings (man -> woman) by exhaustive rotation elimination.

    Output size can be exponential; ``limit`` caps the number returned.
    """
    root = _initial_state(men, women)
    seen = {root.match}
    stack = [root]
    while stack:
        st = stack.pop()
        for cyc in _exposed(st):
            nst = _eliminate(st, cyc, women)
            if nst.match not in seen:
                seen.add(nst.match)
                if limit is not None and len(seen) >= limit:
                    return [list(m) for m in sorted(seen)]
                stack.append(nst)
    return [list(m) for m in sorted(seen)]


def stable_matching_lattice(men: Prefs, women: Prefs) -> dict:
    """Summary of the lattice: man-optimal, woman-optimal, count and number of rotations."""
    allm = enumerate_stable_matchings(men, women)
    mo = gale_shapley(men, women).match
    wo_inv = gale_shapley(women, men).match
    wo = [0] * len(men)
    for w, m in enumerate(wo_inv):
        wo[m] = w
    return {"man_optimal": mo, "woman_optimal": wo, "count": len(allm),
            "rotations": len(find_all_rotations(men, women))}


def find_all_rotations(men: Prefs, women: Prefs) -> set[Rotation]:
    """Every rotation of the instance, normalised to start at its smallest man."""
    root = _initial_state(men, women)
    seen = {root.match}
    rots: set[Rotation] = set()
    stack = [root]
    while stack:
        st = stack.pop()
        for cyc in _exposed(st):
            r = list(_rotation_of(st, cyc))
            k = r.index(min(r))
            rots.add(tuple(r[k:] + r[:k]))
            nst = _eliminate(st, cyc, women)
            if nst.match not in seen:
                seen.add(nst.match)
                stack.append(nst)
    return rots


def _costs(men: Prefs, women: Prefs, match: list[int]) -> tuple[int, int, int]:
    mr, wr = rank_table(men), rank_table(women)
    cm = sum(mr[m][w] for m, w in enumerate(match))
    cw = sum(wr[w][m] for m, w in enumerate(match))
    regret = max(max(mr[m][w] for m, w in enumerate(match)),
                 max(wr[w][m] for m, w in enumerate(match)))
    return cm, cw, regret


def egalitarian_matching(men: Prefs, women: Prefs) -> list[int]:
    """Stable matching minimising total rank over both sides."""
    return min(enumerate_stable_matchings(men, women),
               key=lambda m: (sum(_costs(men, women, m)[:2]), m))


def minimum_regret_matching(men: Prefs, women: Prefs) -> list[int]:
    """Stable matching minimising the worst rank any participant receives."""
    return min(enumerate_stable_matchings(men, women),
               key=lambda m: (_costs(men, women, m)[2], sum(_costs(men, women, m)[:2]), m))


def sex_equal_matching(men: Prefs, women: Prefs) -> list[int]:
    """Stable matching minimising |men's cost - women's cost|."""
    def key(m: list[int]):
        cm, cw, _ = _costs(men, women, m)
        return (abs(cm - cw), cm + cw, m)
    return min(enumerate_stable_matchings(men, women), key=key)
