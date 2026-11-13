"""Stable matching: Gale-Shapley, hospitals/residents and enumeration."""

from __future__ import annotations

from itertools import permutations


def gale_shapley(proposer_prefs: list[list[int]], receiver_prefs: list[list[int]]) -> list[int]:
    """Deferred acceptance for equal-sized one-to-one markets, O(n^2).

    Returns ``match[p]`` = receiver for each proposer.  The result is the
    proposer-optimal (and receiver-pessimal) stable matching.
    """
    n = len(proposer_prefs)
    rank = [{r: i for i, r in enumerate(prefs)} for prefs in receiver_prefs]
    nxt = [0] * n
    holder = [-1] * n  # receiver -> proposer
    free = list(range(n - 1, -1, -1))
    while free:
        p = free.pop()
        if nxt[p] >= len(proposer_prefs[p]):
            continue  # exhausted list, stays unmatched
        r = proposer_prefs[p][nxt[p]]
        nxt[p] += 1
        if p not in rank[r]:
            free.append(p)
        elif holder[r] == -1:
            holder[r] = p
        elif rank[r][p] < rank[r][holder[r]]:
            free.append(holder[r])
            holder[r] = p
        else:
            free.append(p)
    match = [-1] * n
    for r, p in enumerate(holder):
        if p != -1:
            match[p] = r
    return match


def hospitals_residents(
    resident_prefs: list[list[int]],
    hospital_prefs: list[list[int]],
    capacities: list[int],
) -> list[list[int]]:
    """Resident-proposing deferred acceptance with hospital capacities.

    Returns the list of residents assigned to each hospital.
    """
    nh = len(hospital_prefs)
    rank = [{r: i for i, r in enumerate(prefs)} for prefs in hospital_prefs]
    nxt = [0] * len(resident_prefs)
    assigned: list[list[int]] = [[] for _ in range(nh)]
    free = list(range(len(resident_prefs) - 1, -1, -1))
    while free:
        r = free.pop()
        if nxt[r] >= len(resident_prefs[r]):
            continue
        h = resident_prefs[r][nxt[r]]
        nxt[r] += 1
        if r not in rank[h] or capacities[h] == 0:
            free.append(r)
            continue
        assigned[h].append(r)
        if len(assigned[h]) > capacities[h]:
            worst = max(assigned[h], key=lambda x: rank[h][x])
            assigned[h].remove(worst)
            free.append(worst)
    return [sorted(a) for a in assigned]


def blocking_pairs(
    match: list[int], proposer_prefs: list[list[int]], receiver_prefs: list[list[int]]
) -> list[tuple[int, int]]:
    """All (proposer, receiver) pairs that prefer each other to their partners."""
    n = len(proposer_prefs)
    prank = [{r: i for i, r in enumerate(p)} for p in proposer_prefs]
    rrank = [{p: i for i, p in enumerate(r)} for r in receiver_prefs]
    partner = [-1] * len(receiver_prefs)
    for p, r in enumerate(match):
        if r != -1:
            partner[r] = p
    out = []
    for p in range(n):
        for r in proposer_prefs[p]:
            if p not in rrank[r]:
                continue
            p_prefers = match[p] == -1 or prank[p][r] < prank[p][match[p]]
            r_prefers = partner[r] == -1 or rrank[r][p] < rrank[r][partner[r]]
            if p_prefers and r_prefers:
                out.append((p, r))
    return out


def all_stable_matchings(
    proposer_prefs: list[list[int]], receiver_prefs: list[list[int]]
) -> list[tuple[int, ...]]:
    """Enumerate every stable perfect matching by brute force (n <= 8)."""
    n = len(proposer_prefs)
    if n > 8:
        raise ValueError("brute-force enumeration limited to n <= 8")
    res = []
    for perm in permutations(range(n)):
        if not blocking_pairs(list(perm), proposer_prefs, receiver_prefs):
            res.append(perm)
    return res
