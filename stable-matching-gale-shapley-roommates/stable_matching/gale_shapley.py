"""Deferred acceptance for stable marriage (with incomplete lists) and hospitals/residents."""
from __future__ import annotations

from dataclasses import dataclass

from .preferences import Prefs, rank_table


@dataclass
class MatchingResult:
    """Result of a deferred-acceptance run.

    ``match`` maps proposer -> receiver (or -1 if unmatched); ``proposals`` counts proposals made.
    """
    match: list[int]
    proposals: int

    def inverse(self, n_receivers: int) -> list[int]:
        inv = [-1] * n_receivers
        for p, r in enumerate(self.match):
            if r >= 0:
                inv[r] = p
        return inv


def gale_shapley(proposers: Prefs, receivers: Prefs) -> MatchingResult:
    """Gale-Shapley deferred acceptance; proposer-optimal stable matching. O(total list length).

    Lists may be incomplete; a pair is acceptable only if each lists the other.
    """
    rrank = rank_table(receivers)
    nxt = [0] * len(proposers)
    holder = [-1] * len(receivers)
    match = [-1] * len(proposers)
    free = list(range(len(proposers) - 1, -1, -1))
    proposals = 0
    while free:
        p = free.pop()
        while nxt[p] < len(proposers[p]):
            r = proposers[p][nxt[p]]
            nxt[p] += 1
            proposals += 1
            if p not in rrank[r]:
                continue
            cur = holder[r]
            if cur == -1:
                holder[r], match[p] = p, r
                break
            if rrank[r][p] < rrank[r][cur]:
                holder[r], match[p], match[cur] = p, r, -1
                free.append(cur)
                break
    return MatchingResult(match, proposals)


def blocking_pairs(men: Prefs, women: Prefs, match: list[int]) -> list[tuple[int, int]]:
    """All (man, woman) pairs that mutually prefer each other to their current situation."""
    mr, wr = rank_table(men), rank_table(women)
    inv = [-1] * len(women)
    for m, w in enumerate(match):
        if w >= 0:
            inv[w] = m
    out = []
    for m, plist in enumerate(men):
        cur = mr[m][match[m]] if match[m] >= 0 else len(plist)
        for w in plist[:cur]:
            if m not in wr[w]:
                continue
            wcur = wr[w][inv[w]] if inv[w] >= 0 else len(women[w])
            if wr[w][m] < wcur:
                out.append((m, w))
    return out


def is_stable(men: Prefs, women: Prefs, match: list[int]) -> bool:
    """True iff ``match`` is a valid matching with no blocking pair."""
    mr, wr = rank_table(men), rank_table(women)
    seen = set()
    for m, w in enumerate(match):
        if w >= 0:
            if w in seen or w not in mr[m] or m not in wr[w]:
                return False
            seen.add(w)
    return not blocking_pairs(men, women, match)


def hospitals_residents(residents: Prefs, hospitals: Prefs, capacities: list[int],
                        resident_oriented: bool = True) -> list[list[int]]:
    """Many-to-one stable matching. Returns the list of assigned residents per hospital.

    Resident-oriented gives the resident-optimal stable assignment; hospital-oriented the
    hospital-optimal one.
    """
    if resident_oriented:
        hrank = rank_table(hospitals)
        nxt = [0] * len(residents)
        assigned: list[list[int]] = [[] for _ in hospitals]
        free = list(range(len(residents)))
        while free:
            r = free.pop()
            while nxt[r] < len(residents[r]):
                h = residents[r][nxt[r]]
                nxt[r] += 1
                if r not in hrank[h] or capacities[h] == 0:
                    continue
                assigned[h].append(r)
                if len(assigned[h]) > capacities[h]:
                    worst = max(assigned[h], key=lambda x: hrank[h][x])
                    assigned[h].remove(worst)
                    if worst == r:
                        continue
                    free.append(worst)
                break
        return [sorted(a, key=lambda x: hrank[h][x]) for h, a in enumerate(assigned)]
    rrank = rank_table(residents)
    nxt = [0] * len(hospitals)
    cur = [-1] * len(residents)
    count = [0] * len(hospitals)
    queue = [h for h in range(len(hospitals))]
    while queue:
        h = queue.pop()
        while count[h] < capacities[h] and nxt[h] < len(hospitals[h]):
            r = hospitals[h][nxt[h]]
            nxt[h] += 1
            if h not in rrank[r]:
                continue
            old = cur[r]
            if old == -1:
                cur[r] = h
                count[h] += 1
            elif rrank[r][h] < rrank[r][old]:
                cur[r] = h
                count[h] += 1
                count[old] -= 1
                queue.append(old)
    out: list[list[int]] = [[] for _ in hospitals]
    hrank = rank_table(hospitals)
    for r, h in enumerate(cur):
        if h >= 0:
            out[h].append(r)
    return [sorted(a, key=lambda x: hrank[h][x]) for h, a in enumerate(out)]


def hr_blocking_pairs(residents: Prefs, hospitals: Prefs, capacities: list[int],
                      assigned: list[list[int]]) -> list[tuple[int, int]]:
    """Blocking (resident, hospital) pairs of a many-to-one assignment."""
    rr, hr = rank_table(residents), rank_table(hospitals)
    where = {}
    for h, rs in enumerate(assigned):
        for r in rs:
            where[r] = h
    out = []
    for r, plist in enumerate(residents):
        cur = rr[r][where[r]] if r in where else len(plist)
        for h in plist[:cur]:
            if r not in hr[h]:
                continue
            if len(assigned[h]) < capacities[h]:
                out.append((r, h))
            elif capacities[h] > 0 and hr[h][r] < max(hr[h][x] for x in assigned[h]):
                out.append((r, h))
    return out
