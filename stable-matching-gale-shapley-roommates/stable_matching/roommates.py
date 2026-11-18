"""Irving's algorithm for the stable roommates problem."""
from __future__ import annotations

from .preferences import Prefs, rank_table


class _Table:
    """Mutable symmetric preference table with O(1) deletion of pairs."""

    def __init__(self, prefs: Prefs) -> None:
        self.rank = rank_table(prefs)
        self.alive = [list(p) for p in prefs]
        self.removed: list[set[int]] = [set() for _ in prefs]

    def delete(self, a: int, b: int) -> None:
        self.removed[a].add(b)
        self.removed[b].add(a)

    def lst(self, a: int) -> list[int]:
        return [x for x in self.alive[a] if x not in self.removed[a]]

    def first(self, a: int) -> int:
        return self.lst(a)[0]


def irving_roommates(prefs: Prefs) -> list[int] | None:
    """Return a stable perfect matching (partner list) or ``None`` if none exists.

    Phase 1 is a proposal sequence that trims lists; phase 2 repeatedly finds and
    eliminates rotations. Each person must rank all others strictly.
    """
    n = len(prefs)
    if n % 2:
        return None
    t = _Table(prefs)
    holder = [-1] * n  # holder[y] = person whose proposal y holds
    free = list(range(n - 1, -1, -1))
    while free:
        x = free.pop()
        lx = t.lst(x)
        if not lx:
            return None
        y = lx[0]
        z = holder[y]
        if z != -1:
            t.delete(y, z)
            free.append(z)
        holder[y] = x
        # y keeps only people at least as good as x
        pos = t.rank[y][x]
        for w in prefs[y]:
            if t.rank[y][w] > pos:
                t.delete(y, w)
    if any(not t.lst(i) for i in range(n)):
        return None
    # Phase 2
    while True:
        start = next((i for i in range(n) if len(t.lst(i)) > 1), None)
        if start is None:
            break
        seq: list[int] = []
        seen: dict[int, int] = {}
        p = start
        while p not in seen:
            seen[p] = len(seq)
            seq.append(p)
            q = t.lst(p)[1]
            p = t.lst(q)[-1]
        cycle = seq[seen[p]:]
        pairs = []
        for pi in cycle:
            q = t.lst(pi)[1]
            pairs.append((q, t.lst(q)[-1]))
        for q, last in pairs:
            t.delete(q, last)
        for pi in cycle:  # each p_i now proposes to its new first choice, which trims its list
            lst = t.lst(pi)
            if lst:
                y = lst[0]
                pos = t.rank[y][pi]
                for w in prefs[y]:
                    if t.rank[y][w] > pos:
                        t.delete(y, w)
        if any(not t.lst(i) for i in range(n)):
            return None
    match = [t.lst(i)[0] for i in range(n)]
    if any(match[match[i]] != i for i in range(n)):
        return None
    return match


def roommates_blocking_pairs(prefs: Prefs, match: list[int]) -> list[tuple[int, int]]:
    """Pairs of people who both prefer each other to their assigned partners."""
    rk = rank_table(prefs)
    out = []
    for a in range(len(prefs)):
        for b in range(a + 1, len(prefs)):
            if match[a] == b:
                continue
            if rk[a][b] < rk[a][match[a]] and rk[b][a] < rk[b][match[b]]:
                out.append((a, b))
    return out
