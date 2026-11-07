"""Conflict-driven clause learning (CDCL) SAT solver.

Features: two-watched-literal propagation, first-UIP clause learning with
local minimisation, VSIDS branching (lazy heap), phase saving, Luby or
geometric restarts, and LBD-based learned-clause database reduction.
Every learned clause can be logged as a DRUP-style proof (see ``proof.py``).
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field

from .cnf import CNF, normalise_clause


def luby(i: int) -> int:
    """The i-th (1-based) element of the Luby sequence 1,1,2,1,1,2,4,..."""
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i -= (1 << (k - 1)) - 1
        k = 1
        while (1 << k) - 1 < i:
            k += 1


@dataclass
class Stats:
    decisions: int = 0
    propagations: int = 0
    conflicts: int = 0
    restarts: int = 0
    learned: int = 0
    deleted: int = 0
    minimised_lits: int = 0


@dataclass
class CDCLResult:
    status: str  # "SAT", "UNSAT" or "UNKNOWN"
    model: dict[int, bool] | None
    stats: Stats = field(default_factory=Stats)
    proof: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)


def _ext(l: int) -> int:
    """Internal literal index -> DIMACS literal."""
    v = (l >> 1) + 1
    return -v if l & 1 else v


def _int(l: int) -> int:
    """DIMACS literal -> internal literal index (2*(v-1) + sign)."""
    return 2 * (abs(l) - 1) + (1 if l < 0 else 0)


class CDCLSolver:
    """CDCL solver over a :class:`~sat_solver.cnf.CNF`.

    Parameters
    ----------
    restart:
        ``"luby"`` (default), ``"geometric"`` or ``"none"``.
    restart_base:
        Conflicts in the first restart interval.
    var_decay:
        VSIDS decay factor (activity increment grows by ``1/var_decay``).
    reduce_first:
        Learned-clause count that triggers the first database reduction.
    log_proof:
        Record learned/deleted clauses so the result can be RUP-checked.
    """

    def __init__(self, cnf: CNF, *, seed: int = 0, restart: str = "luby",
                 restart_base: int = 32, var_decay: float = 0.95,
                 reduce_first: int = 2000, minimise: bool = True,
                 phase_saving: bool = True, log_proof: bool = False):
        if restart not in ("luby", "geometric", "none"):
            raise ValueError(f"unknown restart policy {restart!r}")
        self.n = cnf.num_vars
        self.restart = restart
        self.restart_base = restart_base
        self.var_decay = var_decay
        self.reduce_limit = reduce_first
        self.minimise = minimise
        self.phase_saving = phase_saving
        self.log_proof = log_proof
        self.stats = Stats()
        self.proof: list[tuple[str, tuple[int, ...]]] = []

        rng = random.Random(seed)
        self.vals = [0] * (2 * self.n)          # per literal: 1 true, -1 false
        self.level = [0] * self.n
        self.reason: list[int] = [-1] * self.n
        self.trail: list[int] = []
        self.trail_lim: list[int] = []
        self.qhead = 0
        self.watches: list[list[int]] = [[] for _ in range(2 * self.n)]
        self.clauses: list[list[int] | None] = []
        self.learnt_idx: list[int] = []
        self.lbd: dict[int, int] = {}
        self.activity = [rng.random() * 1e-5 for _ in range(self.n)]
        self.var_inc = 1.0
        self.phase = [1] * self.n               # saved sign bit (1 = negative)
        self.heap = [(-a, v) for v, a in enumerate(self.activity)]
        heapq.heapify(self.heap)
        self.ok = True

        for c in cnf.clauses:
            if not self.ok:
                break
            self._add_input_clause(c)

    # -- clause management ---------------------------------------------------
    def _add_input_clause(self, c) -> None:
        nc = normalise_clause(c)
        if nc is None:
            return
        lits = [_int(l) for l in nc]
        if not lits:
            self.ok = False
        elif len(lits) == 1:
            if self.vals[lits[0]] == -1:
                self.ok = False
            elif self.vals[lits[0]] == 0:
                self._enqueue(lits[0], -1)
        else:
            self._attach(lits)

    def _attach(self, lits: list[int]) -> int:
        ci = len(self.clauses)
        self.clauses.append(lits)
        self.watches[lits[0]].append(ci)
        self.watches[lits[1]].append(ci)
        return ci

    def _enqueue(self, lit: int, reason: int) -> None:
        self.vals[lit] = 1
        self.vals[lit ^ 1] = -1
        v = lit >> 1
        self.level[v] = len(self.trail_lim)
        self.reason[v] = reason
        self.trail.append(lit)

    # -- propagation -----------------------------------------------------------
    def _propagate(self) -> int:
        """Unit propagation; returns a conflicting clause index or -1."""
        vals, clauses, watches = self.vals, self.clauses, self.watches
        while self.qhead < len(self.trail):
            p = self.trail[self.qhead]
            self.qhead += 1
            self.stats.propagations += 1
            fl = p ^ 1
            ws = watches[fl]
            i = j = 0
            n = len(ws)
            while i < n:
                ci = ws[i]
                i += 1
                c = clauses[ci]
                if c is None:
                    continue
                if c[0] == fl:
                    c[0], c[1] = c[1], c[0]
                if vals[c[0]] == 1:
                    ws[j] = ci
                    j += 1
                    continue
                for k in range(2, len(c)):
                    if vals[c[k]] != -1:
                        c[1], c[k] = c[k], c[1]
                        watches[c[1]].append(ci)
                        break
                else:
                    ws[j] = ci
                    j += 1
                    if vals[c[0]] == -1:
                        while i < n:
                            ws[j] = ws[i]
                            j += 1
                            i += 1
                        del ws[j:]
                        self.qhead = len(self.trail)
                        return ci
                    self._enqueue(c[0], ci)
            del ws[j:]
        return -1

    # -- conflict analysis -----------------------------------------------------
    def _bump(self, v: int) -> None:
        self.activity[v] += self.var_inc
        if self.activity[v] > 1e100:
            self.activity = [a * 1e-100 for a in self.activity]
            self.var_inc *= 1e-100
            self.heap = [(-self.activity[u], u) for u in range(self.n) if self.vals[2 * u] == 0]
            heapq.heapify(self.heap)
        elif self.vals[2 * v] == 0:
            heapq.heappush(self.heap, (-self.activity[v], v))

    def _analyze(self, confl: int) -> tuple[list[int], int]:
        """First-UIP learning. Returns (learned clause, backjump level)."""
        seen = [False] * self.n
        learnt = [0]
        counter = 0
        p = -1
        idx = len(self.trail) - 1
        cur = len(self.trail_lim)
        while True:
            for q in self.clauses[confl]:
                if q == p:
                    continue
                v = q >> 1
                if not seen[v] and self.level[v] > 0:
                    self._bump(v)
                    seen[v] = True
                    if self.level[v] >= cur:
                        counter += 1
                    else:
                        learnt.append(q)
            while not seen[self.trail[idx] >> 1]:
                idx -= 1
            p = self.trail[idx]
            idx -= 1
            confl = self.reason[p >> 1]
            seen[p >> 1] = False
            counter -= 1
            if counter == 0:
                break
        learnt[0] = p ^ 1

        if self.minimise and len(learnt) > 2:
            keep = [learnt[0]]
            for q in learnt[1:]:
                r = self.reason[q >> 1]
                if r < 0:
                    keep.append(q)
                    continue
                redundant = all(
                    (x >> 1) == (q >> 1) or seen[x >> 1] or self.level[x >> 1] == 0
                    for x in self.clauses[r])
                if redundant:
                    self.stats.minimised_lits += 1
                else:
                    keep.append(q)
            learnt = keep

        if len(learnt) == 1:
            return learnt, 0
        best = max(range(1, len(learnt)), key=lambda k: self.level[learnt[k] >> 1])
        learnt[1], learnt[best] = learnt[best], learnt[1]
        return learnt, self.level[learnt[1] >> 1]

    def _backtrack(self, lvl: int) -> None:
        if len(self.trail_lim) <= lvl:
            return
        stop = self.trail_lim[lvl]
        for lit in reversed(self.trail[stop:]):
            v = lit >> 1
            if self.phase_saving:
                self.phase[v] = lit & 1
            self.vals[lit] = self.vals[lit ^ 1] = 0
            self.reason[v] = -1
            heapq.heappush(self.heap, (-self.activity[v], v))
        del self.trail[stop:]
        del self.trail_lim[lvl:]
        self.qhead = stop

    def _pick_branch(self) -> int:
        while self.heap:
            a, v = heapq.heappop(self.heap)
            if self.vals[2 * v] == 0 and -a == self.activity[v]:
                return 2 * v + self.phase[v]
        return -1

    # -- database reduction ----------------------------------------------------
    def _reduce_db(self) -> None:
        locked = set()
        for lit in self.trail:
            r = self.reason[lit >> 1]
            if r >= 0:
                locked.add(r)
        cand = [ci for ci in self.learnt_idx
                if ci not in locked and self.lbd[ci] > 2 and len(self.clauses[ci]) > 2]
        cand.sort(key=lambda ci: (self.lbd[ci], len(self.clauses[ci])), reverse=True)
        drop = set(cand[: len(cand) // 2])
        for ci in drop:
            if self.log_proof:
                self.proof.append(("d", tuple(_ext(l) for l in self.clauses[ci])))
            self.clauses[ci] = None
            del self.lbd[ci]
        self.learnt_idx = [ci for ci in self.learnt_idx if ci not in drop]
        self.stats.deleted += len(drop)

    # -- main loop -------------------------------------------------------------
    def solve(self, max_conflicts: int | None = None) -> CDCLResult:
        if not self.ok or self._propagate() >= 0:
            if self.log_proof:
                self.proof.append(("a", ()))
            return CDCLResult("UNSAT", None, self.stats, self.proof)

        restart_no = 0
        budget = self._next_interval(restart_no)
        since = 0
        while True:
            confl = self._propagate()
            if confl >= 0:
                self.stats.conflicts += 1
                since += 1
                if not self.trail_lim:
                    if self.log_proof:
                        self.proof.append(("a", ()))
                    return CDCLResult("UNSAT", None, self.stats, self.proof)
                learnt, bt = self._analyze(confl)
                self._backtrack(bt)
                if self.log_proof:
                    self.proof.append(("a", tuple(_ext(l) for l in learnt)))
                if len(learnt) == 1:
                    self._enqueue(learnt[0], -1)
                else:
                    ci = self._attach(learnt)
                    self.learnt_idx.append(ci)
                    self.lbd[ci] = len({self.level[l >> 1] for l in learnt})
                    self._enqueue(learnt[0], ci)
                self.stats.learned += 1
                self.var_inc /= self.var_decay
                if max_conflicts is not None and self.stats.conflicts >= max_conflicts:
                    return CDCLResult("UNKNOWN", None, self.stats, self.proof)
            else:
                if budget is not None and since >= budget:
                    self.stats.restarts += 1
                    restart_no += 1
                    budget = self._next_interval(restart_no)
                    since = 0
                    self._backtrack(0)
                    continue
                if len(self.learnt_idx) >= self.reduce_limit:
                    self._reduce_db()
                    self.reduce_limit = int(self.reduce_limit * 1.1) + 1
                lit = self._pick_branch()
                if lit < 0:
                    model = {v + 1: self.vals[2 * v] == 1 for v in range(self.n)}
                    return CDCLResult("SAT", model, self.stats, self.proof)
                self.stats.decisions += 1
                self.trail_lim.append(len(self.trail))
                self._enqueue(lit, -1)

    def _next_interval(self, k: int) -> int | None:
        if self.restart == "none":
            return None
        if self.restart == "luby":
            return self.restart_base * luby(k + 1)
        return int(self.restart_base * 1.5 ** k)


def solve(cnf: CNF, **kw) -> CDCLResult:
    """Convenience wrapper: ``solve(cnf, seed=1, log_proof=True)``."""
    max_conflicts = kw.pop("max_conflicts", None)
    return CDCLSolver(cnf, **kw).solve(max_conflicts)
