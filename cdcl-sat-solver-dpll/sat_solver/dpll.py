"""Classic DPLL baseline: unit propagation, pure literals, branching heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from .cnf import CNF, normalise_clause


@dataclass
class DPLLResult:
    status: str  # "SAT", "UNSAT" or "UNKNOWN"
    model: dict[int, bool] | None
    decisions: int = 0
    propagations: int = 0
    backtracks: int = 0


class _Budget(Exception):
    pass


class DPLLSolver:
    """Recursive DPLL with chronological backtracking.

    ``heuristic`` is one of ``"first"`` (smallest variable), ``"moms"``
    (maximum occurrences in minimum-size clauses) or ``"jw"``
    (two-sided Jeroslow-Wang weights).
    """

    def __init__(self, cnf: CNF, heuristic: str = "jw", max_decisions: int | None = None):
        if heuristic not in ("first", "moms", "jw"):
            raise ValueError(f"unknown heuristic {heuristic!r}")
        self.cnf = cnf
        self.heuristic = heuristic
        self.max_decisions = max_decisions
        self.decisions = self.propagations = self.backtracks = 0

    # -- simplification -----------------------------------------------------
    @staticmethod
    def _assign(clauses: list[tuple[int, ...]], lit: int):
        """Set ``lit`` true. Returns new clause list or ``None`` on conflict."""
        out = []
        for c in clauses:
            if lit in c:
                continue
            if -lit in c:
                c = tuple(l for l in c if l != -lit)
                if not c:
                    return None
            out.append(c)
        return out

    def _propagate(self, clauses, model):
        while True:
            unit = None
            for c in clauses:
                if len(c) == 1:
                    unit = c[0]
                    break
            if unit is None:
                pol: dict[int, set[bool]] = {}
                for c in clauses:
                    for l in c:
                        pol.setdefault(abs(l), set()).add(l > 0)
                for v, s in pol.items():
                    if len(s) == 1:
                        unit = v if True in s else -v
                        break
                if unit is None:
                    return clauses
            else:
                self.propagations += 1
            model[abs(unit)] = unit > 0
            clauses = self._assign(clauses, unit)
            if clauses is None:
                return None

    def _pick(self, clauses) -> int:
        if self.heuristic == "first":
            return min(abs(l) for c in clauses for l in c)
        if self.heuristic == "moms":
            m = min(len(c) for c in clauses)
            cnt = Counter(l for c in clauses if len(c) == m for l in c)
            return max(cnt, key=lambda l: (cnt[l], -abs(l), l > 0))
        w: Counter = Counter()
        for c in clauses:
            for l in c:
                w[l] += 2.0 ** -len(c)
        return max(w, key=lambda l: (w[l], -abs(l), l > 0))

    def _search(self, clauses, model):
        clauses = self._propagate(clauses, model)
        if clauses is None:
            self.backtracks += 1
            return False
        if not clauses:
            return True
        lit = self._pick(clauses)
        for choice in (lit, -lit):
            self.decisions += 1
            if self.max_decisions is not None and self.decisions > self.max_decisions:
                raise _Budget
            sub = self._assign(clauses, choice)
            if sub is None:
                self.backtracks += 1
                continue
            m2 = dict(model)
            m2[abs(choice)] = choice > 0
            if self._search(sub, m2):
                model.update(m2)
                return True
        return False

    def solve(self) -> DPLLResult:
        clauses = []
        for c in self.cnf.clauses:
            n = normalise_clause(c)
            if n is None:
                continue
            if not n:
                return DPLLResult("UNSAT", None)
            clauses.append(n)
        model: dict[int, bool] = {}
        try:
            ok = self._search(clauses, model)
        except _Budget:
            return DPLLResult("UNKNOWN", None, self.decisions, self.propagations, self.backtracks)
        if ok:
            for v in range(1, self.cnf.num_vars + 1):
                model.setdefault(v, False)
        return DPLLResult("SAT" if ok else "UNSAT", model if ok else None,
                          self.decisions, self.propagations, self.backtracks)
