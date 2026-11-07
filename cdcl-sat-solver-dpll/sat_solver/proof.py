"""Reverse unit propagation (RUP) proof checking for UNSAT answers."""
from __future__ import annotations

from .cnf import CNF, normalise_clause


class _Propagator:
    """Tiny counter-free unit propagator over a multiset of clauses."""

    def __init__(self) -> None:
        self.clauses: dict[tuple[int, ...], int] = {}

    def add(self, c: tuple[int, ...]) -> None:
        self.clauses[c] = self.clauses.get(c, 0) + 1

    def remove(self, c: tuple[int, ...]) -> None:
        if c in self.clauses:
            self.clauses[c] -= 1
            if self.clauses[c] == 0:
                del self.clauses[c]

    def conflicts_under(self, assumed: list[int]) -> bool:
        """True iff unit propagation from ``assumed`` derives a conflict."""
        assign: dict[int, bool] = {}
        for l in assumed:
            if assign.get(abs(l), l > 0) != (l > 0):
                return True
            assign[abs(l)] = l > 0
        changed = True
        while changed:
            changed = False
            for c in self.clauses:
                unassigned = None
                n_un = 0
                sat = False
                for l in c:
                    a = assign.get(abs(l))
                    if a is None:
                        n_un += 1
                        unassigned = l
                    elif a == (l > 0):
                        sat = True
                        break
                if sat:
                    continue
                if n_un == 0:
                    return True
                if n_un == 1:
                    assign[abs(unassigned)] = unassigned > 0
                    changed = True
        return False


def check_rup_proof(cnf: CNF, proof: list[tuple[str, tuple[int, ...]]]) -> bool:
    """Verify a DRUP proof of unsatisfiability.

    Each ``("a", clause)`` step must be implied by unit propagation over the
    formula plus earlier lemmas (negate the clause, propagate, expect a
    conflict). ``("d", clause)`` steps delete lemmas. The proof is accepted
    when the empty clause has been derived.
    """
    prop = _Propagator()
    for c in cnf.clauses:
        nc = normalise_clause(c)
        if nc is not None:
            prop.add(nc)
    derived_empty = False
    for kind, clause in proof:
        if kind == "d":
            prop.remove(tuple(sorted(clause, key=lambda l: (abs(l), l))))
            continue
        if not prop.conflicts_under([-l for l in clause]):
            return False
        if not clause:
            derived_empty = True
            break
        prop.add(tuple(sorted(clause, key=lambda l: (abs(l), l))))
    return derived_empty
