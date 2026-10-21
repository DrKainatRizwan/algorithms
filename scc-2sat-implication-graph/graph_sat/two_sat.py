"""2-SAT via the implication graph and SCCs (Aspvall-Plass-Tarjan 1979).

Literal encoding: variable ``i`` (0-based) has positive literal node ``2*i``
and negative literal node ``2*i+1``; negation is ``x ^ 1``.  The public API
uses signed 1-based integers (DIMACS style): ``+3`` is x3, ``-3`` is NOT x3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .digraph import Digraph
from .scc import tarjan_scc, condensation


def lit_node(lit: int) -> int:
    """Signed DIMACS literal -> implication-graph node."""
    if lit == 0:
        raise ValueError("0 is not a valid literal")
    v = abs(lit) - 1
    return 2 * v + (1 if lit < 0 else 0)


def node_lit(node: int) -> int:
    v = node // 2 + 1
    return -v if node & 1 else v


@dataclass
class SatResult:
    satisfiable: bool
    assignment: dict[int, bool] | None = None
    # When unsatisfiable: a variable x with x ~> -x and -x ~> x, and the
    # two witnessing implication chains as literal lists.
    conflict_var: int | None = None
    proof: tuple[list[int], list[int]] | None = field(default=None, repr=False)


class TwoSat:
    """Incremental 2-SAT instance over variables ``1..n``."""

    def __init__(self, n_vars: int) -> None:
        self.n_vars = n_vars
        self.graph = Digraph(2 * n_vars)
        self.clauses: list[tuple[int, int]] = []

    # ---- building -------------------------------------------------------
    def _check(self, lit: int) -> None:
        if lit == 0 or abs(lit) > self.n_vars:
            raise ValueError(f"literal {lit} out of range for {self.n_vars} variables")

    def add_clause(self, a: int, b: int) -> None:
        """Add clause (a OR b)."""
        self._check(a)
        self._check(b)
        self.clauses.append((a, b))
        self.graph.add_edge(lit_node(a) ^ 1, lit_node(b))
        self.graph.add_edge(lit_node(b) ^ 1, lit_node(a))

    def add_unit(self, a: int) -> None:
        """Force literal ``a`` true."""
        self.add_clause(a, a)

    def add_implication(self, a: int, b: int) -> None:
        self.add_clause(-a, b)

    def add_equiv(self, a: int, b: int) -> None:
        self.add_implication(a, b)
        self.add_implication(b, a)

    def add_xor(self, a: int, b: int) -> None:
        """a XOR b (exactly one true)."""
        self.add_clause(a, b)
        self.add_clause(-a, -b)

    def add_at_most_one(self, lits: Iterable[int]) -> None:
        """Pairwise at-most-one constraint."""
        lits = list(lits)
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                self.add_clause(-lits[i], -lits[j])

    # ---- solving --------------------------------------------------------
    def solve(self, assumptions: Iterable[int] = ()) -> SatResult:
        """Solve, optionally under unit ``assumptions``.

        Assumptions are handled by adding temporary unit clauses on a copy.
        """
        assumptions = list(assumptions)
        if assumptions:
            sub = TwoSat(self.n_vars)
            for a, b in self.clauses:
                sub.add_clause(a, b)
            for a in assumptions:
                sub.add_unit(a)
            return sub.solve()
        g = self.graph
        comp = tarjan_scc(g)
        for v in range(self.n_vars):
            if comp[2 * v] == comp[2 * v + 1]:
                return SatResult(False, None, v + 1, self._proof(v + 1))
        # Tarjan ids are in reverse topological order: a literal whose
        # component id is smaller lies later in topological order, so it is
        # set true when comp[x] < comp[not x].
        assignment = {v + 1: comp[2 * v] < comp[2 * v + 1] for v in range(self.n_vars)}
        return SatResult(True, assignment)

    def _proof(self, var: int) -> tuple[list[int], list[int]]:
        p, q = lit_node(var), lit_node(-var)
        a = self.graph.shortest_path(p, q)
        b = self.graph.shortest_path(q, p)
        assert a is not None and b is not None
        return [node_lit(x) for x in a], [node_lit(x) for x in b]

    def is_satisfiable(self) -> bool:
        return self.solve().satisfiable

    def evaluate(self, assignment: dict[int, bool]) -> bool:
        def val(l: int) -> bool:
            return assignment[abs(l)] == (l > 0)
        return all(val(a) or val(b) for a, b in self.clauses)

    def implied_literals(self, lit: int) -> set[int]:
        """All literals forced true whenever ``lit`` is true."""
        return {node_lit(x) for x in self.graph.reachable_from(lit_node(lit))}

    def backbone(self) -> set[int] | None:
        """Literals true in every satisfying assignment (``None`` if UNSAT).

        Literal ``l`` is in the backbone iff the formula with ``-l`` assumed
        is unsatisfiable, i.e. iff ``-l ~> l`` in the implication graph or
        ``-l`` reaches a complementary pair.
        """
        if not self.is_satisfiable():
            return None
        out: set[int] = set()
        for v in range(1, self.n_vars + 1):
            for l in (v, -v):
                if not self.solve([-l]).satisfiable:
                    out.add(l)
        return out

    def enumerate_solutions(self, limit: int | None = None) -> Iterator[dict[int, bool]]:
        """Enumerate all models by branching on variables (small instances)."""
        count = 0
        stack: list[list[int]] = [[]]
        while stack:
            fixed = stack.pop()
            if not self.solve(fixed).satisfiable:
                continue
            if len(fixed) == self.n_vars:
                yield {abs(l): l > 0 for l in fixed}
                count += 1
                if limit is not None and count >= limit:
                    return
                continue
            v = len(fixed) + 1
            stack.append(fixed + [-v])
            stack.append(fixed + [v])

    def count_solutions_dp(self) -> int:
        """Model count by exhaustive enumeration (exponential, small n)."""
        return sum(1 for _ in self.enumerate_solutions())

    def equivalence_classes(self) -> list[list[int]]:
        """Groups of literals that are logically equivalent (same SCC)."""
        comp = tarjan_scc(self.graph)
        groups: dict[int, list[int]] = {}
        for node, c in enumerate(comp):
            groups.setdefault(c, []).append(node_lit(node))
        return [sorted(g, key=abs) for g in groups.values() if len(g) > 1]

    def condensed(self) -> tuple[Digraph, list[int]]:
        return condensation(self.graph)

    def to_dimacs(self) -> str:
        lines = [f"p cnf {self.n_vars} {len(self.clauses)}"]
        lines += [f"{a} {b} 0" for a, b in self.clauses]
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dimacs(cls, text: str) -> "TwoSat":
        n_vars = 0
        clauses: list[tuple[int, int]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line[0] == "c":
                continue
            parts = line.split()
            if parts[0] == "p":
                n_vars = int(parts[2])
                continue
            lits = [int(x) for x in parts if x != "0"]
            if len(lits) == 1:
                lits = [lits[0], lits[0]]
            if len(lits) != 2:
                raise ValueError(f"not a 2-clause: {line!r}")
            clauses.append((lits[0], lits[1]))
        inst = cls(n_vars)
        for a, b in clauses:
            inst.add_clause(a, b)
        return inst
