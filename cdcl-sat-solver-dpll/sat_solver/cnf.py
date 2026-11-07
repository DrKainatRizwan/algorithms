"""CNF formulas, DIMACS I/O and model checking.

Literals follow the DIMACS convention: variable ``v`` (1-based) is the
positive literal ``v`` and its negation is ``-v``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Clause = tuple[int, ...]


def normalise_clause(lits: list[int] | tuple[int, ...]) -> Clause | None:
    """Sort/deduplicate literals. Returns ``None`` for tautologies."""
    s = set(lits)
    if any(-l in s for l in s):
        return None
    return tuple(sorted(s, key=lambda l: (abs(l), l)))


@dataclass
class CNF:
    """A propositional formula in conjunctive normal form."""

    num_vars: int
    clauses: list[Clause] = field(default_factory=list)

    def add_clause(self, lits: list[int] | tuple[int, ...]) -> None:
        for l in lits:
            if l == 0:
                raise ValueError("0 is not a valid literal")
            self.num_vars = max(self.num_vars, abs(l))
        self.clauses.append(tuple(lits))

    @property
    def num_clauses(self) -> int:
        return len(self.clauses)

    def copy(self) -> "CNF":
        return CNF(self.num_vars, list(self.clauses))

    def to_dimacs(self) -> str:
        out = [f"p cnf {self.num_vars} {len(self.clauses)}"]
        out += [" ".join(map(str, c)) + " 0" for c in self.clauses]
        return "\n".join(out) + "\n"

    @classmethod
    def from_dimacs(cls, text: str) -> "CNF":
        """Parse DIMACS CNF text (comments and multi-line clauses allowed)."""
        num_vars = 0
        declared = None
        cnf = cls(0)
        cur: list[int] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line[0] in "c%":
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) != 4 or parts[1] != "cnf":
                    raise ValueError(f"bad problem line: {line!r}")
                num_vars, declared = int(parts[2]), int(parts[3])
                continue
            for tok in line.split():
                v = int(tok)
                if v == 0:
                    cnf.add_clause(cur)
                    cur = []
                else:
                    cur.append(v)
        if cur:
            raise ValueError("last clause not terminated by 0")
        cnf.num_vars = max(cnf.num_vars, num_vars)
        if declared is not None and declared != len(cnf.clauses):
            raise ValueError(
                f"declared {declared} clauses but found {len(cnf.clauses)}")
        return cnf


def evaluate(cnf: CNF, model: dict[int, bool]) -> bool:
    """True iff ``model`` (var -> bool) satisfies every clause."""
    for c in cnf.clauses:
        if not any(model.get(abs(l), False) == (l > 0) for l in c):
            return False
    return True


def count_models(cnf: CNF) -> int:
    """Brute-force model counter (only for tiny formulas; used in tests)."""
    n = cnf.num_vars
    if n > 22:
        raise ValueError("too many variables for brute force")
    total = 0
    for bits in range(1 << n):
        model = {v + 1: bool(bits >> v & 1) for v in range(n)}
        if evaluate(cnf, model):
            total += 1
    return total
