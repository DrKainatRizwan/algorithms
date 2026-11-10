"""Clifford circuit container, generators and runners for both back-ends."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .statevector import StateVector
from .tableau import Tableau

_ONE = {"h", "s", "sdg", "x", "y", "z"}
_TWO = {"cx", "cz", "swap"}
_INVERSE = {"h": "h", "s": "sdg", "sdg": "s", "x": "x", "y": "y", "z": "z",
            "cx": "cx", "cz": "cz", "swap": "swap"}


@dataclass
class Circuit:
    """Ordered list of ``(gate, qubits)`` Clifford operations."""

    n: int
    ops: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    def add(self, name: str, *qubits: int) -> "Circuit":
        arity = 1 if name in _ONE else 2 if name in _TWO else None
        if arity is None:
            raise ValueError(f"unknown gate {name!r}")
        if len(qubits) != arity or any(not 0 <= q < self.n for q in qubits):
            raise ValueError(f"bad qubits {qubits} for gate {name}")
        if arity == 2 and qubits[0] == qubits[1]:
            raise ValueError("two-qubit gate needs distinct qubits")
        self.ops.append((name, tuple(qubits)))
        return self

    def h(self, q): return self.add("h", q)
    def s(self, q): return self.add("s", q)
    def x(self, q): return self.add("x", q)
    def z(self, q): return self.add("z", q)
    def cx(self, a, b): return self.add("cx", a, b)
    def cz(self, a, b): return self.add("cz", a, b)

    def inverse(self) -> "Circuit":
        return Circuit(self.n, [(_INVERSE[g], q) for g, q in reversed(self.ops)])

    def __add__(self, other: "Circuit") -> "Circuit":
        if other.n != self.n:
            raise ValueError("qubit counts differ")
        return Circuit(self.n, self.ops + other.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def run_tableau(self, t: Tableau | None = None) -> Tableau:
        t = Tableau(self.n) if t is None else t
        for name, q in self.ops:
            fn = {"x": t.x_gate, "y": t.y_gate, "z": t.z_gate}.get(name) or getattr(t, name)
            fn(*q)
        return t

    def run_statevector(self) -> StateVector:
        sv = StateVector(self.n)
        for name, q in self.ops:
            if name in _ONE:
                sv.apply1(name, *q)
            else:
                getattr(sv, name)(*q)
        return sv

    def sample(self, shots: int, rng: np.random.Generator) -> dict[str, int]:
        """Measure all qubits ``shots`` times; keys are bitstrings (qubit 0 first)."""
        base = self.run_tableau()
        counts: dict[str, int] = {}
        for _ in range(shots):
            t = base.copy()
            key = "".join(str(t.measure(q, rng)) for q in range(self.n))
            counts[key] = counts.get(key, 0) + 1
        return counts


def ghz_circuit(n: int) -> Circuit:
    """Prepare (|0..0> + |1..1>)/sqrt(2)."""
    c = Circuit(n).h(0)
    for q in range(1, n):
        c.cx(q - 1, q)
    return c


def random_clifford_circuit(n: int, depth: int, rng: np.random.Generator) -> Circuit:
    """Layers of random single-qubit Cliffords followed by random CX pairings."""
    c = Circuit(n)
    for _ in range(depth):
        for q in range(n):
            for g in rng.choice(["h", "s", "x", "z"], size=int(rng.integers(0, 3))):
                c.add(str(g), q)
        if n >= 2:
            perm = rng.permutation(n)
            for i in range(0, n - 1, 2):
                c.cx(int(perm[i]), int(perm[i + 1]))
    return c
