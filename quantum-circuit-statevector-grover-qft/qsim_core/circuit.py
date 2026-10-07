"""Gate-list circuit builder that executes on a :class:`StateVector`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import gates
from .state import StateVector, check_unitary


@dataclass(frozen=True)
class Op:
    """One circuit instruction."""

    name: str
    matrix: np.ndarray
    targets: tuple[int, ...]
    controls: tuple[int, ...] = ()

    @property
    def qubits(self) -> tuple[int, ...]:
        return self.controls + self.targets

    def inverse(self) -> "Op":
        return Op(self.name + "^-1", self.matrix.conj().T, self.targets, self.controls)


class Circuit:
    """A sequence of operations on ``n`` qubits."""

    def __init__(self, n_qubits: int) -> None:
        self.n = n_qubits
        self.ops: list[Op] = []

    def __len__(self) -> int:
        return len(self.ops)

    # -------------------------------------------------------------- core
    def add(
        self,
        name: str,
        matrix: np.ndarray,
        targets: Sequence[int],
        controls: Sequence[int] = (),
    ) -> "Circuit":
        qs = list(controls) + list(targets)
        if len(set(qs)) != len(qs) or any(not 0 <= q < self.n for q in qs):
            raise ValueError(f"bad qubits for {name}: {qs}")
        check_unitary(matrix)
        self.ops.append(Op(name, matrix, tuple(targets), tuple(controls)))
        return self

    # ------------------------------------------------------ gate sugar
    def h(self, q: int) -> "Circuit":
        return self.add("H", gates.H, [q])

    def x(self, q: int) -> "Circuit":
        return self.add("X", gates.X, [q])

    def y(self, q: int) -> "Circuit":
        return self.add("Y", gates.Y, [q])

    def z(self, q: int) -> "Circuit":
        return self.add("Z", gates.Z, [q])

    def s(self, q: int) -> "Circuit":
        return self.add("S", gates.S, [q])

    def t(self, q: int) -> "Circuit":
        return self.add("T", gates.T, [q])

    def p(self, theta: float, q: int) -> "Circuit":
        return self.add(f"P({theta:.3f})", gates.phase(theta), [q])

    def rx(self, theta: float, q: int) -> "Circuit":
        return self.add(f"RX({theta:.3f})", gates.rx(theta), [q])

    def ry(self, theta: float, q: int) -> "Circuit":
        return self.add(f"RY({theta:.3f})", gates.ry(theta), [q])

    def rz(self, theta: float, q: int) -> "Circuit":
        return self.add(f"RZ({theta:.3f})", gates.rz(theta), [q])

    def cx(self, c: int, t: int) -> "Circuit":
        return self.add("CX", gates.X, [t], [c])

    def cz(self, c: int, t: int) -> "Circuit":
        return self.add("CZ", gates.Z, [t], [c])

    def cp(self, theta: float, c: int, t: int) -> "Circuit":
        return self.add(f"CP({theta:.3f})", gates.phase(theta), [t], [c])

    def swap(self, a: int, b: int) -> "Circuit":
        return self.add("SWAP", gates.SWAP, [a, b])

    def ccx(self, c1: int, c2: int, t: int) -> "Circuit":
        return self.add("CCX", gates.X, [t], [c1, c2])

    def mcz(self, qubits: Sequence[int]) -> "Circuit":
        """Multi-controlled Z: flips the sign of the all-ones component."""
        qubits = list(qubits)
        return self.add("MCZ", gates.Z, [qubits[-1]], qubits[:-1])

    # --------------------------------------------------------- composition
    def compose(self, other: "Circuit", qubit_map: Sequence[int] | None = None) -> "Circuit":
        """Append ``other``, optionally remapping its qubits onto ours."""
        m = list(qubit_map) if qubit_map is not None else list(range(other.n))
        if len(m) != other.n:
            raise ValueError("qubit_map length mismatch")
        for op in other.ops:
            self.add(op.name, op.matrix, [m[q] for q in op.targets], [m[q] for q in op.controls])
        return self

    def inverse(self) -> "Circuit":
        out = Circuit(self.n)
        out.ops = [op.inverse() for op in reversed(self.ops)]
        return out

    # ----------------------------------------------------------- running
    def run(self, state: StateVector | None = None) -> StateVector:
        """Execute on ``state`` (a fresh |0..0> if omitted); returns the state."""
        st = state if state is not None else StateVector(self.n)
        if st.n != self.n:
            raise ValueError("state size mismatch")
        for op in self.ops:
            st.apply_controlled(op.matrix, op.controls, op.targets)
        return st

    def unitary(self) -> np.ndarray:
        """Full ``2^n x 2^n`` matrix, built by evolving each basis state."""
        dim = 2**self.n
        cols = []
        for j in range(dim):
            basis = np.zeros(dim, dtype=complex)
            basis[j] = 1.0
            st = StateVector.from_vector(basis)
            self.run(st)
            cols.append(st.vector)
        return np.stack(cols, axis=1)

    def depth(self) -> int:
        """Circuit depth under greedy as-soon-as-possible scheduling."""
        level = [0] * self.n
        for op in self.ops:
            d = max(level[q] for q in op.qubits) + 1
            for q in op.qubits:
                level[q] = d
        return max(level, default=0)

    def gate_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.ops:
            counts[op.name] = counts.get(op.name, 0) + 1
        return dict(sorted(counts.items()))
