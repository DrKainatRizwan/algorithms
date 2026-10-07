"""State-vector container with in-place gate application.

The state of ``n`` qubits is stored as a tensor of shape ``(2,)*n``.  Qubit 0
is the most significant bit of the basis-state index (big-endian), so the
flattened vector matches the usual ``|q0 q1 ... q_{n-1}>`` ordering.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .gates import is_unitary


class StateVector:
    """Pure state of ``n`` qubits, initialised to ``|0...0>``."""

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("need at least one qubit")
        self.n = n_qubits
        self.tensor = np.zeros((2,) * n_qubits, dtype=complex)
        self.tensor[(0,) * n_qubits] = 1.0

    # ------------------------------------------------------------------ io
    @classmethod
    def from_vector(cls, vec: Sequence[complex]) -> "StateVector":
        v = np.asarray(vec, dtype=complex)
        n = int(round(np.log2(v.size)))
        if 2**n != v.size:
            raise ValueError("length must be a power of two")
        if not np.isclose(np.linalg.norm(v), 1.0):
            raise ValueError("state must be normalised")
        s = cls(n)
        s.tensor = v.reshape((2,) * n).copy()
        return s

    @property
    def vector(self) -> np.ndarray:
        return self.tensor.reshape(-1)

    def copy(self) -> "StateVector":
        s = StateVector(self.n)
        s.tensor = self.tensor.copy()
        return s

    # --------------------------------------------------------------- gates
    def _check(self, qubits: Sequence[int]) -> None:
        if len(set(qubits)) != len(qubits):
            raise ValueError("repeated qubit index")
        for q in qubits:
            if not 0 <= q < self.n:
                raise ValueError(f"qubit {q} out of range")

    def apply(self, u: np.ndarray, qubits: Sequence[int]) -> None:
        """Apply a ``2^k x 2^k`` unitary to the listed qubits (in order)."""
        qubits = list(qubits)
        k = len(qubits)
        self._check(qubits)
        if u.shape != (2**k, 2**k):
            raise ValueError("matrix size does not match qubit count")
        ut = u.reshape((2,) * (2 * k))
        out = np.tensordot(ut, self.tensor, axes=(list(range(k, 2 * k)), qubits))
        self.tensor = np.moveaxis(out, list(range(k)), qubits)

    def apply_controlled(
        self, u: np.ndarray, controls: Sequence[int], targets: Sequence[int]
    ) -> None:
        """Apply ``u`` to ``targets`` when every control qubit is 1.

        Works on a slice of the tensor, so cost is ``2^(n - #controls)``.
        """
        controls, targets = list(controls), list(targets)
        self._check(controls + targets)
        if not controls:
            self.apply(u, targets)
            return
        idx = [slice(None)] * self.n
        for c in controls:
            idx[c] = 1
        sub = self.tensor[tuple(idx)]
        # axes of ``sub`` are the non-control qubits in increasing order
        remaining = [q for q in range(self.n) if q not in controls]
        pos = [remaining.index(t) for t in targets]
        k = len(pos)
        ut = u.reshape((2,) * (2 * k))
        out = np.tensordot(ut, sub, axes=(list(range(k, 2 * k)), pos))
        self.tensor[tuple(idx)] = np.moveaxis(out, list(range(k)), pos)

    # -------------------------------------------------------- measurement
    def probabilities(self, qubits: Sequence[int] | None = None) -> np.ndarray:
        """Marginal outcome probabilities over ``qubits`` (default: all)."""
        p = np.abs(self.tensor) ** 2
        if qubits is None:
            return p.reshape(-1)
        qubits = list(qubits)
        self._check(qubits)
        other = tuple(q for q in range(self.n) if q not in qubits)
        m = p.sum(axis=other) if other else p
        # remaining axes are in increasing qubit order; reorder to requested
        order = sorted(qubits)
        m = np.transpose(m, [order.index(q) for q in qubits])
        return m.reshape(-1)

    def sample(self, shots: int, rng: np.random.Generator) -> dict[str, int]:
        """Sample computational-basis bitstrings."""
        p = self.probabilities()
        p = p / p.sum()
        draws = rng.choice(p.size, size=shots, p=p)
        counts: dict[str, int] = {}
        for d in draws:
            key = format(int(d), f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def measure(self, qubit: int, rng: np.random.Generator) -> int:
        """Projectively measure one qubit and collapse the state."""
        self._check([qubit])
        p1 = float(self.probabilities([qubit])[1])
        outcome = int(rng.random() < p1)
        idx = [slice(None)] * self.n
        idx[qubit] = 1 - outcome
        self.tensor[tuple(idx)] = 0.0
        self.tensor /= np.linalg.norm(self.tensor)
        return outcome

    def expectation(self, op: np.ndarray, qubit: int) -> float:
        """<psi| op |psi> for a single-qubit Hermitian ``op``."""
        other = self.copy()
        other.apply(op, [qubit])
        return float(np.real(np.vdot(self.vector, other.vector)))

    def fidelity(self, other: "StateVector") -> float:
        return float(abs(np.vdot(self.vector, other.vector)) ** 2)

    def is_normalised(self) -> bool:
        return bool(np.isclose(np.linalg.norm(self.vector), 1.0))


def check_unitary(u: np.ndarray) -> None:
    if not is_unitary(u):
        raise ValueError("matrix is not unitary")
