"""Parameterised circuits built from RX/RY/RZ, CNOT and Pauli rotations.

Every parametrised gate has the form ``exp(-i theta P / 2)`` with ``P`` a
Pauli string, so the two-term parameter-shift rule is exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pauli import PauliHamiltonian, apply_pauli_string


@dataclass(frozen=True)
class Op:
    kind: str                 # "rot" or "cx" or "x"
    qubits: tuple[int, ...]
    pauli: str = ""           # Pauli string for "rot"
    param: int = -1           # parameter index for "rot"
    sign: float = 1.0         # multiplier on the parameter (the angle if param < 0)


def _apply_x(state: np.ndarray, q: int, n: int) -> np.ndarray:
    t = state.reshape((2,) * n)
    ax = n - 1 - q
    return np.flip(t, axis=ax).reshape(-1).copy()


def _apply_cx(state: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
    idx = np.arange(1 << n)
    mask = ((idx >> c) & 1).astype(bool)
    out = state.copy()
    out[mask] = state[idx[mask] ^ (1 << t)]
    return out


def _apply_rot(state: np.ndarray, pauli: str, theta: float) -> np.ndarray:
    return (np.cos(theta / 2) * state
            - 1j * np.sin(theta / 2) * apply_pauli_string(pauli, state))


@dataclass
class Ansatz:
    """A list of gates acting on ``|0...0>``."""

    num_qubits: int
    ops: list[Op] = field(default_factory=list)
    num_params: int = 0

    def x(self, q: int) -> "Ansatz":
        self.ops.append(Op("x", (q,)))
        return self

    def cx(self, c: int, t: int) -> "Ansatz":
        self.ops.append(Op("cx", (c, t)))
        return self

    def rot(self, pauli: str, param: int, sign: float = 1.0) -> "Ansatz":
        if len(pauli) != self.num_qubits:
            raise ValueError("Pauli length mismatch")
        self.ops.append(Op("rot", (), pauli, param, sign))
        self.num_params = max(self.num_params, param + 1)
        return self

    def fixed_rot(self, pauli: str, angle: float) -> "Ansatz":
        """Rotation with a constant (non-trainable) angle."""
        self.ops.append(Op("rot", (), pauli, -1, angle))
        return self

    def single(self, axis: str, q: int, param: int) -> "Ansatz":
        label = "".join(axis if k == q else "I" for k in range(self.num_qubits))
        return self.rot(label, param)

    def run(self, params: np.ndarray, shifts: dict[int, float] | None = None
            ) -> np.ndarray:
        """Simulate the circuit; ``shifts`` maps op position to an angle
        offset (used by the parameter-shift rule)."""
        params = np.asarray(params, dtype=float)
        if params.shape != (self.num_params,):
            raise ValueError(f"expected {self.num_params} parameters")
        n = self.num_qubits
        state = np.zeros(1 << n, dtype=complex)
        state[0] = 1.0
        for pos, op in enumerate(self.ops):
            if op.kind == "x":
                state = _apply_x(state, op.qubits[0], n)
            elif op.kind == "cx":
                state = _apply_cx(state, op.qubits[0], op.qubits[1], n)
            else:
                theta = op.sign if op.param < 0 else op.sign * params[op.param]
                if shifts and pos in shifts:
                    theta += shifts[pos]
                state = _apply_rot(state, op.pauli, theta)
        return state

    def energy(self, params: np.ndarray, h: PauliHamiltonian) -> float:
        return h.expectation(self.run(params))

    def depth_stats(self) -> dict[str, int]:
        return {"gates": len(self.ops),
                "cx": sum(o.kind == "cx" for o in self.ops),
                "params": self.num_params}


def hardware_efficient(n: int, layers: int) -> Ansatz:
    """RY rotations followed by a ladder of CNOTs, repeated ``layers`` times
    with a final RY layer; ``n * (layers + 1)`` parameters."""
    a = Ansatz(n)
    p = 0
    for _ in range(layers):
        for q in range(n):
            a.single("Y", q, p)
            p += 1
        for q in range(n - 1):
            a.cx(q, q + 1)
    for q in range(n):
        a.single("Y", q, p)
        p += 1
    return a


def h2_uccsd() -> Ansatz:
    """One-parameter unitary-coupled-cluster style ansatz for the reduced
    two-qubit H2 Hamiltonian: Hartree-Fock |01> then exp(-i t/2 Y X)."""
    a = Ansatz(2)
    a.x(0)
    a.rot("YX", 0)
    return a


def hamiltonian_variational(h: PauliHamiltonian, layers: int,
                            init_hadamard_like: bool = True) -> Ansatz:
    """Hamiltonian-variational ansatz: each layer applies one shared angle per
    non-X term group (ZZ-type) and one for the X field.

    The start state is a product of RY(pi/2) rotations (|+>^n) which is the
    ground state of the pure-field limit.  Terms are grouped by whether they
    contain X/Y operators.
    """
    n = h.num_qubits
    a = Ansatz(n)
    if init_hadamard_like:
        for q in range(n):
            a.fixed_rot("".join("Y" if k == q else "I" for k in range(n)),
                        np.pi / 2)
    groups: list[list[str]] = [[], []]
    for label in h.terms:
        groups[0 if set(label) <= {"I", "Z"} else 1].append(label)
    p = 0
    for _ in range(layers):
        for g in groups:
            if not g:
                continue
            for label in g:
                a.rot(label, p)
            p += 1
    return a
