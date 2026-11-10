"""Dense state-vector reference simulator used to validate the tableau."""

from __future__ import annotations

import numpy as np

from .pauli import Pauli
from .tableau import Tableau

_SQ = 1 / np.sqrt(2)
_GATES = {
    "h": np.array([[_SQ, _SQ], [_SQ, -_SQ]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class StateVector:
    """2^n amplitude vector; qubit ``q`` is bit ``q`` of the basis index."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.psi = np.zeros(2 ** n, dtype=complex)
        self.psi[0] = 1.0

    def apply1(self, name: str, q: int) -> None:
        u = _GATES[name]
        t = self.psi.reshape((2,) * self.n)
        ax = self.n - 1 - q
        t = np.moveaxis(np.tensordot(u, t, axes=([1], [ax])), 0, ax)
        self.psi = np.ascontiguousarray(t).reshape(-1)

    def cx(self, a: int, b: int) -> None:
        idx = np.arange(self.psi.size)
        sel = ((idx >> a) & 1) == 1
        new = self.psi.copy()
        new[sel] = self.psi[idx[sel] ^ (1 << b)]
        self.psi = new

    def cz(self, a: int, b: int) -> None:
        idx = np.arange(self.psi.size)
        sel = (((idx >> a) & 1) == 1) & (((idx >> b) & 1) == 1)
        self.psi[sel] *= -1

    def swap(self, a: int, b: int) -> None:
        idx = np.arange(self.psi.size)
        ba, bb = (idx >> a) & 1, (idx >> b) & 1
        j = idx ^ ((ba ^ bb) * ((1 << a) | (1 << b)))
        self.psi = self.psi[j]

    def expectation(self, p: Pauli) -> float:
        return float(np.real(np.vdot(self.psi, p.to_matrix() @ self.psi)))

    def probabilities(self) -> np.ndarray:
        return np.abs(self.psi) ** 2


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>|^2 for normalised vectors (global-phase insensitive)."""
    return float(abs(np.vdot(a, b)) ** 2)


def tableau_to_statevector(t: Tableau) -> np.ndarray:
    """Reconstruct the pure state stabilised by ``t`` (feasible for small n).

    The projector prod_i (I + g_i)/2 is applied to computational basis vectors
    until a non-zero image is found.
    """
    n = t.n
    projectors = [(np.eye(2 ** n) + g.to_matrix()) / 2 for g in t.stabilizers()]
    for k in range(2 ** n):
        v = np.zeros(2 ** n, dtype=complex)
        v[k] = 1
        for p in projectors:
            v = p @ v
        nrm = np.linalg.norm(v)
        if nrm > 1e-9:
            return v / nrm
    raise RuntimeError("stabilizer group has no common +1 eigenvector")
