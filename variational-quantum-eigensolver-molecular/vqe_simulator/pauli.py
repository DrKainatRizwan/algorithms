"""Pauli strings and Pauli-sum Hamiltonians acting on state vectors.

Qubit ``k`` corresponds to character ``k`` of a Pauli string and to bit ``k``
(least significant first) of the computational-basis index.
"""
from __future__ import annotations

import numpy as np

_SINGLE = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _masks(label: str) -> tuple[int, int, int]:
    """Return (x_mask, z_mask, number_of_Y) for a Pauli string."""
    x = z = ny = 0
    for k, ch in enumerate(label):
        if ch not in _SINGLE:
            raise ValueError(f"invalid Pauli character {ch!r}")
        if ch in "XY":
            x |= 1 << k
        if ch in "YZ":
            z |= 1 << k
        if ch == "Y":
            ny += 1
    return x, z, ny


def _parity(arr: np.ndarray) -> np.ndarray:
    """Parity of the popcount of each integer in ``arr``."""
    p = np.zeros_like(arr)
    a = arr.copy()
    while a.any():
        p ^= a & 1
        a >>= 1
    return p


def apply_pauli_string(label: str, state: np.ndarray) -> np.ndarray:
    """Return ``P |state>`` for a Pauli string ``P`` in O(2^n)."""
    n = len(label)
    if state.shape != (1 << n,):
        raise ValueError("state size does not match Pauli string length")
    x, z, ny = _masks(label)
    idx = np.arange(1 << n)
    sign = 1 - 2 * _parity(idx & z)
    phase = (1j) ** ny
    out = np.empty_like(state, dtype=complex)
    # Y = i X Z, so P|j> = i^ny (-1)^{popcount(j & z)} |j xor x>
    out[idx ^ x] = phase * sign * state
    return out


def pauli_matrix(label: str) -> np.ndarray:
    """Dense matrix of a Pauli string (qubit 0 is the least significant bit)."""
    m = np.ones((1, 1), dtype=complex)
    for ch in label:  # later characters are more significant
        m = np.kron(_SINGLE[ch], m)
    return m


class PauliHamiltonian:
    """Real linear combination of Pauli strings."""

    def __init__(self, terms: dict[str, float]):
        if not terms:
            raise ValueError("Hamiltonian needs at least one term")
        lengths = {len(k) for k in terms}
        if len(lengths) != 1:
            raise ValueError("all Pauli strings must have equal length")
        self.num_qubits = lengths.pop()
        self.terms: dict[str, float] = {}
        for k, v in terms.items():
            _masks(k)
            self.terms[k] = self.terms.get(k, 0.0) + float(v)

    def __add__(self, other: "PauliHamiltonian") -> "PauliHamiltonian":
        merged = dict(self.terms)
        for k, v in other.terms.items():
            merged[k] = merged.get(k, 0.0) + v
        return PauliHamiltonian(merged)

    def scale(self, c: float) -> "PauliHamiltonian":
        return PauliHamiltonian({k: c * v for k, v in self.terms.items()})

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Return ``H |state>``."""
        out = np.zeros_like(state, dtype=complex)
        for label, c in self.terms.items():
            out += c * apply_pauli_string(label, state)
        return out

    def expectation(self, state: np.ndarray) -> float:
        """Exact ``<state|H|state>`` for a normalised state."""
        return float(np.real(np.vdot(state, self.apply(state))))

    def term_expectations(self, state: np.ndarray) -> dict[str, float]:
        return {k: float(np.real(np.vdot(state, apply_pauli_string(k, state))))
                for k in self.terms}

    def variance(self, state: np.ndarray) -> float:
        hp = self.apply(state)
        e = np.real(np.vdot(state, hp))
        return float(np.real(np.vdot(hp, hp)) - e * e)

    def matrix(self) -> np.ndarray:
        dim = 1 << self.num_qubits
        m = np.zeros((dim, dim), dtype=complex)
        for label, c in self.terms.items():
            m += c * pauli_matrix(label)
        return m

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.matrix())

    def ground_energy(self) -> float:
        return float(self.eigenvalues()[0])

    def ground_state(self) -> np.ndarray:
        _, v = np.linalg.eigh(self.matrix())
        return v[:, 0]

    def __len__(self) -> int:
        return len(self.terms)
