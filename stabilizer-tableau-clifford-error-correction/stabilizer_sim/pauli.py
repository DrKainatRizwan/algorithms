"""Hermitian n-qubit Pauli strings in the binary symplectic representation."""

from __future__ import annotations

import numpy as np

_PAULI_BITS = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
_BITS_PAULI = {v: k for k, v in _PAULI_BITS.items()}


def phase_exponent(x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray) -> int:
    """Exponent k (mod 4) such that ``P1 * P2 = i**k * P3`` summed over all qubits.

    Single-qubit Paulis are encoded as (x, z) with Y = (1, 1) and the Hermitian
    convention, so e.g. X*Z = -iY contributes k = -1.
    """
    x1 = x1.astype(np.int64); z1 = z1.astype(np.int64)
    x2 = x2.astype(np.int64); z2 = z2.astype(np.int64)
    g = np.where(
        (x1 == 1) & (z1 == 1), z2 - x2,
        np.where((x1 == 1) & (z1 == 0), z2 * (2 * x2 - 1),
                 np.where((x1 == 0) & (z1 == 1), x2 * (1 - 2 * z2), 0)),
    )
    return int(g.sum())


class Pauli:
    """A signed Pauli string ``(-1)^sign * P_0 (x) P_1 ... (x) P_{n-1}``."""

    def __init__(self, x: np.ndarray, z: np.ndarray, sign: int = 0) -> None:
        self.x = np.asarray(x, dtype=np.uint8) % 2
        self.z = np.asarray(z, dtype=np.uint8) % 2
        if self.x.shape != self.z.shape or self.x.ndim != 1:
            raise ValueError("x and z must be 1-D arrays of equal length")
        self.sign = int(sign) % 2

    @classmethod
    def from_label(cls, label: str) -> "Pauli":
        """Parse e.g. ``"+XZIY"`` or ``"-ZZ"`` (sign optional); qubit 0 is first."""
        sign = 0
        if label and label[0] in "+-":
            sign = 1 if label[0] == "-" else 0
            label = label[1:]
        bits = [_PAULI_BITS[c] for c in label.upper()]
        return cls(np.array([b[0] for b in bits]), np.array([b[1] for b in bits]), sign)

    @classmethod
    def single(cls, n: int, qubit: int, kind: str) -> "Pauli":
        """Pauli ``kind`` acting on ``qubit`` of an ``n``-qubit register."""
        chars = ["I"] * n
        chars[qubit] = kind
        return cls.from_label("".join(chars))

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def weight(self) -> int:
        return int(np.count_nonzero(self.x | self.z))

    @property
    def label(self) -> str:
        body = "".join(_BITS_PAULI[(int(a), int(b))] for a, b in zip(self.x, self.z))
        return ("-" if self.sign else "+") + body

    def commutes(self, other: "Pauli") -> bool:
        """True if the two Paulis commute (symplectic inner product is zero)."""
        s = int(np.dot(self.x.astype(int), other.z.astype(int))
                + np.dot(self.z.astype(int), other.x.astype(int)))
        return s % 2 == 0

    def multiply(self, other: "Pauli") -> tuple[int, "Pauli"]:
        """Return ``(k, R)`` with ``self * other = i**k * R`` and ``R`` sign-free."""
        k = 2 * self.sign + 2 * other.sign + phase_exponent(self.x, self.z, other.x, other.z)
        return k % 4, Pauli(self.x ^ other.x, self.z ^ other.z, 0)

    def to_matrix(self) -> np.ndarray:
        """Dense matrix; qubit ``q`` corresponds to bit ``q`` of the basis index."""
        mats = {
            (0, 0): np.eye(2, dtype=complex),
            (1, 0): np.array([[0, 1], [1, 0]], dtype=complex),
            (0, 1): np.array([[1, 0], [0, -1]], dtype=complex),
            (1, 1): np.array([[0, -1j], [1j, 0]], dtype=complex),
        }
        out = np.eye(1, dtype=complex)
        for q in range(self.n):  # qubit 0 ends up least significant
            out = np.kron(mats[(int(self.x[q]), int(self.z[q]))], out)
        return -out if self.sign else out

    def copy(self) -> "Pauli":
        return Pauli(self.x.copy(), self.z.copy(), self.sign)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Pauli) and self.sign == other.sign
                and np.array_equal(self.x, other.x) and np.array_equal(self.z, other.z))

    def __repr__(self) -> str:
        return f"Pauli('{self.label}')"
