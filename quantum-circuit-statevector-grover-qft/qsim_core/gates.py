"""Standard gate matrices and helpers for building new ones."""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)
T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def phase(theta: float) -> np.ndarray:
    """Phase gate diag(1, e^{i theta})."""
    return np.diag([1, np.exp(1j * theta)]).astype(complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)


def is_unitary(u: np.ndarray, atol: float = 1e-10) -> bool:
    """Check U^dagger U = I."""
    u = np.asarray(u)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        return False
    return bool(np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=atol))


def controlled(u: np.ndarray, n_controls: int = 1) -> np.ndarray:
    """Dense matrix of ``u`` controlled on ``n_controls`` leading qubits."""
    dim = u.shape[0]
    total = dim * 2**n_controls
    out = np.eye(total, dtype=complex)
    out[total - dim :, total - dim :] = u
    return out


def kron_all(*ms: np.ndarray) -> np.ndarray:
    """Kronecker product of several matrices (first = most significant)."""
    out = np.eye(1, dtype=complex)
    for m in ms:
        out = np.kron(out, m)
    return out
