"""State-vector kernels for QAOA: qubit 0 is the most significant bit of the index."""
from __future__ import annotations

import numpy as np


def plus_state(n: int) -> np.ndarray:
    """Uniform superposition |+>^n."""
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=complex)


def apply_phase(psi: np.ndarray, diag: np.ndarray, gamma: float) -> np.ndarray:
    """Apply exp(-i gamma * diag) for a diagonal Hamiltonian."""
    return psi * np.exp(-1j * gamma * diag)


def apply_rx_all(psi: np.ndarray, n: int, theta: float) -> np.ndarray:
    """Apply RX(theta) = exp(-i theta X / 2) to every qubit."""
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    out = psi.reshape((2,) * n)
    for ax in range(n):
        a = np.take(out, 0, axis=ax)
        b = np.take(out, 1, axis=ax)
        out = np.stack((c * a + s * b, s * a + c * b), axis=ax)
    return out.reshape(-1)


def apply_mixer(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """Apply the transverse-field mixer exp(-i beta sum_j X_j)."""
    return apply_rx_all(psi, n, 2.0 * beta)


def apply_sum_x(psi: np.ndarray, n: int) -> np.ndarray:
    """Apply the operator sum_j X_j (used by the adjoint gradient)."""
    t = psi.reshape((2,) * n)
    out = np.zeros_like(t)
    for ax in range(n):
        out += np.flip(t, axis=ax)
    return out.reshape(-1)


def probabilities(psi: np.ndarray) -> np.ndarray:
    p = np.abs(psi) ** 2
    return p / p.sum()


def expectation(psi: np.ndarray, diag: np.ndarray) -> float:
    """<psi| D |psi> for a diagonal operator."""
    return float(np.real(np.vdot(psi, diag * psi)))


def sample_bitstrings(psi: np.ndarray, shots: int, seed: int = 0) -> np.ndarray:
    """Draw ``shots`` computational-basis outcomes as integer indices."""
    rng = np.random.default_rng(seed)
    return rng.choice(psi.size, size=shots, p=probabilities(psi))
