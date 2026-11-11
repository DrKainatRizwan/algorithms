"""Quantum Fourier transform on an n-qubit state vector.

Qubit 0 is the most significant bit of the basis-state index.  Three
implementations are provided and cross-checked in the tests: the dense DFT
matrix, a gate-level circuit (Hadamards, controlled phases, swaps) and a
recursive radix-2 FFT.
"""
from __future__ import annotations

import numpy as np


def qft_matrix(n_qubits: int, inverse: bool = False) -> np.ndarray:
    """Dense unitary ``F[j,k] = exp(+-2 pi i jk / Q) / sqrt(Q)``."""
    q = 1 << n_qubits
    j = np.arange(q)
    sign = -1.0 if inverse else 1.0
    return np.exp(sign * 2j * np.pi * np.outer(j, j) / q) / np.sqrt(q)


def _apply_h(state: np.ndarray, qubit: int, n: int) -> np.ndarray:
    t = state.reshape((2,) * n)
    t = np.moveaxis(t, qubit, 0)
    a, b = t[0].copy(), t[1].copy()
    t = np.stack([(a + b), (a - b)]) / np.sqrt(2)
    return np.moveaxis(t, 0, qubit).reshape(-1)


def _apply_cphase(state: np.ndarray, c: int, tgt: int, n: int, theta: float) -> np.ndarray:
    t = state.reshape((2,) * n).copy()
    idx = [slice(None)] * n
    idx[c], idx[tgt] = 1, 1
    t[tuple(idx)] *= np.exp(1j * theta)
    return t.reshape(-1)


def _swap_reverse(state: np.ndarray, n: int) -> np.ndarray:
    t = state.reshape((2,) * n)
    return np.transpose(t, tuple(reversed(range(n)))).reshape(-1).copy()


def qft_gates(state: np.ndarray, n_qubits: int, inverse: bool = False) -> np.ndarray:
    """Apply the QFT circuit: H and controlled-R_k ladders, then bit reversal.

    Uses ``n(n-1)/2`` controlled phases and ``n`` Hadamards.
    """
    state = np.asarray(state, dtype=complex)
    n = n_qubits
    ops: list[tuple[str, int, int, float]] = []
    for j in range(n):
        ops.append(("h", j, -1, 0.0))
        for k in range(j + 1, n):
            ops.append(("cp", k, j, 2 * np.pi / (1 << (k - j + 1))))
    if inverse:
        # inverse circuit = reversed order with negated angles; bit reversal first
        state = _swap_reverse(state, n)
        for kind, a, b, th in reversed(ops):
            state = _apply_h(state, a, n) if kind == "h" else _apply_cphase(state, a, b, n, -th)
        return state
    for kind, a, b, th in ops:
        state = _apply_h(state, a, n) if kind == "h" else _apply_cphase(state, a, b, n, th)
    return _swap_reverse(state, n)


def fft_recursive(x: np.ndarray, sign: int = 1) -> np.ndarray:
    """Unnormalised radix-2 Cooley-Tukey transform ``sum x_j exp(sign 2 pi i jk/N)``."""
    n = len(x)
    if n == 1:
        return x.astype(complex)
    even = fft_recursive(x[0::2], sign)
    odd = fft_recursive(x[1::2], sign)
    tw = np.exp(sign * 2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + tw, even - tw])


def qft_fft(state: np.ndarray, inverse: bool = False) -> np.ndarray:
    """QFT of a state vector through the recursive FFT (O(Q log Q))."""
    sign = -1 if inverse else 1
    return fft_recursive(np.asarray(state, dtype=complex), sign) / np.sqrt(len(state))


def qft_gate_count(n_qubits: int) -> dict[str, int]:
    """Gate counts of the exact QFT circuit."""
    return {"hadamard": n_qubits, "controlled_phase": n_qubits * (n_qubits - 1) // 2,
            "swap": n_qubits // 2}
