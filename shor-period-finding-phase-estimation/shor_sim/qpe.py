"""Quantum phase estimation for an arbitrary small unitary.

The counting register of ``t`` qubits controls ``U^(2^j)``; an inverse QFT
turns the accumulated phases into a measurable binary fraction of the
eigenphase.  The joint state is simulated explicitly.
"""
from __future__ import annotations

import numpy as np

from .qft import qft_fft


def modular_multiplication_unitary(a: int, n: int) -> np.ndarray:
    """Permutation matrix of ``|y> -> |a*y mod n>`` for ``y < n`` (identity above)."""
    dim = n
    u = np.zeros((dim, dim))
    for y in range(n):
        u[(a * y) % n, y] = 1.0
    return u


def phase_estimation_state(u: np.ndarray, target: np.ndarray, t: int) -> np.ndarray:
    """Joint state ``psi[x, y]`` after controlled powers, before the inverse QFT.

    Counting basis index ``x`` accumulates ``U^x |target>``.
    """
    q = 1 << t
    dim = len(target)
    psi = np.zeros((q, dim), dtype=complex)
    v = np.asarray(target, dtype=complex)
    for x in range(q):
        psi[x] = v
        v = u @ v
    return psi / np.sqrt(q)


def phase_estimation_distribution(u: np.ndarray, target: np.ndarray, t: int) -> np.ndarray:
    """Probability of each counting outcome ``y in [0, 2^t)``."""
    psi = phase_estimation_state(u, target, t)
    out = np.apply_along_axis(lambda col: qft_fft(col, inverse=True), 0, psi)
    return (np.abs(out) ** 2).sum(axis=1)


def estimate_phase(u: np.ndarray, target: np.ndarray, t: int,
                   rng: np.random.Generator) -> float:
    """Sample one estimate of the eigenphase in [0, 1)."""
    p = phase_estimation_distribution(u, target, t)
    y = rng.choice(len(p), p=p / p.sum())
    return y / (1 << t)


def eigenvectors_of_modmul(a: int, n: int, r: int) -> list[tuple[float, np.ndarray]]:
    """Eigenpairs ``(s/r, |u_s>)`` of the multiplication-by-a unitary.

    ``|u_s> = r^-1/2 sum_k exp(-2 pi i s k / r) |a^k mod n>``.
    """
    orbit = [pow(a, k, n) for k in range(r)]
    pairs = []
    for s in range(r):
        v = np.zeros(n, dtype=complex)
        for k, val in enumerate(orbit):
            v[val] += np.exp(-2j * np.pi * s * k / r)
        pairs.append((s / r, v / np.sqrt(r)))
    return pairs
