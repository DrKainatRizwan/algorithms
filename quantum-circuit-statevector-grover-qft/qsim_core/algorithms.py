"""Quantum algorithms built from the circuit primitives."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from . import gates
from .circuit import Circuit
from .state import StateVector


# ------------------------------------------------------------------ QFT
def qft_circuit(n: int, swaps: bool = True) -> Circuit:
    """Quantum Fourier transform on ``n`` qubits (qubit 0 most significant).

    Implements |j> -> N^{-1/2} sum_k exp(2 pi i j k / N) |k> using
    n Hadamards, n(n-1)/2 controlled phases and floor(n/2) swaps.
    """
    c = Circuit(n)
    for i in range(n):
        c.h(i)
        for j in range(i + 1, n):
            c.cp(math.pi / 2 ** (j - i), j, i)
    if swaps:
        for i in range(n // 2):
            c.swap(i, n - 1 - i)
    return c


def inverse_qft_circuit(n: int, swaps: bool = True) -> Circuit:
    return qft_circuit(n, swaps).inverse()


def dft_matrix(n: int) -> np.ndarray:
    """Reference unitary DFT matrix for ``n`` qubits."""
    N = 2**n
    jk = np.outer(np.arange(N), np.arange(N))
    return np.exp(2j * np.pi * jk / N) / np.sqrt(N)


# --------------------------------------------------------------- Grover
def optimal_grover_iterations(n: int, n_marked: int) -> int:
    """floor(pi/4 * sqrt(N/M)) iterations maximise success probability."""
    if not 0 < n_marked <= 2**n:
        raise ValueError("n_marked out of range")
    if n_marked == 2**n:
        return 0
    theta = math.asin(math.sqrt(n_marked / 2**n))
    return max(0, int(math.floor(math.pi / (4 * theta))))


def grover_oracle(n: int, marked: Iterable[int]) -> Circuit:
    """Phase oracle flipping the sign of each marked basis state."""
    c = Circuit(n)
    qubits = list(range(n))
    for m in sorted(set(marked)):
        if not 0 <= m < 2**n:
            raise ValueError(f"marked state {m} out of range")
        zeros = [q for q in qubits if not (m >> (n - 1 - q)) & 1]
        for q in zeros:
            c.x(q)
        _phase_flip_all_ones(c, qubits)
        for q in zeros:
            c.x(q)
    return c


def _phase_flip_all_ones(c: Circuit, qubits: Sequence[int]) -> None:
    if len(qubits) == 1:
        c.z(qubits[0])
    else:
        c.mcz(qubits)


def grover_diffusion(n: int) -> Circuit:
    """Inversion about the mean, 2|s><s| - I (up to a global phase)."""
    c = Circuit(n)
    qs = list(range(n))
    for q in qs:
        c.h(q)
        c.x(q)
    _phase_flip_all_ones(c, qs)
    for q in qs:
        c.x(q)
        c.h(q)
    return c


def grover_circuit(n: int, marked: Iterable[int], iterations: int | None = None) -> Circuit:
    marked = list(marked)
    k = optimal_grover_iterations(n, len(set(marked))) if iterations is None else iterations
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    oracle, diff = grover_oracle(n, marked), grover_diffusion(n)
    for _ in range(k):
        c.compose(oracle)
        c.compose(diff)
    return c


def grover_success_curve(n: int, marked: Iterable[int], max_iters: int) -> list[float]:
    """Probability of measuring a marked state after 0..max_iters iterations."""
    marked = sorted(set(marked))
    oracle, diff = grover_oracle(n, marked), grover_diffusion(n)
    st = StateVector(n)
    for q in range(n):
        st.apply(gates.H, [q])
    out = []
    for _ in range(max_iters + 1):
        out.append(float(st.probabilities()[marked].sum()))
        oracle.run(st)
        diff.run(st)
    return out


def grover_search(
    n: int, marked: Iterable[int], rng: np.random.Generator, shots: int = 1
) -> dict[str, int]:
    """Run Grover with the optimal iteration count and sample the result."""
    return grover_circuit(n, marked).run().sample(shots, rng)


# ------------------------------------------------------ phase estimation
def phase_estimation_circuit(t: int, theta: float) -> Circuit:
    """QPE of the phase gate P(2 pi theta) acting on eigenstate |1>.

    Qubits 0..t-1 are the counting register, qubit t is the eigenstate qubit.
    """
    c = Circuit(t + 1)
    c.x(t)
    for q in range(t):
        c.h(q)
    for q in range(t):
        # qubit q (big-endian) controls U^(2^(t-1-q))
        c.cp(2 * math.pi * theta * 2 ** (t - 1 - q), q, t)
    c.compose(inverse_qft_circuit(t), list(range(t)))
    return c


def estimate_phase(t: int, theta: float) -> tuple[float, float]:
    """Most likely phase estimate and its probability."""
    st = phase_estimation_circuit(t, theta).run()
    p = st.probabilities(list(range(t)))
    k = int(np.argmax(p))
    return k / 2**t, float(p[k])


# ------------------------------------------------- small textbook circuits
def ghz_circuit(n: int) -> Circuit:
    c = Circuit(n)
    c.h(0)
    for q in range(1, n):
        c.cx(0, q)
    return c


def deutsch_jozsa(n: int, f: "callable") -> str:  # type: ignore[valid-type]
    """Return ``'constant'`` or ``'balanced'`` for a promise function f."""
    c = Circuit(n + 1)
    c.x(n)
    for q in range(n + 1):
        c.h(q)
    # phase-kickback oracle |x>|y> -> |x>|y xor f(x)> as permutation
    dim = 2 ** (n + 1)
    perm = np.zeros((dim, dim), dtype=complex)
    for x in range(2**n):
        fx = int(f(x)) & 1
        for y in (0, 1):
            perm[(x << 1) | (y ^ fx), (x << 1) | y] = 1.0
    c.add("Uf", perm, list(range(n + 1)))
    for q in range(n):
        c.h(q)
    p = c.run().probabilities(list(range(n)))
    return "constant" if p[0] > 0.5 else "balanced"
