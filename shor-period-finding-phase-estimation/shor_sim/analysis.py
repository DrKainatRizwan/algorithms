"""Exact success statistics and resource estimates for order finding."""
from __future__ import annotations

import numpy as np

from .modular import euler_phi, gcd, multiplicative_order, recover_denominator
from .period import counting_qubits, outcome_distribution
from .qft import qft_gate_count


def peak_positions(r: int, q: int) -> np.ndarray:
    """Ideal outcomes ``round(s q / r)`` for s = 0..r-1."""
    return np.array([int(round(s * q / r)) % q for s in range(r)])


def order_recovery_probability(a: int, n: int, t: int | None = None) -> float:
    """Probability that one run gives the exact order via continued fractions."""
    t = counting_qubits(n) if t is None else t
    q = 1 << t
    r = multiplicative_order(a, n)
    dist = outcome_distribution(a, n, t)
    total = 0.0
    for y in np.nonzero(dist > 1e-14)[0]:
        if recover_denominator(int(y), q, n) == r:
            total += dist[y]
    return float(total)


def peak_mass(a: int, n: int, t: int) -> float:
    """Probability mass within 1/2 of an ideal peak ``s q / r``."""
    q = 1 << t
    r = multiplicative_order(a, n)
    dist = outcome_distribution(a, n, t)
    ys = np.arange(q)
    mask = np.zeros(q, dtype=bool)
    for s in range(r):
        mask |= np.abs(ys - s * q / r) <= 0.5
    return float(dist[mask].sum())


def theoretical_success_floor(r: int) -> float:
    """Lower bound ``phi(r)/r * 4/pi^2`` for obtaining a coprime numerator."""
    return euler_phi(r) / r * 4 / np.pi**2


def split_probability(n: int, bases: list[int] | None = None) -> float:
    """Fraction of coprime bases whose order yields a non-trivial factorisation."""
    bases = bases or [a for a in range(2, n) if gcd(a, n) == 1]
    good = 0
    for a in bases:
        r = multiplicative_order(a, n)
        if r % 2 == 0 and pow(a, r // 2, n) != n - 1:
            good += 1
    return good / len(bases)


def resource_estimate(n: int) -> dict[str, int]:
    """Qubit and gate counts for the textbook circuit (2L counting, L work)."""
    L = n.bit_length()
    t = 2 * L
    qft = qft_gate_count(t)
    return {"bits": L, "counting_qubits": t, "work_qubits": L,
            "total_qubits": t + L, "qft_controlled_phases": qft["controlled_phase"],
            "modmul_blocks": t}
