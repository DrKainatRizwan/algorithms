"""Quantum order finding: the core subroutine of Shor's algorithm.

Two simulation back-ends share one interface:

* ``full``       - explicit joint state over (counting register x work register),
                   including the modular-exponentiation permutation and a
                   projective measurement of the work register.
* ``compressed`` - uses the fact that after measuring the work register the
                   counting register is a uniform comb on one residue class, so
                   only the counting register is stored.  Memory O(Q).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modular import gcd, recover_denominator
from .qft import qft_fft, qft_gates


def counting_qubits(n: int) -> int:
    """Smallest t with ``2^t >= n^2`` (guarantees the continued-fraction bound)."""
    return max(1, int(np.ceil(2 * np.log2(n))))


def oracle_table(a: int, n: int, q: int) -> np.ndarray:
    """``a^x mod n`` for all ``x < q`` (the modular-exponentiation oracle)."""
    table = np.empty(q, dtype=np.int64)
    v = 1
    for x in range(q):
        table[x] = v
        v = v * a % n
    return table


def joint_state_after_oracle(a: int, n: int, t: int) -> np.ndarray:
    """State ``2^-t/2 sum_x |x>|a^x mod n>`` as a (Q, n) array."""
    q = 1 << t
    psi = np.zeros((q, n), dtype=complex)
    psi[np.arange(q), oracle_table(a, n, q)] = 1.0 / np.sqrt(q)
    return psi


def outcome_distribution(a: int, n: int, t: int, backend: str = "compressed",
                         use_gates: bool = False) -> np.ndarray:
    """Exact distribution of the counting-register measurement.

    The distribution is averaged over the (unobserved) work-register value,
    which is what a single run of the circuit samples from.
    """
    q = 1 << t
    if backend == "full":
        psi = joint_state_after_oracle(a, n, t)
        dist = np.zeros(q)
        for v in range(n):
            col = psi[:, v]
            w = np.linalg.norm(col) ** 2
            if w < 1e-15:
                continue
            col = col / np.sqrt(w)
            out = qft_gates(col, t) if use_gates else qft_fft(col)
            dist += w * np.abs(out) ** 2
        return dist
    if backend != "compressed":
        raise ValueError("backend must be 'full' or 'compressed'")
    table = oracle_table(a, n, q)
    dist = np.zeros(q)
    # Each residue class is a shifted comb; shifts only change phases, so one
    # representative (the class of x=0) gives the exact outcome distribution.
    comb = np.zeros(q, dtype=complex)
    comb[table == 1] = 1.0
    comb /= np.linalg.norm(comb)
    out = qft_gates(comb, t) if use_gates else qft_fft(comb)
    dist = np.abs(out) ** 2
    return dist


@dataclass
class OrderFindingRun:
    """One execution of quantum order finding."""
    measurement: int
    t: int
    candidate: int          # denominator recovered from the continued fraction
    order: int              # verified order, 0 if recovery failed


def sample_order(a: int, n: int, rng: np.random.Generator, backend: str = "compressed",
                 t: int | None = None, max_multiple: int = 8) -> OrderFindingRun:
    """Run the quantum subroutine once and post-process the outcome classically."""
    t = counting_qubits(n) if t is None else t
    q = 1 << t
    if backend == "full":
        psi = joint_state_after_oracle(a, n, t)
        weights = (np.abs(psi) ** 2).sum(axis=0)
        v = rng.choice(n, p=weights / weights.sum())       # measure work register
        col = psi[:, v] / np.sqrt(weights[v])
        probs = np.abs(qft_fft(col)) ** 2
    else:
        probs = outcome_distribution(a, n, t, "compressed")
    probs = probs / probs.sum()
    y = int(rng.choice(q, p=probs))
    cand = recover_denominator(y, q, n)
    order = 0
    for m in range(1, max_multiple + 1):
        r = cand * m
        if r and pow(a, r, n) == 1:
            order = _reduce_order(a, n, r)
            break
    return OrderFindingRun(y, t, cand, order)


def _reduce_order(a: int, n: int, r: int) -> int:
    """Strip redundant prime factors from a valid multiple of the order."""
    m, p = r, 2
    primes = []
    while p * p <= m:
        if m % p == 0:
            primes.append(p)
            while m % p == 0:
                m //= p
        p += 1
    if m > 1:
        primes.append(m)
    for p in primes:
        while r % p == 0 and pow(a, r // p, n) == 1:
            r //= p
    return r


def combine_orders(a: int, n: int, candidates: list[int]) -> int:
    """LCM of candidate denominators that yields a verified order, else 0."""
    lcm = 1
    for c in candidates:
        if c:
            lcm = lcm * c // gcd(lcm, c)
            if pow(a, lcm, n) == 1:
                return _reduce_order(a, n, lcm)
    return 0
