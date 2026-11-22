"""Siegenthaler-style correlation attacks, accelerated with a Walsh-Hadamard transform.

For a target LFSR with unknown state ``x`` the keystream satisfies
``z_t = s_t(x) + e_t`` where ``e_t`` is 1 with probability ``p < 1/2``.  Define
``F(m) = sum_{t : mask_t = m} (-1)^{z_t}``.  Then the agreement score of the
candidate ``x`` is ``sum_t (-1)^{z_t + <mask_t, x>}`` = ``WHT(F)(x)``, so all
``2^n`` candidates are scored in ``O(n 2^n)`` instead of ``O(N 2^n)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .generators import GeffeGenerator
from .lfsr import LFSR, recover_state


def walsh_hadamard(a: np.ndarray) -> np.ndarray:
    """Unnormalised fast Walsh-Hadamard transform (length must be a power of 2)."""
    a = np.array(a, dtype=np.int64)
    n = len(a)
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1)
        a = a.reshape(-1)
        h *= 2
    return a


def correlation_scores(lfsr: LFSR, z: np.ndarray) -> np.ndarray:
    """Score ``sum_t (-1)^(z_t + s_t(x))`` for every state ``x`` (index = state)."""
    n = lfsr.length
    f = np.zeros(1 << n, dtype=np.int64)
    ms = lfsr.masks(len(z))
    signs = 1 - 2 * z.astype(np.int64)
    for m, s in zip(ms, signs):
        f[m] += s
    return walsh_hadamard(f)


def brute_force_scores(lfsr: LFSR, z: np.ndarray) -> np.ndarray:
    """Reference ``O(N 2^n)`` scoring used to validate the transform."""
    n = lfsr.length
    out = np.zeros(1 << n, dtype=np.int64)
    for x in range(1, 1 << n):
        seq = lfsr.stream(x, len(z))
        out[x] = int(np.sum(1 - 2 * (seq ^ z).astype(np.int64)))
    out[0] = int(np.sum(1 - 2 * z.astype(np.int64)))  # all-zero stream
    return out


@dataclass(frozen=True)
class AttackResult:
    state: int
    score: int
    z_value: float  # standard deviations above the random-guess mean
    runner_up_score: int


def correlation_attack(lfsr: LFSR, z: np.ndarray) -> AttackResult:
    """Return the non-zero initial state with maximal agreement with ``z``."""
    sc = correlation_scores(lfsr, z)
    sc[0] = -(10**12)  # the all-zero state is not a valid key
    order = np.argsort(sc)
    best, second = int(order[-1]), int(order[-2])
    zval = sc[best] / math.sqrt(len(z))
    return AttackResult(best, int(sc[best]), float(zval), int(sc[second]))


def required_length(n: int, p: float, margin: float = 1.0) -> int:
    """Rough keystream length for reliable detection of the right state.

    The right state has expected score ``N(1-2p)``; wrong states behave like
    ``N(0, N)``.  Requiring the true score to exceed the maximum of ``2^n``
    Gaussians, about ``sqrt(2 n ln 2 N)``, gives ``N ~ 2 n ln2 / (1-2p)^2``.
    """
    bias = 1 - 2 * p
    return math.ceil(margin * 2 * n * math.log(2) / bias**2)


def geffe_attack(gen: GeffeGenerator, z: np.ndarray) -> tuple[int, int, int] | None:
    """Divide-and-conquer attack on the Geffe generator.

    R2 and R3 are each recovered independently through their 3/4 correlation
    with the keystream.  The selector R1 has no correlation, so its
    ``2^n1`` states are searched exhaustively against the keystream, which
    costs far less than the ``2^(n1+n2+n3)`` joint search.
    Returns the three initial states, or ``None`` if no consistent R1 exists.
    """
    s2 = correlation_attack(gen.r2, z).state
    s3 = correlation_attack(gen.r3, z).state
    b = gen.r2.stream(s2, len(z))
    c = gen.r3.stream(s3, len(z))
    # R1 selects b where its bit is 1 and c where 0; where b == c the keystream
    # is uninformative, otherwise z fixes the selector bit directly.
    informative = np.nonzero(b != c)[0]
    known = {int(t): int(b[t] == z[t]) for t in informative}  # 1 -> selector is 1
    if len(known) < gen.r1.length:
        return None
    ts = sorted(known)
    ms = gen.r1.masks(max(ts) + 1)
    from .gf2 import solve

    res = solve([ms[t] for t in ts], [known[t] for t in ts], gen.r1.length)
    if res is None:
        return None
    s1 = res[0]
    if np.array_equal(gen.keystream((s1, s2, s3), len(z)), z):
        return s1, s2, s3
    return None


def empirical_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of positions in which two bit-streams agree."""
    return float(np.mean(a == b))
