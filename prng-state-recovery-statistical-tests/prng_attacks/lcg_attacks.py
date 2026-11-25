"""Attacks on linear congruential generators (full and truncated output)."""

from __future__ import annotations

from math import gcd

import numpy as np

from .generators import LCG


def recover_modulus(outputs: list[int]) -> int | None:
    """Estimate the modulus from full-state outputs (Stern/Plumstead).

    With ``t_i = x_{i+1} - x_i`` the quantities ``t_{i+2} t_i - t_{i+1}^2`` are
    multiples of ``m``; their gcd converges to ``m`` after a few dozen samples.
    """
    t = [b - a for a, b in zip(outputs, outputs[1:])]
    g = 0
    for i in range(len(t) - 2):
        g = gcd(g, t[i + 2] * t[i] - t[i + 1] ** 2)
    return abs(g) or None


def recover_multiplier_increment(outputs: list[int], m: int) -> tuple[int, int] | None:
    """Given the modulus, solve for ``(a, c)`` from consecutive outputs."""
    for i in range(len(outputs) - 2):
        x0, x1, x2 = outputs[i:i + 3]
        d = (x1 - x0) % m
        if gcd(d, m) == 1:
            a = ((x2 - x1) * pow(d, -1, m)) % m
            c = (x1 - a * x0) % m
            return a, c
    return None


def recover_lcg_parameters(outputs: list[int]) -> tuple[int, int, int] | None:
    """Recover ``(m, a, c)`` from raw consecutive states, with no prior knowledge."""
    m = recover_modulus(outputs)
    if m is None:
        return None
    ac = recover_multiplier_increment(outputs, m)
    if ac is None:
        return None
    a, c = ac
    if all((a * x + c) % m == y for x, y in zip(outputs, outputs[1:])):
        return m, a, c
    return None


def recover_truncated_state(outputs: list[int], a: int, c: int, m: int,
                            shift: int) -> list[int]:
    """Recover all internal states consistent with truncated outputs.

    Outputs are ``state >> shift`` for known ``(a, c, m)`` with ``m`` a power
    of two.  The ``shift`` hidden low bits of the first state are enumerated
    (vectorised with numpy) and each candidate is validated against all later
    outputs.  Returns the candidate states *at the first observation*.
    """
    if m & (m - 1):
        raise ValueError("modulus must be a power of two for the wraparound trick")
    if shift > 26:
        raise ValueError("search space too large")
    mask = np.uint64(m - 1)
    cand = (np.uint64(outputs[0]) << np.uint64(shift)) | np.arange(1 << shift, dtype=np.uint64)
    alive = cand.copy()
    cur = cand
    au, cu = np.uint64(a), np.uint64(c)
    keep = np.ones(cand.shape, dtype=bool)
    for o in outputs[1:]:
        cur = (cur * au + cu) & mask          # uint64 wrap is harmless since m | 2^64
        keep &= (cur >> np.uint64(shift)) == np.uint64(o)
    return [int(v) for v in alive[keep]]


def recover_java_random_state(outputs: list[int]) -> LCG | None:
    """Recover a ``java.util.Random`` from ≥2 consecutive ``next(32)`` outputs.

    Returns a generator positioned after the last observed output, or ``None``.
    """
    if len(outputs) < 2:
        raise ValueError("need at least two outputs")
    a, c, m = LCG_PARAMS_JAVA
    states = recover_truncated_state([o & 0xFFFFFFFF for o in outputs], a, c, m, 16)
    if len(states) != 1:
        return None
    x = states[0]
    for _ in range(len(outputs) - 1):
        x = (a * x + c) % m
    g = LCG(0, a, c, m, shift=16, out_bits=32)
    g.x = x
    return g


LCG_PARAMS_JAVA = (0x5DEECE66D, 0xB, 1 << 48)
