"""Fast correlation attack (Meier-Staffelbach style) using low-weight parity checks.

When the feedback polynomial ``f`` is sparse, its powers ``f(x)^(2^k) = f(x^(2^k))``
are equally sparse and every multiple gives a parity check on the LFSR
sequence: ``sum_j s_{t + e_j} = 0``.  Each noisy keystream bit then takes part
in many checks; bits whose checks are mostly violated are likely corrupted.
Iterating this "flip the worst bits" step corrects the noise without the
``2^n`` search of the plain correlation attack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lfsr import LFSR, recover_state


def parity_checks(lfsr: LFSR, nbits: int, max_checks: int | None = None) -> np.ndarray:
    """Return an array of shape ``(k, w)`` of relative offsets of weight-``w`` checks.

    Offsets come from ``f(x^(2^k))`` for all ``k`` with degree ``n 2^k < nbits``.
    Each row ``e`` means ``s_{t+e_0} + ... + s_{t+e_{w-1}} = 0`` for all valid ``t``.
    """
    exps = np.array(lfsr.poly_exponents(), dtype=np.int64)
    rows = []
    k = 0
    while lfsr.length * (1 << k) < nbits:
        rows.append(exps * (1 << k))
        k += 1
        if max_checks is not None and len(rows) >= max_checks:
            break
    return np.array(rows, dtype=np.int64)


def check_satisfaction(z: np.ndarray, checks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every position count satisfied / total checks that involve it."""
    n = len(z)
    sat = np.zeros(n, dtype=np.int64)
    tot = np.zeros(n, dtype=np.int64)
    zi = z.astype(np.int64)
    for e in checks:
        span = int(e.max())
        if span >= n:
            continue
        m = n - span
        s = np.zeros(m, dtype=np.int64)
        for off in e:
            s ^= zi[off : off + m]
        ok = (s == 0).astype(np.int64)
        for off in e:
            sat[off : off + m] += ok
            tot[off : off + m] += 1
    return sat, tot


@dataclass
class FCAResult:
    state: int | None
    iterations: int
    residual_violations: int
    corrected: np.ndarray


def fast_correlation_attack(
    lfsr: LFSR,
    z: np.ndarray,
    p: float,
    max_rounds: int = 60,
    seed: int = 0,
) -> FCAResult:
    """Iteratively decode the noisy keystream and recover the LFSR state.

    Each round flips every bit whose fraction of satisfied checks is below a
    threshold (relaxed as decoding progresses).  The loop stops when all checks
    hold; the state is then read from the first ``n`` corrected bits.  If
    flipping stalls, a few random low-confidence bits are perturbed
    (deterministically, from ``seed``).
    """
    rng = np.random.default_rng(seed)
    cur = z.copy().astype(np.uint8)
    checks = parity_checks(lfsr, len(z))
    n = lfsr.length
    best_viol = None
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        sat, tot = check_satisfaction(cur, checks)
        viol = int(np.sum(tot - sat)) // len(checks[0])
        if best_viol is None or viol < best_viol:
            best_viol = viol
        if viol == 0:
            break
        frac = np.where(tot > 0, sat / np.maximum(tot, 1), 1.0)
        # threshold: a bit with agreement below 1/2 of checks is likely wrong
        thr = 0.5 if rounds <= 3 else 0.45
        flip = (frac < thr) & (tot > 0)
        if not flip.any():
            worst = np.argsort(frac)[: max(1, len(cur) // 200)]
            flip = np.zeros(len(cur), dtype=bool)
            flip[rng.choice(worst, size=min(len(worst), 3), replace=False)] = True
        cur[flip] ^= 1
    sat, tot = check_satisfaction(cur, checks)
    viol = int(np.sum(tot - sat)) // len(checks[0])
    state = recover_state(lfsr, [int(b) for b in cur[:n]])
    if state is not None and viol != 0:
        # accept a state only if it explains almost all of the keystream
        if np.mean(lfsr.stream(state, len(z)) == z) < 0.5 + (1 - 2 * p) / 4:
            state = None
    return FCAResult(state, rounds, viol, cur)
