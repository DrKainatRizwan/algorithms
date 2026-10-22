"""Closed-form predictions and parameter sweeps."""

from __future__ import annotations

import math

import numpy as np

from .channel import Channel, InterceptResend
from .privacy import secret_fraction
from .protocol import BB84Config, run_bb84


def expected_qber_intercept(fraction: float, misalignment: float = 0.0) -> float:
    """Expected QBER with intercept-resend on ``fraction`` of photons.

    Eve induces a 25% error rate on attacked photons; a rotation by theta
    adds ``sin^2(theta)`` independent flips.
    """
    e0 = math.sin(misalignment) ** 2
    ee = 0.25 * fraction
    return e0 + ee - 2 * e0 * ee


def detection_probability(fraction: float, n_check: int) -> float:
    """Probability that comparing ``n_check`` sifted bits exposes Eve."""
    return 1.0 - (1.0 - 0.25 * fraction) ** n_check


def hoeffding_margin(n: int, eps: float) -> float:
    """One-sided deviation of an empirical rate from its mean w.p. ``1-eps``."""
    return math.sqrt(math.log(1.0 / eps) / (2.0 * n))


def qber_threshold(tol: float = 1e-4) -> float:
    """Largest QBER with positive asymptotic key rate (about 11%)."""
    lo, hi = 0.0, 0.5
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if secret_fraction(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def sweep_eavesdropping(
    fractions: list[float], n_photons: int = 4000, trials: int = 5, seed: int = 0
) -> list[dict]:
    """Empirical QBER, abort rate and final key length per attack fraction."""
    rows = []
    for f in fractions:
        qs, aborts, lens = [], 0, []
        for t in range(trials):
            ch = Channel(loss=0.1, eve=InterceptResend(f) if f > 0 else None)
            r = run_bb84(BB84Config(n_photons=n_photons, seed=seed + 97 * t), ch)
            qs.append(r.true_qber)
            aborts += r.aborted
            lens.append(r.final_key_alice.size)
        rows.append({
            "fraction": f, "qber": float(np.mean(qs)),
            "expected": expected_qber_intercept(f),
            "abort_rate": aborts / trials, "key_bits": float(np.mean(lens)),
        })
    return rows
