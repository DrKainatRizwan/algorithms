"""Analytical predictions for gossip on the complete graph."""
from __future__ import annotations

import math


def push_rounds_estimate(n: int) -> float:
    """Frieze-Grimmett / Pittel: push needs log2(n) + ln(n) + O(1) rounds w.h.p."""
    return math.log2(n) + math.log(n)


def mean_field_curve(mode: str, n: int, rounds: int, fanout: int = 1,
                     loss: float = 0.0) -> list[float]:
    """Expected informed fraction per round (mean-field recurrence, K_n).

    With informed fraction p and per-contact success q = 1 - loss:
      push:       p' = 1 - (1-p) * exp(-fanout*q*p)
      pull:       p' = p + (1-p) * fanout*q*p  (capped via 1-(1-q p)^fanout)
      push-pull:  both mechanisms applied independently.
    """
    if mode not in ("push", "pull", "push-pull"):
        raise ValueError("bad mode")
    q = 1.0 - loss
    p = 1.0 / n
    out = [p]
    for _ in range(rounds):
        s = 1.0 - p
        if mode == "push":
            s *= math.exp(-fanout * q * p)
        elif mode == "pull":
            s *= (1.0 - q * p) ** fanout
        else:
            s *= math.exp(-fanout * q * p) * (1.0 - q * p) ** fanout
        p = 1.0 - s
        out.append(p)
    return out


def mean_field_rounds(mode: str, n: int, fanout: int = 1, loss: float = 0.0,
                      threshold: float | None = None, max_rounds: int = 500) -> int:
    """First round at which the expected uninformed count drops below 1/2
    (or the informed fraction reaches ``threshold`` if given)."""
    q = 1.0 - loss
    p = 1.0 / n
    target = threshold if threshold is not None else 1.0 - 0.5 / n
    for r in range(max_rounds + 1):
        if p >= target:
            return r
        s = 1.0 - p
        if mode == "push":
            s *= math.exp(-fanout * q * p)
        elif mode == "pull":
            s *= (1.0 - q * p) ** fanout
        else:
            s *= math.exp(-fanout * q * p) * (1.0 - q * p) ** fanout
        p = 1.0 - s
    return max_rounds


def rumor_residue(k: float, tol: float = 1e-12) -> float:
    """Solve s = exp(-(k+1)(1-s)) for the residue of feedback-coin rumor
    mongering (Demers et al.). Returns the smaller (non-trivial) root."""
    if k <= 0:
        raise ValueError("k must be positive")
    f = lambda s: s - math.exp(-(k + 1.0) * (1.0 - s))
    lo, hi = 0.0, 1.0 - 1e-9
    # f(0) < 0 and f is negative up to the root, then positive until the trivial root at 1.
    # Locate a point with f > 0 by scanning.
    xs = [i / 1000.0 for i in range(1, 1000)]
    for x in xs:
        if f(x) > 0:
            hi = x
            break
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
