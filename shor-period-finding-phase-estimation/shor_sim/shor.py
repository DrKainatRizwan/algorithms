"""Shor's factoring algorithm driven by simulated quantum order finding."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .modular import gcd, is_probable_prime, perfect_power
from .period import OrderFindingRun, sample_order


@dataclass
class ShorResult:
    """Outcome of a factoring attempt, with a trace of the reduction."""
    n: int
    factors: tuple[int, int] | None
    method: str = ""                      # how the factor was found
    attempts: list[dict] = field(default_factory=list)
    quantum_runs: int = 0

    @property
    def success(self) -> bool:
        return self.factors is not None


def factors_from_order(a: int, r: int, n: int) -> tuple[int, int] | None:
    """Classical post-processing: turn an even order into a non-trivial split."""
    if r % 2:
        return None
    x = pow(a, r // 2, n)
    if x == n - 1:
        return None
    for cand in (gcd(x - 1, n), gcd(x + 1, n)):
        if 1 < cand < n:
            return tuple(sorted((cand, n // cand)))  # type: ignore[return-value]
    return None


def shor_factor(n: int, rng: np.random.Generator, backend: str = "compressed",
                max_attempts: int = 30) -> ShorResult:
    """Factor ``n`` (composite, > 3) with simulated Shor order finding."""
    if n < 4:
        raise ValueError("n must be at least 4")
    res = ShorResult(n, None)
    if n % 2 == 0:
        res.factors, res.method = (2, n // 2), "even"
        return res
    if is_probable_prime(n):
        res.method = "prime"
        return res
    pp = perfect_power(n)
    if pp is not None:
        b, k = pp
        res.factors, res.method = (b, n // b), f"perfect-power(k={k})"
        return res
    for _ in range(max_attempts):
        a = int(rng.integers(2, n))
        rec: dict = {"a": a}
        g = gcd(a, n)
        if g > 1:
            rec["lucky_gcd"] = g
            res.attempts.append(rec)
            res.factors, res.method = tuple(sorted((g, n // g))), "lucky-gcd"  # type: ignore
            return res
        run: OrderFindingRun = sample_order(a, n, rng, backend)
        res.quantum_runs += 1
        rec.update(measurement=run.measurement, candidate=run.candidate, order=run.order)
        res.attempts.append(rec)
        if run.order:
            f = factors_from_order(a, run.order, n)
            if f:
                res.factors, res.method = f, "order-finding"
                return res
    return res
