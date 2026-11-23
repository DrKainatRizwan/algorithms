"""Analytical coverage predictions and empirical success measurement."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from .oneway import OneWayFunction


class _Attack(Protocol):
    false_alarms: int

    def invert(self, y: int) -> int | None: ...


def hellman_coverage(n: int, m: int, t: int, tables: int = 1) -> float:
    """Hellman's lower bound on the fraction of points covered.

    Single table: ``(1/N) * sum_{i=1..m} sum_{j=0..t-1} max(0, 1 - i*t/N)**(j+1)``;
    independent tables combine as ``1 - (1 - p)**l``.
    """
    total = 0.0
    for i in range(1, m + 1):
        base = max(0.0, 1.0 - i * t / n)
        if base == 0.0:
            break
        # geometric series sum_{j=1..t} base**j
        if base == 1.0:
            total += t
        else:
            total += base * (1.0 - base**t) / (1.0 - base)
    p = min(1.0, total / n)
    return 1.0 - (1.0 - p) ** tables


def rainbow_coverage(n: int, m: int, t: int) -> float:
    """Oechslin's success probability for one rainbow table.

    ``m_1 = m`` and ``m_{k+1} = N (1 - exp(-m_k / N))`` give the number of
    distinct points per column; ``P = 1 - prod_k (1 - m_k / N)``.
    """
    mk = float(m)
    prod = 1.0
    for _ in range(t):
        prod *= 1.0 - mk / n
        mk = n * (1.0 - math.exp(-mk / n))
    return 1.0 - prod


def dp_expected_chain_length(d: int) -> float:
    """Mean chain length for ``d`` distinguished bits (geometric: ``2**d``)."""
    return float(1 << d)


@dataclass
class AttackStats:
    """Empirical outcome of an inversion campaign."""

    trials: int
    successes: int
    mean_online_evals: float
    false_alarms: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


def measure_success(
    attack: _Attack, f: OneWayFunction, trials: int = 200, seed: int = 1
) -> AttackStats:
    """Invert ``f(x)`` for random ``x``; each returned preimage is verified."""
    rng = random.Random(seed)
    attack.false_alarms = 0
    f.reset_counter()
    ok = 0
    for _ in range(trials):
        y = f(rng.randrange(f.size))
        found = attack.invert(y)
        if found is not None and f(found) == y:
            ok += 1
    # exclude the challenge evaluation and the verification of each found preimage
    evals = f.reset_counter() - trials - ok
    return AttackStats(trials, ok, evals / trials, attack.false_alarms)
