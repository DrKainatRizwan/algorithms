"""PERT three-point estimation, analytic completion probability and Monte Carlo."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .cpm import Activity, critical_path_method, build_graph
from .graph import kahn_order


@dataclass(frozen=True)
class PertActivity:
    name: str
    optimistic: float
    likely: float
    pessimistic: float
    predecessors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.optimistic <= self.likely <= self.pessimistic:
            raise ValueError(f"{self.name}: need optimistic <= likely <= pessimistic")

    @property
    def mean(self) -> float:
        return (self.optimistic + 4 * self.likely + self.pessimistic) / 6.0

    @property
    def variance(self) -> float:
        return ((self.pessimistic - self.optimistic) / 6.0) ** 2


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class PertResult:
    expected: float
    std: float
    critical: list[str]

    def prob_complete_by(self, deadline: float) -> float:
        if self.std == 0:
            return 1.0 if deadline >= self.expected else 0.0
        return normal_cdf((deadline - self.expected) / self.std)

    def deadline_for(self, confidence: float) -> float:
        """Deadline met with given probability (inverse normal by bisection)."""
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        lo, hi = -10.0, 10.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if normal_cdf(mid) < confidence:
                lo = mid
            else:
                hi = mid
        return self.expected + self.std * (lo + hi) / 2


def pert_analysis(acts: Sequence[PertActivity]) -> PertResult:
    """Classic PERT: CPM on mean durations; variance summed along the critical path."""
    base = [Activity(a.name, a.mean, a.predecessors) for a in acts]
    res = critical_path_method(base)
    var = {a.name: a.variance for a in acts}
    # pick the critical chain with the largest variance (conservative)
    g = build_graph(base)
    crit = set(res.critical_activities)
    best: Dict[str, tuple[float, list[str]]] = {}
    for u in res.order:
        if u not in crit:
            continue
        cands = [best[p] for p in g.pred[u] if p in best and abs(res.ef[p] - res.es[u]) < 1e-9]
        pv, pp = max(cands, key=lambda t: t[0]) if cands else (0.0, [])
        best[u] = (pv + var[u], pp + [u])
    ends = [best[u] for u in crit if abs(res.ef[u] - res.duration) < 1e-9]
    v, path = max(ends, key=lambda t: t[0]) if ends else (0.0, [])
    return PertResult(res.duration, math.sqrt(v), path)


def _beta_pert_samples(a: PertActivity, rng: np.random.Generator, n: int) -> np.ndarray:
    lo, hi = a.optimistic, a.pessimistic
    if hi == lo:
        return np.full(n, lo)
    alpha = 1 + 4 * (a.likely - lo) / (hi - lo)
    beta = 1 + 4 * (hi - a.likely) / (hi - lo)
    return lo + (hi - lo) * rng.beta(alpha, beta, size=n)


def monte_carlo_duration(acts: Sequence[PertActivity], n: int = 5000, seed: int = 0) -> np.ndarray:
    """Sample project durations with Beta-PERT activity durations (vectorised forward pass)."""
    rng = np.random.default_rng(seed)
    base = [Activity(a.name, a.mean, a.predecessors) for a in acts]
    g = build_graph(base)
    samples = {a.name: _beta_pert_samples(a, rng, n) for a in acts}
    ef: Dict[str, np.ndarray] = {}
    for u in kahn_order(g):
        start = np.zeros(n)
        for p in g.pred[u]:
            start = np.maximum(start, ef[p])
        ef[u] = start + samples[u]
    return np.max(np.stack(list(ef.values())), axis=0)


def criticality_index(acts: Sequence[PertActivity], n: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Fraction of simulations in which each activity has zero total float."""
    rng = np.random.default_rng(seed)
    counts = {a.name: 0 for a in acts}
    for _ in range(n):
        drawn = [Activity(a.name, float(_beta_pert_samples(a, rng, 1)[0]), a.predecessors) for a in acts]
        for name in critical_path_method(drawn).critical_activities:
            counts[name] += 1
    return {k: v / n for k, v in counts.items()}
