"""Continuous trust scoring in log-odds space with exponential decay.

Each user carries a log-odds trust value ``l``.  Evidence (signals) adds a
signed weight, and between events the value relaxes towards the prior with a
configurable half-life, so that both bad and good evidence fade::

    l(t) = l0 + (l - l0) * 2^(-dt / half_life)
    score = sigmoid(l)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1.0 - p))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class Signal:
    """A named piece of evidence with a log-odds weight (negative = suspicious)."""

    name: str
    weight: float


@dataclass
class _State:
    logit: float
    last_ts: float
    history: list[tuple[float, float]] = field(default_factory=list)


class TrustScorer:
    """Per-subject trust tracker."""

    def __init__(
        self,
        prior: float = 0.8,
        half_life: float = 3600.0,
        min_score: float = 0.001,
        max_score: float = 0.999,
    ):
        if not 0 < prior < 1:
            raise ValueError("prior must be in (0, 1)")
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.prior = prior
        self._prior_logit = logit(prior)
        self.half_life = half_life
        self._lo = logit(min_score)
        self._hi = logit(max_score)
        self._states: dict[str, _State] = {}

    def _decayed(self, st: _State, ts: float) -> float:
        dt = max(0.0, ts - st.last_ts)
        k = 0.5 ** (dt / self.half_life)
        return self._prior_logit + (st.logit - self._prior_logit) * k

    def score(self, subject: str, ts: float | None = None) -> float:
        """Current trust of ``subject`` (optionally projected to time ``ts``)."""
        st = self._states.get(subject)
        if st is None:
            return self.prior
        return sigmoid(self._decayed(st, st.last_ts if ts is None else ts))

    def update(self, subject: str, ts: float, signals: list[Signal]) -> float:
        """Apply ``signals`` at time ``ts`` and return the new score."""
        st = self._states.get(subject)
        if st is None:
            st = _State(self._prior_logit, ts)
            self._states[subject] = st
        value = self._decayed(st, ts) + sum(s.weight for s in signals)
        value = min(max(value, self._lo), self._hi)
        st.logit, st.last_ts = value, ts
        st.history.append((ts, sigmoid(value)))
        return sigmoid(value)

    def history(self, subject: str) -> list[tuple[float, float]]:
        st = self._states.get(subject)
        return list(st.history) if st else []

    def reset(self, subject: str) -> None:
        self._states.pop(subject, None)
