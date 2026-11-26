"""RAPPOR with permanent and instantaneous randomised response.

Erlingsson, Pihur and Korolova (2014): the client first computes a *permanent*
noisy bit vector ``B'`` (memoised forever, so repeated reports cannot be
averaged away) and then, for every report, a fresh *instantaneous* response
``S`` derived from ``B'``.  Here the encoding is one-hot (no Bloom filter), so
the longitudinal and single-report guarantees can be stated exactly.
"""
from __future__ import annotations

import numpy as np

from .oracles import FrequencyOracle, UnaryEncoding, ue_channel, ue_epsilon


class Rappor(FrequencyOracle):
    """One-hot RAPPOR parameterised by ``f``, ``p`` and ``q``.

    ``B'_i = 1`` w.p. ``f/2``, ``0`` w.p. ``f/2``, otherwise the true bit.
    ``S_i = 1`` w.p. ``q`` if ``B'_i = 0`` and w.p. ``p`` if ``B'_i = 1``.
    """

    name = "RAPPOR"

    def __init__(self, d: int, f: float = 0.5, p: float = 0.75, q: float = 0.25) -> None:
        if not 0 < f < 1:
            raise ValueError("f must lie in (0, 1)")
        if not 0 <= q < p <= 1:
            raise ValueError("require 0 <= q < p <= 1")
        self.f, self.p_inst, self.q_inst = f, p, q
        hi = 1 - f / 2  # P(B'=1 | bit=1)
        lo = f / 2      # P(B'=1 | bit=0)
        p_star = hi * p + (1 - hi) * q
        q_star = lo * p + (1 - lo) * q
        super().__init__(d, ue_epsilon(p_star, q_star), p_star, q_star)

    @property
    def epsilon_permanent(self) -> float:
        """Longitudinal privacy loss ``eps_inf`` (bounded however many rounds)."""
        return ue_epsilon(1 - self.f / 2, self.f / 2)

    @classmethod
    def from_permanent_epsilon(cls, d: int, eps_inf: float, p: float = 0.75,
                               q: float = 0.25) -> "Rappor":
        """Choose ``f`` so that the permanent response is ``eps_inf``-LDP."""
        f = 2.0 / (1.0 + np.exp(eps_inf / 2))
        return cls(d, f=f, p=p, q=q)

    # -- client ----------------------------------------------------------
    def permanent_response(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Memoisable noisy bit matrix ``B'`` (``n x d``)."""
        values = np.asarray(values, dtype=np.int64)
        n = values.size
        bits = np.zeros((n, self.d), dtype=bool)
        bits[np.arange(n), values] = True
        r = rng.random((n, self.d))
        out = bits.copy()
        out[r < self.f / 2] = True
        out[(r >= self.f / 2) & (r < self.f)] = False
        return out

    def instantaneous_response(self, permanent: np.ndarray,
                               rng: np.random.Generator) -> np.ndarray:
        """Fresh per-report randomisation of the memoised vector."""
        r = rng.random(permanent.shape)
        return np.where(permanent, r < self.p_inst, r < self.q_inst)

    def perturb(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.instantaneous_response(self.permanent_response(values, rng), rng)

    # -- server ----------------------------------------------------------
    def num_reports(self, reports: np.ndarray) -> int:
        return int(reports.shape[0])

    def support_counts(self, reports: np.ndarray) -> np.ndarray:
        return reports.sum(axis=0).astype(float)

    def channel_matrix(self) -> np.ndarray:
        return ue_channel(self.d, self.p, self.q)

    @property
    def report_bits(self) -> float:
        return float(self.d)

    def equivalent_unary(self) -> UnaryEncoding:
        """The symmetric unary encoding with the same single-report privacy."""
        return UnaryEncoding(self.d, self.epsilon, "symmetric")


def longitudinal_attack_accuracy(rappor: Rappor, values: np.ndarray, rounds: int,
                                 memoize: bool, rng: np.random.Generator) -> float:
    """Accuracy of an attacker who averages ``rounds`` reports per user.

    With memoisation the average converges to the permanent response, which is
    still ``eps_inf``-private.  Without it every round carries independent
    noise and the true value is eventually recovered exactly.
    """
    values = np.asarray(values, dtype=np.int64)
    perm = rappor.permanent_response(values, rng) if memoize else None
    total = np.zeros((values.size, rappor.d))
    for _ in range(rounds):
        base = perm if memoize else rappor.permanent_response(values, rng)
        total += rappor.instantaneous_response(base, rng)
    return float(np.mean(total.argmax(axis=1) == values))
