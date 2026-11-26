"""Local-differential-privacy frequency oracles.

Every oracle here follows the same pattern.  A user holding a private value
``x`` in ``{0, ..., d-1}`` perturbs it locally with a randomiser that satisfies
``epsilon``-LDP and sends the *report* to an untrusted aggregator.  The
aggregator counts, for each candidate value ``v``, how many reports "support"
``v``.  If a report supports the user's true value with probability ``p`` and
supports any other value with probability ``q``, then

    f_hat(v) = (C(v) / n - q) / (p - q)

is an unbiased estimate of the population frequency of ``v`` with variance

    pi (1 - pi) / (n (p - q)^2),   pi = q + f (p - q).

Implemented randomisers: generalised randomised response (GRR / direct
encoding), unary encoding (symmetric = basic RAPPOR, optimised = OUE),
optimised local hashing (OLH) and Hadamard response (HR).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

HASH_PRIME = 2_147_483_647  # 2**31 - 1, keeps a*x + b inside int64


def _check_domain(d: int, epsilon: float) -> None:
    if d < 2:
        raise ValueError("domain size must be at least 2")
    if not epsilon > 0:
        raise ValueError("epsilon must be positive")


def _check_values(values: np.ndarray, d: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("values must be a 1-D array")
    if values.size and (values.min() < 0 or values.max() >= d):
        raise ValueError("values outside the domain")
    return values


def ue_epsilon(p: float, q: float) -> float:
    """Privacy loss of a one-hot unary encoding with bit probabilities (p, q)."""
    return float(np.log(p * (1 - q) / (q * (1 - p))))


def ue_channel(d: int, p: float, q: float) -> np.ndarray:
    """Exact ``d x 2**d`` channel matrix of a unary-encoding randomiser."""
    if d > 14:
        raise ValueError("channel enumeration limited to d <= 14")
    outs = np.arange(2 ** d)
    bits = ((outs[:, None] >> np.arange(d)) & 1).astype(bool)
    chan = np.empty((d, 2 ** d))
    for x in range(d):
        probs = np.full(d, q)
        probs[x] = p
        chan[x] = np.prod(np.where(bits, probs, 1 - probs), axis=1)
    return chan


class FrequencyOracle(ABC):
    """Base class: perturb on the client, estimate on the server."""

    name = "oracle"

    def __init__(self, d: int, epsilon: float, p: float, q: float) -> None:
        _check_domain(d, epsilon)
        self.d = int(d)
        self.epsilon = float(epsilon)
        self.p = float(p)
        self.q = float(q)

    # -- client side -----------------------------------------------------
    @abstractmethod
    def perturb(self, values: np.ndarray, rng: np.random.Generator) -> Any:
        """Randomise a batch of private values, one report per user."""

    # -- server side -----------------------------------------------------
    @abstractmethod
    def support_counts(self, reports: Any) -> np.ndarray:
        """Number of reports supporting each of the ``d`` values."""

    @abstractmethod
    def num_reports(self, reports: Any) -> int:
        """Number of users represented in ``reports``."""

    def estimate(self, reports: Any) -> np.ndarray:
        """Unbiased frequency estimates (may be negative or exceed 1)."""
        n = self.num_reports(reports)
        counts = self.support_counts(reports)
        return (counts / n - self.q) / (self.p - self.q)

    def run(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Perturb and aggregate in one call."""
        return self.estimate(self.perturb(values, rng))

    def variance(self, n: int, f: float = 0.0) -> float:
        """Variance of the estimate of an item with true frequency ``f``."""
        pi = self.q + f * (self.p - self.q)
        return float(pi * (1 - pi) / (n * (self.p - self.q) ** 2))

    # -- analysis --------------------------------------------------------
    @abstractmethod
    def channel_matrix(self) -> np.ndarray:
        """Exact conditional distribution of the report given the input."""

    @property
    @abstractmethod
    def report_bits(self) -> float:
        """Approximate communication cost of one report in bits."""


class GeneralizedRandomizedResponse(FrequencyOracle):
    """k-ary randomised response: report the truth w.p. ``p``, else a lie."""

    name = "GRR"

    def __init__(self, d: int, epsilon: float) -> None:
        _check_domain(d, epsilon)
        e = np.exp(epsilon)
        p = e / (e + d - 1)
        super().__init__(d, epsilon, p, (1 - p) / (d - 1))

    def perturb(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        values = _check_values(values, self.d)
        keep = rng.random(values.size) < self.p
        shift = rng.integers(1, self.d, size=values.size)  # uniform over the other d-1
        return np.where(keep, values, (values + shift) % self.d)

    def num_reports(self, reports: np.ndarray) -> int:
        return int(reports.size)

    def support_counts(self, reports: np.ndarray) -> np.ndarray:
        return np.bincount(reports, minlength=self.d).astype(float)

    def channel_matrix(self) -> np.ndarray:
        chan = np.full((self.d, self.d), self.q)
        np.fill_diagonal(chan, self.p)
        return chan

    @property
    def report_bits(self) -> float:
        return float(np.log2(self.d))


class UnaryEncoding(FrequencyOracle):
    """One-hot encode, then flip each bit independently.

    ``variant='symmetric'`` is basic RAPPOR (``p = 1 - q``); ``'optimized'`` is
    OUE with ``p = 1/2`` and ``q = 1/(e^eps + 1)``, which minimises variance.
    """

    def __init__(self, d: int, epsilon: float, variant: str = "optimized") -> None:
        _check_domain(d, epsilon)
        if variant == "symmetric":
            h = np.exp(epsilon / 2)
            p, q = h / (h + 1), 1 / (h + 1)
            self.name = "SUE"
        elif variant == "optimized":
            p, q = 0.5, 1 / (np.exp(epsilon) + 1)
            self.name = "OUE"
        else:
            raise ValueError("variant must be 'symmetric' or 'optimized'")
        self.variant = variant
        super().__init__(d, epsilon, p, q)

    def perturb(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        values = _check_values(values, self.d)
        n = values.size
        out = np.empty((n, self.d), dtype=bool)
        for lo in range(0, n, 20_000):  # chunked to bound the float temporary
            hi = min(lo + 20_000, n)
            out[lo:hi] = rng.random((hi - lo, self.d)) < self.q
        out[np.arange(n), values] = rng.random(n) < self.p
        return out

    def num_reports(self, reports: np.ndarray) -> int:
        return int(reports.shape[0])

    def support_counts(self, reports: np.ndarray) -> np.ndarray:
        return reports.sum(axis=0).astype(float)

    def channel_matrix(self) -> np.ndarray:
        return ue_channel(self.d, self.p, self.q)

    @property
    def report_bits(self) -> float:
        return float(self.d)


def universal_hash(a: np.ndarray, b: np.ndarray, x: np.ndarray, g: int) -> np.ndarray:
    """Carter-Wegman hash ``((a x + b) mod P) mod g`` (vectorised)."""
    return ((a * x + b) % HASH_PRIME) % g


class OptimizedLocalHashing(FrequencyOracle):
    """OLH: hash to ``g ~ e^eps + 1`` buckets, then apply GRR on the bucket.

    A report is the triple ``(a, b, y)`` identifying the user's hash function
    and the perturbed bucket.  Communication is O(log g) and the variance does
    not depend on ``d``.
    """

    name = "OLH"

    def __init__(self, d: int, epsilon: float, g: int | None = None) -> None:
        _check_domain(d, epsilon)
        e = np.exp(epsilon)
        self.g = int(g) if g is not None else max(2, int(round(e + 1)))
        if self.g < 2:
            raise ValueError("g must be at least 2")
        p = e / (e + self.g - 1)
        super().__init__(d, epsilon, p, 1.0 / self.g)

    def perturb(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        values = _check_values(values, self.d)
        n = values.size
        a = rng.integers(1, HASH_PRIME, size=n, dtype=np.int64)
        b = rng.integers(0, HASH_PRIME, size=n, dtype=np.int64)
        z = universal_hash(a, b, values, self.g)
        keep = rng.random(n) < self.p
        shift = rng.integers(1, self.g, size=n)
        y = np.where(keep, z, (z + shift) % self.g)
        return np.stack([a, b, y], axis=1)

    def num_reports(self, reports: np.ndarray) -> int:
        return int(reports.shape[0])

    def support_counts(self, reports: np.ndarray) -> np.ndarray:
        a, b, y = reports[:, 0], reports[:, 1], reports[:, 2]
        counts = np.empty(self.d)
        step = 64
        for lo in range(0, self.d, step):
            vs = np.arange(lo, min(lo + step, self.d), dtype=np.int64)
            h = universal_hash(a[:, None], b[:, None], vs[None, :], self.g)
            counts[lo:lo + step] = (h == y[:, None]).sum(axis=0)
        return counts

    def channel_matrix(self, a: int = 1, b: int = 0) -> np.ndarray:
        """Channel ``d x g`` for one fixed hash function (the hash is public)."""
        z = universal_hash(np.int64(a), np.int64(b), np.arange(self.d, dtype=np.int64), self.g)
        grr = np.full((self.g, self.g), (1 - self.p) / (self.g - 1))
        np.fill_diagonal(grr, self.p)
        return grr[z]

    @property
    def report_bits(self) -> float:
        return 2 * 31 + float(np.log2(self.g))


def fwht(vec: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform (unnormalised), O(K log K)."""
    out = np.array(vec, dtype=float)
    k = out.size
    if k & (k - 1):
        raise ValueError("length must be a power of two")
    h = 1
    while h < k:
        out = out.reshape(-1, 2, h)
        out = np.stack([out[:, 0] + out[:, 1], out[:, 0] - out[:, 1]], axis=1)
        out = out.reshape(-1)
        h *= 2
    return out


def parity(x: np.ndarray) -> np.ndarray:
    """Bit parity of each (non-negative, < 2**32) integer."""
    x = np.asarray(x, dtype=np.int64).copy()
    for s in (16, 8, 4, 2, 1):
        x ^= x >> s
    return x & 1


class HadamardResponse(FrequencyOracle):
    """Hadamard response (Acharya-Sun-Zhang style).

    Value ``x`` is associated with row ``x + 1`` of a ``K x K`` Hadamard matrix
    (``K`` the smallest power of two above ``d``).  The report is a column index
    ``y``, drawn uniformly from the ``K/2`` columns where that row is ``+1``
    with probability ``e^eps / (e^eps + 1)`` and from the others otherwise.
    Reports need only ``log2 K`` bits and the server decodes every item at once
    with one fast Walsh-Hadamard transform.
    """

    name = "HR"

    def __init__(self, d: int, epsilon: float) -> None:
        _check_domain(d, epsilon)
        k = 1
        while k < d + 1:
            k *= 2
        self.k = k
        e = np.exp(epsilon)
        super().__init__(d, epsilon, e / (e + 1), 0.5)

    def perturb(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        values = _check_values(values, self.d)
        rows = values + 1
        want = np.where(rng.random(values.size) < self.p, 0, 1)  # target parity
        y = np.zeros(values.size, dtype=np.int64)
        todo = np.arange(values.size)
        while todo.size:  # rejection sampling, accepts w.p. 1/2 per round
            cand = rng.integers(0, self.k, size=todo.size)
            ok = parity(rows[todo] & cand) == want[todo]
            y[todo[ok]] = cand[ok]
            todo = todo[~ok]
        return y

    def num_reports(self, reports: np.ndarray) -> int:
        return int(reports.size)

    def support_counts(self, reports: np.ndarray) -> np.ndarray:
        hist = np.bincount(reports, minlength=self.k).astype(float)
        w = fwht(hist)[1:self.d + 1]  # sum_y H[v+1, y] hist[y]
        return (reports.size + w) / 2

    def channel_matrix(self) -> np.ndarray:
        cols = np.arange(self.k)
        chan = np.empty((self.d, self.k))
        for x in range(self.d):
            plus = parity((x + 1) & cols) == 0
            chan[x] = np.where(plus, self.p, 1 - self.p) / (self.k / 2)
        return chan

    @property
    def report_bits(self) -> float:
        return float(np.log2(self.k))
