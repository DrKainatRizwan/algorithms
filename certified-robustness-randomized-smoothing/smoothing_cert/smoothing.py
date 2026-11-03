"""Randomized smoothing (Cohen, Rosenfeld & Kolter, 2019) with Gaussian noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import binom_two_sided_p, clopper_pearson_lower, norm_ppf

ABSTAIN = -1


@dataclass(frozen=True)
class CertifyResult:
    """Outcome of :meth:`SmoothedClassifier.certify`."""
    label: int          # predicted class, or ABSTAIN
    radius: float       # certified L2 radius (0.0 when abstaining)
    p_lower: float      # lower confidence bound on top-class probability


class SmoothedClassifier:
    """g(x) = argmax_c P[f(x + N(0, sigma^2 I)) = c] built from a base classifier ``f``.

    ``base`` must expose ``predict(X) -> int array`` over a batch.
    """

    def __init__(self, base, num_classes: int, sigma: float, seed: int = 0,
                 chunk: int = 2000):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.base = base
        self.num_classes = num_classes
        self.sigma = float(sigma)
        self.chunk = chunk
        self.rng = np.random.default_rng(seed)

    def sample_counts(self, x: np.ndarray, n: int) -> np.ndarray:
        """Class histogram of base predictions on ``n`` noisy copies of ``x``."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        left = n
        while left > 0:
            m = min(left, self.chunk)
            noisy = x[None, :] + self.sigma * self.rng.normal(size=(m, x.size))
            counts += np.bincount(self.base.predict(noisy), minlength=self.num_classes)
            left -= m
        return counts

    def predict(self, x: np.ndarray, n: int = 1000, alpha: float = 0.001) -> int:
        """PREDICT: abstain unless the top class beats the runner-up in a binomial test."""
        counts = self.sample_counts(x, n)
        order = np.argsort(-counts, kind="stable")
        na, nb = int(counts[order[0]]), int(counts[order[1]])
        if binom_two_sided_p(na, na + nb, 0.5) > alpha:
            return ABSTAIN
        return int(order[0])

    def certify(self, x: np.ndarray, n0: int = 100, n: int = 1000,
                alpha: float = 0.001) -> CertifyResult:
        """CERTIFY: select a class with ``n0`` samples, bound its probability with ``n`` more."""
        guess = int(np.argmax(self.sample_counts(x, n0)))
        k = int(self.sample_counts(x, n)[guess])
        p_lower = clopper_pearson_lower(k, n, alpha)
        if p_lower < 0.5:
            return CertifyResult(ABSTAIN, 0.0, p_lower)
        return CertifyResult(guess, self.sigma * norm_ppf(p_lower), p_lower)

    @staticmethod
    def radius_from_pa(p_a: float, sigma: float, p_b: float | None = None) -> float:
        """Certified radius (sigma/2)(Phi^-1(pA) - Phi^-1(pB)); pB defaults to 1 - pA."""
        if p_b is None:
            p_b = 1.0 - p_a
        p_a = min(p_a, 1 - 1e-12)
        p_b = max(p_b, 1e-12)
        return 0.5 * sigma * (norm_ppf(p_a) - norm_ppf(p_b))

    def soft_probs(self, x: np.ndarray, n: int = 200) -> np.ndarray:
        """Monte-Carlo estimate of the smoothed class probabilities."""
        return self.sample_counts(x, n) / n
