"""Theory helpers, privacy auditing, synthetic data and error metrics."""
from __future__ import annotations

import numpy as np

from .oracles import (FrequencyOracle, GeneralizedRandomizedResponse,
                      HadamardResponse, OptimizedLocalHashing, UnaryEncoding)

ORACLE_NAMES = ("GRR", "SUE", "OUE", "OLH", "HR")


def make_oracle(name: str, d: int, epsilon: float) -> FrequencyOracle:
    """Construct an oracle from its short name."""
    key = name.upper()
    if key == "GRR":
        return GeneralizedRandomizedResponse(d, epsilon)
    if key == "SUE":
        return UnaryEncoding(d, epsilon, "symmetric")
    if key == "OUE":
        return UnaryEncoding(d, epsilon, "optimized")
    if key == "OLH":
        return OptimizedLocalHashing(d, epsilon)
    if key == "HR":
        return HadamardResponse(d, epsilon)
    raise ValueError(f"unknown oracle {name!r}")


def variance_table(d: int, epsilon: float, n: int) -> dict[str, float]:
    """Approximate (``f ~ 0``) estimator variance of every oracle."""
    return {nm: make_oracle(nm, d, epsilon).variance(n) for nm in ORACLE_NAMES}


def best_oracle(d: int, epsilon: float, candidates=("GRR", "OUE", "OLH")) -> FrequencyOracle:
    """Lowest-variance oracle among ``candidates`` (GRR wins for small ``d``)."""
    best = min((make_oracle(nm, d, epsilon) for nm in candidates),
               key=lambda o: o.variance(1))
    return best


def grr_threshold(epsilon: float) -> float:
    """Domain size below which GRR beats OUE/OLH: ``3 e^eps + 2``."""
    return 3 * np.exp(epsilon) + 2


def required_users(oracle: FrequencyOracle, target_std: float) -> int:
    """Smallest ``n`` so that the per-item standard deviation is <= target."""
    return int(np.ceil(oracle.variance(1) / target_std ** 2))


def privacy_loss(channel: np.ndarray) -> float:
    """Worst-case log likelihood ratio over inputs and outputs of a channel."""
    chan = np.asarray(channel, dtype=float)
    if not np.allclose(chan.sum(axis=1), 1.0):
        raise ValueError("channel rows must sum to one")
    with np.errstate(divide="ignore"):
        logs = np.log(chan)
    return float((logs.max(axis=0) - logs.min(axis=0)).max())


def audit(oracle: FrequencyOracle) -> float:
    """Exact epsilon actually delivered by the oracle's randomiser."""
    return privacy_loss(oracle.channel_matrix())


# -- data and metrics ----------------------------------------------------
def zipf_distribution(d: int, s: float = 1.1) -> np.ndarray:
    """Zipf probabilities over ``d`` items (item 0 is the most frequent)."""
    w = 1.0 / np.arange(1, d + 1) ** s
    return w / w.sum()


def sample_users(probs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` private values i.i.d. from ``probs``."""
    return rng.choice(probs.size, size=n, p=probs)


def true_frequencies(values: np.ndarray, d: int) -> np.ndarray:
    """Empirical frequency vector of ``values``."""
    return np.bincount(values, minlength=d) / max(len(values), 1)


def mse(true: np.ndarray, est: np.ndarray) -> float:
    """Mean squared error across the domain."""
    return float(np.mean((np.asarray(true) - np.asarray(est)) ** 2))


def l1_error(true: np.ndarray, est: np.ndarray) -> float:
    """Total variation style L1 distance."""
    return float(np.abs(np.asarray(true) - np.asarray(est)).sum())


def topk_recall(true: np.ndarray, est: np.ndarray, k: int) -> float:
    """Fraction of the true top-k items found in the estimated top-k."""
    t = set(np.argsort(-np.asarray(true), kind="stable")[:k].tolist())
    e = set(np.argsort(-np.asarray(est), kind="stable")[:k].tolist())
    return len(t & e) / k
