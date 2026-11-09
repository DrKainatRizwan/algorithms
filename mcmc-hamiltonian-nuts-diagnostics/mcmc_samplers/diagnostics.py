"""Convergence diagnostics: autocorrelation, ESS, split R-hat, MCSE."""
from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of a 1-D series via FFT (biased estimator)."""
    x = np.asarray(x, float)
    n = x.size
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    if acov[0] <= 0:
        return np.zeros(n)
    return acov / acov[0]


def effective_sample_size(chains: np.ndarray) -> float:
    """ESS for one parameter from ``chains`` of shape (n_chains, n_draws).

    Uses the multi-chain autocorrelation estimator with Geyer's initial positive
    sequence truncation (Vehtari et al. 2021, Stan reference manual).
    """
    chains = np.atleast_2d(np.asarray(chains, float))
    m, n = chains.shape
    if n < 4:
        return float(m * n)
    acov = np.array([autocorrelation(c) * c.var() for c in chains])  # per-chain autocov
    chain_var = acov[:, 0] * n / (n - 1)
    w = chain_var.mean()
    var_plus = w * (n - 1) / n
    if m > 1:
        var_plus += chains.mean(axis=1).var(ddof=1)
    if var_plus <= 0:
        return float(m * n)
    rho = 1.0 - (w - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = []
    for t in range(0, n - 1, 2):
        g = rho[t] + rho[t + 1]
        if g < 0:
            break
        pairs.append(g)
    if not pairs:
        return float(m * n)
    pairs = np.minimum.accumulate(np.array(pairs))
    tau = -1.0 + 2.0 * pairs.sum()
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def split_rhat(chains: np.ndarray) -> float:
    """Split potential scale reduction factor for (n_chains, n_draws)."""
    chains = np.atleast_2d(np.asarray(chains, float))
    n = chains.shape[1] // 2
    if n < 2:
        return float("nan")
    parts = np.vstack([chains[:, :n], chains[:, n:2 * n]])
    means = parts.mean(axis=1)
    b = n * means.var(ddof=1)
    w = parts.var(axis=1, ddof=1).mean()
    if w == 0:
        return 1.0
    var_hat = (n - 1) / n * w + b / n
    return float(np.sqrt(var_hat / w))


def mcse(chains: np.ndarray) -> float:
    """Monte Carlo standard error of the posterior mean."""
    chains = np.atleast_2d(chains)
    return float(chains.std(ddof=1) / np.sqrt(effective_sample_size(chains)))


def summarize(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Per-dimension summary for ``samples`` of shape (n_chains, n_draws, dim)
    (a 2-D array is treated as a single chain)."""
    s = np.asarray(samples, float)
    if s.ndim == 2:
        s = s[None]
    d = s.shape[2]
    return {
        "mean": s.reshape(-1, d).mean(axis=0),
        "std": s.reshape(-1, d).std(axis=0, ddof=1),
        "ess": np.array([effective_sample_size(s[:, :, k]) for k in range(d)]),
        "rhat": np.array([split_rhat(s[:, :, k]) for k in range(d)]),
    }
