"""Statistical helpers: normal distribution and exact binomial confidence bounds."""

from __future__ import annotations

import math

import numpy as np


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation + Halley refinement)."""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must lie in [0, 1]")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    # one Halley step
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _log_binom_pmf(n: int, p: float) -> np.ndarray:
    ks = np.arange(n + 1)
    lg = np.array([math.lgamma(i + 1) for i in range(n + 1)])
    logc = lg[n] - lg[ks] - lg[n - ks]
    if p <= 0.0:
        out = np.full(n + 1, -np.inf)
        out[0] = 0.0
        return out
    if p >= 1.0:
        out = np.full(n + 1, -np.inf)
        out[n] = 0.0
        return out
    return logc + ks * math.log(p) + (n - ks) * math.log1p(-p)


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lp = _log_binom_pmf(n, p)[k:]
    m = lp.max()
    if not np.isfinite(m):
        return 0.0
    return float(min(1.0, math.exp(m) * np.exp(lp - m).sum()))


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided (1-alpha) Clopper-Pearson lower confidence bound on a binomial proportion."""
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if binom_sf(k, n, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """One-sided (1-alpha) Clopper-Pearson upper confidence bound."""
    if k == n:
        return 1.0
    return 1.0 - clopper_pearson_lower(n - k, n, alpha)


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (sum of pmf <= pmf(k))."""
    lp = _log_binom_pmf(n, p)
    thr = lp[k] + 1e-12
    return float(min(1.0, np.exp(lp[lp <= thr]).sum()))
