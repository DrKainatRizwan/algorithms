"""One-dimensional optimal transport and sliced Wasserstein distances.

In one dimension the optimal plan is the monotone coupling, so ``W_p`` reduces
to an integral of quantile-function differences, computed here exactly for
weighted empirical measures by merging cumulative weights.
"""
from __future__ import annotations

import numpy as np

from .cost import as_histogram


def _sorted_measure(x: np.ndarray, w: np.ndarray | None):
    x = np.asarray(x, dtype=float).ravel()
    w = np.ones(x.size) if w is None else np.asarray(w, dtype=float).ravel()
    w = as_histogram(w, "weights")
    if w.size != x.size:
        raise ValueError("weights and samples differ in length")
    order = np.argsort(x, kind="stable")
    return x[order], w[order]


def monotone_plan(x, y, a=None, b=None):
    """Return the monotone coupling as a list of ``(i, j, mass)`` over sorted samples.

    Indices refer to the *sorted* order of ``x`` and ``y``.
    """
    xs, aw = _sorted_measure(x, a)
    ys, bw = _sorted_measure(y, b)
    i = j = 0
    ra, rb = aw[0], bw[0]
    out = []
    while i < len(xs) and j < len(ys):
        m = min(ra, rb)
        if m > 0:
            out.append((i, j, m))
        ra -= m
        rb -= m
        if ra <= 1e-15:
            i += 1
            if i < len(xs):
                ra = aw[i]
        if rb <= 1e-15:
            j += 1
            if j < len(ys):
                rb = bw[j]
    return out


def wasserstein_1d(x, y, a=None, b=None, p: float = 1.0) -> float:
    """Exact ``W_p`` between weighted 1-D empirical measures."""
    xs, _ = _sorted_measure(x, a)
    ys, _ = _sorted_measure(y, b)
    total = 0.0
    for i, j, m in monotone_plan(x, y, a, b):
        total += m * abs(xs[i] - ys[j]) ** p
    return float(total ** (1.0 / p))


def quantile_function(x, w, q: np.ndarray) -> np.ndarray:
    """Generalised inverse CDF of a weighted empirical measure at levels ``q``."""
    xs, ws = _sorted_measure(x, w)
    cdf = np.cumsum(ws)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, np.asarray(q, dtype=float), side="left")
    return xs[np.minimum(idx, xs.size - 1)]


def random_directions(n_proj: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed unit vectors on the sphere ``S^{dim-1}``."""
    v = rng.standard_normal((n_proj, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sliced_wasserstein(X, Y, a=None, b=None, p: float = 2.0, n_proj: int = 100,
                       seed: int = 0) -> float:
    """Monte-Carlo sliced Wasserstein distance ``SW_p``.

    Averages ``W_p^p`` of the 1-D projections over random directions.
    Cost is ``O(n_proj * n log n)``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if X.shape[0] == 1 and X.shape[1] > 1 and Y.shape[1] == 1:
        X = X.T
    rng = np.random.default_rng(seed)
    thetas = random_directions(n_proj, X.shape[1], rng)
    acc = 0.0
    for th in thetas:
        acc += wasserstein_1d(X @ th, Y @ th, a, b, p=p) ** p
    return float((acc / n_proj) ** (1.0 / p))


def wasserstein_1d_barycenter(samples: list, weights: list[float], n_atoms: int = 200) -> np.ndarray:
    """Exact barycenter of 1-D measures (equal atom weights): average quantile functions."""
    lam = np.asarray(weights, dtype=float)
    lam = lam / lam.sum()
    q = (np.arange(n_atoms) + 0.5) / n_atoms
    out = np.zeros(n_atoms)
    for l, s in zip(lam, samples):
        out += l * quantile_function(s, None, q)
    return out
