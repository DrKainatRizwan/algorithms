"""Shapley value estimators: exact, permutation sampling, KernelSHAP and linear closed form.

The coalition value function is the interventional (marginal) expectation
``v(S) = E_b[ f(x_S, b_{~S}) ]`` over a background sample ``b``.
"""
from __future__ import annotations
from math import comb, factorial
from typing import Callable
import numpy as np

Model = Callable[[np.ndarray], np.ndarray]


class ValueFunction:
    """Coalition value function with a background dataset."""

    def __init__(self, f: Model, x: np.ndarray, background: np.ndarray):
        self.f = f
        self.x = np.asarray(x, dtype=float)
        self.bg = np.atleast_2d(np.asarray(background, dtype=float))
        if self.bg.shape[1] != self.x.shape[0]:
            raise ValueError("background and x dimensionality differ")
        self.d = self.x.shape[0]
        self.calls = 0  # number of coalition evaluations
        self.base = float(np.mean(self.f(self.bg)))
        self.full = float(np.asarray(self.f(self.x[None, :])).ravel()[0])

    def __call__(self, mask: np.ndarray) -> float:
        return float(self.batch(np.asarray(mask, dtype=bool)[None, :])[0])

    def batch(self, masks: np.ndarray) -> np.ndarray:
        """Evaluate many coalitions (rows of a boolean matrix) in one model call."""
        masks = np.asarray(masks, dtype=bool)
        m, nb = len(masks), len(self.bg)
        Z = np.where(masks[:, None, :], self.x[None, None, :], self.bg[None, :, :])
        out = np.asarray(self.f(Z.reshape(m * nb, self.d))).reshape(m, nb)
        self.calls += m
        return out.mean(axis=1)


def exact_shapley(vf: ValueFunction) -> np.ndarray:
    """Exact Shapley values by enumerating all 2^d coalitions. O(2^d) evaluations."""
    d = vf.d
    if d > 20:
        raise ValueError("exact enumeration limited to d <= 20")
    n = 1 << d
    bits = ((np.arange(n)[:, None] >> np.arange(d)) & 1).astype(bool)
    vals = vf.batch(bits)
    size = bits.sum(axis=1)
    w = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
    phi = np.zeros(d)
    for i in range(d):
        without = np.nonzero(~bits[:, i])[0]
        with_i = without | (1 << i)
        phi[i] = np.sum(w[size[without]] * (vals[with_i] - vals[without]))
    return phi


def permutation_shapley(vf: ValueFunction, n_perm: int = 100, seed: int = 0,
                        antithetic: bool = True) -> np.ndarray:
    """Monte-Carlo Shapley via random permutations (optionally with reversed pairs).

    Each permutation costs d+1 coalition evaluations and satisfies efficiency exactly.
    """
    rng = np.random.default_rng(seed)
    d = vf.d
    phi = np.zeros(d)
    count = 0
    perms = []
    for _ in range(n_perm):
        p = rng.permutation(d)
        perms.append(p)
        if antithetic:
            perms.append(p[::-1])
    for p in perms:
        masks = np.zeros((d + 1, d), dtype=bool)
        for k in range(d):
            masks[k + 1:, p[k]] = True
        v = vf.batch(masks)
        phi[p] += np.diff(v)
        count += 1
    return phi / count


def shapley_kernel_weight(d: int, s: int) -> float:
    """Shapley kernel pi(s) = (d-1) / (C(d,s) s (d-s)) for 0 < s < d."""
    return (d - 1) / (comb(d, s) * s * (d - s))


def kernel_shap(vf: ValueFunction, n_samples: int | None = None, seed: int = 0,
                l2: float = 0.0) -> np.ndarray:
    """KernelSHAP: weighted least squares over coalitions with the efficiency constraint.

    If ``n_samples`` is None or >= 2^d - 2 all coalitions are enumerated (exact).
    Otherwise coalition sizes are drawn proportionally to the kernel mass, with
    complementary (paired) sampling, and duplicate coalitions are merged.
    """
    d = vf.d
    total = vf.full - vf.base
    if d == 1:
        return np.array([total])
    rng = np.random.default_rng(seed)
    if n_samples is None or n_samples >= (1 << d) - 2:
        n = 1 << d
        allm = ((np.arange(1, n - 1)[:, None] >> np.arange(d)) & 1).astype(bool)
        sizes = allm.sum(axis=1)
        w = np.array([shapley_kernel_weight(d, int(s)) for s in sizes])
        masks = allm
    else:
        ss = np.arange(1, d)
        mass = np.array([(d - 1) / (s * (d - s)) for s in ss])
        mass /= mass.sum()
        table: dict[bytes, list] = {}
        for _ in range(max(1, n_samples // 2)):
            s = rng.choice(ss, p=mass)
            m = np.zeros(d, dtype=bool)
            m[rng.choice(d, s, replace=False)] = True
            for mm in (m, ~m):
                key = mm.tobytes()
                if key in table:
                    table[key][1] += 1
                else:
                    table[key] = [mm, 1]
        masks = np.array([v[0] for v in table.values()])
        w = np.array([v[1] for v in table.values()], dtype=float)
    y = vf.batch(masks) - vf.base
    Z = masks.astype(float)
    # eliminate last variable using sum(phi) = total
    Zr = Z[:, :-1] - Z[:, -1:]
    yr = y - Z[:, -1] * total
    A = Zr.T @ (w[:, None] * Zr) + l2 * np.eye(d - 1)
    b = Zr.T @ (w * yr)
    head = np.linalg.solve(A, b)
    return np.append(head, total - head.sum())


def linear_shap(coef: np.ndarray, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Closed form for a linear model: phi_i = w_i (x_i - E[b_i])."""
    return np.asarray(coef) * (np.asarray(x) - np.atleast_2d(background).mean(axis=0))
