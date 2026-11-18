"""Binary soft-margin SVM trained with Platt's Sequential Minimal Optimisation.

The dual problem solved is::

    max_a  sum(a) - 1/2 sum_ij a_i a_j y_i y_j K_ij
    s.t.   0 <= a_i <= C,   sum_i a_i y_i = 0

Two multipliers are optimised analytically at each step; the second one is
selected with the maximal-|E1 - E2| heuristic, falling back to a sweep over
non-bound and then all examples.
"""

from __future__ import annotations

import numpy as np

from .kernels import Kernel, make_kernel

Array = np.ndarray


class SVC:
    """Binary kernel SVM with labels in ``{-1, +1}`` (any two labels accepted).

    Parameters
    ----------
    C: box constraint.
    kernel: kernel name or :class:`Kernel`.
    tol: KKT tolerance.
    max_passes: cap on outer sweeps.
    seed: seed for the randomised sweep start.
    """

    def __init__(self, C: float = 1.0, kernel: str | Kernel = "rbf", tol: float = 1e-3,
                 max_passes: int = 200, seed: int = 0, **kernel_params) -> None:
        if C <= 0:
            raise ValueError("C must be positive")
        self.C = float(C)
        self.kernel = make_kernel(kernel, **kernel_params)
        self.tol = tol
        self.max_passes = max_passes
        self.seed = seed
        self.n_steps_ = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X: Array, y: Array) -> "SVC":
        X = np.asarray(X, dtype=float)
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError("SVC is binary; use OneVsRestSVC for more classes")
        self.classes_ = labels
        t = np.where(np.asarray(y) == labels[1], 1.0, -1.0)
        n = len(t)
        K = self.kernel(X, X)
        self._K, self._t = K, t
        self._rng = np.random.default_rng(self.seed)
        self._alpha = np.zeros(n)
        self._b = 0.0
        self._E = -t.copy()  # f(x) = 0 initially
        self.n_steps_ = 0

        changed, examine_all, passes = 0, True, 0
        while (changed > 0 or examine_all) and passes < self.max_passes:
            changed = 0
            if examine_all:
                idx = range(n)
            else:
                idx = np.where((self._alpha > 1e-12) & (self._alpha < self.C - 1e-12))[0]
            for i in idx:
                changed += self._examine(int(i))
            if examine_all:
                examine_all = False
            elif changed == 0:
                examine_all = True
            passes += 1
        self.n_passes_ = passes

        sv = self._alpha > 1e-8
        self.support_ = np.where(sv)[0]
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (self._alpha * t)[sv]
        self.intercept_ = self._b
        self._X = X
        return self

    def _examine(self, i2: int) -> int:
        a2, E2, t2 = self._alpha[i2], self._E[i2], self._t[i2]
        r2 = E2 * t2
        if not ((r2 < -self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0)):
            return 0
        nb = np.where((self._alpha > 1e-12) & (self._alpha < self.C - 1e-12))[0]
        if len(nb) > 1:
            # second-choice heuristic: largest expected step |E1 - E2|
            i1 = int(nb[np.argmax(np.abs(self._E[nb] - E2))])
            if self._step(i1, i2):
                return 1
        if len(nb):
            start = int(self._rng.integers(len(nb)))
            for k in range(len(nb)):
                if self._step(int(nb[(start + k) % len(nb)]), i2):
                    return 1
        n = len(self._t)
        start = int(self._rng.integers(n))
        for k in range(n):
            if self._step((start + k) % n, i2):
                return 1
        return 0

    def _step(self, i1: int, i2: int) -> bool:
        if i1 == i2:
            return False
        K, t, C = self._K, self._t, self.C
        a1, a2 = self._alpha[i1], self._alpha[i2]
        y1, y2 = t[i1], t[i2]
        E1, E2 = self._E[i1], self._E[i2]
        s = y1 * y2
        if y1 != y2:
            L, H = max(0.0, a2 - a1), min(C, C + a2 - a1)
        else:
            L, H = max(0.0, a1 + a2 - C), min(C, a1 + a2)
        if L >= H:
            return False
        k11, k22, k12 = K[i1, i1], K[i2, i2], K[i1, i2]
        eta = k11 + k22 - 2.0 * k12
        eps = 1e-12
        if eta > eps:
            a2n = float(np.clip(a2 + y2 * (E1 - E2) / eta, L, H))
        else:
            b = self._b
            f1 = y1 * (E1 - b) - a1 * k11 - s * a2 * k12
            f2 = y2 * (E2 - b) - s * a1 * k12 - a2 * k22
            L1, H1 = a1 + s * (a2 - L), a1 + s * (a2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12
            a2n = L if Lobj < Hobj - 1e-10 else (H if Lobj > Hobj + 1e-10 else a2)
        if abs(a2n - a2) < 1e-10 * (a2n + a2 + 1e-10):
            return False
        a1n = a1 + s * (a2 - a2n)
        # keep a1 inside the box against rounding
        a1n = min(max(a1n, 0.0), C)
        d1, d2 = a1n - a1, a2n - a2
        b1 = self._b - E1 - y1 * d1 * k11 - y2 * d2 * k12
        b2 = self._b - E2 - y1 * d1 * k12 - y2 * d2 * k22
        if 1e-12 < a1n < C - 1e-12:
            b_new = b1
        elif 1e-12 < a2n < C - 1e-12:
            b_new = b2
        else:
            b_new = 0.5 * (b1 + b2)
        self._E += y1 * d1 * K[i1] + y2 * d2 * K[i2] + (b_new - self._b)
        self._alpha[i1], self._alpha[i2] = a1n, a2n
        self._b = b_new
        self.n_steps_ += 1
        return True

    # ------------------------------------------------------------ inference
    def decision_function(self, X: Array) -> Array:
        """Signed score ``sum_i a_i y_i k(x_i, x) + b``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.kernel(X, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X: Array) -> Array:
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def score(self, X: Array, y: Array) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def gradient(self, x: Array) -> Array:
        """Gradient of the decision function at a single point ``x``."""
        return self.dual_coef_ @ self.kernel.grad(x, self.support_vectors_)

    # ---------------------------------------------------------- diagnostics
    def dual_objective(self) -> float:
        a, t, K = self._alpha, self._t, self._K
        v = a * t
        return float(a.sum() - 0.5 * v @ K @ v)

    def primal_objective(self) -> float:
        """``1/2 ||w||^2 + C sum hinge`` evaluated at the fitted solution."""
        v = self._alpha * self._t
        reg = 0.5 * v @ self._K @ v
        margins = self._t * (self._K @ v + self._b)
        return float(reg + self.C * np.maximum(0.0, 1.0 - margins).sum())

    def kkt_violation(self) -> float:
        """Largest violation of the KKT conditions over the training set."""
        a, t = self._alpha, self._t
        m = t * (self._K @ (a * t) + self._b)
        lo = np.where(a < 1e-8, np.maximum(0.0, 1.0 - m), 0.0)
        mid = np.where((a >= 1e-8) & (a <= self.C - 1e-8), np.abs(1.0 - m), 0.0)
        hi = np.where(a > self.C - 1e-8, np.maximum(0.0, m - 1.0), 0.0)
        return float(max(lo.max(), mid.max(), hi.max()))
