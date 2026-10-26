"""L2-regularised binary logistic regression with exact derivatives.

Objective (sum form, labels in {0, 1}, bias folded into the weight vector)::

    J(w) = sum_i s_i * log(1 + exp(-y_i' w.x_i)) + (l2 / 2) * ||w||^2

with ``y' = 2y - 1`` and optional per-sample weights ``s_i``.  The gradient and
Hessian are available in closed form which the influence-function code in
:mod:`poison_robust.influence` relies on.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def add_bias(X: np.ndarray) -> np.ndarray:
    """Append a constant-one column to ``X``."""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, np.ones((X.shape[0], 1))])


def log1pexp(z: np.ndarray) -> np.ndarray:
    """Stable ``log(1 + exp(z))``."""
    return np.logaddexp(0.0, z)


class LogisticRegression:
    """Binary logistic regression fitted by damped Newton iterations."""

    def __init__(self, l2: float = 1.0, max_iter: int = 60, tol: float = 1e-10) -> None:
        self.l2 = float(l2)
        self.max_iter = max_iter
        self.tol = tol
        self.w: np.ndarray | None = None

    # ------------------------------------------------------------------ objective
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None, reg: bool = True) -> float:
        """Total (summed) loss, optionally including the L2 penalty."""
        w = self._w(w)
        z = add_bias(X) @ w
        per = log1pexp(z) - y * z
        if sample_weight is not None:
            per = per * sample_weight
        total = float(per.sum())
        if reg:
            total += 0.5 * self.l2 * float(w @ w)
        return total

    def per_sample_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        w = self._w(w)
        z = add_bias(X) @ w
        return log1pexp(z) - y * z

    def per_sample_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        """Matrix of unregularised per-sample gradients, shape ``(n, d+1)``."""
        w = self._w(w)
        Xb = add_bias(X)
        r = sigmoid(Xb @ w) - y
        return Xb * r[:, None]

    def grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None, reg: bool = True) -> np.ndarray:
        w = self._w(w)
        G = self.per_sample_grad(X, y, w)
        if sample_weight is not None:
            G = G * sample_weight[:, None]
        g = G.sum(axis=0)
        if reg:
            g = g + self.l2 * w
        return g

    def hessian(self, X: np.ndarray, w: np.ndarray | None = None,
                sample_weight: np.ndarray | None = None, reg: bool = True) -> np.ndarray:
        w = self._w(w)
        Xb = add_bias(X)
        p = sigmoid(Xb @ w)
        s = p * (1.0 - p)
        if sample_weight is not None:
            s = s * sample_weight
        H = (Xb * s[:, None]).T @ Xb
        if reg:
            H = H + self.l2 * np.eye(H.shape[0])
        return H

    # ---------------------------------------------------------------------- fitting
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
            w0: np.ndarray | None = None) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        d = X.shape[1] + 1
        w = np.zeros(d) if w0 is None else np.array(w0, dtype=float)
        f = self.loss(X, y, w, sample_weight)
        for _ in range(self.max_iter):
            g = self.grad(X, y, w, sample_weight)
            if np.linalg.norm(g) < self.tol:
                break
            H = self.hessian(X, w, sample_weight)
            step = np.linalg.solve(H, g)
            t = 1.0
            while t > 1e-10:
                f_new = self.loss(X, y, w - t * step, sample_weight)
                if f_new <= f - 1e-4 * t * float(g @ step):
                    break
                t *= 0.5
            w = w - t * step
            if abs(f - f_new) < 1e-15 * max(1.0, abs(f)):
                f = f_new
                break
            f = f_new
        self.w = w
        return self

    # -------------------------------------------------------------------- prediction
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return add_bias(X) @ self._w(None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def copy(self) -> "LogisticRegression":
        m = LogisticRegression(self.l2, self.max_iter, self.tol)
        m.w = None if self.w is None else self.w.copy()
        return m

    def _w(self, w: np.ndarray | None) -> np.ndarray:
        if w is not None:
            return w
        if self.w is None:
            raise RuntimeError("model has not been fitted")
        return self.w
