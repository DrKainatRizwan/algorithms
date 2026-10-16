"""A small multilayer perceptron with hand-written backpropagation.

Besides parameter gradients the model exposes the gradient of the loss with
respect to its *input*, which is the primitive every gradient-based attack needs.
"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample cross-entropy loss computed with log-sum-exp."""
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(y)), y]


class MLP:
    """Fully connected ReLU network with softmax cross-entropy output.

    ``sizes=(2, 32, 32, 3)`` builds two hidden layers.  With ``sizes=(d, k)`` the
    model is plain multinomial logistic regression.
    """

    def __init__(self, sizes: tuple[int, ...], seed: int = 0) -> None:
        if len(sizes) < 2:
            raise ValueError("need at least input and output sizes")
        rng = np.random.default_rng(seed)
        self.sizes = tuple(sizes)
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    # ---- forward -------------------------------------------------------
    def _forward(self, X: np.ndarray):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return acts

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.atleast_2d(X))[-1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(cross_entropy(self.logits(X), y).mean())

    # ---- backward ------------------------------------------------------
    def _backward(self, acts, y: np.ndarray, per_sample: bool):
        """Return (delta at input, list of dW, list of db)."""
        n = len(y)
        delta = softmax(acts[-1])
        delta[np.arange(n), y] -= 1.0          # dL_i / dlogits_i
        if not per_sample:
            delta = delta / n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            delta = delta @ self.W[i].T
            if i > 0:
                delta = delta * (acts[i] > 0)
        return delta, gW, gb

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        """Mean loss and gradients w.r.t. weights and biases."""
        acts = self._forward(X)
        loss = float(cross_entropy(acts[-1], y).mean())
        _, gW, gb = self._backward(acts, y, per_sample=False)
        return loss, gW, gb

    def input_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-sample gradient dL_i/dx_i, shape (n, d)."""
        acts = self._forward(np.atleast_2d(X))
        dx, _, _ = self._backward(acts, np.asarray(y), per_sample=True)
        return dx

    def margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Logit of the true class minus the best other logit (>0 = correct)."""
        z = self.logits(X)
        idx = np.arange(len(y))
        true = z[idx, y]
        other = z.copy()
        other[idx, y] = -np.inf
        return true - other.max(axis=1)

    # ---- parameters ----------------------------------------------------
    def get_params(self):
        return [p.copy() for p in self.W + self.b]

    def set_params(self, params) -> None:
        n = len(self.W)
        self.W = [p.copy() for p in params[:n]]
        self.b = [p.copy() for p in params[n:]]
