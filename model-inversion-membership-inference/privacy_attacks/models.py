"""NumPy MLP / softmax-regression classifier with hand-written backprop.

Supports per-example gradients (needed for DP-SGD), input gradients (needed
for model inversion) and an Adam trainer with optional L2 regularisation,
early stopping and differentially-private noisy clipped updates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable log-softmax."""
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class MLP:
    """Fully connected ReLU network; ``hidden=()`` gives softmax regression."""

    def __init__(self, dim: int, n_classes: int, hidden: tuple[int, ...] = (64,), seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        sizes = [dim, *hidden, n_classes]
        self.W = [rng.standard_normal((a, b)) * math.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.n_classes = n_classes

    # ---- forward -------------------------------------------------------
    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        acts, pre = [np.atleast_2d(X)], []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            pre.append(z)
            acts.append(np.maximum(z, 0.0) if l < len(self.W) - 1 else z)
        return acts, pre

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0][-1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logits(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-example cross-entropy."""
        return -log_softmax(self.logits(X))[np.arange(len(y)), y]

    # ---- backward ------------------------------------------------------
    def _deltas(self, X: np.ndarray, delta_out: np.ndarray):
        acts, pre = self._forward(X)
        deltas = [delta_out]
        for l in range(len(self.W) - 1, 0, -1):
            deltas.append((deltas[-1] @ self.W[l].T) * (pre[l - 1] > 0))
        deltas.reverse()
        return acts, deltas

    def _ce_delta(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = self.predict_proba(X)
        d[np.arange(len(y)), y] -= 1.0
        return d

    def per_example_grads(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Per-example gradients, one ``(B, *param_shape)`` array per parameter.

        Order: ``[W0, b0, W1, b1, ...]``.
        """
        acts, deltas = self._deltas(X, self._ce_delta(X, y))
        out: list[np.ndarray] = []
        for a, dl in zip(acts[:-1], deltas):
            out.append(np.einsum("bi,bj->bij", a, dl))
            out.append(dl)
        return out

    def mean_grads(self, X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        """Batch-mean gradients in the same order as :meth:`per_example_grads`."""
        acts, deltas = self._deltas(X, self._ce_delta(X, y))
        B = len(y)
        out: list[np.ndarray] = []
        for a, dl in zip(acts[:-1], deltas):
            out.append(a.T @ dl / B)
            out.append(dl.mean(axis=0))
        return out

    def input_gradient(self, x: np.ndarray, target: int) -> np.ndarray:
        """Gradient of ``log p(target | x)`` with respect to the input ``x``."""
        x = np.atleast_2d(x)
        d = -self.predict_proba(x)
        d[0, target] += 1.0  # d log p_t / d logits = onehot - p
        _, deltas = self._deltas(x, d)
        return (deltas[0] @ self.W[0].T)[0]

    # ---- parameter plumbing -------------------------------------------
    def params(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for W, b in zip(self.W, self.b):
            out += [W, b]
        return out

    def copy(self) -> "MLP":
        m = MLP.__new__(MLP)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        m.n_classes = self.n_classes
        return m


@dataclass
class TrainConfig:
    """Training hyper-parameters, including the training-time defences."""

    epochs: int = 150
    lr: float = 0.01
    batch_size: int = 32
    l2: float = 0.0
    clip_norm: float | None = None       # DP-SGD per-example clipping bound C
    noise_multiplier: float = 0.0        # DP-SGD sigma (noise std = sigma * C)
    patience: int | None = None          # early stopping on validation loss
    seed: int = 0


@dataclass
class TrainResult:
    history: list[float] = field(default_factory=list)
    steps: int = 0
    epochs_run: int = 0


def train(model: MLP, X: np.ndarray, y: np.ndarray, cfg: TrainConfig,
          X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> TrainResult:
    """Train in place with Adam; optionally DP-SGD and early stopping."""
    rng = np.random.default_rng(cfg.seed)
    params = model.params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    n = len(y)
    res = TrainResult()
    best, best_params, bad = math.inf, None, 0
    t = 0
    for ep in range(cfg.epochs):
        perm = rng.permutation(n)
        for s in range(0, n, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            if cfg.clip_norm is not None:
                pe = model.per_example_grads(X[idx], y[idx])
                norms = np.sqrt(sum((g.reshape(len(idx), -1) ** 2).sum(axis=1) for g in pe))
                scale = np.minimum(1.0, cfg.clip_norm / (norms + 1e-12))
                grads = []
                for g in pe:
                    tot = np.tensordot(scale, g, axes=(0, 0))
                    tot = tot + cfg.noise_multiplier * cfg.clip_norm * rng.standard_normal(tot.shape)
                    grads.append(tot / len(idx))
            else:
                grads = model.mean_grads(X[idx], y[idx])
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                if cfg.l2 and p.ndim == 2:
                    g = g + cfg.l2 * p
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                mh, vh = m[i] / (1 - 0.9**t), v[i] / (1 - 0.999**t)
                p -= cfg.lr * mh / (np.sqrt(vh) + 1e-8)
        res.epochs_run = ep + 1
        res.history.append(float(model.loss(X, y).mean()))
        if cfg.patience is not None and X_val is not None:
            vl = float(model.loss(X_val, y_val).mean())
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_params = [p.copy() for p in params]
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_params is not None:
        for p, q in zip(params, best_params):
            p[...] = q
    res.steps = t
    return res


def zcdp_epsilon(steps: int, noise_multiplier: float, delta: float = 1e-5) -> float:
    """Conservative (no subsampling amplification) (eps, delta) bound for DP-SGD.

    Each step is a Gaussian mechanism with sensitivity C and std sigma*C, which
    is ``1/(2 sigma^2)``-zCDP; composition adds, and zCDP converts to
    ``eps = rho + 2 sqrt(rho ln(1/delta))``.
    """
    if noise_multiplier <= 0:
        return math.inf
    rho = steps / (2.0 * noise_multiplier**2)
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
