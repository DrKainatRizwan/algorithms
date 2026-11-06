"""Gradient-matching attack (Zhu et al., "Deep Leakage from Gradients").

The attacker minimises ``0.5 * || grad(model; X, y) - target ||^2`` over the
dummy inputs ``X`` using Adam. The gradient of this objective with respect to
``X`` (a second-order quantity) is derived by hand and checked against finite
differences in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import MLP, softmax


class Adam:
    """Minimal Adam optimiser for a single array."""

    def __init__(self, shape, lr: float = 0.05, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return x - self.lr * mh / (np.sqrt(vh) + self.eps)


def matching_loss_and_grad(model: MLP, X: np.ndarray, y: np.ndarray,
                           target: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    """Return ``L`` and ``dL/dX`` for the gradient-matching objective (mean-reduced)."""
    B = len(y)
    h, p = model.forward(X)
    dz = p.copy(); dz[np.arange(B), y] -= 1.0
    s = 1.0 - h ** 2
    c = dz @ model.W2                       # (B,H)  W2^T dz
    da = c * s
    g = {"W1": da.T @ X / B, "b1": da.sum(0) / B, "W2": dz.T @ h / B, "b2": dz.sum(0) / B}
    r = {k: g[k] - target[k] for k in g}
    loss = 0.5 * sum(float((v ** 2).sum()) for v in r.values())

    u = X @ r["W1"].T + r["b1"]             # (B,H)
    v = h @ r["W2"].T + r["b2"]             # (B,K)
    w = dz @ r["W2"]                        # (B,H)
    # Softmax Jacobian applied row-wise: Js q = p*q - p*(p.q)
    def Js(q):
        return p * q - p * np.sum(p * q, axis=1, keepdims=True)
    G_h = (Js((s * u) @ model.W2.T) @ model.W2        # W2^T Js W2 (s*u)
           - 2.0 * c * h * u
           + Js(v) @ model.W2
           + w)
    grad_x = (G_h * s) @ model.W1 + da @ r["W1"]
    return loss, grad_x / B


@dataclass
class AttackResult:
    """Outcome of a gradient-matching attack."""
    X: np.ndarray
    loss: float
    history: list[float] = field(default_factory=list)


def gradient_matching_attack(model: MLP, target: dict[str, np.ndarray], labels: np.ndarray,
                             rng: np.random.Generator, iters: int = 600, lr: float = 0.05,
                             restarts: int = 3, box: bool = True,
                             tv_weight: float = 0.0, side: int | None = None) -> AttackResult:
    """Reconstruct ``len(labels)`` inputs from the (mean) gradient ``target``.

    Labels are assumed known (they can be recovered analytically for B=1).
    ``box`` clips pixels to [0,1]; ``tv_weight`` adds a total-variation prior on
    ``side x side`` images. Returns the best restart by final matching loss.
    """
    B, d = len(labels), model.W1.shape[1]
    best: AttackResult | None = None
    for _ in range(restarts):
        X = rng.uniform(0.3, 0.7, size=(B, d)) if box else rng.standard_normal((B, d)) * 0.5
        opt = Adam(X.shape, lr=lr)
        hist: list[float] = []
        for _ in range(iters):
            loss, gx = matching_loss_and_grad(model, X, labels, target)
            if tv_weight > 0 and side:
                gx = gx + tv_weight * _tv_grad(X, side)
            X = opt.step(X, gx)
            if box:
                X = np.clip(X, 0.0, 1.0)
            hist.append(loss)
        final, _ = matching_loss_and_grad(model, X, labels, target)
        if best is None or final < best.loss:
            best = AttackResult(X=X, loss=final, history=hist)
    assert best is not None
    return best


def _tv_grad(X: np.ndarray, side: int, eps: float = 1e-6) -> np.ndarray:
    """Gradient of a smoothed isotropic total variation penalty."""
    I = X.reshape(len(X), side, side)
    dx = np.diff(I, axis=2, append=I[:, :, -1:])
    dy = np.diff(I, axis=1, append=I[:, -1:, :])
    n = np.sqrt(dx ** 2 + dy ** 2 + eps)
    px, py = dx / n, dy / n
    g = -px - py
    g[:, :, 1:] += px[:, :, :-1]
    g[:, 1:, :] += py[:, :-1, :]
    return g.reshape(X.shape)
