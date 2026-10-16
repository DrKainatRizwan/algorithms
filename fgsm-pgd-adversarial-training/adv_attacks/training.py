"""Standard and adversarial training loops (Adam optimiser)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attacks import fgsm, pgd
from .model import MLP


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.01
    eps: float = 0.3
    attack: str = "pgd"           # 'pgd' or 'fgsm'
    pgd_steps: int = 5
    mix: float = 0.5              # fraction of adversarial samples in each batch
    norm: str = "linf"
    seed: int = 0
    history: list = field(default_factory=list)


class _Adam:
    def __init__(self, params, lr: float, b1=0.9, b2=0.999, eps=1e-8) -> None:
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads) -> None:
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _train(model: MLP, X: np.ndarray, y: np.ndarray, cfg: TrainConfig, adversarial: bool):
    rng = np.random.default_rng(cfg.seed)
    opt = _Adam(model.W + model.b, cfg.lr)
    n = len(y)
    cfg.history = []
    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        total = 0.0
        for s in range(0, n, cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            xb, yb = X[idx], y[idx]
            if adversarial:
                k = int(round(cfg.mix * len(idx)))
                if k > 0:
                    if cfg.attack == "fgsm":
                        adv = fgsm(model, xb[:k], yb[:k], cfg.eps, cfg.norm)
                    else:
                        adv = pgd(model, xb[:k], yb[:k], cfg.eps, steps=cfg.pgd_steps,
                                  norm=cfg.norm, seed=int(rng.integers(1 << 31)))
                    xb = np.vstack([adv, xb[k:]]) if k < len(idx) else adv
            loss, gW, gb = model.loss_and_grads(xb, yb)
            opt.step(model.W + model.b, gW + gb)
            total += loss * len(idx)
        cfg.history.append(total / n)
    return model


def fit(model: MLP, X: np.ndarray, y: np.ndarray, cfg: TrainConfig | None = None) -> MLP:
    """Standard (natural) training with Adam."""
    return _train(model, X, y, cfg or TrainConfig(), adversarial=False)


def adversarial_fit(model: MLP, X: np.ndarray, y: np.ndarray,
                    cfg: TrainConfig | None = None) -> MLP:
    """Adversarial training: fit on attacked examples generated each step
    against the current weights (min-max formulation of Madry et al.)."""
    return _train(model, X, y, cfg or TrainConfig(), adversarial=True)
