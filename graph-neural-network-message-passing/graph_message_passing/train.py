"""Training loop, evaluation and numerical gradient checking."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import Graph
from .model import GNN, Adam, cross_entropy


@dataclass
class History:
    """Training curves and the selected best epoch."""

    loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val: float = 0.0
    best_val_loss: float = float("inf")


def accuracy(logits: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Classification accuracy over masked nodes."""
    return float((logits[mask].argmax(axis=1) == y[mask]).mean())


def train(model: GNN, g: Graph, train_mask: np.ndarray, val_mask: np.ndarray,
          epochs: int = 200, lr: float = 0.01, weight_decay: float = 5e-4,
          patience: int = 50) -> History:
    """Full-batch training with early stopping on validation accuracy
    (ties broken by validation loss).

    The parameters of the best validation epoch are restored on exit.
    """
    opt = Adam(model.layers, lr=lr, weight_decay=weight_decay)
    hist = History()
    best_state = None
    since = 0
    for ep in range(epochs):
        logits = model.forward(g.x, train=True)
        loss, grad = cross_entropy(logits, g.y, train_mask)
        model.backward(grad)
        opt.step()
        val_logits = model.forward(g.x)
        val = accuracy(val_logits, g.y, val_mask)
        val_loss, _ = cross_entropy(val_logits, g.y, val_mask)
        hist.loss.append(loss)
        hist.val_acc.append(val)
        better = val > hist.best_val or (val == hist.best_val and val_loss < hist.best_val_loss)
        if better or best_state is None:
            hist.best_val, hist.best_val_loss = val, val_loss
            hist.best_epoch, since = ep, 0
            best_state = [{k: v.copy() for k, v in l.params.items()} for l in model.layers]
        else:
            since += 1
            if since >= patience:
                break
    for l, st in zip(model.layers, best_state):
        for k in st:
            l.params[k][...] = st[k]
    return hist


def evaluate(model: GNN, g: Graph, mask: np.ndarray) -> float:
    """Accuracy of ``model`` on ``mask`` (evaluation mode)."""
    return accuracy(model.forward(g.x), g.y, mask)


def gradient_check(model: GNN, g: Graph, mask: np.ndarray, eps: float = 1e-5,
                   n_checks: int = 5, seed: int = 0) -> float:
    """Max relative error between analytic and central-difference gradients.

    The model must have ``dropout == 0`` so that the loss is deterministic.
    """
    rng = np.random.default_rng(seed)
    loss, grad = cross_entropy(model.forward(g.x), g.y, mask)
    model.backward(grad)
    worst = 0.0
    for layer in model.layers:
        for name, p in layer.params.items():
            flat = p.reshape(-1)
            ana = layer.grads[name].reshape(-1)
            for i in rng.choice(flat.size, size=min(n_checks, flat.size), replace=False):
                old = flat[i]
                flat[i] = old + eps
                lp, _ = cross_entropy(model.forward(g.x), g.y, mask)
                flat[i] = old - eps
                lm, _ = cross_entropy(model.forward(g.x), g.y, mask)
                flat[i] = old
                num = (lp - lm) / (2 * eps)
                denom = max(abs(num), abs(ana[i]), 1e-8)
                worst = max(worst, abs(num - ana[i]) / denom if denom > 1e-7 else 0.0)
    return worst
