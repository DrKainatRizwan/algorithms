"""Reusable experiment drivers shared by the demo and the tests."""
from __future__ import annotations

import numpy as np

from .data import DatasetSpec, make_dataset
from .defenses import apply_defense
from .metrics import match_reconstructions, psnr
from .model import MLP
from .optimize import gradient_matching_attack
from .training import train_sgd


def build_victim(seed: int = 0, hidden: int = 32, pretrain_epochs: int = 3,
                 spec: DatasetSpec | None = None):
    """Create a lightly trained victim model and its dataset."""
    spec = spec or DatasetSpec()
    rng = np.random.default_rng(seed)
    X, y = make_dataset(spec, rng)
    model = MLP(spec.dim, hidden, spec.n_classes, rng)
    train_sgd(model, X, y, epochs=pretrain_epochs, lr=0.1, batch=16, rng=rng)
    return model, X, y, spec


def attack_batch(model: MLP, X: np.ndarray, y: np.ndarray, batch: int, seed: int,
                 defense: str = "none", strength: float = 0.0, iters: int = 400,
                 restarts: int = 2) -> dict[str, float]:
    """Attack the first ``batch`` samples; return mean PSNR after optimal matching."""
    rng = np.random.default_rng(seed)
    xb, yb = X[:batch], y[:batch]
    target = apply_defense(model.gradients(xb, yb), defense, strength, rng)
    res = gradient_matching_attack(model, target, yb, rng, iters=iters, restarts=restarts)
    perm, m = match_reconstructions(xb, res.X)
    p = float(np.mean([psnr(xb[i], res.X[perm[i]]) for i in range(batch)]))
    return {"psnr": p, "mse": float(m), "match_loss": float(res.loss)}
