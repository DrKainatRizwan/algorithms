"""Sanitisation + retraining and backdoor metrics."""
from __future__ import annotations

import numpy as np

from .data import apply_trigger
from .model import MLP, train


def clean_accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(X) == y).mean())


def attack_success_rate(model: MLP, X: np.ndarray, y: np.ndarray, target: int,
                        size: int = 3, value: float = 4.0) -> float:
    """Fraction of triggered non-target test inputs classified as ``target``."""
    keep = y != target
    pred = model.predict(apply_trigger(X[keep], size=size, value=value))
    return float((pred == target).mean())


def sanitize_and_retrain(X: np.ndarray, y: np.ndarray, flagged: np.ndarray, n_classes: int,
                         **train_kwargs) -> MLP:
    """Drop flagged samples and train a fresh model on the remainder."""
    keep = np.ones(len(y), bool)
    keep[flagged] = False
    return train(X[keep], y[keep], n_classes, **train_kwargs)
