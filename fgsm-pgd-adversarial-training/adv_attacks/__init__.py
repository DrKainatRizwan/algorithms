"""FGSM / PGD adversarial attacks and adversarial training in pure NumPy."""

from .data import make_blobs, make_spirals, make_rings, make_fragile, train_test_split
from .model import MLP, softmax, cross_entropy
from .attacks import (
    project_linf,
    project_l2,
    fgsm,
    pgd,
    mi_fgsm,
    random_noise,
)
from .training import TrainConfig, fit, adversarial_fit
from .evaluation import (
    accuracy,
    robust_accuracy,
    epsilon_sweep,
    linear_margin_drop,
    gradient_check,
    attack_success_rate,
)

__all__ = [
    "make_blobs", "make_spirals", "make_rings", "make_fragile", "train_test_split",
    "MLP", "softmax", "cross_entropy",
    "project_linf", "project_l2", "fgsm", "pgd", "mi_fgsm", "random_noise",
    "TrainConfig", "fit", "adversarial_fit",
    "accuracy", "robust_accuracy", "epsilon_sweep", "linear_margin_drop",
    "gradient_check", "attack_success_rate",
]
