"""Data poisoning attacks and robust training for logistic regression."""

from .model import LogisticRegression, sigmoid
from .data import make_blobs, train_val_test_split, Dataset
from .influence import (
    removal_influence,
    flip_influence,
    poison_input_gradient,
    validation_gradient,
)
from .attacks import (
    AttackResult,
    random_label_flip,
    influence_label_flip,
    gradient_poisoning,
    boundary_flip,
)
from .defenses import (
    DefenseResult,
    no_defense,
    trim_fit,
    sphere_slab_filter,
    knn_sanitize,
    trimmed_gradient_fit,
    DEFENSES,
)
from .evaluate import evaluate_grid, format_table

__all__ = [
    "LogisticRegression", "sigmoid", "make_blobs", "train_val_test_split", "Dataset",
    "removal_influence", "flip_influence", "poison_input_gradient", "validation_gradient",
    "AttackResult", "random_label_flip", "influence_label_flip", "gradient_poisoning",
    "boundary_flip", "DefenseResult", "no_defense", "trim_fit", "sphere_slab_filter",
    "knn_sanitize", "trimmed_gradient_fit", "DEFENSES", "evaluate_grid", "format_table",
]
