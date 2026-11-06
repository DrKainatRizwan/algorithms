"""Backdoor trigger detection via activation clustering and spectral signatures."""

from .data import make_dataset, apply_trigger, poison_dataset, PoisonedData
from .model import MLP, train
from .clustering import pca, kmeans, silhouette_score
from .detection import (
    activation_clustering,
    spectral_signature,
    ClassReport,
    DetectionResult,
    evaluate_detection,
)
from .defense import sanitize_and_retrain, attack_success_rate, clean_accuracy

__all__ = [
    "make_dataset", "apply_trigger", "poison_dataset", "PoisonedData",
    "MLP", "train", "pca", "kmeans", "silhouette_score",
    "activation_clustering", "spectral_signature", "ClassReport",
    "DetectionResult", "evaluate_detection", "sanitize_and_retrain",
    "attack_success_rate", "clean_accuracy",
]
