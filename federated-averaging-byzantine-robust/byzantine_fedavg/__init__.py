"""Federated averaging with Byzantine-robust aggregation rules."""

from .aggregators import (
    AGGREGATORS,
    coordinate_median,
    fedavg_mean,
    geometric_median,
    krum,
    multi_krum,
    trimmed_mean,
)
from .attacks import (
    ATTACKS,
    gaussian_attack,
    little_is_enough,
    scaled_negative_attack,
    sign_flip_attack,
)
from .data import make_blobs_dataset, partition_iid, partition_label_skew
from .model import SoftmaxModel
from .simulator import FederatedSimulator, SimulationResult

__all__ = [
    "AGGREGATORS", "ATTACKS", "FederatedSimulator", "SimulationResult",
    "SoftmaxModel", "coordinate_median", "fedavg_mean", "gaussian_attack",
    "geometric_median", "krum", "little_is_enough", "make_blobs_dataset",
    "multi_krum", "partition_iid", "partition_label_skew",
    "scaled_negative_attack", "sign_flip_attack", "trimmed_mean",
]
