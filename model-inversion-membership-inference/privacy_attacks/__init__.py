"""Membership-inference and model-inversion attack simulator with defences."""
from .attacks import (CalibratedAttack, LabelOnlyAttack, ShadowAttack, score_attack,
                      train_shadow_models)
from .data import World, make_world
from .defenses import LabelOnly, OutputNoise, Rounding, Temperature, TopK
from .inversion import (invert_all_classes, invert_blackbox, invert_whitebox,
                        reconstruction_quality)
from .metrics import auc, membership_advantage, summarize, tpr_at_fpr
from .models import MLP, TrainConfig, train, zcdp_epsilon

__all__ = [
    "CalibratedAttack", "LabelOnlyAttack", "ShadowAttack", "score_attack", "train_shadow_models",
    "World", "make_world", "LabelOnly", "OutputNoise", "Rounding", "Temperature", "TopK",
    "invert_all_classes", "invert_blackbox", "invert_whitebox", "reconstruction_quality",
    "auc", "membership_advantage", "summarize", "tpr_at_fpr",
    "MLP", "TrainConfig", "train", "zcdp_epsilon",
]
__version__ = "1.0.0"
