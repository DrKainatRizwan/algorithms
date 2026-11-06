"""Gradient inversion attacks and defences for federated learning (NumPy only)."""

from .data import make_dataset, DatasetSpec
from .model import MLP
from .analytic import analytic_reconstruct, recover_label
from .optimize import Adam, gradient_matching_attack, matching_loss_and_grad, AttackResult
from .defenses import clip_and_noise, top_k_sparsify, quantize, apply_defense
from .metrics import mse, psnr, match_reconstructions, ssim_global
from .training import train_sgd
from .experiments import build_victim, attack_batch

__all__ = [
    "make_dataset", "DatasetSpec", "MLP", "analytic_reconstruct", "recover_label",
    "Adam", "gradient_matching_attack", "matching_loss_and_grad", "AttackResult",
    "clip_and_noise", "top_k_sparsify", "quantize", "apply_defense",
    "mse", "psnr", "match_reconstructions", "ssim_global", "train_sgd", "build_victim", "attack_batch",
]
