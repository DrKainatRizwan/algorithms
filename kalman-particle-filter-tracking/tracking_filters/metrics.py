"""Accuracy and consistency metrics for tracking filters."""
from __future__ import annotations
import numpy as np


def rmse(est: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error over all samples and components."""
    return float(np.sqrt(np.mean((np.asarray(est) - np.asarray(truth)) ** 2)))


def position_errors(est: np.ndarray, truth: np.ndarray, dim: int = 2) -> np.ndarray:
    """Euclidean position error per step."""
    return np.linalg.norm(np.asarray(est)[:, :dim] - np.asarray(truth)[:, :dim], axis=1)


def nees(est, truth, Ps) -> np.ndarray:
    """Normalised estimation error squared per step (expected value = state dim)."""
    e = np.asarray(est) - np.asarray(truth)
    return np.array([ei @ np.linalg.solve(P, ei) for ei, P in zip(e, Ps)])


def nis(innovations, Ss) -> np.ndarray:
    """Normalised innovation squared per step (expected value = measurement dim)."""
    return np.array([y @ np.linalg.solve(S, y) for y, S in zip(innovations, Ss)])


def mean_consistency(values: np.ndarray, dof: int) -> float:
    """Ratio of the average NEES/NIS to its expected value; ~1 for a consistent filter."""
    return float(np.mean(values) / dof)
