"""Backdoor detectors: Activation Clustering (Chen et al. 2018) and
Spectral Signatures (Tran et al. 2018)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clustering import kmeans, pca, silhouette_score
from .model import MLP


@dataclass
class ClassReport:
    """Per-class statistics from activation clustering."""
    label: int
    n: int
    silhouette: float
    relative_size: float  # fraction of the class in the smaller cluster
    suspicious: bool
    flagged: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, int))


@dataclass
class DetectionResult:
    """Indices (into the inspected dataset) flagged as poisoned."""
    flagged: np.ndarray
    reports: list[ClassReport] = field(default_factory=list)

    @property
    def mask_size(self) -> int:
        return int(len(self.flagged))


def activation_clustering(model: MLP, X: np.ndarray, y: np.ndarray, n_classes: int,
                          n_components: int = 5, sil_threshold: float = 0.30,
                          size_threshold: float = 0.35, seed: int = 0) -> DetectionResult:
    """For each class: project last-layer activations with PCA, run 2-means,
    and call the class suspicious if the clusters are well separated
    (silhouette >= ``sil_threshold``) *and* one cluster is small
    (relative size <= ``size_threshold``). The smaller cluster is flagged.
    """
    acts = model.activations(X)
    flagged: list[np.ndarray] = []
    reports: list[ClassReport] = []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        if len(idx) < 4:
            reports.append(ClassReport(c, len(idx), 0.0, 0.5, False))
            continue
        Z, _, _ = pca(acts[idx], n_components)
        labels, _, _ = kmeans(Z, 2, seed=seed + c)
        sizes = np.bincount(labels, minlength=2)
        small = int(sizes.argmin())
        rel = sizes[small] / len(idx)
        sil = silhouette_score(Z, labels)
        susp = bool(sil >= sil_threshold and rel <= size_threshold)
        sel = idx[labels == small] if susp else np.zeros(0, int)
        flagged.append(sel)
        reports.append(ClassReport(c, len(idx), sil, float(rel), susp, sel))
    out = np.sort(np.concatenate(flagged)) if flagged else np.zeros(0, int)
    return DetectionResult(out, reports)


def spectral_signature(model: MLP, X: np.ndarray, y: np.ndarray, n_classes: int,
                       eps: float = 0.1, removal_factor: float = 1.5) -> DetectionResult:
    """Per class, centre the activations, take the top singular vector and
    score each sample by its squared projection. The top ``removal_factor*eps``
    fraction of each class is flagged (the original paper uses 1.5*eps)."""
    acts = model.activations(X)
    flagged = []
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        if len(idx) < 2:
            continue
        A = acts[idx] - acts[idx].mean(0)
        _, _, vt = np.linalg.svd(A, full_matrices=False)
        scores = (A @ vt[0]) ** 2
        k = int(np.ceil(removal_factor * eps * len(idx)))
        flagged.append(idx[np.argsort(-scores)[:k]])
    out = np.sort(np.concatenate(flagged)) if flagged else np.zeros(0, int)
    return DetectionResult(out)


def evaluate_detection(flagged: np.ndarray, is_poison: np.ndarray) -> dict[str, float]:
    """Precision/recall/F1 of ``flagged`` indices against the true poison mask."""
    pred = np.zeros(len(is_poison), bool)
    pred[flagged] = True
    tp = int((pred & is_poison).sum())
    fp = int((pred & ~is_poison).sum())
    fn = int((~pred & is_poison).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
