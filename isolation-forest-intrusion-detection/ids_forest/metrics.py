"""Ranking and threshold metrics for anomaly scores."""

from __future__ import annotations

import numpy as np


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties handled."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under ROC via the Mann-Whitney U statistic."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes required")
    r = _rank(score)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision: mean precision at each true positive in descending-score order."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.sum() == 0:
        raise ValueError("no positives")
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float(prec[ys == 1].mean())


def precision_recall_f1(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 (predict anomaly when score > threshold); returns (thr, f1)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    best_t, best_f = float(score.min() - 1e-12), -1.0
    for t in np.unique(score):
        f = precision_recall_f1(y, (score >= t).astype(int))[2]
        if f > best_f:
            best_f, best_t = f, float(t) - 1e-12
    return best_t, best_f
