"""Evaluation metrics for membership-inference attacks and reconstructions."""
from __future__ import annotations

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties handled by mid-ranks."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    ``scores_in`` are scores of true members, ``scores_out`` of non-members;
    a higher score means "more likely a member".
    """
    si = np.asarray(scores_in, dtype=float)
    so = np.asarray(scores_out, dtype=float)
    n1, n0 = len(si), len(so)
    if n1 == 0 or n0 == 0:
        raise ValueError("both score sets must be non-empty")
    ranks = _rankdata(np.concatenate([si, so]))
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def roc_curve(scores_in: np.ndarray, scores_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays sweeping the decision threshold downwards."""
    si = np.asarray(scores_in, dtype=float)
    so = np.asarray(scores_out, dtype=float)
    thr = np.unique(np.concatenate([si, so]))[::-1]
    tpr = np.array([(si >= t).mean() for t in thr])
    fpr = np.array([(so >= t).mean() for t in thr])
    return np.concatenate([[0.0], fpr, [1.0]]), np.concatenate([[0.0], tpr, [1.0]])


def tpr_at_fpr(scores_in: np.ndarray, scores_out: np.ndarray, target_fpr: float = 0.05) -> float:
    """Largest true-positive rate attainable with FPR <= ``target_fpr``."""
    fpr, tpr = roc_curve(scores_in, scores_out)
    ok = fpr <= target_fpr + 1e-12
    return float(tpr[ok].max())


def membership_advantage(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """Yeom et al. advantage ``max_t (TPR(t) - FPR(t))`` (a KS statistic)."""
    fpr, tpr = roc_curve(scores_in, scores_out)
    return float((tpr - fpr).max())


def best_threshold(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """Threshold maximising balanced accuracy on the given calibration scores."""
    cand = np.unique(np.concatenate([scores_in, scores_out]))
    best_t, best_v = float(cand[0]), -1.0
    for t in cand:
        v = 0.5 * ((scores_in >= t).mean() + (scores_out < t).mean())
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t


def summarize(scores_in: np.ndarray, scores_out: np.ndarray) -> dict[str, float]:
    """Standard bundle of attack-quality numbers."""
    return {
        "auc": auc(scores_in, scores_out),
        "advantage": membership_advantage(scores_in, scores_out),
        "tpr@5%fpr": tpr_at_fpr(scores_in, scores_out, 0.05),
        "tpr@1%fpr": tpr_at_fpr(scores_in, scores_out, 0.01),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two flattened vectors (0 if either is zero)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def centered_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened arrays."""
    return cosine(np.asarray(a, float).ravel() - np.mean(a), np.asarray(b, float).ravel() - np.mean(b))
