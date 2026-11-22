"""CUR / interpolative decompositions via pivoting, leverage-score sampling and DEIM."""
from __future__ import annotations

import numpy as np


def pivoted_qr_columns(A: np.ndarray, k: int) -> np.ndarray:
    """Indices of k columns chosen by greedy column-pivoted Gram-Schmidt (Businger-Golub)."""
    R = np.array(A, dtype=float, copy=True)
    n = R.shape[1]
    k = min(k, n, R.shape[0])
    chosen: list[int] = []
    for _ in range(k):
        norms = np.sum(R * R, axis=0)
        norms[chosen] = -1.0
        j = int(np.argmax(norms))
        chosen.append(j)
        q = R[:, j]
        nq = np.linalg.norm(q)
        if nq < 1e-14:
            break
        q = q / nq
        R = R - np.outer(q, q @ R)
    return np.array(chosen, dtype=int)


def leverage_scores(A: np.ndarray, k: int, axis: int = 0) -> np.ndarray:
    """Rank-k statistical leverage scores of the rows (axis=0) or columns (axis=1); they sum to k."""
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    V = U[:, :k] if axis == 0 else Vt[:k].T
    return np.sum(V * V, axis=1)


def deim_indices(V: np.ndarray) -> np.ndarray:
    """Discrete empirical interpolation: pick one row index per column of an orthonormal basis V."""
    n, k = V.shape
    idx = [int(np.argmax(np.abs(V[:, 0])))]
    for j in range(1, k):
        c = np.linalg.solve(V[idx, :j], V[idx, j])
        r = V[:, j] - V[:, :j] @ c
        idx.append(int(np.argmax(np.abs(r))))
    return np.array(idx, dtype=int)


def _sample(p: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    count = min(count, int(np.count_nonzero(p)))
    return np.sort(rng.choice(len(p), size=count, replace=False, p=p / p.sum()))


def cur_decomposition(A: np.ndarray, k: int, method: str = "leverage", c: int | None = None,
                      r: int | None = None, rng: np.random.Generator | None = None):
    """Return (C, U, R, cols, rows) with A ≈ C U R, U = C^+ A R^+ (optimal for the chosen C, R).

    method: 'leverage' (sampling w/o replacement), 'uniform', 'pivot' (QR pivots) or 'deim'.
    """
    rng = rng or np.random.default_rng(0)
    m, n = A.shape
    c = c or k
    r = r or k
    if method == "leverage":
        cols = _sample(leverage_scores(A, k, axis=1), c, rng)
        rows = _sample(leverage_scores(A, k, axis=0), r, rng)
    elif method == "uniform":
        cols = np.sort(rng.choice(n, size=min(c, n), replace=False))
        rows = np.sort(rng.choice(m, size=min(r, m), replace=False))
    elif method == "pivot":
        cols = np.sort(pivoted_qr_columns(A, c))
        rows = np.sort(pivoted_qr_columns(A.T, r))
    elif method == "deim":
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        cols = np.sort(deim_indices(Vt[:k].T))
        rows = np.sort(deim_indices(U[:, :k]))
    else:
        raise ValueError(f"unknown method {method!r}")
    C, R = A[:, cols], A[rows, :]
    U = np.linalg.pinv(C) @ A @ np.linalg.pinv(R)
    return C, U, R, cols, rows


def interpolative_decomposition(A: np.ndarray, k: int):
    """Column ID: A ≈ A[:, J] @ T with J from pivoted QR and T = A[:, J]^+ A. Returns (J, T)."""
    J = pivoted_qr_columns(A, k)
    T = np.linalg.lstsq(A[:, J], A, rcond=None)[0]
    return J, T
