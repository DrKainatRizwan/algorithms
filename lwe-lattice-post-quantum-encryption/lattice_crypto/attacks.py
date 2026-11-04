"""Cryptanalysis of small LWE instances: LLL, Kannan embedding and brute force."""
from __future__ import annotations

import itertools
import math

import numpy as np

from .sampling import center


# --- LLL ------------------------------------------------------------------
def _gso_from(B: np.ndarray, Bs: np.ndarray, mu: np.ndarray, nn: np.ndarray, start: int) -> None:
    """(Re)compute Gram-Schmidt data in place for rows ``start`` onward."""
    for i in range(start, B.shape[0]):
        v = B[i].copy()
        if i:
            mu[i, :i] = (Bs[:i] @ B[i]) / np.where(nn[:i] > 0, nn[:i], 1.0)
            # modified Gram-Schmidt for numerical stability
            v = B[i].copy()
            for j in range(i):
                c = (v @ Bs[j]) / nn[j] if nn[j] > 0 else 0.0
                mu[i, j] = c
                v -= c * Bs[j]
        Bs[i] = v
        nn[i] = v @ v


def lll_reduce(basis, delta: float = 0.99, max_iter: int = 200000) -> np.ndarray:
    """LLL-reduce the rows of an integer basis.

    Integer rows are kept exactly (Python ints); Gram-Schmidt data is floating point and
    is refreshed only for the rows affected by each swap.
    """
    B = np.array(basis, dtype=object)
    k = B.shape[0]
    Bf = np.array(B.tolist(), dtype=float)
    Bs, mu, nn = np.zeros_like(Bf), np.zeros((k, k)), np.zeros(k)
    _gso_from(Bf, Bs, mu, nn, 0)
    i, it = 1, 0
    while i < k:
        it += 1
        if it > max_iter:
            raise RuntimeError("LLL did not converge")
        for j in range(i - 1, -1, -1):  # size reduction
            c = int(round(mu[i, j]))
            if c:
                B[i] = B[i] - c * B[j]
                Bf[i] -= c * Bf[j]
                mu[i, :j] -= c * mu[j, :j]
                mu[i, j] -= c
        if nn[i] >= (delta - mu[i, i - 1] ** 2) * nn[i - 1]:
            i += 1
        else:
            B[[i, i - 1]] = B[[i - 1, i]]
            Bf[[i, i - 1]] = Bf[[i - 1, i]]
            _gso_from(Bf, Bs, mu, nn, i - 1)
            i = max(i - 1, 1)
    return np.array(B.tolist(), dtype=np.int64)


# --- modular linear algebra -------------------------------------------------
def mod_inverse_matrix(M: np.ndarray, q: int) -> np.ndarray | None:
    """Inverse of a square matrix mod prime q by Gauss-Jordan; None if singular."""
    n = M.shape[0]
    aug = [[int(x) % q for x in row] + [int(i == j) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        piv = next((r for r in range(c, n) if aug[r][c] % q), None)
        if piv is None:
            return None
        aug[c], aug[piv] = aug[piv], aug[c]
        inv = pow(aug[c][c], q - 2, q)
        aug[c] = [x * inv % q for x in aug[c]]
        for r in range(n):
            if r != c and aug[r][c]:
                f = aug[r][c]
                aug[r] = [(x - f * y) % q for x, y in zip(aug[r], aug[c])]
    return np.array([row[n:] for row in aug], dtype=np.int64)


def _pick_invertible_rows(A: np.ndarray, q: int):
    """Greedy choice of n rows of A forming an invertible matrix mod q."""
    m, n = A.shape
    chosen: list[int] = []
    for r in range(m):
        trial = chosen + [r]
        # rank test via inverse of padded square only at the end; use incremental elimination
        if _rank_mod(A[trial], q) == len(trial):
            chosen = trial
            if len(chosen) == n:
                break
    if len(chosen) < n:
        return None
    return chosen


def _rank_mod(M: np.ndarray, q: int) -> int:
    rows = [[int(x) % q for x in r] for r in M]
    rank, cols = 0, M.shape[1]
    for c in range(cols):
        piv = next((r for r in range(rank, len(rows)) if rows[r][c]), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        inv = pow(rows[rank][c], q - 2, q)
        rows[rank] = [x * inv % q for x in rows[rank]]
        for r in range(len(rows)):
            if r != rank and rows[r][c]:
                f = rows[r][c]
                rows[r] = [(x - f * y) % q for x, y in zip(rows[r], rows[rank])]
        rank += 1
    return rank


def embedding_basis(A: np.ndarray, b: np.ndarray, q: int, M: int = 1):
    """Kannan embedding basis for the q-ary lattice of A, with target b.

    Returns (basis, perm, A1inv) where perm reorders samples so the first n rows of A are
    invertible. Basis dimension is m + 1.
    """
    m, n = A.shape
    rows = _pick_invertible_rows(A, q)
    if rows is None:
        raise ValueError("A has rank < n mod q")
    perm = rows + [i for i in range(m) if i not in rows]
    Ap, bp = A[perm] % q, b[perm] % q
    A1inv = mod_inverse_matrix(Ap[:n], q)
    T = (Ap[n:] @ A1inv) % q          # (m-n, n)
    basis = np.zeros((m + 1, m + 1), dtype=np.int64)
    for i in range(n):
        basis[i, i] = 1
        basis[i, n:m] = T[:, i]
    for j in range(m - n):
        basis[n + j, n + j] = q
    basis[m, :m] = bp
    basis[m, m] = M
    return basis, perm, A1inv


def primal_attack(A: np.ndarray, b: np.ndarray, q: int, sigma: float = 1.0, delta: float = 0.99):
    """Solve LWE by embedding + LLL. Returns recovered secret s or None."""
    m, n = A.shape
    basis, perm, A1inv = embedding_basis(A, b, q, M=max(1, int(round(sigma))))
    red = lll_reduce(basis, delta=delta)
    Mval = basis[m, m]
    Ap, bp = A[perm] % q, b[perm] % q
    for v in red:
        if abs(int(v[m])) == Mval:
            sign = 1 if v[m] == Mval else -1
            e = sign * v[:m]  # the row equals (bp - Ap s, M) = (e, M) up to sign
            s = (A1inv @ ((bp[:n] - e[:n]) % q)) % q
            if np.array_equal(center((Ap @ s - bp + e) % q, q), np.zeros(m, dtype=np.int64)):
                return s
    return None


def brute_force_lwe(A: np.ndarray, b: np.ndarray, q: int, bound: int):
    """Exhaustive search over s in Z_q^n for one with all residuals within +-bound."""
    m, n = A.shape
    for cand in itertools.product(range(q), repeat=n):
        s = np.array(cand, dtype=np.int64)
        resid = center((b - A @ s) % q, q)
        if np.abs(resid).max() <= bound:
            return s
    return None


# --- heuristic security estimate -----------------------------------------------
def _delta_of_beta(beta: int) -> float:
    """Root-Hermite factor of BKZ-beta (Chen's heuristic)."""
    return ((beta / (2 * math.pi * math.e)) * (math.pi * beta) ** (1.0 / beta)) ** (1.0 / (2 * (beta - 1)))


def estimate_beta(n: int, q: int, sigma: float, m: int | None = None) -> tuple[int, float]:
    """Smallest BKZ block size for which the uSVP primal attack is expected to succeed.

    Uses the usual success condition sqrt(beta/d)*sigma <= delta^(2*beta-d-1) * q^((m-n)/d).
    Returns (beta, core_svp_bits) with the classical sieving cost 2^(0.292 beta).
    """
    best = None
    for mm in ([m] if m else range(n, 4 * n + 1, max(1, n // 8))):
        d = mm + 1
        vol = (mm - n) / d
        for beta in range(40, d + 1):
            lhs = math.sqrt(beta / d) * sigma
            rhs = _delta_of_beta(beta) ** (2 * beta - d - 1) * q ** vol
            if lhs <= rhs:
                if best is None or beta < best:
                    best = beta
                break
    if best is None:
        best = 4 * n
    return best, 0.292 * best
