"""Preconditioners: Jacobi, SSOR, incomplete Cholesky IC(0) and ILU(0)."""
from __future__ import annotations

import numpy as np

from .sparse import CSRMatrix


class Preconditioner:
    """Interface: apply(r) returns an approximation of A^{-1} r."""

    def apply(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Identity(Preconditioner):
    def apply(self, r: np.ndarray) -> np.ndarray:
        return r.copy()


class Jacobi(Preconditioner):
    """Diagonal scaling."""

    def __init__(self, a: CSRMatrix) -> None:
        d = a.diagonal()
        if np.any(d == 0):
            raise ValueError("zero on diagonal")
        self.inv = 1.0 / d

    def apply(self, r: np.ndarray) -> np.ndarray:
        return self.inv * r


def _forward(a: CSRMatrix, b: np.ndarray, diag: np.ndarray | None, unit: bool) -> np.ndarray:
    """Solve (L) y = b using strictly lower entries of a; diag overrides diagonal."""
    n = a.shape[0]
    y = np.zeros(n)
    for i in range(n):
        cols, vals = a.row(i)
        m = cols < i
        s = b[i] - vals[m] @ y[cols[m]]
        if unit:
            y[i] = s
        else:
            y[i] = s / (diag[i] if diag is not None else vals[cols == i][0])
    return y


def _backward(a: CSRMatrix, b: np.ndarray, diag: np.ndarray | None, unit: bool) -> np.ndarray:
    """Solve U x = b using strictly upper entries of a."""
    n = a.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        cols, vals = a.row(i)
        m = cols > i
        s = b[i] - vals[m] @ x[cols[m]]
        if unit:
            x[i] = s
        else:
            x[i] = s / (diag[i] if diag is not None else vals[cols == i][0])
    return x


class SSOR(Preconditioner):
    """Symmetric SOR: M = (D/w + L) (D/w)^{-1} (D/w + U) * w/(2-w)."""

    def __init__(self, a: CSRMatrix, omega: float = 1.0) -> None:
        if not 0 < omega < 2:
            raise ValueError("omega must be in (0, 2)")
        self.a, self.omega = a, omega
        self.d = a.diagonal() / omega

    def apply(self, r: np.ndarray) -> np.ndarray:
        w = self.omega
        y = _forward(self.a, r, self.d, False)
        z = y * self.d
        x = _backward(self.a, z, self.d, False)
        return x * (2 - w) / w


class IC0(Preconditioner):
    """Incomplete Cholesky with zero fill: A ~ L L^T on the sparsity of tril(A)."""

    def __init__(self, a: CSRMatrix) -> None:
        n = a.shape[0]
        rows: list[dict[int, float]] = []
        for i in range(n):
            cols, vals = a.row(i)
            rows.append({int(c): float(v) for c, v in zip(cols, vals) if c <= i})
        for i in range(n):
            ri = rows[i]
            for j in sorted(ri):
                if j == i:
                    break
                rj = rows[j]
                s = ri[j]
                for k, v in ri.items():
                    if k >= j:
                        break
                    if k in rj:
                        s -= v * rj[k]
                ri[j] = s / rj[j]
            d = ri[i] - sum(v * v for k, v in ri.items() if k < i)
            if d <= 0:
                raise np.linalg.LinAlgError("IC(0) breakdown: non-positive pivot")
            ri[i] = np.sqrt(d)
        rr, cc, vv = [], [], []
        for i, ri in enumerate(rows):
            for j, v in ri.items():
                rr.append(i); cc.append(j); vv.append(v)
        self.L = CSRMatrix.from_triplets(rr, cc, vv, a.shape)
        self.U = self.L.transpose()

    def apply(self, r: np.ndarray) -> np.ndarray:
        y = _forward(self.L, r, None, False)
        return _backward(self.U, y, None, False)


class ILU0(Preconditioner):
    """Incomplete LU with zero fill (IKJ variant); unit lower L, upper U."""

    def __init__(self, a: CSRMatrix) -> None:
        n = a.shape[0]
        rows = []
        for i in range(n):
            cols, vals = a.row(i)
            rows.append({int(c): float(v) for c, v in zip(cols, vals)})
        for i in range(1, n):
            ri = rows[i]
            for k in sorted(c for c in ri if c < i):
                rk = rows[k]
                ri[k] /= rk[k]
                f = ri[k]
                for j, v in rk.items():
                    if j > k and j in ri:
                        ri[j] -= f * v
        rr, cc, vv = [], [], []
        for i, ri in enumerate(rows):
            if ri.get(i, 0.0) == 0.0:
                raise np.linalg.LinAlgError("ILU(0) zero pivot")
            for j, v in ri.items():
                rr.append(i); cc.append(j); vv.append(v)
        self.LU = CSRMatrix.from_triplets(rr, cc, vv, a.shape)

    def apply(self, r: np.ndarray) -> np.ndarray:
        y = _forward(self.LU, r, None, True)
        return _backward(self.LU, y, None, False)
