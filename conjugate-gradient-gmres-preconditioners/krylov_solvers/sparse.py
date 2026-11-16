"""Compressed sparse row matrix implemented with numpy arrays."""
from __future__ import annotations

import numpy as np


class CSRMatrix:
    """Square or rectangular sparse matrix in CSR format."""

    def __init__(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
                 shape: tuple[int, int]) -> None:
        self.data = np.asarray(data, dtype=float)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.shape = (int(shape[0]), int(shape[1]))
        if len(self.indptr) != self.shape[0] + 1:
            raise ValueError("indptr length must be nrows + 1")

    @classmethod
    def from_triplets(cls, rows, cols, vals, shape) -> "CSRMatrix":
        """Build from COO triplets; duplicate entries are summed."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=float)
        order = np.lexsort((cols, rows))
        rows, cols, vals = rows[order], cols[order], vals[order]
        if len(rows):
            new = np.ones(len(rows), dtype=bool)
            new[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
            grp = np.cumsum(new) - 1
            summed = np.zeros(grp[-1] + 1)
            np.add.at(summed, grp, vals)
            rows, cols, vals = rows[new], cols[new], summed
        indptr = np.zeros(shape[0] + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        return cls(vals, cols, np.cumsum(indptr), shape)

    @classmethod
    def from_dense(cls, a: np.ndarray) -> "CSRMatrix":
        r, c = np.nonzero(a)
        return cls.from_triplets(r, c, a[r, c], a.shape)

    @property
    def nnz(self) -> int:
        return len(self.data)

    def row_ids(self) -> np.ndarray:
        return np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute A @ x in O(nnz)."""
        prod = self.data * x[self.indices]
        out = np.zeros(self.shape[0])
        np.add.at(out, self.row_ids(), prod)
        return out

    __matmul__ = matvec

    def transpose(self) -> "CSRMatrix":
        return CSRMatrix.from_triplets(self.indices, self.row_ids(), self.data,
                                       (self.shape[1], self.shape[0]))

    def diagonal(self) -> np.ndarray:
        d = np.zeros(min(self.shape))
        rid = self.row_ids()
        mask = rid == self.indices
        d[rid[mask]] = self.data[mask]
        return d

    def to_dense(self) -> np.ndarray:
        a = np.zeros(self.shape)
        a[self.row_ids(), self.indices] = self.data
        return a

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.indptr[i], self.indptr[i + 1]
        return self.indices[s:e], self.data[s:e]

    def is_symmetric(self, tol: float = 1e-12) -> bool:
        if self.shape[0] != self.shape[1]:
            return False
        return bool(np.abs(self.to_dense() - self.transpose().to_dense()).max() <= tol)
