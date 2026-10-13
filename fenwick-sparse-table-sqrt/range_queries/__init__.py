"""Range-query data structures: Fenwick trees, sparse tables, sqrt decomposition."""
from .fenwick import Fenwick, RangeFenwick, Fenwick2D
from .sparse_table import SparseTable, ArgMinSparseTable, DisjointSparseTable
from .sqrt_decomp import SqrtDecomposition, SqrtRangeAdd, mo_distinct_counts
from .naive import NaiveArray

__all__ = [
    "Fenwick", "RangeFenwick", "Fenwick2D", "SparseTable", "ArgMinSparseTable", "DisjointSparseTable",
    "SqrtDecomposition", "SqrtRangeAdd", "mo_distinct_counts", "NaiveArray",
]
