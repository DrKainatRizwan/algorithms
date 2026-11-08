"""Locality-sensitive hashing toolkit: MinHash, SimHash, hyperplane and p-stable LSH."""

from .shingles import fnv1a_64, char_shingles, word_shingles, jaccard
from .minhash import MinHasher
from .simhash import SimHasher, SimHashIndex, hamming
from .hyperplane import HyperplaneHasher, PStableHasher
from .index import MinHashLSHIndex, VectorLSHIndex, collision_probability, optimal_bands
from .datasets import make_documents, make_clusters, exact_knn, recall_at_k

__all__ = [
    "fnv1a_64", "char_shingles", "word_shingles", "jaccard", "MinHasher",
    "SimHasher", "SimHashIndex", "hamming", "HyperplaneHasher", "PStableHasher",
    "MinHashLSHIndex", "VectorLSHIndex", "collision_probability", "optimal_bands",
    "make_documents", "make_clusters", "exact_knn", "recall_at_k",
]
