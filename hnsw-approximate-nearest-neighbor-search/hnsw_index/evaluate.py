"""Recall and benchmarking helpers."""
from __future__ import annotations

import time

import numpy as np

from .exact import BruteForceIndex


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Mean fraction of true neighbours recovered (rows are queries)."""
    hits = [len(set(f.tolist()) & set(t.tolist())) / len(t) for f, t in zip(found, truth)]
    return float(np.mean(hits))


def benchmark_index(index, data: np.ndarray, queries: np.ndarray, k: int,
                    ef_values: list[int], metric: str = "l2") -> list[dict]:
    """Measure recall, query time and distance evaluations for each ``ef``."""
    bf = BruteForceIndex(metric)
    bf.add(data)
    t0 = time.perf_counter()
    truth = bf.batch_query(queries, k)
    brute_t = (time.perf_counter() - t0) / len(queries)
    rows = []
    for ef in ef_values:
        index.reset_counter()
        t0 = time.perf_counter()
        found = np.array([index.query(q, k, ef=ef)[0] for q in queries])
        dt = (time.perf_counter() - t0) / len(queries)
        rows.append({
            "ef": ef,
            "recall": recall_at_k(found, truth),
            "ms_per_query": dt * 1e3,
            "dist_evals": index.dist_count / len(queries),
            "brute_ms": brute_t * 1e3,
        })
    return rows
