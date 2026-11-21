"""Save / load an HNSW index to a single ``.npz`` file."""
from __future__ import annotations

import numpy as np

from .hnsw import HNSWIndex


def save_index(index: HNSWIndex, path: str) -> None:
    """Serialise vectors, graph, parameters and tombstones."""
    flat, offsets = [], []   # rows of (node, layer, count) then neighbours
    for node, layers in enumerate(index._links):
        for layer, nbrs in enumerate(layers):
            offsets.append((node, layer, len(nbrs)))
            flat.extend(nbrs)
    params = np.array([index.dim, index.M, index.ef_construction,
                       int(index.heuristic), int(index.single_layer),
                       index.entry, index.max_level])
    np.savez_compressed(
        path, vecs=index._vecs[:index._n], offsets=np.array(offsets, dtype=int).reshape(-1, 3),
        flat=np.array(flat, dtype=int), deleted=np.array(sorted(index._deleted), dtype=int),
        params=params, metric=np.array(index.metric_name))


def load_index(path: str) -> HNSWIndex:
    """Inverse of :func:`save_index`."""
    z = np.load(path)
    dim, M, efc, heur, single, entry, max_level = (int(v) for v in z["params"])
    idx = HNSWIndex(dim, metric=str(z["metric"]), M=M, ef_construction=efc,
                    heuristic=bool(heur), single_layer=bool(single))
    vecs = z["vecs"]
    n = len(vecs)
    idx._vecs = np.vstack([vecs, np.empty((max(16, n), dim))]) if n else idx._vecs
    idx._n = n
    idx._links = [[] for _ in range(n)]
    pos = 0
    flat = z["flat"]
    for node, layer, cnt in z["offsets"]:
        assert layer == len(idx._links[node])
        idx._links[node].append([int(v) for v in flat[pos:pos + cnt]])
        pos += cnt
    idx._deleted = {int(v) for v in z["deleted"]}
    idx.entry, idx.max_level = entry, max_level
    return idx
