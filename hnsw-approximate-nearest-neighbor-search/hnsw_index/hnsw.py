"""Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018)."""
from __future__ import annotations

import heapq
import math

import numpy as np

from .distance import get_metric


class HNSWIndex:
    """Multi-layer proximity graph supporting incremental insert, k-NN query
    and deletion (tombstones).

    Parameters
    ----------
    M: max neighbours per node on upper layers (layer 0 allows ``2*M``).
    ef_construction: beam width used while inserting.
    heuristic: use the diversity-based neighbour selection (Alg. 4) instead
        of plain closest-M selection.
    single_layer: if True every node lives on layer 0 (plain NSW baseline).
    """

    def __init__(self, dim: int, metric: str = "l2", M: int = 12,
                 ef_construction: int = 100, heuristic: bool = True,
                 single_layer: bool = False, seed: int = 0) -> None:
        if M < 2:
            raise ValueError("M must be >= 2")
        self.dim = dim
        self.metric_name = metric
        self._dist = get_metric(metric)
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.heuristic = heuristic
        self.single_layer = single_layer
        self._ml = 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)
        self._vecs = np.empty((16, dim))
        self._n = 0
        self._links: list[list[list[int]]] = []   # node -> layer -> neighbours
        self._deleted: set[int] = set()
        self.entry: int = -1
        self.max_level: int = -1
        self.dist_count = 0

    # ------------------------------------------------------------ basics
    def __len__(self) -> int:
        return self._n - len(self._deleted)

    def reset_counter(self) -> None:
        self.dist_count = 0

    def level_of(self, node: int) -> int:
        return len(self._links[node]) - 1

    def neighbors(self, node: int, layer: int) -> list[int]:
        return list(self._links[node][layer])

    def _d(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        self.dist_count += len(ids)
        return self._dist(q, self._vecs[ids])

    def _random_level(self) -> int:
        if self.single_layer:
            return 0
        u = 1.0 - self._rng.random()
        return int(-math.log(u) * self._ml)

    # ------------------------------------------------------------ search
    def _search_layer(self, q: np.ndarray, entries: list[int], ef: int,
                      layer: int) -> list[tuple[float, int]]:
        """Beam search on one layer; returns up to ``ef`` (dist, id), ascending."""
        visited = set(entries)
        d0 = self._d(q, entries)
        cand = [(float(d), e) for d, e in zip(d0, entries)]
        heapq.heapify(cand)
        best = [(-d, e) for d, e in cand]          # max-heap of results
        heapq.heapify(best)
        while cand:
            d, c = heapq.heappop(cand)
            if d > -best[0][0] and len(best) >= ef:
                break
            nb = [x for x in self._links[c][layer] if x not in visited]
            if not nb:
                continue
            visited.update(nb)
            for dist, e in zip(self._d(q, nb), nb):
                dist = float(dist)
                if len(best) < ef or dist < -best[0][0]:
                    heapq.heappush(cand, (dist, e))
                    heapq.heappush(best, (-dist, e))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-nd, e) for nd, e in best)

    def _select(self, cands: list[tuple[float, int]], m: int) -> list[int]:
        """Choose at most ``m`` neighbours from sorted (dist, id) candidates."""
        if not self.heuristic or len(cands) <= m:
            return [e for _, e in cands[:m]]
        chosen: list[int] = []
        skipped: list[int] = []
        for d, e in cands:
            if len(chosen) >= m:
                break
            # keep e only if it is closer to the query than to any chosen one
            if chosen:
                de = self._d(self._vecs[e], chosen)
                if np.any(de < d):
                    skipped.append(e)
                    continue
            chosen.append(e)
        for e in skipped:                           # keepPrunedConnections
            if len(chosen) >= m:
                break
            chosen.append(e)
        return chosen

    # ------------------------------------------------------------ insert
    def add(self, x: np.ndarray) -> int:
        """Insert one vector and return its id."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {x.shape}")
        if self._n == len(self._vecs):
            self._vecs = np.vstack([self._vecs, np.empty_like(self._vecs)])
        node = self._n
        self._vecs[node] = x
        self._n += 1
        level = self._random_level()
        self._links.append([[] for _ in range(level + 1)])
        if self.entry < 0:
            self.entry, self.max_level = node, level
            return node
        ep = [self.entry]
        for layer in range(self.max_level, level, -1):
            ep = [self._search_layer(x, ep, 1, layer)[0][1]]
        for layer in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(x, ep, self.ef_construction, layer)
            cap = self.M0 if layer == 0 else self.M
            nbrs = self._select(found, self.M)
            self._links[node][layer] = list(nbrs)
            for nb in nbrs:
                lst = self._links[nb][layer]
                lst.append(node)
                if len(lst) > cap:
                    ds = self._d(self._vecs[nb], lst)
                    ranked = sorted(zip(ds.tolist(), lst))
                    self._links[nb][layer] = self._select(ranked, cap)
            ep = [e for _, e in found]
        if level > self.max_level:
            self.entry, self.max_level = node, level
        return node

    def add_batch(self, X: np.ndarray) -> list[int]:
        return [self.add(x) for x in np.asarray(X, dtype=float)]

    # ------------------------------------------------------------ query
    def query(self, q: np.ndarray, k: int = 10, ef: int | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k nearest live neighbours as (ids, distances)."""
        if self.entry < 0 or len(self) == 0:
            return np.empty(0, dtype=int), np.empty(0)
        q = np.asarray(q, dtype=float)
        ef = max(ef or 50, k)
        if self._deleted:
            ef += len(self._deleted)
        ep = [self.entry]
        for layer in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, layer)[0][1]]
        res = [(d, e) for d, e in self._search_layer(q, ep, ef, 0)
               if e not in self._deleted][:k]
        return (np.array([e for _, e in res], dtype=int),
                np.array([d for d, _ in res]))

    # ------------------------------------------------------------ delete
    def delete(self, node: int) -> None:
        """Mark a node deleted (it still routes queries but is never returned)."""
        if not 0 <= node < self._n:
            raise IndexError(node)
        self._deleted.add(node)

    # ------------------------------------------------------------ stats
    def level_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for links in self._links:
            hist[len(links) - 1] = hist.get(len(links) - 1, 0) + 1
        return dict(sorted(hist.items()))

    def mean_degree(self, layer: int = 0) -> float:
        deg = [len(l[layer]) for l in self._links if len(l) > layer]
        return float(np.mean(deg)) if deg else 0.0

    def is_connected_layer0(self) -> bool:
        """Whether every node is reachable from the entry point on layer 0."""
        if self._n == 0:
            return True
        seen = {self.entry}
        stack = [self.entry]
        while stack:
            for nb in self._links[stack.pop()][0]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == self._n
