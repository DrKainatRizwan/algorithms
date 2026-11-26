"""Heavy-hitter discovery over huge domains with a prefix tree.

Items are ``bits``-bit integers, so enumerating the domain is infeasible.  Users
are split at random into one group per tree level.  A group at level ``j``
reports the length-``L_j`` prefix of its item through a frequency oracle whose
domain is only the *candidate* prefixes (children of the previous level's
survivors) plus one catch-all bucket.  The top ``k`` candidates survive.  Every
user reports once, with the full budget ``epsilon``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import best_oracle


@dataclass
class LevelTrace:
    """Diagnostics for one level of the prefix tree."""
    prefix_bits: int
    num_candidates: int
    group_size: int
    oracle: str
    survivors: list[int] = field(default_factory=list)


@dataclass
class HeavyHitterResult:
    """Output: ``items`` are (value, estimated frequency) sorted descending."""
    items: list[tuple[int, float]]
    levels: list[LevelTrace]


class PrefixTreeHeavyHitters:
    """Level-by-level heavy-hitter identification under ``epsilon``-LDP."""

    def __init__(self, bits: int, epsilon: float, k: int, step: int = 4,
                 first_bits: int | None = None) -> None:
        if bits < 1 or k < 1 or step < 1:
            raise ValueError("bits, k and step must be positive")
        self.bits, self.epsilon, self.k, self.step = bits, epsilon, k, step
        first = first_bits if first_bits is not None else min(step, bits)
        lengths = [min(first, bits)]
        while lengths[-1] < bits:
            lengths.append(min(lengths[-1] + step, bits))
        self.lengths = lengths

    def discover(self, values: np.ndarray, rng: np.random.Generator) -> HeavyHitterResult:
        values = np.asarray(values, dtype=np.int64)
        if values.size and (values.min() < 0 or values.max() >= 1 << self.bits):
            raise ValueError("values outside the bit range")
        m = len(self.lengths)
        groups = np.array_split(rng.permutation(values.size), m)
        candidates = np.arange(1 << self.lengths[0], dtype=np.int64)
        levels: list[LevelTrace] = []
        est = np.zeros(0)
        for j, (length, idx) in enumerate(zip(self.lengths, groups)):
            if j > 0:
                s = length - self.lengths[j - 1]
                kids = np.arange(1 << s, dtype=np.int64)
                candidates = ((candidates[:, None] << s) | kids[None, :]).ravel()
            candidates = np.sort(candidates)
            prefixes = values[idx] >> (self.bits - length)
            pos = np.searchsorted(candidates, prefixes)
            pos = np.minimum(pos, candidates.size - 1)
            hit = candidates[pos] == prefixes
            mapped = np.where(hit, pos, candidates.size)  # catch-all bucket
            oracle = best_oracle(candidates.size + 1, self.epsilon)
            est = oracle.run(mapped, rng)[:candidates.size]
            keep = np.argsort(-est, kind="stable")[:self.k]
            levels.append(LevelTrace(length, int(candidates.size), int(idx.size),
                                     oracle.name, [int(c) for c in candidates[keep]]))
            if j < m - 1:
                candidates = candidates[keep]
        keep = np.argsort(-est, kind="stable")[:self.k]
        items = [(int(candidates[i]), float(est[i])) for i in keep]
        return HeavyHitterResult(items, levels)
