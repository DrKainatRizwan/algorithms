"""Learned environment models for Dyna-style planning."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

Outcome = tuple[int, bool]  # (next_state, done)


class TabularModel:
    """Learned model of ``(s, a) -> (r, s', done)``.

    ``mode="last"`` remembers only the most recent outcome (ideal for
    deterministic, possibly changing worlds). ``mode="counts"`` keeps
    empirical outcome frequencies and mean rewards (needed under noise).
    """

    def __init__(self, mode: str = "last") -> None:
        if mode not in ("last", "counts"):
            raise ValueError("mode must be 'last' or 'counts'")
        self.mode = mode
        self._last: dict[tuple[int, int], tuple[float, int, bool]] = {}
        self._counts: dict[tuple[int, int], dict[Outcome, int]] = defaultdict(dict)
        self._rsum: dict[tuple[int, int, int, bool], float] = defaultdict(float)
        self.keys: list[tuple[int, int]] = []
        self._seen: set[tuple[int, int]] = set()
        self.predecessors: dict[int, set[tuple[int, int]]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._seen

    def update(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        key = (s, a)
        if key not in self._seen:
            self._seen.add(key)
            self.keys.append(key)
        self.predecessors[s2].add(key)
        if self.mode == "last":
            old = self._last.get(key)
            if old is not None and old[1] != s2:
                self.predecessors[old[1]].discard(key)
            self._last[key] = (r, s2, done)
        else:
            out = (s2, done)
            self._counts[key][out] = self._counts[key].get(out, 0) + 1
            self._rsum[(s, a, s2, done)] += r

    def sample(self, s: int, a: int, rng: np.random.Generator) -> tuple[float, int, bool]:
        """Draw ``(r, s', done)`` from the learned model."""
        if self.mode == "last":
            return self._last[(s, a)]
        outs = self._counts[(s, a)]
        items = list(outs.items())
        w = np.array([n for _, n in items], dtype=float)
        (s2, done), n = items[int(rng.choice(len(items), p=w / w.sum()))]
        return self._rsum[(s, a, s2, done)] / n, s2, done

    def distribution(self, s: int, a: int) -> list[tuple[float, float, int, bool]]:
        """Return ``[(prob, reward, s', done), ...]`` for expected backups."""
        if self.mode == "last":
            r, s2, d = self._last[(s, a)]
            return [(1.0, r, s2, d)]
        outs = self._counts[(s, a)]
        total = sum(outs.values())
        return [
            (n / total, self._rsum[(s, a, s2, d)] / n, s2, d) for (s2, d), n in outs.items()
        ]
