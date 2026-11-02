"""Synchronous push / pull / push-pull rumor spreading (SI epidemic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .network import Graph

MODES = ("push", "pull", "push-pull")


@dataclass
class SpreadResult:
    """Outcome of one dissemination run."""

    n_alive: int
    informed_history: list[int] = field(default_factory=list)  # informed count after each round (index 0 = start)
    messages: int = 0
    rounds_to_full: int | None = None  # None if never reached all alive nodes
    rounds_to_90: int | None = None

    @property
    def coverage(self) -> float:
        return self.informed_history[-1] / self.n_alive

    @property
    def rounds(self) -> int:
        return len(self.informed_history) - 1


def spread(graph: Graph, mode: str, rng: np.random.Generator, source: int = 0,
           fanout: int = 1, loss: float = 0.0, crashed: Iterable[int] = (),
           max_rounds: int = 200) -> SpreadResult:
    """Run one rumor-spreading simulation.

    Each round every (alive) node contacts ``fanout`` random neighbours.
    ``push``: informed nodes send the rumor; ``pull``: uninformed nodes request
    it; ``push-pull``: both. Each contact is lost independently with
    probability ``loss``. Crashed nodes neither send nor receive; they stay in
    the neighbour lists so contacts to them are wasted, as in a real cluster
    that has not yet detected the failure. Updates apply at round end.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not 0.0 <= loss < 1.0:
        raise ValueError("loss must be in [0, 1)")
    dead = set(crashed)
    if source in dead:
        raise ValueError("source node is crashed")
    informed = np.zeros(graph.n, dtype=bool)
    informed[source] = True
    alive_n = graph.n - len(dead)
    res = SpreadResult(n_alive=alive_n, informed_history=[1])
    if alive_n == 1:
        res.rounds_to_full = 0
        res.rounds_to_90 = 0
        return res
    alive_nodes = [v for v in range(graph.n) if v not in dead]
    for r in range(1, max_rounds + 1):
        new = informed.copy()
        for u in alive_nodes:
            is_inf = bool(informed[u])
            if is_inf and mode == "pull":
                continue
            if not is_inf and mode == "push":
                continue
            for _ in range(fanout):
                v = graph.random_neighbor(u, rng)
                if v < 0:
                    continue
                res.messages += 1
                if v in dead or (loss > 0 and rng.random() < loss):
                    continue
                if is_inf:           # push
                    new[v] = True
                elif informed[v]:    # pull reply
                    new[u] = True
        informed = new
        cnt = int(informed.sum())
        res.informed_history.append(cnt)
        if res.rounds_to_90 is None and cnt >= 0.9 * alive_n:
            res.rounds_to_90 = r
        if cnt == alive_n:
            res.rounds_to_full = r
            break
    return res
