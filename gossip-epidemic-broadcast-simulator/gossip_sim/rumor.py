"""Rumor mongering with loss of interest (Demers et al., 1987)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import Graph

S, I, R = 0, 1, 2  # susceptible, infective ("hot"), removed


@dataclass
class RumorResult:
    """Final state of an SIR-style rumor run."""

    n: int
    residue: float          # fraction that never heard the rumor
    rounds: int             # rounds until no infective node remained
    messages: int
    traffic_per_node: float


def rumor_mongering(graph: Graph, rng: np.random.Generator, k: int = 2,
                    variant: str = "feedback-coin", source: int = 0,
                    max_rounds: int = 1000) -> RumorResult:
    """Push-based rumor mongering.

    Every infective node pushes to a random neighbour each round. If the target
    already knows the rumor the sender loses interest:

    * ``feedback-coin``: with probability ``1/k``;
    * ``feedback-counter``: after ``k`` unnecessary contacts;
    * ``blind-coin``: with probability ``1/k`` after *every* contact.
    """
    if variant not in ("feedback-coin", "feedback-counter", "blind-coin"):
        raise ValueError("unknown variant")
    if k < 1:
        raise ValueError("k must be >= 1")
    state = np.full(graph.n, S, dtype=np.int8)
    state[source] = I
    strikes = np.zeros(graph.n, dtype=np.int32)
    messages = 0
    rounds = 0
    while rounds < max_rounds:
        hot = np.flatnonzero(state == I)
        if hot.size == 0:
            break
        rounds += 1
        newly: list[int] = []
        for u in hot:
            v = graph.random_neighbor(int(u), rng)
            if v < 0:
                state[u] = R
                continue
            messages += 1
            known = state[v] != S or v in newly
            if not known:
                newly.append(v)
                if variant == "blind-coin" and rng.random() < 1.0 / k:
                    state[u] = R
            else:
                if variant == "feedback-counter":
                    strikes[u] += 1
                    if strikes[u] >= k:
                        state[u] = R
                elif rng.random() < 1.0 / k:
                    state[u] = R
        for v in newly:
            state[v] = I
    residue = float(np.mean(state == S))
    return RumorResult(graph.n, residue, rounds, messages, messages / graph.n)
