"""Monte-Carlo trial runner with summary statistics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .network import Graph
from .protocols import SpreadResult, spread


@dataclass
class TrialSummary:
    trials: int
    mean_rounds: float
    std_rounds: float
    mean_rounds_90: float
    mean_messages: float
    completion_rate: float


def run_trials(graph: Graph, mode: str, trials: int, seed: int = 0,
               **kwargs) -> TrialSummary:
    """Run ``trials`` independent spreads (seeded) and summarise them.

    Rounds statistics are over completed runs only; ``completion_rate`` is the
    fraction of runs that informed every alive node.
    """
    rng = np.random.default_rng(seed)
    results: list[SpreadResult] = [spread(graph, mode, rng, **kwargs) for _ in range(trials)]
    done = [r for r in results if r.rounds_to_full is not None]
    rounds = np.array([r.rounds_to_full for r in done], dtype=float)
    r90 = np.array([r.rounds_to_90 for r in results if r.rounds_to_90 is not None], dtype=float)
    return TrialSummary(
        trials=trials,
        mean_rounds=float(rounds.mean()) if rounds.size else float("nan"),
        std_rounds=float(rounds.std()) if rounds.size else float("nan"),
        mean_rounds_90=float(r90.mean()) if r90.size else float("nan"),
        mean_messages=float(np.mean([r.messages for r in results])),
        completion_rate=len(done) / trials,
    )
