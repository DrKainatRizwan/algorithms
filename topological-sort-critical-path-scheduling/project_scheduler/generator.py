"""Random layered project network generator."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .cpm import Activity


def random_project(n: int, seed: int = 0, layers: int = 5, edge_prob: float = 0.3,
                   resources: Sequence[str] = (), max_demand: int = 3, max_duration: int = 9) -> List[Activity]:
    """Generate a DAG of ``n`` activities spread over ``layers`` layers (always acyclic)."""
    rng = np.random.default_rng(seed)
    layer_of = np.sort(rng.integers(0, layers, size=n))
    names = [f"A{i:03d}" for i in range(n)]
    acts: List[Activity] = []
    for i in range(n):
        preds = tuple(names[j] for j in range(i) if layer_of[j] < layer_of[i] and rng.random() < edge_prob)
        if not preds and layer_of[i] > 0:
            earlier = [j for j in range(i) if layer_of[j] < layer_of[i]]
            if earlier:
                preds = (names[int(rng.choice(earlier))],)
        demands = {r: int(rng.integers(0, max_demand + 1)) for r in resources}
        demands = {r: d for r, d in demands.items() if d > 0}
        acts.append(Activity(names[i], int(rng.integers(1, max_duration + 1)), preds, demands))
    return acts
