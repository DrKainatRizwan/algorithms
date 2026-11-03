"""Scheme comparison harness: balance, churn cost and lookup speed."""
from __future__ import annotations

import time
from typing import Callable

from .jump_maglev import Maglev
from .metrics import load_counts, moved_fraction, peak_to_mean, snapshot
from .rendezvous import RendezvousHash
from .ring import HashRing


def make_schemes(nodes: list[str], vnodes: int = 100) -> dict[str, Callable[[object], str]]:
    return {
        f"ring(v={vnodes})": HashRing(nodes, vnodes).get_node,
        "rendezvous": RendezvousHash(nodes).get_node,
        "maglev": Maglev(nodes, 10007).get_node,
    }


def compare(n_nodes: int = 20, n_keys: int = 20000) -> list[dict]:
    """Balance on n nodes, then remove one node and measure movement + lookup time."""
    nodes = [f"node-{i}" for i in range(n_nodes)]
    keys = [f"key-{i}" for i in range(n_keys)]
    rows = []
    for name, fn in make_schemes(nodes).items():
        t0 = time.perf_counter()
        before = snapshot(fn, keys)
        dt = (time.perf_counter() - t0) / n_keys * 1e6
        counts = load_counts(fn, keys, nodes)
        victim = nodes[-1]
        rest = nodes[:-1]
        after_fn = make_schemes(rest)[name]
        after = snapshot(after_fn, keys)
        rows.append({
            "scheme": name,
            "peak_to_mean": peak_to_mean(counts),
            "moved_on_removal": moved_fraction(before, after),
            "ideal_moved": counts[victim] / n_keys,
            "us_per_lookup": dt,
        })
    return rows
