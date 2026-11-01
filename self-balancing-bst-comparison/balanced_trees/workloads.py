"""Workload generators and a benchmarking harness for the tree implementations."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

from .base import SortedMap
from .redblack import RedBlackTree
from .splay import SplayTree
from .treap import Treap

TREES: dict[str, Callable[[], SortedMap]] = {
    "treap": lambda: Treap(seed=7),
    "splay": SplayTree,
    "red-black": RedBlackTree,
}


def sequential(n: int) -> list[int]:
    return list(range(n))


def uniform(n: int, seed: int = 0) -> list[int]:
    keys = list(range(n))
    random.Random(seed).shuffle(keys)
    return keys


def zipf_accesses(universe: int, count: int, s: float = 1.2, seed: int = 0) -> list[int]:
    """Access sequence where key ``i`` has probability proportional to 1/(i+1)^s."""
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** s for i in range(universe)]
    return rng.choices(range(universe), weights=weights, k=count)


def working_set_accesses(universe: int, count: int, window: int = 8, seed: int = 0) -> list[int]:
    """Locality-heavy sequence: slowly drifting window of ``window`` hot keys."""
    rng = random.Random(seed)
    base = 0
    out = []
    for i in range(count):
        if i % 50 == 0:
            base = rng.randrange(max(1, universe - window))
        out.append(base + rng.randrange(window))
    return out


@dataclass
class BenchResult:
    name: str
    workload: str
    height: int
    avg_depth: float
    insert_cmp: float      # comparisons per insert
    insert_rot: float      # rotations per insert
    access_cmp: float      # comparisons per access
    seconds: float


def run_benchmark(tree_name: str, insert_keys: list[int], accesses: list[int], workload: str) -> BenchResult:
    """Insert ``insert_keys`` then perform ``accesses`` lookups, collecting counters."""
    tree = TREES[tree_name]()
    t0 = time.perf_counter()
    for k in insert_keys:
        tree.insert(k, k)
    ins_cmp, ins_rot = tree.comparisons, tree.rotations
    height, avg_depth = tree.height(), tree.average_depth()
    tree.reset_counters()
    for k in accesses:
        tree.get(k)
    acc_cmp = tree.comparisons
    secs = time.perf_counter() - t0
    n = max(1, len(insert_keys))
    return BenchResult(tree_name, workload, height, avg_depth, ins_cmp / n,
                       ins_rot / n, acc_cmp / max(1, len(accesses)), secs)


def compare_all(n: int = 2000, seed: int = 0) -> list[BenchResult]:
    """Run the standard four-workload comparison across all trees."""
    scenarios = {
        "sequential": (sequential(n), uniform(n, seed)),
        "random": (uniform(n, seed), uniform(n, seed + 1)),
        "zipf": (uniform(n, seed), zipf_accesses(n, 4 * n, seed=seed)),
        "working-set": (uniform(n, seed), working_set_accesses(n, 4 * n, seed=seed)),
    }
    results = []
    for wl, (ins, acc) in scenarios.items():
        for name in TREES:
            results.append(run_benchmark(name, ins, acc, wl))
    return results


def format_table(results: list[BenchResult]) -> str:
    head = f"{'workload':<12}{'tree':<10}{'height':>7}{'avg depth':>11}{'cmp/ins':>9}{'rot/ins':>9}{'cmp/get':>9}"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(f"{r.workload:<12}{r.name:<10}{r.height:>7}{r.avg_depth:>11.2f}"
                     f"{r.insert_cmp:>9.2f}{r.insert_rot:>9.2f}{r.access_cmp:>9.2f}")
    return "\n".join(lines)
