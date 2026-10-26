"""Trace-driven comparison harness and a memoising decorator."""
from __future__ import annotations

import functools
from typing import Callable, Hashable, Mapping, Sequence

from .arc import ARCCache, TwoQCache
from .base import CachePolicy, CacheStats
from .classic import ClockCache, FIFOCache, LFUCache, LRUCache, RandomCache
from .optimal import simulate_opt

POLICIES: dict[str, Callable[[int], CachePolicy]] = {
    "FIFO": FIFOCache,
    "Random": RandomCache,
    "CLOCK": ClockCache,
    "LRU": LRUCache,
    "LFU": LFUCache,
    "2Q": TwoQCache,
    "ARC": ARCCache,
}


def make_policy(name: str, capacity: int) -> CachePolicy:
    """Instantiate a policy by name (see :data:`POLICIES`)."""
    try:
        return POLICIES[name](capacity)
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None


def hit_rate(name: str, trace: Sequence[Hashable], capacity: int) -> float:
    """Hit rate of one policy on one trace ('OPT' gives Belady's bound)."""
    if name == "OPT":
        return simulate_opt(trace, capacity).hit_rate
    return make_policy(name, capacity).run(trace).hit_rate


def compare(
    trace: Sequence[Hashable],
    capacities: Sequence[int],
    policies: Sequence[str] | None = None,
    include_opt: bool = True,
) -> dict[str, list[float]]:
    """Hit-rate curves: ``{policy: [hit rate at each capacity]}``."""
    names = list(policies) if policies is not None else list(POLICIES)
    if include_opt:
        names.append("OPT")
    return {n: [hit_rate(n, trace, c) for c in capacities] for n in names}


def best_policy(curves: Mapping[str, Sequence[float]], index: int) -> str:
    """Name of the best non-OPT policy at capacity position ``index``."""
    return max((n for n in curves if n != "OPT"), key=lambda n: curves[n][index])


def format_table(capacities: Sequence[int], curves: Mapping[str, Sequence[float]]) -> str:
    """Render curves as an aligned text table of percentages."""
    head = "policy  " + " ".join(f"{c:>7d}" for c in capacities)
    rows = [head, "-" * len(head)]
    for n, vals in curves.items():
        rows.append(f"{n:<7} " + " ".join(f"{100 * v:>6.1f}%" for v in vals))
    return "\n".join(rows)


def cached(policy: str = "ARC", capacity: int = 128) -> Callable:
    """Memoising decorator whose eviction is governed by a named policy."""

    def deco(fn: Callable) -> Callable:
        cache = make_policy(policy, capacity)
        store: dict[Hashable, object] = {}

        @functools.wraps(fn)
        def wrapper(*args: Hashable):
            hit = cache.access(args)
            if hit:
                return store[args]
            for old in cache.last_evicted:
                store.pop(old, None)
            store[args] = fn(*args)
            return store[args]

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.stats = lambda: cache.stats  # type: ignore[attr-defined]
        return wrapper

    return deco


__all__ = ["POLICIES", "make_policy", "hit_rate", "compare", "best_policy", "format_table", "cached", "CacheStats"]
