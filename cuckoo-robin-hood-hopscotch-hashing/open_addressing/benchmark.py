"""Workloads and measurement helpers for comparing the tables."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .base import OpenAddressingTable


def random_keys(n: int, seed: int = 0) -> list[int]:
    """``n`` distinct pseudo-random 48-bit integers."""
    rng = random.Random(seed)
    seen: set[int] = set()
    while len(seen) < n:
        seen.add(rng.getrandbits(48))
    out = sorted(seen)
    rng.shuffle(out)
    return out


def sequential_keys(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def strided_keys(n: int, stride: int = 1024) -> list[int]:
    """Keys that are multiples of a power of two -- the identity hash's nightmare."""
    return [i * stride for i in range(n)]


def make_workload(kind: str, n: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Return ``(present, absent)`` key lists, each of length ``n``."""
    if kind == "random":
        keys = random_keys(2 * n, seed)
    elif kind == "sequential":
        keys = sequential_keys(2 * n)
    elif kind == "strided":
        keys = strided_keys(2 * n)
    else:
        raise ValueError(f"unknown workload {kind!r}")
    return keys[:n], keys[n:]


@dataclass(frozen=True)
class ProbeProfile:
    """Probe-length summary of a populated table."""

    mean_hit: float
    mean_miss: float
    max_hit: int
    p99_hit: int
    max_miss: int


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def probe_profile(
    table: OpenAddressingTable, present: list[int], absent: list[int]
) -> ProbeProfile:
    """Exact probe counts of successful and unsuccessful lookups."""
    hits = sorted(table.probe_count(k) for k in present)
    misses = [table.probe_count(k) for k in absent]
    return ProbeProfile(
        mean_hit=sum(hits) / max(1, len(hits)),
        mean_miss=sum(misses) / max(1, len(misses)),
        max_hit=hits[-1] if hits else 0,
        p99_hit=_percentile(hits, 0.99),
        max_miss=max(misses, default=0),
    )


def fill_to_load(
    factory: Callable[[int], OpenAddressingTable],
    capacity: int,
    load: float,
    keys: list[int],
) -> tuple[OpenAddressingTable, bool]:
    """Insert ``load * capacity`` keys; report whether the table had to grow."""
    table = factory(capacity)
    start = table.capacity
    for k in keys[: int(load * start)]:
        table[k] = k
    return table, table.capacity == start


def load_sweep(
    factories: dict[str, Callable[[int], OpenAddressingTable]],
    loads: list[float],
    capacity: int = 2048,
    workload: str = "random",
    seed: int = 0,
) -> dict[str, dict[float, ProbeProfile | None]]:
    """Probe profiles per scheme and target load (``None`` if it had to grow)."""
    present, absent = make_workload(workload, capacity, seed)
    results: dict[str, dict[float, ProbeProfile | None]] = {}
    for name, factory in factories.items():
        row: dict[float, ProbeProfile | None] = {}
        for load in loads:
            table, ok = fill_to_load(factory, capacity, load, present)
            used = present[: len(table)]
            row[load] = probe_profile(table, used, absent) if ok else None
        results[name] = row
    return results


def churn(
    table: OpenAddressingTable, n: int, rounds: int, seed: int = 0
) -> tuple[float, float]:
    """Steady-state delete+insert churn at ``n`` live keys.

    Returns ``(mean hit probes, mean miss probes)`` after ``rounds`` rounds.
    """
    rng = random.Random(seed)
    pool = random_keys(n + rounds + n, seed)
    live = pool[:n]
    for k in live:
        table[k] = k
    fresh = pool[n : n + rounds]
    for k in fresh:
        victim = rng.randrange(len(live))
        del table[live[victim]]
        live[victim] = k
        table[k] = k
    absent = pool[n + rounds :]
    prof = probe_profile(table, live, absent)
    return prof.mean_hit, prof.mean_miss
