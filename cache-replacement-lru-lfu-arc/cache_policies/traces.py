"""Synthetic request-trace generators (all seeded, return lists of ints)."""
from __future__ import annotations

import numpy as np


def zipf_trace(n: int, universe: int, alpha: float = 1.0, seed: int = 0) -> list[int]:
    """Independent references with P(rank r) proportional to r**-alpha."""
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, universe + 1) ** alpha
    return rng.choice(universe, size=n, p=w / w.sum()).tolist()


def loop_trace(n: int, loop_len: int) -> list[int]:
    """Cyclic scan 0..loop_len-1, the classic LRU worst case."""
    return [i % loop_len for i in range(n)]


def scan_pollution_trace(
    n: int, hot: int = 50, scan_len: int = 400, hot_burst: int = 400, seed: int = 0
) -> list[int]:
    """Uniform hot-set bursts interleaved with one-shot sequential scans."""
    rng = np.random.default_rng(seed)
    out: list[int] = []
    next_cold = hot
    while len(out) < n:
        out.extend(rng.integers(0, hot, size=hot_burst).tolist())
        out.extend(range(next_cold, next_cold + scan_len))
        next_cold += scan_len
    return out[:n]


def shifting_working_set_trace(
    n: int, set_size: int = 60, phases: int = 5, alpha: float = 0.8, seed: int = 0
) -> list[int]:
    """Zipf-skewed working sets that move to a disjoint key range each phase."""
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, set_size + 1) ** alpha
    w /= w.sum()
    per = n // phases
    out: list[int] = []
    for ph in range(phases):
        count = per if ph < phases - 1 else n - per * (phases - 1)
        out.extend((rng.choice(set_size, size=count, p=w) + ph * set_size).tolist())
    return out


def mixed_trace(n: int, seed: int = 0) -> list[int]:
    """Interleave a Zipf stream, a loop and a scan in disjoint key ranges."""
    rng = np.random.default_rng(seed)
    z = zipf_trace(n, 500, 0.9, seed)
    loop = [10_000 + i % 80 for i in range(n)]
    scan = list(range(20_000, 20_000 + n))
    zi = li = si = 0
    out: list[int] = []
    for src in rng.choice(3, size=n, p=[0.6, 0.2, 0.2]):
        if src == 0:
            out.append(z[zi]); zi += 1
        elif src == 1:
            out.append(loop[li]); li += 1
        else:
            out.append(scan[si]); si += 1
    return out


def working_set_size(trace: list[int], window: int) -> float:
    """Mean number of distinct keys over consecutive windows of ``window`` requests."""
    if window < 1:
        raise ValueError("window must be positive")
    sizes = [len(set(trace[i : i + window])) for i in range(0, len(trace) - window + 1, window)]
    return float(np.mean(sizes)) if sizes else 0.0
