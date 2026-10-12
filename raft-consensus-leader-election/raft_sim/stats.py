"""Experiment helpers: election latency and availability statistics."""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from .cluster import Cluster
from .node import RaftConfig


def election_latency(n: int, seed: int, config: Optional[RaftConfig] = None,
                     drop_rate: float = 0.0, max_ms: int = 10000) -> Optional[int]:
    """Milliseconds from cold start until a stable leader exists."""
    c = Cluster(n, seed, config, drop_rate)
    return c.now if c.wait_for_leader(max_ms) else None


def failover_latency(n: int, seed: int, config: Optional[RaftConfig] = None,
                     max_ms: int = 10000) -> Optional[int]:
    """Milliseconds from leader crash until a new stable leader exists."""
    c = Cluster(n, seed, config)
    old = c.wait_for_leader(max_ms)
    if old is None:
        return None
    c.run(200)
    c.crash(old.id)
    t0 = c.now
    if not c.run_until(lambda x: x.stable_leader() is not None, max_ms):
        return None
    return c.now - t0


def summarize(samples: List[Optional[int]]) -> Dict[str, float]:
    ok = sorted(s for s in samples if s is not None)
    if not ok:
        return {"n": 0, "failed": len(samples)}
    return {"n": len(ok), "failed": len(samples) - len(ok),
            "mean": statistics.fmean(ok), "median": float(ok[len(ok) // 2]),
            "p95": float(ok[min(len(ok) - 1, int(0.95 * len(ok)))]), "max": float(ok[-1])}
