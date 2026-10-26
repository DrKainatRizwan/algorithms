"""Cache replacement policies (LRU, LFU, ARC, 2Q, CLOCK, ...) with Belady's OPT
and trace-driven benchmarking tools."""
from .arc import ARCCache, TwoQCache
from .base import CachePolicy, CacheStats
from .classic import ClockCache, FIFOCache, LFUCache, LRUCache, RandomCache
from .optimal import next_use_indices, simulate_opt
from .simulate import POLICIES, best_policy, cached, compare, format_table, hit_rate, make_policy
from .traces import (loop_trace, mixed_trace, scan_pollution_trace,
                     shifting_working_set_trace, working_set_size, zipf_trace)

__all__ = [
    "ARCCache", "TwoQCache", "CachePolicy", "CacheStats", "ClockCache", "FIFOCache",
    "LFUCache", "LRUCache", "RandomCache", "next_use_indices", "simulate_opt", "POLICIES",
    "best_policy", "cached", "compare", "format_table", "hit_rate", "make_policy",
    "loop_trace", "mixed_trace", "scan_pollution_trace", "shifting_working_set_trace",
    "working_set_size", "zipf_trace",
]
