"""Compare replacement policies on several synthetic workloads."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cache_policies import (ARCCache, compare, format_table, loop_trace, mixed_trace,
                            scan_pollution_trace, shifting_working_set_trace, zipf_trace)

N = 30000
CAPS = [16, 32, 64, 128]
WORKLOADS = {
    "zipf (alpha=1.0, 2000 keys)": zipf_trace(N, 2000, 1.0, seed=1),
    "loop (len 100)": loop_trace(N, 100),
    "scan pollution (hot=50)": scan_pollution_trace(N, seed=1),
    "shifting working set": shifting_working_set_trace(N, seed=1),
    "mixed zipf+loop+scan": mixed_trace(N, seed=1),
}

t0 = time.time()
for name, trace in WORKLOADS.items():
    print(f"\n== {name}: {len(trace)} requests, capacities {CAPS}")
    print(format_table(CAPS, compare(trace, CAPS)))

arc = ARCCache(64)
arc.run(WORKLOADS["mixed zipf+loop+scan"])
print(f"\nARC on mixed workload: final p={arc.p:.1f}, |T1|={len(arc.t1)}, |T2|={len(arc.t2)}, ghosts={arc.ghost_size()}")
print(f"elapsed {time.time() - t0:.1f}s")
