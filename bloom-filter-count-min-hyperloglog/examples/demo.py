"""Demo: membership, frequency and cardinality sketches on a Zipf stream."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stream_sketches import (BloomFilter, CountMinSketch, HeavyHitters, HyperLogLog,
                             ScalableBloomFilter, exact_counts, zipf_stream)
from stream_sketches.analytics import (cms_errors, empirical_fp_rate, hll_relative_error,
                                       top_k_recall)

N, UNIVERSE = 100_000, 20_000
stream = zipf_stream(N, UNIVERSE, 1.1, seed=7)
truth = exact_counts(stream)
print(f"stream: {N} events, {len(truth)} distinct keys")

print("\n== Bloom filter ==")
for fp in (0.1, 0.01, 0.001):
    bf = BloomFilter(len(truth), fp)
    bf.update(truth)
    emp = empirical_fp_rate(bf, [f"absent{i}" for i in range(30000)])
    print(f"target {fp:<6} empirical {emp:.4f} theory {bf.expected_fp_rate():.4f} "
          f"memory {bf.size_bytes/1024:.1f} KiB (exact set ~{len(truth)*60/1024:.0f} KiB)")
sb = ScalableBloomFilter(256, 0.01)
for k in truth:
    sb.add(k)
print(f"scalable: {len(sb.slices)} slices, {len(sb)} items, {sb.size_bytes/1024:.1f} KiB")

print("\n== Count-Min sketch ==")
for cons in (False, True):
    cms = CountMinSketch(0.002, 0.01, conservative=cons)
    t = time.perf_counter()
    for x in stream:
        cms.add(x)
    dt = time.perf_counter() - t
    s = cms_errors(cms, truth)
    print(f"conservative={cons!s:<5} mean err {s['mean_error']:.2f} max err {s['max_error']:.0f} "
          f"bound {s['bound']:.0f} within {s['frac_within_bound']:.4f} "
          f"{cms.size_bytes/1024:.0f} KiB {N/dt/1000:.0f}k upd/s")

print("\n== Heavy hitters (top 10) ==")
hh = HeavyHitters(30, epsilon=0.001)
for x in stream:
    hh.add(x)
top = hh.top(10)
for key, est in top:
    print(f"{key:<10} est {est:>6} true {truth[key]:>6}")
print(f"top-10 recall: {top_k_recall([k for k, _ in top], truth, 10):.2f}")

print("\n== HyperLogLog ==")
print(f"{'p':>3} {'registers':>9} {'bytes':>7} {'theory':>8} {'empirical':>10}")
for p in (6, 8, 10, 12, 14):
    h = HyperLogLog(p)
    print(f"{p:>3} {h.m:>9} {h.size_bytes:>7} {h.standard_error():>8.4f} "
          f"{hll_relative_error(p, 50_000, trials=4):>10.4f}")
h = HyperLogLog(14)
h.update(stream)
print(f"stream distinct: true {len(truth)} estimated {h.estimate():.0f}")
a, b = HyperLogLog(14), HyperLogLog(14)
a.update(range(0, 60000)); b.update(range(40000, 100000))
print(f"union true 100000 est {a.merge(b).estimate():.0f}; "
      f"intersection true 20000 est {a.intersection_estimate(b):.0f}")
