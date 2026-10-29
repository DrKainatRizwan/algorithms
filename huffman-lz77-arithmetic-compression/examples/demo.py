"""Benchmark all codecs on synthetic datasets."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compression_suite import suite
from compression_suite.entropy import (conditional_entropy, entropy, gen_random,
                                       gen_repetitive, gen_skewed, gen_text)

data = {
    "text": gen_text(20000, 1),
    "repetitive": gen_repetitive(20000, 2),
    "skewed": gen_skewed(20000, 3),
    "random": gen_random(20000, 4),
}
for k, v in data.items():
    print(f"{k:11s} n={len(v)}  H0={entropy(v):.3f}  H1={conditional_entropy(v):.3f} bits/byte")
print()
print(f"{'dataset':11s}{'codec':9s}{'size':>7s}{'bits/B':>8s}{'enc s':>8s}{'dec s':>8s}  ok")
for r in suite.benchmark(data):
    print(f"{r['dataset']:11s}{r['codec']:9s}{r['size']:7d}{r['bpc']:8.3f}"
          f"{r['enc_s']:8.2f}{r['dec_s']:8.2f}  {r['ok']}")
print()
for k, v in data.items():
    b = suite.pack(v)
    assert suite.unpack(b) == v
    print(f"auto pack {k:11s}: {len(v)} -> {len(b)} (method id {b[0]})")
