"""Demo: compare LDP frequency oracles, post-processing, RAPPOR and heavy hitters."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldp_frequency import (PrefixTreeHeavyHitters, Rappor, audit, best_oracle, make_oracle,
                           longitudinal_attack_accuracy, mse, project_simplex, sample_users,
                           topk_recall, true_frequencies, variance_table, zipf_distribution)

t0 = time.time()
rng = np.random.default_rng(2024)

print("== 1. Estimator error vs domain size (n=50000, eps=1.0, Zipf s=1.1) ==")
n, eps = 50_000, 1.0
print(f"{'d':>5} " + " ".join(f"{m:>10}" for m in ("GRR", "OUE", "OLH", "HR")) + "   best-by-theory")
for d in (8, 32, 128, 512):
    probs = zipf_distribution(d, 1.1)
    values = sample_users(probs, n, rng)
    truth = true_frequencies(values, d)
    row = []
    for name in ("GRR", "OUE", "OLH", "HR"):
        row.append(mse(truth, make_oracle(name, d, eps).run(values, rng)))
    print(f"{d:>5} " + " ".join(f"{x:>10.2e}" for x in row) + f"   {best_oracle(d, eps).name}")

print("\n== 2. Theoretical variance (n=50000, eps=2, d=1024) and report sizes ==")
for name, v in variance_table(1024, 2.0, 50_000).items():
    o = make_oracle(name, 1024, 2.0)
    print(f"{name:>4}: std={np.sqrt(v):.5f}  report={o.report_bits:7.1f} bits")

print("\n== 3. Exact privacy audit of the randomisers (d=6, eps=1.5) ==")
for name in ("GRR", "SUE", "OUE", "OLH", "HR"):
    print(f"{name:>4}: audited epsilon = {audit(make_oracle(name, 6, 1.5)):.4f}")

print("\n== 4. Simplex post-processing (OLH, d=256, n=30000) ==")
d, n = 256, 30_000
values = sample_users(zipf_distribution(d, 1.3), n, rng)
truth = true_frequencies(values, d)
est = make_oracle("OLH", d, 1.0).run(values, rng)
proj = project_simplex(est)
print(f"MSE raw={mse(truth, est):.3e}  projected={mse(truth, proj):.3e}")
print(f"top-10 recall raw={topk_recall(truth, est, 10):.2f}  projected={topk_recall(truth, proj, 10):.2f}")

print("\n== 5. RAPPOR longitudinal privacy (6 values, 300 rounds) ==")
r = Rappor(6, f=0.6, p=0.7, q=0.3)
vals = rng.integers(0, 6, size=2000)
print(f"eps_inf={r.epsilon_permanent:.3f}  eps_1={r.epsilon:.3f}")
for memo in (False, True):
    acc = longitudinal_attack_accuracy(r, vals, 300, memo, rng)
    print(f"memoised={memo!s:5}: averaging attacker recovers the value {acc:.1%} of the time")

print("\n== 6. Heavy hitters over a 2^20 domain (n=200000, eps=4) ==")
bits, n = 20, 200_000
secret = [0xABCDE, 0x13579, 0x2468A, 0xFEDCB]
u = rng.random(n)
values = rng.integers(0, 1 << bits, size=n)
for i, (item, lo, hi) in enumerate(zip(secret, (0.0, 0.2, 0.35, 0.45), (0.2, 0.35, 0.45, 0.52))):
    values[(u >= lo) & (u < hi)] = item
res = PrefixTreeHeavyHitters(bits, 4.0, k=6, step=4).discover(values, rng)
print("planted:", {hex(s): f for s, f in zip(secret, (0.20, 0.15, 0.10, 0.07))})
for item, f in res.items[:5]:
    print(f"  found {hex(item):>8}  est freq {f:.3f}")
print("levels:", [(l.prefix_bits, l.num_candidates, l.oracle) for l in res.levels])
print(f"\nDone in {time.time() - t0:.1f}s")
