"""Demo: Hungarian vs auction timing, certificates, bottleneck and gated matching."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assignment_solvers import (  # noqa: E402
    auction_assignment,
    bottleneck_assignment,
    forced_pair_cost,
    gated_assignment,
    hungarian,
    random_cost_matrix,
    verify_duals,
)


def timed(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    return r, (time.perf_counter() - t) * 1000


print("== Hungarian vs auction (integer costs in [0, 1000)) ==")
print(f"{'n':>5} {'hung ms':>9} {'auct ms':>9} {'bids':>7} {'phases':>7} {'equal':>6}")
for n in (25, 50, 100, 200):
    c = random_cost_matrix(n, seed=n, high=1000)
    h, th = timed(hungarian, c)
    (a, st), ta = timed(auction_assignment, c, return_stats=True)
    print(f"{n:>5} {th:>9.1f} {ta:>9.1f} {st.bids:>7} {st.phases:>7} {h.cost == a.cost!s:>6}")

print("\n== Dual certificate ==")
c = random_cost_matrix(60, 80, seed=1)
r = hungarian(c)
print("optimal cost:", r.cost, "| certificate valid:", verify_duals(c, r))

print("\n== Bid effect of epsilon scaling (n=60) ==")
c = random_cost_matrix(60, seed=2, high=10000)
for scale in (2.0, 5.0, 20.0, 1e12):
    _, st = auction_assignment(c, scale=scale, return_stats=True)
    label = "no scaling" if scale > 1e6 else f"scale {scale:g}"
    print(f"{label:>12}: {st.bids} bids, {st.phases} phases")

print("\n== Bottleneck vs min-sum (n=40) ==")
c = random_cost_matrix(40, seed=3)
s, b = hungarian(c), bottleneck_assignment(c)
print(f"min-sum : total {s.cost:.0f}, max edge {c[s.rows, s.cols].max():.0f}")
print(f"bottleneck: total {b.cost:.0f}, max edge {c[b.rows, b.cols].max():.0f}")

print("\n== Gated tracking (8 tracks, 6 detections) ==")
rng = np.random.default_rng(4)
tracks = rng.uniform(0, 20, size=(8, 2))
dets = tracks[:6] + rng.normal(0, 0.4, size=(6, 2))
dets[5] += 30  # spurious detection far from any track
d = np.linalg.norm(tracks[:, None] - dets[None], axis=2)
matches, ut, ud = gated_assignment(d, gate=2.0)
print("matches:", matches)
print("unmatched tracks:", ut, "unmatched detections:", ud)

print("\n== Regret of forcing a pair ==")
c = random_cost_matrix(8, seed=5)
opt = hungarian(c)
print("optimal:", opt.cost, "| forcing (0,0):", forced_pair_cost(c, 0, 0))
