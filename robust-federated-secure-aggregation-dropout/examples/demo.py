"""Demo: secure aggregation with dropouts, then robust federated training."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from secure_aggregation import FLConfig, make_dataset, run_protocol, train  # noqa: E402
from secure_aggregation.quantize import decode, encode  # noqa: E402


def part1() -> None:
    print("== 1. One secure-aggregation round with dropouts ==")
    rng = np.random.default_rng(1)
    n, d = 20, 50
    xs = {u: rng.normal(scale=0.3, size=d) for u in range(1, n + 1)}
    enc = {u: encode(x, 2.0**16) for u, x in xs.items()}
    drops = {"share": {3}, "masked": {5, 6, 7}, "unmask": {8}}
    t0 = time.perf_counter()
    res = run_protocol(enc, threshold=10, dropouts=drops, seed=3)
    dt = time.perf_counter() - t0
    got = decode(res.total, 2.0**16)
    want = sum(xs[u] for u in res.survivors)
    print(f"clients={n} threshold=10 survivors={len(res.survivors)} dropped-after-share={res.dropped}")
    print(f"max |secure - plaintext sum| = {np.max(np.abs(got - want)):.2e}  ({dt:.2f}s)")
    print(f"upload per client ~ {res.stats['bytes_up'] // n} bytes for a {d}-dim update")


def part2() -> None:
    print("\n== 2. Federated logistic regression, 30 clients, 20% dropout ==")
    data = make_dataset(seed=0)
    for agg in ("plain", "secure"):
        out = train(data, FLConfig(aggregator=agg, dropout_rate=0.2, rounds=8, seed=1))
        print(f"{agg:>8}: acc={out['accuracy']:.3f} loss={out['loss']:.3f} skipped={out['skipped_rounds']}")


def part3() -> None:
    print("\n== 3. 20% sign-flipping Byzantine clients ==")
    data = make_dataset(seed=0)
    for name, kw in [("no attack, secure mean", dict(aggregator="secure")),
                     ("attack, plain mean", dict(aggregator="plain", byzantine_frac=0.2)),
                     ("attack, secure mean", dict(aggregator="secure", byzantine_frac=0.2)),
                     ("attack, bucketed median", dict(aggregator="bucketed", byzantine_frac=0.2))]:
        out = train(data, FLConfig(rounds=10, seed=2, clip=2.0, n_buckets=10, **kw))
        print(f"{name:>26}: acc={out['accuracy']:.3f} loss={out['loss']:.3f}")


def part4() -> None:
    print("\n== 4. Protocol cost vs number of clients (d=100) ==")
    rng = np.random.default_rng(0)
    for n in (10, 20, 40):
        enc = {u: encode(rng.normal(scale=0.3, size=100), 2.0**16) for u in range(1, n + 1)}
        t0 = time.perf_counter()
        res = run_protocol(enc, n // 2 + 1, {"masked": set(range(1, n // 10 + 1))}, seed=0)
        print(f"n={n:>3}: {time.perf_counter() - t0:.2f}s, up={res.stats['bytes_up']:>7} B, "
              f"down={res.stats['bytes_down']:>8} B, survivors={len(res.survivors)}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
