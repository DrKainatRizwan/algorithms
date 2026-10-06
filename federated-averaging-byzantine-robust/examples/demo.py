"""Compare aggregation rules under several Byzantine attacks."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from byzantine_fedavg import (ATTACKS, AGGREGATORS, FederatedSimulator,
                              SoftmaxModel, make_blobs_dataset,
                              partition_label_skew)

N_CLIENTS, N_BYZ, ROUNDS = 20, 4, 15


def main() -> None:
    X, y = make_blobs_dataset(4000, 10, 4, spread=1.5, seed=7)
    Xtr, ytr, Xte, yte = X[:3200], y[:3200], X[3200:], y[3200:]
    shards = partition_label_skew(ytr, N_CLIENTS, alpha=5.0, seed=7)
    print(f"{N_CLIENTS} clients, {N_BYZ} Byzantine, {ROUNDS} rounds")
    attacks = ["sign_flip", "scaled_negative", "gaussian", "lie"]
    print(f"{'aggregator':<14}" + "".join(f"{a:>17}" for a in ["none"] + attacks))
    for name, agg in AGGREGATORS.items():
        row = []
        for atk in [None] + attacks:
            sim = FederatedSimulator(
                SoftmaxModel(10, 4), Xtr, ytr, shards, Xte, yte, agg,
                n_byzantine=0 if atk is None else N_BYZ,
                attack=ATTACKS.get(atk), seed=7)
            row.append(sim.run(ROUNDS).final_accuracy)
        print(f"{name:<14}" + "".join(f"{a:>17.3f}" for a in row))


if __name__ == "__main__":
    main()
