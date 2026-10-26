"""Demo: poison a logistic regression and compare defences."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from poison_robust import (LogisticRegression, evaluate_grid, format_table, gradient_poisoning,
                           train_val_test_split)


def main() -> None:
    t0 = time.time()
    ds = train_val_test_split(n_train=300, n_val=300, n_test=1000, d=2, separation=2.5, seed=1)
    clean = LogisticRegression(1.0).fit(ds.X_train, ds.y_train)
    print(f"Clean model test accuracy: {clean.accuracy(ds.X_test, ds.y_test):.3f}")

    rng = np.random.default_rng(1)
    res = gradient_poisoning(ds.X_train, ds.y_train, ds.X_val, ds.y_val, 0.1, rng, steps=25)
    print(f"Gradient attack: validation loss {res.history[0]:.1f} -> {max(res.history):.1f} "
          f"with {res.n_poisoned} injected points")

    rows = evaluate_grid(ds, fracs=(0.1, 0.2, 0.3), seed=1)
    print("\nTest accuracy (rows: attack, poison rate; columns: defence)")
    print(format_table(rows))
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
