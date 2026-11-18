"""Demo: SMO training, kernel comparison, multiclass, tuning and evasion attacks."""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from smo_svm import (SVC, OneVsRestSVC, make_blobs, make_moons, make_circles, make_xor,
                     train_test_split, deepfool, robust_accuracy, minimum_distance, grid_search)

print("== Kernels on non-linear datasets (test accuracy) ==")
for name, gen in [("moons", make_moons), ("circles", make_circles), ("xor", make_xor)]:
    X, y = gen(300, seed=0)
    Xa, Xb, ya, yb = train_test_split(X, y, seed=0)
    row = []
    for k, kw in [("linear", {}), ("poly", {"degree": 3}), ("rbf", {"gamma": 2.0})]:
        t = time.perf_counter()
        m = SVC(C=5.0, kernel=k, **kw).fit(Xa, ya)
        row.append(f"{k}={m.score(Xb, yb):.3f} ({len(m.support_)} SV, {time.perf_counter()-t:.2f}s)")
    print(f"{name:8s}", "  ".join(row))

print("\n== Optimality certificates (moons, RBF) ==")
X, y = make_moons(300, seed=1)
m = SVC(C=5.0, gamma=2.0, tol=1e-4).fit(X, y)
print(f"dual={m.dual_objective():.4f} primal={m.primal_objective():.4f} "
      f"KKT violation={m.kkt_violation():.4f} SMO steps={m.n_steps_}")

print("\n== Multiclass (3 blobs, one-vs-rest RBF) ==")
Xc, yc = make_blobs(300, np.array([[0, 0], [4, 0], [2, 3.5]], float), std=1.0, seed=2)
Xa, Xb, ya, yb = train_test_split(Xc, yc, seed=2)
print(f"test accuracy = {OneVsRestSVC(kernel='rbf', gamma=0.3).fit(Xa, ya).score(Xb, yb):.3f}")

print("\n== Grid search (3-fold CV, moons) ==")
X, y = make_moons(200, noise=0.2, seed=3)
best, score, _ = grid_search(X, y, {"C": [0.1, 1, 10], "gamma": [0.1, 1, 5]})
print(f"best={best} cv accuracy={score:.3f}")

print("\n== Evasion attacks (blobs, std 1.2) ==")
X, y = make_blobs(300, std=1.2, seed=4)
Xa, Xb, ya, yb = train_test_split(X, y, seed=4)
for k, kw in [("linear", {}), ("rbf", {"gamma": 0.2})]:
    m = SVC(C=1.0, kernel=k, **kw).fit(Xa, ya)
    d = minimum_distance(m, Xb[:40])
    accs = [robust_accuracy(m, Xb, yb, e) for e in (0.0, 0.5, 1.0, 2.0)]
    print(f"{k:6s} median min-L2 distance={np.median(d):.3f}  "
          f"robust acc @ eps 0/0.5/1/2 = " + " / ".join(f"{a:.3f}" for a in accs))
