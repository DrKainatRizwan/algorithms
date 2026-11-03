"""Certified accuracy of smoothed classifiers under several noise levels."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from smoothing_cert import (LinearSoftmax, MLPClassifier, SmoothedClassifier, accuracy,
                            exact_linear_radius, make_blobs)
from smoothing_cert.evaluate import (certified_accuracy_curve, certify_dataset,
                                     empirical_robust_accuracy)

t0 = time.time()
Xtr, ytr = make_blobs(1500, dim=20, classes=4, sep=4.0, spread=1.0, seed=0)
Xte, yte = make_blobs(60, dim=20, classes=4, sep=4.0, spread=1.0, seed=99)
radii = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
print("radius        " + "  ".join(f"{r:5.2f}" for r in radii))
for sigma in (0.25, 0.5, 1.0):
    mlp = MLPClassifier(20, 64, 4, seed=0)
    mlp.fit(Xtr, ytr, epochs=40, lr=0.01, noise_sigma=sigma, seed=0)
    sm = SmoothedClassifier(mlp, 4, sigma, seed=0)
    res = certify_dataset(sm, Xte, n0=50, n=500)
    curve = certified_accuracy_curve(res, yte, radii)
    print(f"MLP sigma={sigma:<4} " + "  ".join(f"{a:5.2f}" for a in curve),
          f"| clean base acc {accuracy(mlp, Xte, yte):.2f}")

lin = LinearSoftmax(20, 4); lin.fit(Xtr, ytr, epochs=30, noise_sigma=0.5)
sm = SmoothedClassifier(lin, 4, 0.5, seed=1)
print("\nLinear model: certified radius vs. exact boundary distance")
for x in Xte[:5]:
    r = sm.certify(x, n0=100, n=5000)
    print(f"  certified {r.radius:5.2f}   exact margin {exact_linear_radius(lin, x):5.2f}   pA_lower {r.p_lower:.4f}")

mlp = MLPClassifier(20, 64, 4, seed=0); mlp.fit(Xtr, ytr, epochs=40, lr=0.01, noise_sigma=0.5)
sm = SmoothedClassifier(mlp, 4, 0.5, seed=2)
print("\nEOT-PGD empirical accuracy of smoothed MLP (sigma=0.5):")
for eps in (0.0, 0.5, 1.0, 2.0):
    print(f"  eps={eps:3.1f}: {empirical_robust_accuracy(sm, mlp, Xte[:30], yte[:30], eps):.2f}")
print(f"\nelapsed {time.time()-t0:.1f}s")
