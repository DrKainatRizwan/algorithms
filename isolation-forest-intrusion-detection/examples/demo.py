"""Compare Isolation Forest and Extended Isolation Forest on synthetic network flows."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ids_forest import (IntrusionDetector, make_traffic, roc_auc, average_precision,
                        precision_recall_f1, best_f1_threshold, TRAFFIC_FEATURES)
from ids_forest.data import ATTACKS

Xtr, _, _ = make_traffic(3000, 0, seed=1, attacks=())
Xte, yte, kind = make_traffic(2000, 150, seed=2)
print(f"train flows={len(Xtr)}  test flows={len(Xte)}  attacks={int(yte.sum())}")

dets = {}
for name, ext in (("IForest", False), ("ExtIForest", True)):
    t0 = time.perf_counter()
    det = IntrusionDetector(extended=ext, n_trees=100, contamination=0.05, seed=0,
                            feature_names=TRAFFIC_FEATURES).fit(Xtr)
    fit_t = time.perf_counter() - t0
    s = det.score(Xte)
    p, r, f = precision_recall_f1(yte, det.predict(Xte))
    print(f"{name:11s} AUC={roc_auc(yte, s):.3f} AP={average_precision(yte, s):.3f} "
          f"P={p:.3f} R={r:.3f} F1={f:.3f} bestF1={best_f1_threshold(yte, s)[1]:.3f} fit+score={fit_t:.2f}s")
    dets[name] = det

det = dets["IForest"]
s = det.score(Xte)
print("\nPer-attack AUC vs benign (IForest):")
for k, a in enumerate(ATTACKS):
    m = (kind == -1) | (kind == k)
    print(f"  {a:13s} {roc_auc(yte[m], s[m]):.3f}")

print("\nExplanations for one flow per attack family:")
for k, a in enumerate(ATTACKS):
    i = int(np.flatnonzero(kind == k)[0])
    print(f"  {a:13s}", [(n, round(z, 1)) for n, z in det.explain(Xte[i])])
