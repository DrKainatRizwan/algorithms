# Anomaly-Based Intrusion Detection with Isolation Forest

## Overview
A from-scratch NumPy implementation of the Isolation Forest and its oblique-split variant, the Extended Isolation Forest, wrapped in a small intrusion-detection pipeline. A synthetic flow generator produces benign traffic plus four attack families (port scan, SYN flood, exfiltration, brute force). The detector is trained on benign traffic only and scores unseen flows.

Package `ids_forest`:
- `tree.py` – axis-parallel isolation tree and the normalising constant c(n)
- `forest.py` – `IsolationForest` (sub-sampling, scoring, contamination threshold, feature importance)
- `extended.py` – `ExtendedIsolationForest` with random hyperplane cuts and a tunable extension level
- `data.py` – labelled synthetic network-flow generator
- `metrics.py` – ROC-AUC, average precision, precision/recall/F1, best-F1 threshold
- `detector.py` – `IntrusionDetector`: log1p + robust (median/MAD) scaling, scoring, per-flow explanation

## How it works
Anomalies are few and different, so random partitioning isolates them in fewer splits. Each tree is built on a sub-sample of size ψ by repeatedly choosing a random feature and a uniform split value between the node's min and max, up to depth ⌈log₂ ψ⌉. The path length h(x) of a point is its depth at the external node plus c(size) to account for the unbuilt subtree, where c(n) = 2H(n−1) − 2(n−1)/n. The score is

    s(x) = 2^(−E[h(x)] / c(ψ))

so s near 1 is anomalous and s well below 0.5 is normal. The Extended variant splits with `(x − p)·n < 0` for a random normal n and intercept p, which removes the rectangular score artefacts of axis-parallel cuts. The detector sets its alert threshold at the (1 − contamination) quantile of training scores.

## Complexity
- Training: O(t · ψ log ψ) — independent of the dataset size n beyond sub-sampling.
- Scoring: O(m · t · log ψ) for m points.
- Memory: O(t · ψ) nodes. Extended trees additionally store a d-vector per node.

## Usage
```python
from ids_forest import IntrusionDetector, make_traffic, TRAFFIC_FEATURES, roc_auc

Xtr, _, _ = make_traffic(3000, 0, seed=1, attacks=())
Xte, yte, _ = make_traffic(2000, 150, seed=2)
det = IntrusionDetector(extended=True, feature_names=TRAFFIC_FEATURES).fit(Xtr)
print(roc_auc(yte, det.score(Xte)))
print(det.explain(Xte[0]))
```
Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests.

## Results
Demo run (3000 benign training flows, 2000 benign + 150 attack test flows, 100 trees, ψ = 256, contamination 5%):

| Model | ROC-AUC | Avg. precision | Precision | Recall | F1 @ threshold | Best F1 |
|-------|--------|----|------|------|------|------|
| Isolation Forest | 0.995 | 0.899 | 0.575 | 1.000 | 0.730 | 0.916 |
| Extended Isolation Forest | 0.997 | 0.966 | 0.584 | 1.000 | 0.737 | 0.899 |

Per-family AUC against benign traffic (Isolation Forest): port scan 0.998, DoS 0.995, exfiltration 0.993, brute force 0.995. The fixed 5% contamination threshold over-alerts (the test set holds about 7% attacks, while the threshold assumes 5% anomalies in training data), which is why precision lags behind the best-F1 figure.

## References
- F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation Forest.* ICDM 2008.
- F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation-based anomaly detection.* ACM TKDD 6(1), 2012.
- S. Hariri, M. C. Kind, R. J. Brunner. *Extended Isolation Forest.* IEEE TKDE 33(4), 2021.

---

**Author:** Dr. Kainat Rizwan
