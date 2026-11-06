# Backdoor Trigger Detection in Neural Networks via Activation Clustering

## Overview
A from-scratch NumPy study of data-poisoning backdoors (BadNets-style patch triggers) and two
activation-space detectors: **Activation Clustering** (Chen et al., 2018) and **Spectral
Signatures** (Tran et al., 2018). It includes a synthetic image generator, trigger poisoning with
ground-truth masks, an MLP with manual backprop, PCA / k-means++ / silhouette code, a
sanitise-and-retrain defence, and attack-success-rate metrics.

## How it works
1. **Attack.** A 3x3 bright patch is stamped on non-target images, which are relabelled as the
   target class and appended to the training set. The trained model behaves normally on clean
   inputs but predicts the target whenever the patch is present.
2. **Activation clustering.** For each class, take last-hidden-layer activations of the training
   samples, reduce with PCA, run 2-means. Clean classes split into two roughly balanced, poorly
   separated clusters; a poisoned class splits into a well-separated clean cluster and poison
   cluster. A class is suspicious if the silhouette is high and the smaller cluster is small
   (relative size). The smaller cluster is flagged.
3. **Spectral signatures.** Per class, centre activations and score each sample by its squared
   projection on the top singular vector; flag the top 1.5*eps fraction.
4. **Defence.** Remove flagged samples and retrain from scratch.

## Complexity
- Activation extraction: O(n * W) for W weights.
- PCA by SVD: O(n d^2) per class; k-means: O(n k i) per restart; exact silhouette O(n^2) time and
  memory per class (the dominant cost for large classes; subsample if needed).
- Spectral signatures: O(n d^2) per class. Space O(n d) for activations.

## Usage
```python
from backdoor_detection import *
Xtr, ytr = make_dataset(300, 4, seed=1)
pd = poison_dataset(Xtr, ytr, target=0, rate=0.10, seed=3)
model = train(pd.X, pd.y, 4)
res = activation_clustering(model, pd.X, pd.y, 4)
print(evaluate_detection(res.flagged, pd.is_poison))
clean_model = sanitize_and_retrain(pd.X, pd.y, res.flagged, 4)
```

## Results
Output of `python examples/demo.py` (4 classes, 1200 clean samples, target class 0):

| Poison rate | ASR (backdoored) | AC precision/recall | SS precision/recall | ASR after AC defence | Clean acc after |
|---|---|---|---|---|---|
| 5%  | 1.000 | 1.00 / 0.75 | 0.29 / 0.44 | 0.767 | 1.000 |
| 10% | 1.000 | 1.00 / 1.00 | 0.32 / 0.49 | 0.010 | 1.000 |
| 20% | 1.000 | 0.00 / 0.00 | 0.26 / 0.39 | 1.000 | 1.000 |

At 10% poisoning, class 0 has silhouette 0.57 and relative cluster size 0.31 against silhouettes
of 0.23-0.26 for clean classes. Limitations are visible too: at 5% only part of the poison is
found, so the backdoor survives retraining, and at 20% poison outnumbers clean samples in the
target class, so the flagged "small" cluster is the clean one and the size criterion fails.
Spectral signatures with a fixed removal budget suffer from flagging samples in every class.

## Tests
`python -m pytest -q` runs 14 deterministic tests (gradient check, clustering correctness,
backdoor implantation, detection precision/recall, no false alarms on clean data, defence).

## References
- B. Chen et al., "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering", 2018.
- B. Tran, J. Li, A. Madry, "Spectral Signatures in Backdoor Attacks", NeurIPS 2018.
- T. Gu, B. Dolan-Gavitt, S. Garg, "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain", 2017.
- D. Arthur, S. Vassilvitskii, "k-means++: The Advantages of Careful Seeding", 2007.

---

**Author:** Dr. Kainat Rizwan
