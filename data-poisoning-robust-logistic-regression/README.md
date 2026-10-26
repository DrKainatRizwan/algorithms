# Data Poisoning Attacks and Robust Training for Logistic Regression

## Overview

A from-scratch NumPy study of training-set poisoning against L2-regularised logistic regression, and of the defences that mitigate it. It implements four attacks (random, boundary and influence-ranked label flips, plus gradient-based injection of new poison points) and four defences (TRIM, sphere/slab filtering, k-NN label sanitisation and a trimmed-mean robust gradient method), with exact gradients, Hessians and influence functions so every attack is verified against brute-force retraining.

## How it works

**Model.** Objective `J(w) = sum_i log(1 + exp(-y_i' w.x_i)) + (lambda/2)||w||^2`, fitted by damped Newton iterations with closed-form gradient and Hessian `H`.

**Influence attacks.** For a perturbation of the training gradient `dg`, the optimum moves by `dw = -H^-1 dg`. Flipping label `i` therefore changes the validation loss by `-(g_flip - g_orig)^T H^-1 g_val`. The attack ranks points by this score and flips the best ones greedily in a few rounds, refitting between rounds.

**Gradient poisoning (Biggio et al.).** A poison point `(x_c, y_c)` is appended; implicit differentiation through the retraining map gives `dL_val/dx_c = -J^T H^-1 g_val` with `J = d(grad_w l)/dx_c`. Poison features take normalised projected gradient steps inside a bounding box; the best iterate is kept. The gradient is tested against finite differences of a full retrain.

**Defences.**
- *TRIM*: alternate fitting and keeping the `(1-alpha)n` lowest-loss points until the kept set is stable.
- *Sphere + slab*: drop points far from the median class centroid, or far along the centroid-difference axis.
- *k-NN sanitisation*: drop points whose label disagrees with most of their `k` neighbours.
- *Trimmed-gradient descent*: coordinate-wise trimmed mean of per-sample gradients.

## Complexity

With `n` samples, `d` features, `k` poison points and `T` attack iterations:

| Component | Time | Space |
|-----------|------|-------|
| Newton fit | O(iters * (n d^2 + d^3)) | O(n d) |
| Influence flip ranking | O(n d^2 + d^3) | O(n d) |
| Gradient poisoning | O(T (fit + k d^3)) | O(n d) |
| TRIM | O(iters * fit + n log n) | O(n d) |
| k-NN sanitisation | O(n^2 d) | O(n^2) |
| Trimmed-gradient | O(iters * n d log n) | O(n d) |

## Usage

```python
import numpy as np
from poison_robust import (train_val_test_split, gradient_poisoning, LogisticRegression, trim_fit)

ds = train_val_test_split(n_train=300, n_val=300, n_test=1000, seed=1)
atk = gradient_poisoning(ds.X_train, ds.y_train, ds.X_val, ds.y_val, 0.2,
                         np.random.default_rng(1), steps=25)
plain = LogisticRegression(1.0).fit(atk.X, atk.y)
robust = trim_fit(atk.X, atk.y, alpha=0.2).model
print(plain.accuracy(ds.X_test, ds.y_test), robust.accuracy(ds.X_test, ds.y_test))
```

Run `python examples/demo.py` for the full grid and `python -m pytest -q` for the 19 tests.

## Results

Two-class Gaussian data (n=300, d=2), clean test accuracy 0.882. Test accuracy under attack, defences told the true poison rate (output of `examples/demo.py`):

| Attack | Rate | none | TRIM | sphere+slab | k-NN | trimmed-grad |
|--------|------|------|------|-------------|------|--------------|
| random-flip | 0.20 | 0.882 | 0.884 | 0.883 | 0.883 | 0.880 |
| boundary-flip | 0.20 | 0.876 | 0.885 | 0.886 | 0.884 | 0.886 |
| influence-flip | 0.10 | 0.833 | 0.862 | 0.847 | 0.843 | 0.857 |
| influence-flip | 0.20 | 0.771 | 0.852 | 0.789 | 0.765 | 0.811 |
| influence-flip | 0.30 | 0.691 | 0.691 | 0.724 | 0.682 | 0.629 |
| gradient | 0.20 | 0.671 | 0.883 | 0.883 | 0.710 | 0.884 |
| gradient | 0.30 | 0.170 | 0.474 | 0.883 | 0.166 | 0.884 |

Observations: random flips are nearly harmless for regularised logistic regression; influence-ranked flips are far more damaging; optimised injections collapse the undefended model (0.17 at 30%) but sit far from the data, so geometric filters and trimmed gradients remove them. Beyond roughly 20% targeted flips, all defences degrade. The whole demo runs in about 1.5 s.

## References

- Biggio, Nelson, Laskov. *Poisoning Attacks against Support Vector Machines.* ICML 2012.
- Koh, Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017.
- Jagielski et al. *Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Regression Learning.* IEEE S&P 2018.
- Steinhardt, Koh, Liang. *Certified Defenses for Data Poisoning Attacks.* NeurIPS 2017.
- Yin, Chen, Ramchandran, Bartlett. *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.* ICML 2018.

---

**Author:** Dr. Kainat Rizwan
