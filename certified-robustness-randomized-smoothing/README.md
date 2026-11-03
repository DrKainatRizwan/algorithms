# Certified Robustness via Randomized Smoothing for Small Classifiers

## Overview
A from-scratch NumPy implementation of randomized smoothing for provable L2 robustness. Any base classifier `f` is turned into a smoothed classifier `g(x) = argmax_c P[f(x + N(0, σ²I)) = c]`. Monte-Carlo sampling plus an exact Clopper-Pearson bound yields, with confidence `1 − α`, a radius inside which the prediction of `g` cannot change.

The package contains softmax-regression and MLP base models with hand-written backprop and noise-augmented training, the `PREDICT` / `CERTIFY` procedures, L2 PGD and EOT-PGD attacks for sanity checks, and certified-accuracy evaluation.

## How it works
1. Train the base model on inputs perturbed with `N(0, σ²I)` so that it behaves well under noise.
2. `CERTIFY(x)`: draw `n0` noisy copies to guess the top class `c`; draw `n` fresh copies and count `k` votes for `c`.
3. Compute `p_A = ` lower `(1−α)` Clopper-Pearson bound on `k/n` (exact binomial tail, solved by bisection). If `p_A < 1/2`, abstain.
4. Otherwise certify radius `R = σ · Φ⁻¹(p_A)` (Cohen et al., 2019). Using a separate `p_B` gives `σ/2 (Φ⁻¹(p_A) − Φ⁻¹(p_B))`.
5. `PREDICT(x)` abstains unless a two-sided exact binomial test separates the top two classes.

For a binary linear classifier, `p_A = Φ(margin / (σ‖w‖))`, so the certified radius equals the exact distance to the decision boundary; this is used as a ground-truth test.

## Complexity
- `CERTIFY`: `O((n0 + n) · T_f)` where `T_f` is the base forward cost, plus `O(n · 60)` for the binomial bisection. Memory `O(chunk · d)`.
- The maximum certifiable radius is `σ Φ⁻¹(α^{1/n})`, so more samples are needed for larger radii.

## Usage
```python
from smoothing_cert import MLPClassifier, SmoothedClassifier, make_blobs

X, y = make_blobs(1500, dim=20, classes=4, sep=4.0, seed=0)
mlp = MLPClassifier(20, 64, 4)
mlp.fit(X, y, epochs=40, lr=0.01, noise_sigma=0.5)
smoothed = SmoothedClassifier(mlp, num_classes=4, sigma=0.5, seed=0)
res = smoothed.certify(X[0], n0=100, n=1000, alpha=0.001)
print(res.label, res.radius)   # label is -1 (ABSTAIN) if no certificate
```

Run `python examples/demo.py` for the benchmark and `python -m pytest -q` for the tests.

## Results
Certified accuracy on 60 test points (20-D, 4 Gaussian classes, MLP with 64 hidden units, `n0=50, n=500, α=0.001`; `python examples/demo.py`):

| σ | r=0 | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|-----|------|-----|-----|-----|-----|
| 0.25 | 0.98 | 0.95 | 0.93 | 0.00 | 0.00 | 0.00 |
| 0.50 | 0.97 | 0.95 | 0.95 | 0.80 | 0.00 | 0.00 |
| 1.00 | 0.97 | 0.97 | 0.95 | 0.85 | 0.62 | 0.28 |

Larger σ buys larger radii at little cost in clean accuracy on this easy task. For a linear model (σ=0.5, n=5000) certified radii of 0.62 vs. exact 0.65 are obtained near the boundary; the others saturate at the sampling cap of 1.50. Empirical EOT-PGD accuracy of the smoothed MLP: 0.93 / 0.93 / 0.87 / 0.57 at ε = 0 / 0.5 / 1.0 / 2.0, consistent with the certificates.

## References
- J. Cohen, E. Rosenfeld, J. Z. Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML 2019.
- M. Lecuyer et al. *Certified Robustness to Adversarial Examples with Differential Privacy.* IEEE S&P 2019.
- C. J. Clopper, E. S. Pearson. *The use of confidence or fiducial limits illustrated in the case of the binomial.* Biometrika 1934.
- A. Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.

---

**Author:** Dr. Kainat Rizwan
