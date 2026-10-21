# Model Inversion and Membership Inference Attack Simulator with Defenses

A from-scratch NumPy laboratory for two classic privacy attacks on machine-learning models, together with the defences that blunt them. A small MLP is trained on a synthetic image-like dataset with known class prototypes, so both attack success (membership AUC) and reconstruction fidelity (correlation with the true prototype) can be measured exactly.

## Overview

| Module | Contents |
|--------|----------|
| `privacy_attacks/data.py` | Smooth class prototypes, noisy samples, member / non-member / shadow / test splits |
| `privacy_attacks/models.py` | ReLU MLP with manual backprop, per-example gradients, input gradients, Adam trainer with L2, early stopping and DP-SGD, zCDP epsilon bound |
| `privacy_attacks/attacks.py` | Loss / confidence / entropy / modified-entropy scores, shadow-model attack, reference-calibrated attack, label-only attack |
| `privacy_attacks/inversion.py` | White-box gradient-ascent inversion and black-box NES inversion, reconstruction metrics |
| `privacy_attacks/defenses.py` | API wrappers: label-only, top-k, rounding, temperature, Laplace output noise |
| `privacy_attacks/metrics.py` | AUC (rank based), ROC, TPR at fixed FPR, membership advantage, correlation |

## How it works

**Membership inference.** Overfit models fit training points more tightly than unseen points, so the per-example loss separates members from non-members. All attacks emit a score where larger means "member" and are evaluated with the same metrics:

* *Threshold scores*: `log p(y|x)`, max-confidence, negative entropy, and the Song-Mittal modified entropy.
* *Shadow attack* (Shokri et al.): train several shadow MLPs on random subsets of a disjoint pool, record logit-scaled true-class probability, entropy, confidence and correctness for their members and non-members, and fit a logistic-regression attack classifier.
* *Reference-calibrated attack*: subtract the average shadow-model loss on the point from the target loss to remove per-example difficulty.
* *Label-only attack*: query noisy copies of `x` and use the fraction still classified correctly; members sit further from the decision boundary. It needs no confidence values at all.

**Model inversion.** For class `c`, ascend `log p(c|x) - (lambda/2)||x - mu||^2` using the exact input gradient (white-box), or estimate that gradient with antithetic natural-evolution-strategy queries (black-box, only `predict_proba` is used). The reconstruction is compared with the true prototype by Pearson correlation and a margin against the best wrong class.

**Defences.** Training-time: L2 weight decay, early stopping on a validation set, and DP-SGD (per-example clipping to norm `C`, Gaussian noise `sigma*C`; a conservative zCDP composition bound gives epsilon). Output-time: label-only release, top-k truncation, rounding, temperature scaling and Laplace output noise.

## Complexity

* Forward/backward pass: `O(B * sum_l d_l d_{l+1})` time, `O(B * max d_l)` activations.
* Per-example gradients (DP-SGD): `O(B * P)` memory for `P` parameters.
* Shadow attack: `S` shadow trainings plus one logistic regression, `O(S * T_train)`.
* Label-only attack: `Q` forward passes per point.
* White-box inversion: `O(steps * P)`; black-box NES: `O(steps * n_samples)` queries.
* AUC: `O(n log n)` from sorting.

## Usage

```python
from privacy_attacks import (MLP, TrainConfig, train, make_world, score_attack, auc,
                             invert_whitebox, reconstruction_quality, TopK)

w = make_world(0)
model = MLP(w.dim, w.n_classes, (64,), seed=1)
train(model, w.X_in, w.y_in, TrainConfig())

print(auc(score_attack(model, w.X_in, w.y_in), score_attack(model, w.X_out, w.y_out)))  # ~0.84
api = TopK(model, 1)                          # defended API
print(auc(score_attack(api, w.X_in, w.y_in), score_attack(api, w.X_out, w.y_out)))      # ~0.58

rec = invert_whitebox(model, target=0, dim=w.dim)
print(reconstruction_quality(rec.x, w.protos, 0))
```

Run `python examples/demo.py` for the full experiment (about 2 seconds) and `python -m pytest -q` for the 27 tests (finite-difference gradient checks, AUC against brute force, attack/defence behaviour, inversion).

## Results

Output of `examples/demo.py` (5 classes, 64 features, 100 members, 100 non-members, seed 0):

| Setting | Test acc | Loss AUC | Shadow AUC | Calibrated AUC | Label-only AUC |
|---------|---------:|---------:|-----------:|---------------:|---------------:|
| Undefended | 0.904 | 0.838 | 0.838 | 0.610 | 0.792 |
| L2 (0.05) | 0.932 | 0.768 | 0.765 | 0.621 | 0.660 |
| Early stopping | 0.840 | 0.841 | 0.841 | 0.747 | 0.798 |
| DP-SGD (sigma=1) | 0.622 | 0.693 | 0.673 | 0.651 | 0.704 |
| DP-SGD (sigma=2) | 0.420 | 0.633 | 0.589 | 0.602 | 0.644 |

Output-side defences on the undefended model (loss AUC): full probabilities 0.838, top-1 only 0.575, rounded to one decimal 0.670, Laplace noise 0.1 gives 0.597, label-only 0.575. The label-only attack is unaffected (0.79), showing that hiding confidences does not remove leakage from the decision boundary.

Model inversion, mean correlation with the true prototype (all 5 classes identified in each case): undefended white-box 0.607, L2 0.752, DP-SGD (sigma=1) 0.302, black-box 0.648, black-box through a noisy API 0.511. Note that L2 regularisation makes inversion *easier* even as it reduces membership leakage, because a smoother model encodes the class prototype more cleanly. The reference-calibrated attack is weaker than the plain loss attack here: the synthetic data has little per-example difficulty variation for calibration to remove.

## References

* R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P 2017.
* S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* CSF 2018.
* M. Fredrikson, S. Jha, T. Ristenpart. *Model Inversion Attacks that Exploit Confidence Information.* CCS 2015.
* L. Song, P. Mittal. *Systematic Evaluation of Privacy Risks of Machine Learning Models.* USENIX Security 2021.
* C. A. Choquette-Choo et al. *Label-Only Membership Inference Attacks.* ICML 2021.
* L. Watson et al. *On the Importance of Difficulty Calibration in Membership Inference Attacks.* ICLR 2022.
* M. Abadi et al. *Deep Learning with Differential Privacy.* CCS 2016.
* T. Salimans et al. *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.* 2017.

---

**Author:** Dr. Kainat Rizwan
