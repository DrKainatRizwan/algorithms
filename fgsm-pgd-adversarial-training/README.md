# FGSM and PGD Adversarial Attacks with Adversarial Training in NumPy

## Overview
A compact, dependency-light study of evasion attacks and the min-max defence against them. The package contains a ReLU multilayer perceptron with hand-written backpropagation (including the gradient with respect to the *input*), the Fast Gradient Sign Method, Projected Gradient Descent (L-inf and L2, targeted and untargeted), Momentum-Iterative FGSM, and an adversarial-training loop with an Adam optimiser. Only NumPy is used.

## How it works
- **Model** – `MLP` computes logits, softmax cross-entropy and exact gradients. `input_gradient` returns per-sample `∇ₓ L(x, y)`; it is validated against central finite differences.
- **FGSM** – a single step `x' = x + ε·sign(∇ₓL)` (or the normalised gradient for L2). For a linear score `w·x + b` this is the exact optimum of the inner maximisation: the margin falls by `ε‖w‖₁`.
- **PGD** – iterates `x ← Π_{B_ε(x₀)}(x + α·steepest(∇ₓL))` from a random start, with projection onto the L-inf box or L2 ball. It is a much stronger first-order adversary for non-linear models.
- **MI-FGSM** – accumulates L1-normalised gradients in a momentum term to stabilise the update direction.
- **Adversarial training** – approximately solves `min_θ E[max_{‖δ‖≤ε} L(θ, x+δ, y)]` by generating PGD (or FGSM) examples against the current weights for each minibatch and taking an Adam step on them.
- **Fragile-feature task** – `make_fragile` follows the construction of Tsipras et al.: one noisy large-margin feature plus many tiny-margin but jointly predictive features. Natural training latches onto the latter, which an attacker with `ε` above the feature offset flips together.

## Complexity
For `n` samples, input dimension `d`, network with `P` parameters, `T` attack steps:
- FGSM: one forward/backward pass, `O(nP)` time, `O(nd)` extra space.
- PGD: `O(T·nP)` time.
- Adversarial training: `O(E·(T+1)·nP)` for `E` epochs, i.e. roughly `T+1` times the cost of natural training. Memory is `O(P + bd)` for batch size `b`.

## Usage
```python
from adv_attacks import (MLP, TrainConfig, fit, adversarial_fit, pgd,
                         make_fragile, train_test_split, robust_accuracy)

X, y = make_fragile(1000, seed=1)
Xtr, ytr, Xte, yte = train_test_split(X, y, 0.3, seed=0)
adv = adversarial_fit(MLP((10, 16, 2), seed=1), Xtr, ytr,
                      TrainConfig(epochs=60, lr=0.01, eps=0.4, mix=1.0))
print(robust_accuracy(adv, Xte, yte, 0.4, "pgd", steps=20))
```
Run `python examples/demo.py` for the full comparison and `python -m pytest -q` for the tests (20 tests).

## Results
Output of `examples/demo.py` (fragile-feature task, L-inf ε = 0.4, 300 test points; the whole demo takes under a second):

| model | clean | FGSM | PGD-20 | attack success rate |
|-------|------:|-----:|-------:|--------------------:|
| natural | 1.000 | 0.117 | 0.117 | 0.883 |
| adversarially trained | 0.817 | 0.693 | 0.690 | 0.155 |

PGD robust accuracy against ε:

| ε | natural | adv-trained |
|--:|--------:|------------:|
| 0.0 | 1.000 | 0.817 |
| 0.2 | 0.993 | 0.740 |
| 0.3 | 0.637 | 0.710 |
| 0.4 | 0.117 | 0.690 |
| 0.6 | 0.000 | 0.637 |

The natural model collapses once ε exceeds the tiny-margin offset (0.3), while the adversarially trained one gives up about 18 points of clean accuracy in exchange for a flat robustness curve – the accuracy/robustness trade-off. On the two-spiral task a naturally trained network falls from 1.000 to 0.800 under PGD at ε = 0.3.

## References
- I. Goodfellow, J. Shlens, C. Szegedy. *Explaining and Harnessing Adversarial Examples.* ICLR 2015.
- A. Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018.
- Y. Dong et al. *Boosting Adversarial Attacks with Momentum.* CVPR 2018.
- D. Tsipras et al. *Robustness May Be at Odds with Accuracy.* ICLR 2019.
- A. Kurakin, I. Goodfellow, S. Bengio. *Adversarial Examples in the Physical World.* 2017.

---

**Author:** Dr. Kainat Rizwan
