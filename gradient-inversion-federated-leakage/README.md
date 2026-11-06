# Gradient Inversion Attacks and Defences for Federated Learning

## Overview

In federated learning clients share gradients rather than data, yet gradients of a dense layer
can reveal the training inputs almost exactly. This project implements, from scratch in NumPy,
the main attack families and a set of client-side defences against them:

- **Analytic inversion** of a single example from first-layer gradients (exact, closed form) with label recovery.
- **Gradient matching** ("Deep Leakage from Gradients") for batches, with a hand-derived second-order gradient, Adam, restarts, box constraints and an optional total-variation prior.
- **Defences**: clipping with Gaussian noise (DP-SGD style), top-k sparsification and uniform quantisation.
- Metrics (MSE, PSNR, global SSIM) with optimal permutation matching of reconstructed batches.

## How it works

Victim: `logits = W2 tanh(W1 x + b1) + b2`, softmax cross-entropy.

**Analytic attack.** With `a = W1 x + b1` and `da = dL/da`, `gW1 = da xᵀ` and `gb1 = da`, so `x = gW1[i] / gb1[i]`
for every hidden unit `i`. We use the least-squares combination `x = (gb1ᵀ gW1)/‖gb1‖²`. The label is the only
coordinate where `gb2 = p − onehot(y)` is negative.

**Gradient matching.** Minimise `L(X) = ½‖∇θ ℓ(X, y) − g*‖²` over dummy inputs `X`. Differentiating through the
gradient requires a Hessian-vector-style term; with residuals `r`, `u = r_W1 x + r_b1`, `v = r_W2 h + r_b2`,
`w = r_W2ᵀ dz`, `c = W2ᵀ dz`, `s = 1 − h²` and the softmax Jacobian `Js = diag(p) − ppᵀ`:

```
G_h  = W2ᵀ Js W2 (s⊙u) − 2 c⊙h⊙u + W2ᵀ Js v + w
∇ₓL = W1ᵀ (s⊙G_h) + r_W1ᵀ da
```

This is verified against central finite differences in the tests (error ~1e-10). When several samples share one
averaged gradient the problem is a joint non-convex fit; reconstructions are compared to the truth under the best
permutation.

## Complexity

Let `d` input dimension, `H` hidden units, `K` classes, `B` batch size, `T` iterations, `R` restarts.

- Analytic attack: `O(H·d)` time and space.
- Gradient matching: `O(R·T·B·(H·d + H·K))` time, `O(B·d + H·d)` space.
- Permutation matching of results: `O(B!)` brute force (used for `B ≤ 7`).
- Defences: `O(P)` for clipping/noise/quantisation, `O(P)` expected for top-k (partition), `P` = parameter count.

## Usage

```python
import numpy as np
from gradient_leakage import build_victim, analytic_reconstruct, gradient_matching_attack

model, X, y, _ = build_victim(seed=0)
g = model.gradients(X[:1], y[:1])
x_hat = analytic_reconstruct(g)                       # exact for B=1

g = model.gradients(X[:3], y[:3])
res = gradient_matching_attack(model, g, y[:3], np.random.default_rng(1), iters=800, restarts=3)
```

Run `python examples/demo.py` for the full experiment and `python -m pytest -q` for the tests.

## Results

Victim: 64-32-4 tanh MLP on 8×8 synthetic images, 3 epochs of pre-training (output of `examples/demo.py`).

| Experiment | Result |
|---|---|
| Analytic, B=1 | MSE 1.1e-32, label recovered |
| Gradient matching B=1 / 2 / 3 / 4 | 55.3 / 75.3 / 43.7 / 30.2 dB PSNR |
| Noise σ=0.01 / 0.1 / 1.0 (B=1) | 28.8 / 9.3 / 4.1 dB |
| Top-k keep 50% / 5% | 37.0 / 12.0 dB |
| 4-level quantisation | 24.3 dB |
| No defence (500 iters, B=1) | 37.7 dB |

Reconstruction quality decays with batch size, and moderate noise or aggressive sparsification removes most of the leak,
at the usual cost in utility. Note that these numbers come from a small model where the gradient has many more equations than unknowns; wider
batches or larger inputs make the problem much harder.

## References

- Zhu, Liu, Han. *Deep Leakage from Gradients.* NeurIPS 2019.
- Zhao, Mopuri, Bilen. *iDLG: Improved Deep Leakage from Gradients.* 2020.
- Geiping et al. *Inverting Gradients – How easy is it to break privacy in federated learning?* NeurIPS 2020.
- Phong et al. *Privacy-Preserving Deep Learning via Additively Homomorphic Encryption.* 2018.
- Abadi et al. *Deep Learning with Differential Privacy.* CCS 2016.

---

**Author:** Dr. Kainat Rizwan
