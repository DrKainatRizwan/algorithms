# Differential Privacy Toolkit: Laplace, Gaussian and Exponential Mechanisms with Privacy Accounting

## Overview

A from-scratch NumPy implementation of the core building blocks of differential privacy: noise mechanisms, private statistical queries, the sparse vector technique, DP-SGD for logistic regression, and three privacy accountants (basic, advanced and Renyi-DP composition including Poisson-subsampled Gaussian).

## How it works

- **Laplace mechanism** – adds `Lap(Δ₁/ε)` noise, giving pure ε-DP.
- **Gaussian mechanism** – adds `N(0, σ²)`; σ is calibrated either with the classical formula `Δ₂·sqrt(2 ln(1.25/δ))/ε` or with the *analytic* calibration of Balle & Wang, solving the exact `δ(ε)` expression by bisection (noticeably tighter noise).
- **Exponential mechanism** – selects a candidate with probability ∝ `exp(ε·u/(2Δu))`; used for private quantiles (interval utilities weighted by interval width).
- **Geometric mechanism / randomized response** – discrete noise for integer counts and local-DP bit reporting with unbiased debiasing.
- **AboveThreshold** – sparse vector technique that pays for only one "above" answer regardless of how many queries it scans.
- **Accountants** – `BasicAccountant` (budget enforcement), `AdvancedAccountant` (DRV composition), `RDPAccountant` (Mironov's Renyi DP; Gaussian, Laplace and subsampled Gaussian curves, converted to (ε, δ) with the Balle et al. 2020 bound; also calibrates the noise multiplier for a target ε).
- **DP-SGD** – Poisson subsampling, per-example gradient clipping, Gaussian noise, with ε reported by the RDP accountant.

## Complexity

- Mechanisms: O(d) time and space for a d-dimensional release.
- Analytic σ calibration: O(log(1/tol)) evaluations of `erfc`.
- Subsampled Gaussian RDP: O(|orders|·α) per composition; composition itself is O(|orders|).
- Quantile: O(n log n) for sorting.
- DP-SGD: O(T·B·d) with T steps, batch size B.

## Usage

```python
import numpy as np
from dp_toolkit import *

rng = np.random.default_rng(0)
print(private_mean(np.random.rand(1000) * 100, 0, 100, epsilon=1.0, rng=rng))

acc = RDPAccountant()
acc.compose_subsampled_gaussian(sigma=1.1, q=0.01, steps=1000)
print(acc.epsilon(delta=1e-5))
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests (24 tests).

## Results (from `examples/demo.py`)

Gaussian calibration for Δ=1, ε=0.5, δ=1e-5: classical σ = 9.690, analytic σ = 7.032.

| k Gaussian queries (σ=5, δ=1e-5) | RDP ε | basic composition ε |
|---|---|---|
| 10 | 2.814 | 8.34 |
| 100 | 10.802 | 93.18 |
| 1000 | 48.757 | 1021.17 |

Subsampled Gaussian (q=0.01, σ=1.1, 1000 steps): ε = 1.726 at δ=1e-5.

DP-SGD logistic regression (n=5000, 5 epochs): accuracy 0.993 without noise, 0.993 with σ=0.8 (ε=6.78) and 0.993 with σ=2.0 (ε=1.24).

## References

- Dwork, McSherry, Nissim, Smith. *Calibrating Noise to Sensitivity in Private Data Analysis*, TCC 2006.
- Dwork, Roth. *The Algorithmic Foundations of Differential Privacy*, 2014.
- McSherry, Talwar. *Mechanism Design via Differential Privacy*, FOCS 2007.
- Balle, Wang. *Improving the Gaussian Mechanism for Differential Privacy*, ICML 2018.
- Mironov. *Renyi Differential Privacy*, CSF 2017; Mironov, Talwar, Zhang. *Renyi DP of the Sampled Gaussian Mechanism*, 2019.
- Balle et al. *Hypothesis Testing Interpretations and Renyi Differential Privacy*, AISTATS 2020.
- Abadi et al. *Deep Learning with Differential Privacy*, CCS 2016.

---

**Author:** Dr. Kainat Rizwan
