# Explainable AI Toolkit: SHAP-Style Shapley Values and LIME from Scratch

## Overview
A NumPy-only implementation of post-hoc explanation methods for tabular models: exact Shapley values, permutation-sampling Shapley, KernelSHAP, the linear closed form, tabular LIME, and integrated gradients. Small models (logistic regression, a tanh MLP, CART trees and a random forest) are included so every explainer can be tested against known ground truth, plus metrics for efficiency, faithfulness (deletion curves), and stability.

## How it works
- **Value function.** For an instance `x` and a background sample `B`, `v(S) = mean_b f(x_S, b_{~S})` (interventional expectation). `v(∅)` is the mean prediction and `v(N)=f(x)`.
- **Exact Shapley.** `phi_i = Σ_{S⊆N\{i}} |S|!(d-|S|-1)!/d! · (v(S∪i) − v(S))`, computed from one batched evaluation of all 2^d coalitions.
- **Permutation sampling.** Average marginal contributions along random feature orderings (with reversed antithetic partners). Each ordering costs d+1 evaluations and satisfies efficiency exactly.
- **KernelSHAP.** Weighted least squares over coalitions with kernel `π(s) = (d−1)/(C(d,s)·s·(d−s))`, subject to `Σ phi = f(x) − E f` (solved by eliminating one variable). Full enumeration is exact; otherwise sizes are sampled proportionally to kernel mass with complementary pairs.
- **LIME.** Gaussian perturbations around `x`, exponential locality kernel, greedy forward feature selection, and a weighted ridge surrogate whose coefficients are mapped back to original units.
- **Integrated gradients.** Midpoint-rule path integral from a baseline; satisfies completeness.

## Complexity
| Method | Model evaluations | Extra time | Space |
|---|---|---|---|
| Exact Shapley | 2^d · \|B\| | O(d·2^d) | O(2^d) |
| Permutation | 2·P·(d+1)·\|B\| | O(P·d²) | O(d) |
| KernelSHAP | m·\|B\| | O(m·d² + d³) | O(m·d) |
| LIME | N | O(N·k·d²) forward selection | O(N·d) |
| Integrated gradients | steps gradients | O(steps·d) | O(steps·d) |

## Usage
```python
from xai_toolkit import *
X, y, w = make_tabular(800, 8, 4, seed=0)
model = LogisticModel().fit(X, y)
vf = ValueFunction(model.predict, X[0], X[:50])
phi = kernel_shap(vf, n_samples=100, seed=0)
print(phi, efficiency_gap(phi, vf.base, vf.full))
lime = LimeTabular(X, seed=0).explain(X[0], model.predict, n_features=3)
print(lime.top(3), lime.r2)
```

## Results
Output of `python examples/demo.py` (MLP, d=8, 50 background rows):

| Budget | KernelSHAP max error (calls) | Permutation max error (calls) |
|---|---|---|
| ~30 | 0.0385 (28) | 0.0300 (72) |
| ~100 | 0.0162 (56) | 0.0107 (216) |
| ~300 | 0.0000 (254, full enumeration) | 0.0028 (612) |

- Logistic accuracy 0.895, MLP 0.880, forest 0.860 on the held-out split.
- On a linear model, exact Shapley equals `w_i(x_i − mean_i)` and LIME's local contributions match to three decimals (R² = 1.0).
- Random forest: efficiency gap 5.6e-17; deletion AUC (supportive features removed first) −0.263 for Shapley order versus −0.096 for reversed order.
- Stability (mean Spearman over 6 seeds): LIME with 300 samples 0.957, KernelSHAP with 40 samples 0.937.
- Integrated gradients sum −0.0359 vs f(x)−f(baseline) −0.0360.

Run the tests with `python -m pytest -q` (20 tests).

## References
- Shapley, L. S. (1953). A value for n-person games.
- Lundberg, S. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Ribeiro, M., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?" Explaining the Predictions of Any Classifier. KDD.
- Sundararajan, M., Taly, A., Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML.
- Covert, I. & Lee, S.-I. (2021). Improving KernelSHAP: Practical Shapley Value Estimation Using Linear Regression. AISTATS.
- Štrumbelj, E. & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions.

---

**Author:** Dr. Kainat Rizwan
