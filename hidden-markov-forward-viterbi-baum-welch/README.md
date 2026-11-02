# Hidden Markov Models: Forward-Backward, Viterbi and Baum-Welch from Scratch

## Overview
A compact NumPy implementation of discrete-emission hidden Markov models: exact
likelihood evaluation, smoothing, MAP decoding, unsupervised learning by EM and
supervised estimation, plus brute-force oracles and BIC model selection. All
recursions are available in two numerically stable forms (per-step scaling and
log-space) that are cross-checked against each other and against exhaustive
enumeration on small problems.

## How it works
- **Forward / backward (scaled).** `alpha_hat_t = normalise((alpha_hat_{t-1} A) * B[:, x_t])`,
  with scale factors `c_t = P(x_t | x_{1:t-1})`, so `log P(x) = sum_t log c_t`
  without underflow on long sequences. The backward pass reuses the same `c_t`.
- **Smoothing.** `gamma_t(i) = alpha_hat_t(i) beta_hat_t(i)`; pairwise posteriors
  `xi_t(i, j)` are accumulated in one matrix product.
- **Viterbi.** Max-product dynamic programming in log space with back-pointers
  gives the single most probable state path; posterior decoding maximises the
  expected number of correct positions instead.
- **Baum-Welch.** EM: the E-step computes `gamma` and `xi` for every sequence; the
  M-step renormalises expected counts (with a tiny pseudocount). Log-likelihood
  is monotonically non-decreasing; random restarts mitigate local optima.
- **Model selection.** BIC with `(N-1) + N(N-1) + N(M-1)` free parameters.

Modules: `models` (HMM, sampler), `inference`, `logspace`, `learning`, `analysis`.

## Complexity
For `N` states, `M` symbols, sequence length `T`:

| Operation | Time | Space |
|-----------|------|-------|
| Forward / backward / Viterbi | O(T N^2) | O(T N) |
| One Baum-Welch iteration | O(T (N^2 + N M)) | O(T N + N^2) |
| Brute-force oracle | O(T N^T) | O(T) |

## Usage
```python
import numpy as np
from hmm_toolkit import HMM, viterbi, baum_welch, likelihood

rng = np.random.default_rng(0)
hmm = HMM([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
states, obs = hmm.sample(500, rng)
path, logp = viterbi(hmm, obs)
fit = baum_welch(obs, n_states=2, rng=rng, n_symbols=3, n_restarts=3)
print(likelihood(hmm, obs), fit.final_ll)
```

Run `python examples/demo.py` for the full demo and `python -m pytest -q` for the tests.

## Results
Demo on a synthetic "GC-island" task (2 states, 4 symbols, `T = 3000`, seed 42):

| Metric | Value |
|--------|-------|
| Viterbi accuracy (true model) | 0.820 |
| Posterior-decoding accuracy (true model) | 0.844 |
| Scaled vs log-space log-likelihood | -4101.4241 / -4101.4241 |
| Baum-Welch (3 restarts, 3 x 800 symbols): iterations | 102 (converged) |
| Fitted LL vs true-model LL | -3292.7 vs -3294.6 |
| Viterbi accuracy, learned / supervised model | 0.822 / 0.810 |
| BIC for N = 1, 2, 3 (well-separated source) | 1912.3, **1678.1**, 1726.5 |

Posterior decoding yields higher per-position accuracy than Viterbi, as theory
predicts, and the unsupervised fit matches the supervised one.

## References
- L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition", *Proc. IEEE* 77(2), 1989.
- L. E. Baum, T. Petrie, G. Soules, N. Weiss, "A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains", *Ann. Math. Statist.* 41(1), 1970.
- A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm", *IEEE Trans. Inf. Theory* 13(2), 1967.
- C. M. Bishop, *Pattern Recognition and Machine Learning*, ch. 13, Springer, 2006.

---

**Author:** Dr. Kainat Rizwan
