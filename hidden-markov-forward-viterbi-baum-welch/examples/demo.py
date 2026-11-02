"""Demo: decode, learn and select HMMs on a synthetic gene-finding-style task."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from hmm_toolkit import (HMM, forward, likelihood, log_forward, viterbi,
                         posterior_decode, baum_welch, supervised_fit,
                         state_accuracy, select_n_states)

rng = np.random.default_rng(42)
# States: 0 = background, 1 = GC-rich island. Symbols: A, C, G, T.
true = HMM([0.9, 0.1], [[0.97, 0.03], [0.08, 0.92]],
           [[0.30, 0.20, 0.20, 0.30], [0.12, 0.38, 0.38, 0.12]])
z, x = true.sample(3000, rng)

print("== Decoding with the true model ==")
vp, lp = viterbi(true, x)
pd = posterior_decode(true, x)
print(f"Viterbi accuracy   : {np.mean(vp == z):.3f}  (log joint {lp:.1f})")
print(f"Posterior accuracy : {np.mean(pd == z):.3f}")

print("\n== Numerical stability (T=3000) ==")
print(f"scaled log-lik {likelihood(true, x):.4f} | log-space {log_forward(true, x)[1]:.4f}")
print(f"naive product of unscaled alphas underflows: "
      f"{np.prod(forward(true, x)[1]) == 0.0}")

print("\n== Baum-Welch (unsupervised, 3 restarts) ==")
seqs = [true.sample(800, rng) for _ in range(3)]
t0 = time.perf_counter()
res = baum_welch([s[1] for s in seqs], 2, np.random.default_rng(1),
                 n_symbols=4, n_restarts=3, max_iter=200)
dt = time.perf_counter() - t0
print(f"iterations {res.n_iter}, converged={res.converged}, time {dt:.2f}s")
print(f"fitted LL {res.final_ll:.1f} vs true-model LL "
      f"{sum(likelihood(true, s[1]) for s in seqs):.1f}")
order = np.argsort(res.hmm.B[:, 1])
print("fitted A:\n", np.round(res.hmm.A[np.ix_(order, order)], 3))
print("fitted B:\n", np.round(res.hmm.B[order], 3))
acc = state_accuracy(z, viterbi(res.hmm, x)[0], 2)
print(f"Viterbi accuracy with learned model: {acc:.3f}")

sup = supervised_fit([s[0] for s in seqs], [s[1] for s in seqs], 2, 4)
print(f"Supervised-fit Viterbi accuracy    : {np.mean(viterbi(sup, x)[0] == z):.3f}")

print("\n== Model selection by BIC (well-separated 2-state source) ==")
src = HMM([0.5, 0.5], [[0.95, 0.05], [0.05, 0.95]],
          [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
sel = [src.sample(400, rng)[1] for _ in range(2)]
scores = select_n_states(sel, [1, 2, 3], 4, np.random.default_rng(3), n_restarts=2)
for n, sc in scores.items():
    print(f"  N={n}: BIC={sc:.1f}")
print("selected N =", min(scores, key=scores.get))
