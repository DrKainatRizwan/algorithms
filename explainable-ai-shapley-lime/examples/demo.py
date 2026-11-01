"""Compare Shapley estimators and LIME on a logistic model and a random forest."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from xai_toolkit import *

X, y, w = make_tabular(800, 8, 4, seed=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, seed=0)
lg = LogisticModel().fit(Xtr, ytr)
mlp = MLPClassifier(hidden=12, epochs=60, seed=0).fit(Xtr, ytr)
rf = RandomForest(20, 5, seed=0).fit(Xtr, ytr)
for name, m in [("logistic", lg), ("mlp", mlp)]:
    print(f"{name} test acc: {np.mean((m.predict(Xte) > .5) == yte):.3f}")
print(f"forest test acc: {np.mean((rf.predict(Xte) > .5) == yte):.3f}")
print("true weights:", np.round(w, 2))

bg = Xtr[:50]; x = Xte[0]
print("\n== Estimator accuracy on the MLP (instance 0, d=8) ==")
vf = ValueFunction(mlp.predict, x, bg)
t = time.time(); ex = exact_shapley(vf); te = time.time() - t
print(f"exact      calls={vf.calls:4d} time={te*1000:6.1f}ms phi={np.round(ex,3)}")
for budget in (30, 100, 300):
    vf2 = ValueFunction(mlp.predict, x, bg)
    ks = kernel_shap(vf2, n_samples=budget, seed=0)
    vf3 = ValueFunction(mlp.predict, x, bg)
    ps = permutation_shapley(vf3, budget // 9 + 1, seed=0)
    print(f"budget~{budget:3d}  kernelSHAP maxerr={np.abs(ks-ex).max():.4f} (calls {vf2.calls})  "
          f"permutation maxerr={np.abs(ps-ex).max():.4f} (calls {vf3.calls})")

print("\n== Linear model: Shapley vs closed form vs LIME ==")
vfl = ValueFunction(lg.decision, x, bg)
sh = exact_shapley(vfl)
print("closed-form match:", np.allclose(sh, linear_shap(lg.w, x, bg)))
lime = LimeTabular(Xtr, n_samples=3000, seed=0).explain(x, lg.decision)
print("LIME coef*(x-mean):", np.round(lime.coef * (x - bg.mean(0)), 3), f"R2={lime.r2:.4f}")
print("Shapley          :", np.round(sh, 3))

print("\n== Forest: local attributions and faithfulness ==")
vfr = ValueFunction(rf.predict, x, bg)
phi = kernel_shap(vfr, 200, seed=0)
print("efficiency gap:", f"{efficiency_gap(phi, vfr.base, vfr.full):.2e}")
fill = bg.mean(0)
sgn = 1 if rf.predict(x[None])[0] >= vfr.base else -1
g = deletion_auc(deletion_curve(lambda Z: sgn * rf.predict(Z), x, sgn * phi, fill, signed=True))
r = deletion_auc(deletion_curve(lambda Z: sgn * rf.predict(Z), x, -sgn * phi, fill, signed=True))
print(f"deletion AUC (lower=better, supportive features removed first): shap-ordered {g:.4f} vs reversed {r:.4f}")
print("top-3 features:", np.argsort(-np.abs(phi))[:3])

print("\n== Stability across seeds (mean Spearman) ==")
print("LIME(n=300)  :", round(explanation_stability(lambda s: LimeTabular(Xtr, n_samples=300, seed=s).explain(x, mlp.predict).coef, list(range(6))), 3))
print("KernelSHAP(40):", round(explanation_stability(lambda s: kernel_shap(ValueFunction(mlp.predict, x, bg), 40, seed=s), list(range(6))), 3))

print("\n== Integrated gradients (MLP) ==")
ig = integrated_gradients(mlp.gradient, x, bg.mean(0), 200)
print("sum IG:", round(ig.sum(), 4), "f(x)-f(base):", round(mlp.predict(x[None])[0] - mlp.predict(bg.mean(0)[None])[0], 4))
print("rank corr with Shapley:", round(rank_correlation(ig, ex), 3))
