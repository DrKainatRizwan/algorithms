"""End-to-end demo: attack a deliberately overfit model, then apply defences."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_attacks import (CalibratedAttack, LabelOnly, LabelOnlyAttack, MLP, OutputNoise, Rounding,
                             ShadowAttack, TopK, TrainConfig, auc, invert_all_classes, invert_blackbox,
                             make_world, score_attack, summarize, train, zcdp_epsilon)
from privacy_attacks.metrics import tpr_at_fpr

t0 = time.time()
w = make_world(0)
print(f"World: {w.n_classes} classes, dim {w.dim}, {len(w.y_in)} members / {len(w.y_out)} non-members\n")

shadow = ShadowAttack(n_shadows=6, set_size=100, seed=0).fit(w.X_shadow, w.y_shadow, w.n_classes)
calib = CalibratedAttack(shadow.shadows)
lab = LabelOnlyAttack(n_queries=30, sigma=1.0)


def fit(cfg: TrainConfig) -> MLP:
    m = MLP(w.dim, w.n_classes, (64,), seed=1)
    train(m, w.X_in, w.y_in, cfg, w.X_out[:50], w.y_out[:50])
    return m


def row(name: str, api, model: MLP) -> None:
    attacks = {
        "loss": (score_attack(api, w.X_in, w.y_in), score_attack(api, w.X_out, w.y_out)),
        "shadow": (shadow.score(api, w.X_in, w.y_in), shadow.score(api, w.X_out, w.y_out)),
        "calib": (calib.score(api, w.X_in, w.y_in), calib.score(api, w.X_out, w.y_out)),
        "label": (lab.score(api, w.X_in, w.y_in), lab.score(api, w.X_out, w.y_out)),
    }
    s = " ".join(f"{k}={auc(*v):.3f}" for k, v in attacks.items())
    tp = tpr_at_fpr(*attacks["calib"], 0.05)
    print(f"{name:<22} test_acc={model.accuracy(w.X_test, w.y_test):.3f}  AUC: {s}  calib TPR@5%FPR={tp:.2f}")


print("== Membership inference: training-time defences ==")
cfgs = {
    "undefended": TrainConfig(),
    "L2 (0.05)": TrainConfig(l2=0.05),
    "early stopping": TrainConfig(epochs=300, patience=5, lr=0.005),
    "DP-SGD (sigma=1)": TrainConfig(epochs=40, clip_norm=1.0, noise_multiplier=1.0),
    "DP-SGD (sigma=2)": TrainConfig(epochs=40, clip_norm=1.0, noise_multiplier=2.0),
}
models = {}
for name, cfg in cfgs.items():
    models[name] = fit(cfg)
    row(name, models[name], models[name])
steps = 40 * int(np.ceil(100 / 32))
print(f"(zCDP-style eps bound, sigma=2, no amplification: {zcdp_epsilon(steps, 2.0):.0f})\n")

print("== Membership inference: output-side defences on the undefended model ==")
base = models["undefended"]
for name, api in {"full probabilities": base, "top-1 only": TopK(base, 1), "rounded (1 dec.)": Rounding(base, 1),
                  "Laplace noise 0.1": OutputNoise(base, 0.1, seed=0), "label only": LabelOnly(base)}.items():
    row(name, api, base)

print("\n== Model inversion (class reconstruction vs ground-truth prototype) ==")
for name in ("undefended", "L2 (0.05)", "DP-SGD (sigma=1)"):
    r = invert_all_classes(models[name], w.protos)
    print(f"{name:<18} white-box: mean corr={np.mean([q['corr'] for q in r]):.3f} "
          f"classes identified={int(sum(q['identified'] for q in r))}/{w.n_classes}")
r = invert_all_classes(base, w.protos, blackbox=True, steps=200)
print(f"{'undefended':<18} black-box: mean corr={np.mean([q['corr'] for q in r]):.3f} "
      f"classes identified={int(sum(q['identified'] for q in r))}/{w.n_classes}")
r = invert_all_classes(OutputNoise(base, 0.1, seed=0), w.protos, blackbox=True, steps=200)
print(f"{'noisy API':<18} black-box: mean corr={np.mean([q['corr'] for q in r]):.3f} "
      f"classes identified={int(sum(q['identified'] for q in r))}/{w.n_classes}")
print(f"\nDone in {time.time() - t0:.1f}s")
