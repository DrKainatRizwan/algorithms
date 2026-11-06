"""Backdoor detection demo: poison, train, detect, sanitise, retrain."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backdoor_detection import (activation_clustering, attack_success_rate, clean_accuracy,
                                evaluate_detection, make_dataset, poison_dataset,
                                sanitize_and_retrain, spectral_signature, train)

C, TARGET = 4, 0
Xtr, ytr = make_dataset(300, C, seed=1)
Xte, yte = make_dataset(100, C, seed=2)

print("rate  | ASR(bd) acc(bd) | AC prec/rec | SS prec/rec | ASR(AC) acc(AC)")
for rate in (0.05, 0.10, 0.20):
    pd = poison_dataset(Xtr, ytr, TARGET, rate, seed=3)
    m = train(pd.X, pd.y, C, seed=0)
    asr, acc = attack_success_rate(m, Xte, yte, TARGET), clean_accuracy(m, Xte, yte)
    ac = activation_clustering(m, pd.X, pd.y, C)
    ss = spectral_signature(m, pd.X, pd.y, C, eps=rate)
    ea, es = evaluate_detection(ac.flagged, pd.is_poison), evaluate_detection(ss.flagged, pd.is_poison)
    m2 = sanitize_and_retrain(pd.X, pd.y, ac.flagged, C, seed=0)
    print(f"{rate:.2f}  | {asr:6.3f} {acc:7.3f} | {ea['precision']:.2f}/{ea['recall']:.2f}"
          f"   | {es['precision']:.2f}/{es['recall']:.2f}   | "
          f"{attack_success_rate(m2, Xte, yte, TARGET):6.3f} {clean_accuracy(m2, Xte, yte):7.3f}")
    if rate == 0.10:
        for r in ac.reports:
            print(f"   class {r.label}: n={r.n} sil={r.silhouette:.2f} "
                  f"rel_size={r.relative_size:.2f} suspicious={r.suspicious}")
