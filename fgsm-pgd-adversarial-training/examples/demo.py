"""Natural vs adversarially trained models under FGSM and PGD attacks."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adv_attacks import (MLP, TrainConfig, adversarial_fit, attack_success_rate, epsilon_sweep,
                         fit, fgsm, gradient_check, make_fragile, make_spirals, pgd,
                         robust_accuracy, accuracy, train_test_split)


def main() -> None:
    t0 = time.time()
    X, y = make_fragile(1000, dim=10, eta=0.3, seed=1)
    Xtr, ytr, Xte, yte = train_test_split(X, y, 0.3, seed=0)
    eps = 0.4
    nat = fit(MLP((10, 16, 2), seed=1), Xtr, ytr, TrainConfig(epochs=60, lr=0.01))
    adv = adversarial_fit(MLP((10, 16, 2), seed=1), Xtr, ytr,
                          TrainConfig(epochs=60, lr=0.01, eps=eps, mix=1.0, pgd_steps=5))
    print(f"gradient check (max rel err): {gradient_check(nat, Xte, yte):.2e}")
    print(f"\nFragile-feature task, eps={eps} (L-inf)")
    print(f"{'model':<12}{'clean':>8}{'FGSM':>8}{'PGD-20':>8}{'ASR':>8}")
    for name, m in (("natural", nat), ("adv-trained", adv)):
        xa = pgd(m, Xte, yte, eps, steps=20)
        print(f"{name:<12}{accuracy(m, Xte, yte):8.3f}"
              f"{robust_accuracy(m, Xte, yte, eps, 'fgsm'):8.3f}"
              f"{robust_accuracy(m, Xte, yte, eps, 'pgd', steps=20):8.3f}"
              f"{attack_success_rate(m, Xte, yte, xa):8.3f}")

    print("\nPGD robust accuracy vs epsilon")
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
    rn = epsilon_sweep(nat, Xte, yte, grid, "pgd", steps=20)
    ra = epsilon_sweep(adv, Xte, yte, grid, "pgd", steps=20)
    print(f"{'eps':>6}{'natural':>10}{'adv-trained':>13}")
    for (e, a), (_, b) in zip(rn, ra):
        print(f"{e:6.2f}{a:10.3f}{b:13.3f}")

    Xs, ys = make_spirals(200, seed=1)
    Xs_tr, ys_tr, Xs_te, ys_te = train_test_split(Xs, ys, 0.3, seed=0)
    sp = fit(MLP((2, 32, 32, 2), seed=1), Xs_tr, ys_tr, TrainConfig(epochs=150, lr=0.01))
    print(f"\nSpirals: clean {accuracy(sp, Xs_te, ys_te):.3f}, "
          f"FGSM(0.3) {robust_accuracy(sp, Xs_te, ys_te, 0.3, 'fgsm'):.3f}, "
          f"PGD(0.3) {robust_accuracy(sp, Xs_te, ys_te, 0.3, 'pgd', steps=20):.3f}")
    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
