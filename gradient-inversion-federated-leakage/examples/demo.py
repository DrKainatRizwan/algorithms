"""Gradient inversion demo: analytic leak, optimisation attack, batch size and defences."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gradient_leakage import (analytic_reconstruct, attack_batch, build_victim,  # noqa: E402
                              mse, psnr, recover_label, ssim_global)


def main() -> None:
    model, X, y, _ = build_victim(seed=0)
    print(f"Victim: 64-32-4 tanh MLP, train loss {model.loss(X, y):.3f}\n")

    g = model.gradients(X[:1], y[:1])
    rec = analytic_reconstruct(g)
    print("1) Analytic leak from a single-example gradient")
    print(f"   label recovered: {recover_label(g)} (true {y[0]}), MSE {mse(rec, X[0]):.2e}, "
          f"SSIM {ssim_global(rec, X[0]):.4f}\n")

    print("2) Gradient-matching attack vs batch size (mean PSNR dB after matching)")
    for b in (1, 2, 3, 4):
        r = attack_batch(model, X, y, b, seed=1, iters=800, restarts=3)
        print(f"   B={b}: PSNR {r['psnr']:6.2f} dB  MSE {r['mse']:.2e}")
    print()

    print("3) Defences against the B=1 attack")
    cases = [("none", 0.0), ("noise", 0.01), ("noise", 0.1), ("noise", 1.0),
             ("sparsify", 0.5), ("sparsify", 0.05), ("quantize", 4)]
    for name, s in cases:
        r = attack_batch(model, X, y, 1, seed=1, defense=name, strength=s, iters=500, restarts=2)
        print(f"   {name:9s} strength={s:<5} PSNR {r['psnr']:6.2f} dB")


if __name__ == "__main__":
    main()
