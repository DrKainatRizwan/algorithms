"""Regret of online learners across adversarial, stochastic and regression streams."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from online_learning import *  # noqa: F401,F403
from online_learning import (AdaGrad, Ball, Box, FTRL, FollowTheLeader,
                             FollowThePerturbedLeader, Hedge, OnlineGradientDescent,
                             OnlineNewtonStep, Simplex)


def reg(learner, losses, dom):
    return regret(run_online(learner, losses), losses, dom)


def table(title, header, rows):
    print(f"\n== {title} ==")
    print(header)
    for r in rows:
        print(r)


def main() -> None:
    t0 = time.time()
    H = [200, 800, 3200]

    # 1. FTL adversary
    dom = Box(1, -1, 1)
    rows, series = [], {"FTL": [], "FTRL": [], "OGD": []}
    for T in H:
        ls = ftl_adversary(T)
        series["FTL"].append(reg(FollowTheLeader(dom), ls, dom))
        series["FTRL"].append(reg(FTRL(dom, 1 / np.sqrt(T)), ls, dom))
        series["OGD"].append(reg(OnlineGradientDescent(dom, 1.0), ls, dom))
    for k, v in series.items():
        rows.append(f"{k:6s} " + " ".join(f"{x:9.1f}" for x in v) + f"   exponent {growth_exponent(H, v):.2f}")
    table("FTL adversary (1-D, linear losses)", "       " + " ".join(f"T={T:<7d}" for T in H), rows)

    # 2. experts
    n = 10
    rows, series = [], {"Hedge": [], "FPL": [], "FTL": []}
    for T in H:
        ls = expert_losses(T, n, seed=1)
        series["Hedge"].append(reg(Hedge(n, hedge_eta(n, T)), ls, Simplex(n)))
        series["FPL"].append(reg(FollowThePerturbedLeader(n, np.sqrt((1 + np.log(n)) / T), 0), ls, Simplex(n)))
        series["FTL"].append(reg(FollowTheLeader(Simplex(n)), ls, Simplex(n)))
    for k, v in series.items():
        rows.append(f"{k:6s} " + " ".join(f"{x:9.1f}" for x in v))
    rows.append("bound  " + " ".join(f"{hedge_regret_bound(n, T):9.1f}" for T in H))
    table(f"Experts (N={n}, stochastic losses)", "       " + " ".join(f"T={T:<7d}" for T in H), rows)

    # 3. strongly convex regression
    dom = Ball(5, 1.0)
    rows, series = [], {"OGD const": [], "OGD sqrt": [], "OGD 1/(Ht)": [], "ONS": []}
    for T in H:
        ls = regression_stream(T, 5, seed=3, lam=0.5)
        series["OGD const"].append(reg(OnlineGradientDescent(dom, 0.1, "const"), ls, dom))
        series["OGD sqrt"].append(reg(OnlineGradientDescent(dom, 0.5, "sqrt"), ls, dom))
        series["OGD 1/(Ht)"].append(reg(OnlineGradientDescent(dom, schedule="strong"), ls, dom))
        ls2 = regression_stream(T, 5, seed=3)
        g = OnlineNewtonStep.default_gamma(min(l.exp_concavity for l in ls2), 2.5, dom.diameter())
        series["ONS"].append(reg(OnlineNewtonStep(dom, g, 1 / (g * dom.diameter()) ** 2), ls2, dom))
    for k, v in series.items():
        rows.append(f"{k:11s} " + " ".join(f"{x:8.2f}" for x in v) + f"   exponent {growth_exponent(H, v):.2f}")
    table("Ridge regression stream (ONS uses the unregularised stream)",
          "            " + " ".join(f"T={T:<6d}" for T in H), rows)

    # 4. AdaGrad vs OGD logistic
    dom = Box(8, -2, 2)
    T = 3200
    ls = logistic_stream(T, 8, seed=2)
    rows = [f"AdaGrad {reg(AdaGrad(dom, 0.5), ls, dom):9.2f}",
            f"OGD sqrt {reg(OnlineGradientDescent(dom, 0.5, 'sqrt'), ls, dom):8.2f}"]
    table(f"Logistic stream, box [-2,2]^8, T={T}", "learner  regret", rows)
    print(f"\nElapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
