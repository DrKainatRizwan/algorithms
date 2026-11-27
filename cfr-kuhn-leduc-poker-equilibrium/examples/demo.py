"""Demo: solve Kuhn and Leduc poker with several CFR variants."""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from counterfactual_regret import (CFRSolver, KUHN_VALUE, KuhnPoker, LeducHoldem,
                                   MCCFRSolver, build_tree, expected_value,
                                   exploitability, format_strategy, kuhn_alpha,
                                   kuhn_equilibrium)


def main() -> None:
    kuhn = build_tree(KuhnPoker())
    leduc = build_tree(LeducHoldem())
    print(f"Kuhn : {kuhn.n_nodes} nodes, {kuhn.n_infosets} infosets")
    print(f"Leduc: {leduc.n_nodes} nodes, {leduc.n_infosets} infosets")

    eq = kuhn_equilibrium(kuhn, 0.2)
    print(f"\nAnalytic Kuhn equilibrium (alpha=0.2): value={expected_value(kuhn, eq):+.5f} "
          f"(theory {KUHN_VALUE:+.5f}), exploitability={exploitability(kuhn, eq):.2e}")

    print("\nKuhn exploitability of the average strategy")
    print(f"{'variant':<9}{'100':>11}{'1000':>11}{'5000':>11}")
    for variant in ("vanilla", "plus", "linear", "dcfr"):
        solver = CFRSolver(kuhn, variant)
        row = []
        done = 0
        for target in (100, 1000, 5000):
            solver.run(target - done)
            done = target
            row.append(exploitability(kuhn, solver.average_strategy))
        print(f"{variant:<9}" + "".join(f"{x:11.2e}" for x in row))

    solver = CFRSolver(kuhn, "plus").run(5000)
    avg = solver.average_strategy
    print(f"\nCFR+ Kuhn strategy of player 0 (alpha = P(bet J) = {kuhn_alpha(kuhn, avg):.3f}, "
          f"value {expected_value(kuhn, avg):+.5f}):")
    print(format_strategy(kuhn, avg, 0))

    print("\nLeduc hold'em exploitability (chips, ante = 1)")
    print(f"{'variant':<9}{'50':>11}{'200':>11}{'500':>11}{'seconds':>10}")
    for variant in ("vanilla", "plus", "linear", "dcfr"):
        solver = CFRSolver(leduc, variant)
        t0 = time.time()
        row, done = [], 0
        for target in (50, 200, 500):
            solver.run(target - done)
            done = target
            row.append(exploitability(leduc, solver.average_strategy))
        print(f"{variant:<9}" + "".join(f"{x:11.2e}" for x in row)
              + f"{time.time() - t0:10.2f}")
    plus = CFRSolver(leduc, "plus").run(500).average_strategy
    print(f"Leduc game value under CFR+ average strategy: {expected_value(leduc, plus):+.4f} "
          "(literature: about -0.0856)")

    print("\nMonte-Carlo CFR on Leduc (exploitability)")
    for mode, iters in (("external", 20000), ("external", 100000), ("outcome", 100000)):
        t0 = time.time()
        s = MCCFRSolver(leduc, mode, seed=0).run(iters)
        print(f"  {mode:<9}{iters:>8} iterations: "
              f"{exploitability(leduc, s.average_strategy):.4f}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
