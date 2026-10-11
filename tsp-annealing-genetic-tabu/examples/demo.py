"""Compare nearest-neighbour, 2-opt, annealing, GA and tabu search on random TSP instances."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsp_metaheuristics import compare, held_karp, random_instance, simulated_annealing, \
    genetic_algorithm, tabu_search


def main() -> None:
    small = random_instance(11, seed=1)
    opt, _ = held_karp(small)
    print(f"Exact optimum (Held-Karp, n=11): {opt:.4f}")
    for name, r in (("annealing", simulated_annealing(small, 20000)),
                    ("genetic", genetic_algorithm(small, 40, 60)),
                    ("tabu", tabu_search(small, 80))):
        print(f"  {name:10s} {r.length:.4f}  gap {100 * (r.length / opt - 1):.2f}%")

    for n in (50, 100):
        inst = random_instance(n, seed=7)
        print(f"\nRandom instance n={n} (mean of 3 seeds)")
        print(f"{'method':18s}{'mean':>9s}{'best':>9s}{'sec':>8s}")
        for k, v in compare(inst).items():
            print(f"{k:18s}{v['mean']:9.4f}{v['best']:9.4f}{v['seconds']:8.2f}")


if __name__ == "__main__":
    main()
