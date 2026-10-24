"""Demo: compare MST algorithms, union-find heuristics and tree sensitivity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mst_suite import (is_minimum_spanning_forest, kruskal, random_connected_graph,
                       second_best_mst, sensitivity)
from mst_suite.benchmark import (adversarial_chain_steps, compare_algorithms, density_sweep,
                                 unionfind_steps)


def main() -> None:
    print("== Algorithm agreement (n=2000, m=10000) ==")
    g = random_connected_graph(2000, 10000, seed=1)
    for name, (ms, w) in compare_algorithms(g).items():
        print(f"{name:11s} {ms:8.2f} ms  weight={w:.6f}")
    t = kruskal(g)
    print("cycle-property certificate:", is_minimum_spanning_forest(g, t.edges))

    print("\n== Density sweep (n=1000), ms ==")
    print(f"{'m':>7s} " + " ".join(f"{k:>11s}" for k in ("kruskal", "prim", "prim_dense", "boruvka")))
    for m, row in density_sweep(1000, [2, 8, 32]):
        print(f"{m:7d} " + " ".join(f"{row[k][0]:11.2f}" for k in ("kruskal", "prim", "prim_dense", "boruvka")))

    print("\n== Union-find steps inside Kruskal (n=2000, m=10000) ==")
    for (u, c), s in unionfind_steps(g).items():
        print(f"union={u:5s} compression={c:9s} steps={s}")
    print("\nChain workload n=2000 steps:")
    for u, c in [("naive", "none"), ("rank", "none"), ("naive", "full"), ("rank", "full")]:
        print(f"  {u:5s}/{c:5s} {adversarial_chain_steps(2000, u, c)}")

    print("\n== Sensitivity and second-best tree (n=300, m=1200) ==")
    g2 = random_connected_graph(300, 1200, seed=4)
    t2 = kruskal(g2)
    sb = second_best_mst(g2, t2.edges)
    print(f"MST weight {t2.weight:.5f}, second best {sb.weight:.5f} (swap {sb.removed}->{sb.added})")
    sens = sensitivity(g2, t2.edges)
    tight = sorted(sens, key=lambda e: sens[e] - g2.edges[e][2])[:3]
    for e in tight:
        print(f"  tree edge {e}: w={g2.edges[e][2]:.4f} stays in MST up to {sens[e]:.4f}")


if __name__ == "__main__":
    main()
