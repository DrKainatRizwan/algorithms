"""Compare Dijkstra, bidirectional Dijkstra, ALT and contraction hierarchies."""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from road_routing import (ALT, ContractionHierarchy, bidirectional_dijkstra, dijkstra,
                          grid_road_network, largest_scc_subgraph, random_pairs, run_queries)


def main() -> None:
    g = largest_scc_subgraph(grid_road_network(40, 40, seed=1))
    print(f"road network: {g.n} vertices, {g.m} directed edges")
    t = time.perf_counter()
    alt = ALT(g, k=8, seed=0)
    print(f"ALT preprocessing (8 landmarks): {time.perf_counter() - t:.2f}s")
    t = time.perf_counter()
    ch = ContractionHierarchy(g)
    print(f"CH preprocessing: {time.perf_counter() - t:.2f}s, stats={ch.stats()}")

    pairs = random_pairs(g, 200, seed=42)
    algos = {"dijkstra": lambda s, q: dijkstra(g, s, q),
             "bidirectional": lambda s, q: bidirectional_dijkstra(g, s, q),
             "ALT": alt.query, "CH": ch.query}
    ref = None
    print(f"\n{'algorithm':<15}{'settled/query':>15}{'ms/query':>12}{'exact':>8}")
    for name, fn in algos.items():
        r = run_queries(fn, pairs)
        ref = ref or r["dists"]
        ok = all(abs(a - b) < 1e-9 for a, b in zip(ref, r["dists"]))
        print(f"{name:<15}{r['settled']:>15.1f}{r['ms']:>12.3f}{str(ok):>8}")
    s, q = pairs[0]
    r = ch.query(s, q)
    print(f"\nexample route {s}->{q}: cost {r.dist:.3f}, {len(r.path)} vertices")


if __name__ == "__main__":
    main()
