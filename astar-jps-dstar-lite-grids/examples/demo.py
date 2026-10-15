"""Demo: static planners on four map types, then navigation in unknown terrain."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from grid_pathfinding import Grid, astar, jps
from grid_pathfinding.benchmark import compare_navigation, compare_static


def main() -> None:
    print("== Small example (A* path, '*' marks route) ==")
    g = Grid.from_strings([
        "..........",
        "..####....",
        "..#..#....",
        "..#..####.",
        "..#.......",
        "....###...",
    ])
    r = astar(g, (0, 0), (9, 5))
    print(g.render(r.path))
    print(f"cost={r.cost:.3f} expanded={r.expanded}; JPS cost={jps(g, (0, 0), (9, 5)).cost:.3f}\n")

    print("== Static planners, 160x160 maps, 8 queries each (mean per query) ==")
    for kind in ("random20", "caves", "rooms", "maze"):
        print(f"[{kind}]")
        for st in compare_static(kind, 160, 8, seed=5):
            print(f"  {st.name:12s} {st.mean_ms:8.2f} ms  expanded={st.mean_expanded:9.1f}  cost={st.mean_cost:8.2f}")

    print("\n== Navigation in unknown terrain (60x60, 25% hidden obstacles, 10 episodes) ==")
    res = compare_navigation(60, 10, seed=21)
    for k, v in res.items():
        print(f"  {k:14s} {v:10.2f}")


if __name__ == "__main__":
    main()
