"""Demo: hull algorithms, Delaunay/Voronoi construction and applications."""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geometry_kit import (
    Delaunay, Voronoi, closest_pair, convex_hull, diameter, euclidean_mst,
    hull_area, largest_empty_circle, monotone_chain, nearest_site,
)


def pts(n, seed, disk=False):
    r = random.Random(seed)
    out = []
    while len(out) < n:
        x, y = r.uniform(-1, 1), r.uniform(-1, 1)
        if not disk or x * x + y * y <= 1:
            out.append((x, y))
    return out


def timed(f, *a):
    t = time.perf_counter()
    res = f(*a)
    return res, (time.perf_counter() - t) * 1000


def main() -> None:
    print("== Convex hull benchmark (ms) ==")
    print(f"{'n':>7} {'dist':>8} {'h':>5} " + " ".join(f"{m:>9}" for m in ("monotone", "graham", "jarvis", "quickhull")))
    for n, disk in [(2000, False), (2000, True), (20000, False)]:
        P = pts(n, 1, disk)
        row, h = [], 0
        for m in ("monotone", "graham", "jarvis", "quickhull"):
            if m == "jarvis" and n > 5000:
                row.append("      n/a")
                continue
            hull, ms = timed(convex_hull, P, m)
            h = len(hull)
            row.append(f"{ms:9.1f}")
        print(f"{n:>7} {'disk' if disk else 'square':>8} {h:>5} " + " ".join(row))
    hull = monotone_chain(pts(5000, 2, True))
    d, p, q = diameter(hull)
    print(f"disk hull: area={hull_area(hull):.4f} (pi={math.pi:.4f}), diameter={d:.4f}")

    print("\n== Delaunay triangulation ==")
    for n in (500, 2000, 5000):
        P = pts(n, 3)
        D, ms = timed(Delaunay, P)
        h = len(monotone_chain(P))
        print(f"n={n:5d}: {len(D.triangles):6d} triangles (expected {2*n-2-h}), {ms:7.1f} ms")
    P = pts(150, 4)
    print("empty-circle property holds:", Delaunay(P).is_delaunay())

    print("\n== Voronoi / Lloyd relaxation ==")
    box = (-1, -1, 1, 1)
    S = pts(100, 5)
    for it in range(6):
        v = Voronoi(S, box=box)
        a = v.cell_areas()
        mean = sum(a) / len(a)
        cv = math.sqrt(sum((x - mean) ** 2 for x in a) / len(a)) / mean
        print(f"iter {it}: total area={sum(a):.4f}, cell-area CV={cv:.3f}")
        S = v.lloyd_step()

    print("\n== Applications ==")
    P = pts(1500, 6)
    mst, ms = timed(euclidean_mst, P)
    print(f"EMST of 1500 points: weight={sum(w for *_, w in mst):.4f} in {ms:.1f} ms")
    cp, ms = timed(closest_pair, P)
    print(f"closest pair distance={cp[0]:.6f} ({ms:.1f} ms)")
    r, c = largest_empty_circle(P)
    print(f"largest empty circle: r={r:.4f} at ({c[0]:.3f}, {c[1]:.3f})")
    D = Delaunay(P)
    rng = random.Random(7)
    ok = 0
    for _ in range(200):
        q = (rng.uniform(-1, 1), rng.uniform(-1, 1))
        g = nearest_site(D, q, start=0)
        ok += math.dist(P[g], q) <= min(math.dist(p, q) for p in P) + 1e-12
    print(f"greedy Delaunay nearest-neighbour exact on {ok}/200 queries")


if __name__ == "__main__":
    main()
