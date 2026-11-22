"""Demo: correlation, fast correlation and algebraic attacks on LFSR keystream generators."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stream_cryptanalysis import (  # noqa: E402
    FilterGenerator, GeffeGenerator, LFSR, add_noise, algebraic_attack, berlekamp_massey,
    correlation_attack, fast_correlation_attack, geffe_attack, required_length, unknown_count,
)
from stream_cryptanalysis.correlation import brute_force_scores, correlation_scores  # noqa: E402


def main() -> None:
    print("== 1. Geffe generator (LFSR lengths 11/12/13) ==")
    g = GeffeGenerator(LFSR.primitive(11), LFSR.primitive(12), LFSR.primitive(13))
    states = (1234, 3001, 6000)
    z = g.keystream(states, 400)
    t = time.perf_counter()
    rec = geffe_attack(g, z)
    print(f"divide-and-conquer recovered {rec} (true {states}) in {time.perf_counter()-t:.3f}s")
    print(f"search cost 2^12 + 2^13 + 2^11 vs joint 2^{11+12+13}")
    zl = g.keystream(states, 2 * g.linear_complexity_bound + 10)
    L, _ = berlekamp_massey(list(zl))
    print(f"Berlekamp-Massey: linear complexity {L} (bound {g.linear_complexity_bound}), "
          f"needs only {2*L} keystream bits")

    print("\n== 2. WHT scoring vs brute force (n=12, N=300, p=0.2) ==")
    r = LFSR.primitive(12)
    rng = np.random.default_rng(7)
    zz = add_noise(r.stream(2025, 300), 0.2, rng)
    t = time.perf_counter(); a = correlation_scores(r, zz); t_wht = time.perf_counter() - t
    t = time.perf_counter(); b = brute_force_scores(r, zz); t_bf = time.perf_counter() - t
    print(f"identical={np.array_equal(a, b)}  WHT {t_wht*1e3:.1f} ms  brute force {t_bf*1e3:.0f} ms  "
          f"speed-up x{t_bf/t_wht:.0f}")
    res = correlation_attack(r, zz)
    print(f"recovered state {res.state} (true 2025), score {res.score} vs runner-up {res.runner_up_score}")
    print(f"heuristic keystream length for n=12,p=0.2: {required_length(12, 0.2)}")

    print("\n== 3. Fast correlation attack on a 31-bit LFSR (x^31 + x^3 + 1) ==")
    r31 = LFSR.primitive(31)
    st = 0x1234567
    print("noise p | N=5000  N=20000  N=100000   (successes out of 5 seeds)")
    for p in (0.2, 0.3, 0.4, 0.45):
        row = []
        for N in (5000, 20000, 100000):
            ok = 0
            for sd in range(5):
                rg = np.random.default_rng(sd)
                zn = add_noise(r31.stream(st, N), p, rg)
                ok += fast_correlation_attack(r31, zn, p).state == st
            row.append(f"{ok}/5")
        print(f"  {p:.2f}  |  " + "     ".join(row))
    print("(plain correlation search would need 2^31 candidates)")

    print("\n== 4. Algebraic attack on a filter generator ==")
    for n, pos, anf in (
        (12, (0, 2, 5, 7, 11), ((0,), (1, 2), (3, 4), (2, 3))),
        (14, (0, 3, 4, 8, 13), ((1,), (0, 2), (3, 4), (2, 3))),
    ):
        f = FilterGenerator(LFSR.primitive(n), pos, anf)
        stt = (0x9A5 * 7) % (1 << n) or 1
        k = f.keystream(stt, 400)
        t = time.perf_counter()
        got = algebraic_attack(f, k)
        print(f"n={n} degree={f.degree} unknowns={unknown_count(n, f.degree)} "
              f"recovered={got == stt} in {time.perf_counter()-t:.2f}s")


if __name__ == "__main__":
    main()
