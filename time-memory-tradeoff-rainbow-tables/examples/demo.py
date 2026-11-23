"""Compare Hellman, distinguished-point and rainbow tables on a 20-bit space."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tmto import (  # noqa: E402
    DistinguishedPointAttack,
    HellmanAttack,
    OneWayFunction,
    PasswordSpace,
    RainbowAttack,
    hellman_coverage,
    measure_success,
    rainbow_coverage,
)

BITS = 18
N = 1 << BITS


def run(name, atk, f, theory=None):
    t0 = time.time()
    pre = atk.precompute()
    t1 = time.time()
    st = measure_success(atk, f, trials=200, seed=42)
    t2 = time.time()
    th = f"{theory:.3f}" if theory is not None else "  -  "
    print(
        f"{name:<22}{atk.memory:>8}{pre:>11}{st.success_rate:>9.3f}{th:>8}"
        f"{st.mean_online_evals:>12.0f}{st.false_alarms:>8}  {t1 - t0 + t2 - t1:5.1f}s"
    )


def main() -> None:
    f = OneWayFunction(BITS)
    print(f"Search space N = 2^{BITS} = {N}")
    print(f"{'attack':<22}{'memory':>8}{'precomp':>11}{'success':>9}{'theory':>8}"
          f"{"online f()":>12}{"false-al":>9}  time")
    # Hellman: m t^2 = N -> t=32, m=256, l tables
    run("Hellman l=8", HellmanAttack(f, 256, 32, 8, seed=1), f, hellman_coverage(N, 256, 32, 8))
    run("Hellman l=24", HellmanAttack(f, 256, 32, 24, seed=1), f, hellman_coverage(N, 256, 32, 24))
    run("DP d=5 l=24", DistinguishedPointAttack(f, 256, 5, 24, seed=1), f)
    run("Rainbow t=100", RainbowAttack(f, 6000, 100, 1, seed=1), f, rainbow_coverage(N, 6000, 100))
    run("Rainbow t=100 x2", RainbowAttack(f, 6000, 100, 2, seed=1), f)

    # password cracking illustration
    ps = PasswordSpace(3)  # 36^3 = 46656 passwords
    bits = 16
    g = OneWayFunction(bits)
    n = g.size
    atk = RainbowAttack(g, 4000, 40, 2, seed=9)
    atk.precompute()
    pw = "k7z"
    digest = g(ps.decode(pw) % n)
    found = atk.invert(digest)
    print()
    print(f"Password digest {digest:#06x}: recovered preimage index -> "
          f"{ps.encode(found % ps.size) if found is not None and found < ps.size else found}"
          f" (digest check: {found is not None and g(found) == digest})")


if __name__ == "__main__":
    main()
