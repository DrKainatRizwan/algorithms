"""Demonstration of PRNG state-recovery attacks and the statistical battery."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prng_attacks import (  # noqa: E402
    LCG, MT19937, JavaRandom, Xorshift128Plus, bit_matrix, clone_mt19937,
    lsb_stream, recover_java_random_state, recover_lcg_parameters, recover_seed,
    recover_xorshift128plus, rewind_twist, run_battery, temper, untemper,
)


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"    [{label}: {(time.perf_counter() - t0) * 1000:.1f} ms]")
    return out


def main() -> None:
    print("== Mersenne Twister ==")
    victim = MT19937(20240229)
    seen = victim.outputs(624)
    clone = timed("clone from 624 outputs", lambda: clone_mt19937(seen))
    ok = clone.outputs(1000) == victim.outputs(1000)
    print(f"  cloned from 624 outputs, next 1000 predicted correctly: {ok}")

    first = MT19937(31337).outputs(624)
    seed = timed("seed recovery", lambda: recover_seed(first))
    print(f"  seed recovered from first block: {seed}")

    g = MT19937(99)
    b1, b2 = g.outputs(624), g.outputs(624)
    prev = rewind_twist([untemper(o) for o in b2])
    match = sum(temper(w) == o for w, o in zip(prev[1:], b1[1:]))
    print(f"  rewound one twist: {match}/623 earlier outputs reproduced")

    print("\n== xorshift128+ (128 LSBs) ==")
    xs = Xorshift128Plus(0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9)
    bits = [o & 1 for o in xs.outputs(160)]
    cl = timed("GF(2) solve", lambda: recover_xorshift128plus(bits))
    print(f"  full 64-bit outputs predicted: {cl.outputs(1000) == xs.outputs(1000)}")

    print("\n== LCG ==")
    lcg = LCG(2024, 1103515245, 12345, 2 ** 31)
    states = [lcg.step() for _ in range(40)]
    print(f"  (m, a, c) recovered from 40 states: {recover_lcg_parameters(states)}")
    jr = JavaRandom(42)
    obs = jr.outputs(2)
    rec = timed("2^16 brute force", lambda: recover_java_random_state(obs))
    print(f"  java.util.Random cloned from 2 outputs: {rec.outputs(10) == jr.outputs(10)}")

    print("\n== Statistical battery (p >= 0.001 passes) ==")
    streams = {
        "MT19937 full words": bit_matrix(MT19937(1).outputs(8000), 32),
        "MT19937 LSB": lsb_stream(MT19937(1).outputs(8000)),
        "xorshift128+ full words": bit_matrix(Xorshift128Plus(0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9).outputs(4000), 64),
        "xorshift128+ LSB": lsb_stream(Xorshift128Plus(0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9).outputs(8000)),
        "LCG(2^32) LSB": lsb_stream(LCG(1, 1664525, 1013904223, 2 ** 32).outputs(8000)),
        "JavaRandom LSB": lsb_stream(JavaRandom(7).outputs(8000)),
    }
    for name, bits in streams.items():
        res = run_battery(bits)
        failed = [r.name for r in res if not r.passed()]
        print(f"  {name:26s} {len(res) - len(failed)}/{len(res)} pass"
              + (f"   failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
