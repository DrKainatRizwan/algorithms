"""Demo: RS and BCH decoding, erasures, interleaving and benchmarks."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reed_solomon_bch import (BCH, BlockInterleaver, ReedSolomon, add_burst,
                              add_symbol_errors)
from reed_solomon_bch.experiments import (bch_error_sweep, decoder_timing,
                                          rs_outcome_rates)


def main() -> None:
    rng = random.Random(2024)

    print("== RS(255, 223) over GF(2^8), t = 16 ==")
    rs = ReedSolomon(255, 223)
    msg = [rng.randrange(256) for _ in range(223)]
    noisy, pos = add_symbol_errors(rs.encode(msg), 16, rng)
    _, found = rs.decode_full(noisy)
    print(f"injected 16 symbol errors; located {len(found)}; "
          f"recovered={rs.decode(noisy) == msg}")

    print("\n== Errors + erasures on RS(63, 47) (nsym = 16) ==")
    rs2 = ReedSolomon(63, 47, m=6)
    for errs, eras in [(8, 0), (4, 8), (0, 16), (5, 6)]:
        m2 = [rng.randrange(64) for _ in range(47)]
        cw = rs2.encode(m2)
        p = rng.sample(range(63), errs + eras)
        bad = list(cw)
        for i in p:
            bad[i] ^= rng.randrange(1, 64)
        print(f"  errors={errs:2d} erasures={eras:2d} (2e+s="
              f"{2 * errs + eras:2d}) -> {rs2.decode(bad, p[:eras]) == m2}")

    print("\n== Beyond capacity: RS(31, 21), t = 5, 2000 trials each ==")
    rs3 = ReedSolomon(31, 21, m=5)
    for e in (5, 6, 8, 12):
        r = rs_outcome_rates(rs3, e, 2000, seed=e)
        print(f"  {e:2d} errors: corrected={r['corrected']:.3f} "
              f"detected={r['detected']:.3f} miscorrected={r['miscorrected']:.3f}")

    print("\n== BCH codes ==")
    for m, t in [(4, 2), (6, 3), (8, 8)]:
        c = BCH(m, t)
        print(f"  BCH({c.n},{c.k}) t={t}, designed distance {c.designed_distance}")
    print("  BCH(15,7) exact minimum distance:", BCH(4, 2).minimum_distance())
    c = BCH(6, 3)
    print("  BCH(63,45) success vs #bit errors (300 trials):")
    for e, rate in bch_error_sweep(c, 6, 300, seed=5):
        print(f"    {e} errors -> {rate:.3f}")

    print("\n== Burst errors with a block interleaver ==")
    depth = 8
    il = BlockInterleaver(depth, 31)
    msgs = [[rng.randrange(32) for _ in range(21)] for _ in range(depth)]
    flat = [s for mm in msgs for s in rs3.encode(mm)]
    rx = il.deinterleave(add_burst(il.interleave(flat), 40, 40, rng, 32))
    ok = sum(rs3.decode(rx[i * 31:(i + 1) * 31]) == msgs[i] for i in range(depth))
    print(f"  40-symbol burst: {ok}/{depth} codewords recovered with interleaving")
    raw = add_burst(flat[:31], 0, 10, rng, 32)
    try:
        rs3.decode(raw)
        print("  10-symbol burst without interleaving: decoded")
    except Exception:
        print("  10-symbol burst without interleaving: decoding failure")

    print("\n== Decoder timing, RS(255, 223), 16 errors ==")
    tm = decoder_timing(255, 223, 5)
    print(f"  Berlekamp-Massey: {tm['bm'] * 1e3:.1f} ms/decode")
    print(f"  PGZ (matrix)    : {tm['pgz'] * 1e3:.1f} ms/decode")


if __name__ == "__main__":
    main()
