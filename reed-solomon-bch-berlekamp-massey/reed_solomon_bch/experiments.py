"""Monte-Carlo experiments on decoder behaviour (all seeded)."""

from __future__ import annotations

import random
import time

from .bch import BCH
from .channel import add_bit_errors, add_symbol_errors
from .reed_solomon import DecodingFailure, ReedSolomon


def rs_outcome_rates(code: ReedSolomon, n_errors: int, trials: int,
                     seed: int = 0, n_erasures: int = 0) -> dict[str, float]:
    """Fraction of trials that are corrected / detected / miscorrected."""
    rng = random.Random(seed)
    ok = detected = wrong = 0
    for _ in range(trials):
        msg = [rng.randrange(code.field.size) for _ in range(code.k)]
        cw = code.encode(msg)
        positions = rng.sample(range(code.n), n_errors + n_erasures)
        noisy = list(cw)
        for p in positions:
            noisy[p] ^= rng.randrange(1, code.field.size)
        era = positions[:n_erasures]
        try:
            out = code.decode(noisy, era)
        except DecodingFailure:
            detected += 1
            continue
        if out == msg:
            ok += 1
        else:
            wrong += 1
    return {"corrected": ok / trials, "detected": detected / trials,
            "miscorrected": wrong / trials}


def bch_error_sweep(code: BCH, max_errors: int, trials: int,
                    seed: int = 0) -> list[tuple[int, float]]:
    """Decoding success rate for 0..max_errors random bit flips."""
    rng = random.Random(seed)
    rows = []
    for e in range(max_errors + 1):
        good = 0
        for _ in range(trials):
            msg = [rng.randrange(2) for _ in range(code.k)]
            noisy, _ = add_bit_errors(code.encode(msg), e, rng)
            try:
                good += code.decode(noisy) == msg
            except DecodingFailure:
                pass
        rows.append((e, good / trials))
    return rows


def decoder_timing(n: int, k: int, trials: int, seed: int = 0
                   ) -> dict[str, float]:
    """Average seconds per full-capability decode for BM vs PGZ."""
    out = {}
    for method in ("bm", "pgz"):
        code = ReedSolomon(n, k, m=8, method=method)
        rng = random.Random(seed)
        words = []
        for _ in range(trials):
            cw = code.encode([rng.randrange(256) for _ in range(k)])
            words.append(add_symbol_errors(cw, code.t, rng)[0])
        t0 = time.perf_counter()
        for w in words:
            code.decode(w)
        out[method] = (time.perf_counter() - t0) / trials
    return out
