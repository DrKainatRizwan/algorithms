"""Demo: correctness cross-check and timing of the matchers."""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pattern_search import (AhoCorasick, SearchEngine, kmp_search, naive_search,
                            rabin_karp, rabin_karp_multi, z_search)


def timed(f, *a):
    t = time.perf_counter()
    r = f(*a)
    return r, (time.perf_counter() - t) * 1000


def main() -> None:
    eng = SearchEngine(["he", "she", "his", "hers"])
    print("highlight:", eng.highlight("ushers and his history"))
    sec = SearchEngine(["password", "token", "api_key"], case_sensitive=False, whole_word=True)
    print("redact   :", sec.redact("Send the Password and API_KEY, not tokens or token."))

    rng = random.Random(42)
    text = "".join(rng.choice("acgt") for _ in range(200_000))
    pat = "".join(rng.choice("acgt") for _ in range(6))
    print(f"\nsingle pattern ({len(pat)} chars) in {len(text)} chars")
    ref, t = timed(naive_search, text, pat)
    print(f"  naive        {t:8.1f} ms  hits={len(ref)}")
    for name, f in (("KMP", kmp_search), ("Z", z_search), ("Rabin-Karp", rabin_karp)):
        r, t = timed(f, text, pat)
        assert r == ref
        print(f"  {name:<12} {t:8.1f} ms  hits={len(r)}")

    print("\nmulti pattern, 200 patterns of length 6-10")
    pats = list({"".join(rng.choice("acgt") for _ in range(rng.randint(6, 10))) for _ in range(200)})
    ac, t = timed(AhoCorasick, pats)
    print(f"  AC build     {t:8.1f} ms  states={ac.num_states()}")
    ms, t = timed(ac.find_all, text)
    print(f"  AC scan      {t:8.1f} ms  hits={len(ms)}")
    rk, t = timed(rabin_karp_multi, text, pats)
    print(f"  RK multi     {t:8.1f} ms  hits={sum(len(v) for v in rk.values())}")
    t0 = time.perf_counter()
    nv = sum(len(kmp_search(text, p)) for p in pats)
    print(f"  KMP x{len(pats)}    {(time.perf_counter() - t0) * 1000:8.1f} ms  hits={nv}")
    assert nv == len(ms)


if __name__ == "__main__":
    main()
