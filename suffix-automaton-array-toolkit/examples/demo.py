"""Demo: build indexes on random DNA-like text and compare construction costs."""
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from suffix_toolkit import (SuffixAutomaton, SuffixArrayIndex, build_suffix_array,
                            build_suffix_array_naive, count_distinct_substrings,
                            longest_repeated_substring, longest_common_substring,
                            kth_substring)


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def main() -> None:
    print("banana:", build_suffix_array("banana"), "| LRS:", longest_repeated_substring("banana"),
          "| 5th substring:", kth_substring("banana", 5))
    rng = random.Random(42)
    print("\n n      SA(doubling)  SA(naive)   SAM build   SAM states  distinct")
    for n in (1000, 5000, 20000):
        s = "".join(rng.choice("ACGT") for _ in range(n))
        _, t1 = timed(lambda: build_suffix_array(s))
        _, t2 = timed(lambda: build_suffix_array_naive(s)) if n <= 5000 else (None, float("nan"))
        sam, t3 = timed(lambda: SuffixAutomaton(s))
        d = count_distinct_substrings(s)
        assert d == sam.distinct_substrings()
        print(f" {n:<6} {t1*1000:9.1f} ms {t2*1000:9.1f} ms {t3*1000:9.1f} ms {len(sam):9d} {d:11d}")

    s = "".join(rng.choice("ACGT") for _ in range(20000))
    idx, sam = SuffixArrayIndex(s), SuffixAutomaton(s)
    pat = s[777:785]
    print(f"\npattern {pat}: SA count={idx.count(pat)}, SAM count={sam.count(pat)}, "
          f"first at {sam.first_occurrence(pat)}")
    a, b = s[:10000], s[5000:15000]
    print("longest common substring length:", len(longest_common_substring(a, b)))
    print("longest repeated substring length:", len(longest_repeated_substring(s)))


if __name__ == "__main__":
    main()
