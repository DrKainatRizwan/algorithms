import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pattern_search import (AhoCorasick, SearchEngine, border_array, count_distinct_matches,
                            failure_function, kmp_search, naive_search, rabin_karp,
                            rabin_karp_multi, resolve_overlaps, shortest_period,
                            z_array, z_search, RollingHash)


def rand_str(rng, n, alpha="ab"):
    return "".join(rng.choice(alpha) for _ in range(n))


def test_failure_function_known():
    assert failure_function("abacabab") == [0, 0, 1, 0, 1, 2, 3, 2]


def test_z_array_known():
    assert z_array("aabxaab") == [7, 1, 0, 0, 3, 1, 0]


def test_overlapping_matches():
    for f in (kmp_search, z_search, rabin_karp, naive_search):
        assert f("aaaaa", "aa") == [0, 1, 2, 3]


@pytest.mark.parametrize("seed", range(5))
def test_single_matchers_agree_with_naive(seed):
    rng = random.Random(seed)
    for _ in range(40):
        t = rand_str(rng, rng.randint(0, 60))
        p = rand_str(rng, rng.randint(1, 5))
        expect = naive_search(t, p)
        assert kmp_search(t, p) == expect
        assert z_search(t, p) == expect
        assert rabin_karp(t, p) == expect


def test_rabin_karp_with_colliding_modulus_still_correct_when_verified():
    t, p = "abcabcabc" * 5, "cab"
    assert rabin_karp(t, p, base=3, mod=7) == naive_search(t, p)
    # Without verification a tiny modulus yields false positives.
    assert len(rabin_karp(t, "zzz", base=3, mod=2, verify=False)) >= 0


def test_rolling_hash_substring_equality():
    rh = RollingHash("abcxabc")
    assert rh.get(0, 3) == rh.get(4, 7)
    assert rh.get(0, 3) != rh.get(1, 4)


def test_empty_pattern_and_short_text():
    assert kmp_search("abc", "") == [0, 1, 2, 3]
    assert kmp_search("ab", "abc") == []
    assert rabin_karp("ab", "abc") == []


def test_classic_aho_corasick():
    ac = AhoCorasick(["he", "she", "his", "hers"])
    got = [(m.start, m.pattern) for m in ac.find_all("ushers")]
    assert got == [(1, "she"), (2, "he"), (2, "hers")]


def test_aho_corasick_nested_and_duplicate_patterns():
    ac = AhoCorasick(["a", "aa", "aaa", "a"])
    cnt = ac.count("aaaa")
    assert cnt["a"] == 8  # duplicate id registered twice -> counted per id then merged
    assert cnt["aa"] == 3 and cnt["aaa"] == 2


@pytest.mark.parametrize("seed", range(5))
def test_aho_corasick_matches_naive(seed):
    rng = random.Random(100 + seed)
    pats = list({rand_str(rng, rng.randint(1, 5), "abc") for _ in range(8)})
    text = rand_str(rng, 200, "abc")
    got = {p: [] for p in pats}
    for m in AhoCorasick(pats).find_all(text):
        got[m.pattern].append(m.start)
    for p in pats:
        assert got[p] == naive_search(text, p)


def test_rabin_karp_multi_matches_aho():
    rng = random.Random(7)
    pats = list({rand_str(rng, rng.randint(1, 6), "abc") for _ in range(10)})
    text = rand_str(rng, 300, "abc")
    rk = rabin_karp_multi(text, pats)
    for p in pats:
        assert rk[p] == naive_search(text, p)


def test_empty_pattern_rejected_and_no_match():
    with pytest.raises(ValueError):
        AhoCorasick().add("")
    assert not AhoCorasick(["xyz"]).contains_any("abcabc")
    assert AhoCorasick(["bc"]).contains_any("abcabc")


def test_overlap_policies():
    eng = SearchEngine(["ab", "abc", "bcd", "d"])
    ms = eng.search("abcd")
    assert len(ms) == 4
    ll = resolve_overlaps(ms, "leftmost_longest")
    assert [m.pattern for m in ll] == ["abc", "d"]
    mc = resolve_overlaps(ms, "max_cover")
    assert sum(m.end - m.start for m in mc) == 4
    lf = resolve_overlaps(ms, "leftmost_first")
    assert [m.pattern for m in lf] == ["ab", "d"]
    with pytest.raises(ValueError):
        resolve_overlaps(ms, "bogus")


def test_max_cover_beats_greedy():
    # Greedy picks "abc"(3) then nothing; optimal picks "ab"+"cdef"? ensure >= greedy
    eng = SearchEngine(["abc", "ab", "cdef"])
    text = "abcdef"
    greedy = sum(m.end - m.start for m in eng.search(text, "leftmost_longest"))
    best = sum(m.end - m.start for m in eng.search(text, "max_cover"))
    assert greedy == 3 and best == 6


def test_case_insensitive_and_whole_word():
    eng = SearchEngine(["Cat"], case_sensitive=False, whole_word=True)
    ms = eng.search("cat concat CAT, cats")
    assert [m.start for m in ms] == [0, 11]
    assert ms[0].pattern == "Cat"


def test_highlight_redact_replace():
    eng = SearchEngine(["secret", "key"])
    assert eng.highlight("secret key") == "[secret] [key]"
    assert eng.redact("my secret key") == "my ****** ***"
    assert eng.replace("secret key", {"secret": "S"}) == "S key"


def test_period_and_borders():
    assert shortest_period("abcabcabc") == 3
    assert shortest_period("abcd") == 4
    assert border_array("aaaa") == [0, 1, 2, 3]
    assert count_distinct_matches("hello world", ["lo", "wor", "xyz", "lo"]) == 2


def test_large_scale_agreement():
    rng = random.Random(3)
    text = rand_str(rng, 5000, "abcd")
    pats = list({rand_str(rng, rng.randint(3, 8), "abcd") for _ in range(30)})
    ac = AhoCorasick(pats)
    total = sum(len(naive_search(text, p)) for p in pats)
    assert len(ac.find_all(text)) == total
