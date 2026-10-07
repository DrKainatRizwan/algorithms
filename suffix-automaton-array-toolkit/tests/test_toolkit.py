import random
import pytest
from suffix_toolkit import (
    build_suffix_array, build_suffix_array_naive, kasai_lcp, SuffixAutomaton,
    SuffixArrayIndex, count_distinct_substrings, longest_repeated_substring,
    longest_common_substring, longest_common_substring_sa, kth_substring,
    bwt_from_suffix_array, inverse_bwt,
)


def rand_str(rng, n, alphabet="ab"):
    return "".join(rng.choice(alphabet) for _ in range(n))


def naive_substrings(s):
    return {s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)}


def naive_count(s, p):
    return sum(s.startswith(p, i) for i in range(len(s) - len(p) + 1))


def test_banana_suffix_array():
    assert build_suffix_array("banana") == [5, 3, 1, 0, 4, 2]


def test_empty_and_single():
    assert build_suffix_array("") == []
    assert build_suffix_array("a") == [0]


def test_sa_matches_naive_random():
    rng = random.Random(1)
    for _ in range(100):
        s = rand_str(rng, rng.randint(1, 40), "abc")
        assert build_suffix_array(s) == build_suffix_array_naive(s)


def test_sa_repetitive():
    for s in ["a" * 30, "ab" * 20, "abaabaaab" * 3]:
        assert build_suffix_array(s) == build_suffix_array_naive(s)


def test_kasai_lcp_banana():
    s = "banana"
    assert kasai_lcp(s, build_suffix_array(s)) == [0, 1, 3, 0, 0, 2]


def test_kasai_matches_bruteforce():
    rng = random.Random(2)
    s = rand_str(rng, 60, "abc")
    sa = build_suffix_array(s)
    lcp = kasai_lcp(s, sa)
    for i in range(1, len(s)):
        a, b = s[sa[i - 1]:], s[sa[i]:]
        h = 0
        while h < min(len(a), len(b)) and a[h] == b[h]:
            h += 1
        assert lcp[i] == h


def test_automaton_recognises_exactly_substrings():
    rng = random.Random(3)
    s = rand_str(rng, 25, "ab")
    sam = SuffixAutomaton(s)
    subs = naive_substrings(s)
    for sub in subs:
        assert sam.contains(sub)
    for _ in range(100):
        q = rand_str(rng, rng.randint(1, 8), "abc")
        assert sam.contains(q) == (q in subs)


def test_automaton_state_bound():
    s = "abcbcabcabc" * 5
    assert len(SuffixAutomaton(s)) <= 2 * len(s)


def test_automaton_counts_and_first_occurrence():
    rng = random.Random(4)
    s = rand_str(rng, 80, "ab")
    sam = SuffixAutomaton(s)
    for _ in range(60):
        q = rand_str(rng, rng.randint(1, 5), "ab")
        assert sam.count(q) == naive_count(s, q)
        assert sam.first_occurrence(q) == s.find(q)


def test_incremental_equals_batch():
    sam = SuffixAutomaton()
    for c in "abracadabra":
        sam.extend(c)
        sam.count("a")  # exercise cache invalidation
    assert sam.count("abra") == 2
    assert sam.distinct_substrings() == SuffixAutomaton("abracadabra").distinct_substrings()


def test_distinct_substrings_agree():
    rng = random.Random(5)
    for _ in range(30):
        s = rand_str(rng, rng.randint(1, 30), "abc")
        expected = len(naive_substrings(s))
        assert SuffixAutomaton(s).distinct_substrings() == expected
        assert count_distinct_substrings(s) == expected


def test_suffix_array_index_search():
    rng = random.Random(6)
    s = rand_str(rng, 100, "abc")
    idx = SuffixArrayIndex(s)
    for _ in range(60):
        q = rand_str(rng, rng.randint(1, 4), "abc")
        expected = [i for i in range(len(s)) if s.startswith(q, i)]
        assert idx.find_all(q) == expected
        assert idx.count(q) == len(expected)
    assert not idx.contains("zzz")


def test_longest_repeated_substring():
    assert longest_repeated_substring("banana") == "ana"
    assert longest_repeated_substring("abcd") == ""
    assert longest_repeated_substring("") == ""


def test_longest_common_substring_methods_agree():
    rng = random.Random(7)
    for _ in range(30):
        a, b = rand_str(rng, 30, "abc"), rand_str(rng, 30, "abc")
        subs = naive_substrings(a) & naive_substrings(b)
        best = max(map(len, subs), default=0)
        r1 = longest_common_substring(a, b)
        r2 = longest_common_substring_sa(a, b)
        assert len(r1) == len(r2) == best
        assert r1 in a and r1 in b and r2 in a and r2 in b


def test_kth_substring():
    s = "abca"
    ordered = sorted(naive_substrings(s))
    for k, expected in enumerate(ordered, 1):
        assert kth_substring(s, k) == expected
    with pytest.raises(IndexError):
        kth_substring(s, len(ordered) + 1)


def test_bwt_roundtrip():
    assert bwt_from_suffix_array("banana", "$") == "annb$aa"
    rng = random.Random(8)
    for _ in range(20):
        s = rand_str(rng, rng.randint(1, 40), "abc")
        assert inverse_bwt(bwt_from_suffix_array(s)) == s
    with pytest.raises(ValueError):
        bwt_from_suffix_array("a$b", "$")
