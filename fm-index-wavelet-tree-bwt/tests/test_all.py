import random
import pytest
from compressed_index import (BitVector, suffix_array, bwt, inverse_bwt, lcp_kasai,
                              WaveletTree, FMIndex, naive_count, naive_locate)


def rand_text(seed, n, alpha=b"acgt"):
    r = random.Random(seed)
    return bytes(r.choice(alpha) for _ in range(n))


def test_bitvector_rank_select():
    r = random.Random(1)
    bits = [r.randint(0, 1) for _ in range(500)]
    bv = BitVector(bits)
    for i in range(0, 501, 7):
        assert bv.rank1(i) == sum(bits[:i])
        assert bv.rank0(i) == i - sum(bits[:i])
    ones = [i for i, b in enumerate(bits) if b]
    zeros = [i for i, b in enumerate(bits) if not b]
    for k in (1, 5, len(ones)):
        assert bv.select1(k) == ones[k - 1]
    for k in (1, 9, len(zeros)):
        assert bv.select0(k) == zeros[k - 1]
    assert all(bv.access(i) == bits[i] for i in range(500))


def test_bitvector_errors():
    bv = BitVector([1, 0, 1])
    with pytest.raises(ValueError):
        bv.select1(3)
    with pytest.raises(IndexError):
        bv.rank1(4)


def test_suffix_array_matches_sorted():
    for seed in range(5):
        t = rand_text(seed, 80, b"ab")
        expect = sorted(range(len(t)), key=lambda i: t[i:])
        assert suffix_array(t) == expect


def test_lcp_kasai():
    t = b"banana"
    sa = suffix_array(t)
    assert sa == [5, 3, 1, 0, 4, 2]
    assert lcp_kasai(t, sa) == [0, 1, 3, 0, 0, 2]


def test_bwt_banana_and_inverse():
    assert bwt(b"banana") == b"annb\x00aa"
    for seed in range(5):
        t = rand_text(seed, 100, b"abcd")
        assert inverse_bwt(bwt(t)) == t


def test_bwt_rejects_sentinel():
    with pytest.raises(ValueError):
        bwt(b"a\x00b")


def test_wavelet_access_rank_select():
    seq = rand_text(3, 300, b"abcdefg")
    wt = WaveletTree(seq)
    assert all(wt.access(i) == seq[i] for i in range(len(seq)))
    for c in b"abcdefz":
        for i in (0, 50, 300):
            assert wt.rank(c, i) == seq[:i].count(c)
    pos = [i for i, c in enumerate(seq) if c == ord("d")]
    assert [wt.select(ord("d"), k) for k in range(1, len(pos) + 1)] == pos


def test_wavelet_quantile():
    seq = rand_text(4, 200, b"abcdefgh")
    wt = WaveletTree(seq)
    for l, r in [(0, 200), (10, 60), (150, 151)]:
        srt = sorted(seq[l:r])
        for k in range(0, r - l, max(1, (r - l) // 5)):
            assert wt.quantile(l, r, k) == srt[k]


def test_fm_count_locate_random():
    t = rand_text(5, 400)
    fm = FMIndex(t, sa_sample=4)
    r = random.Random(6)
    for _ in range(60):
        L = r.randint(1, 6)
        s = r.randint(0, len(t) - L)
        p = t[s:s + L] if r.random() < 0.7 else rand_text(r.random(), L)
        assert fm.count(p) == naive_count(t, p)
        assert fm.locate(p) == naive_locate(t, p)


@pytest.mark.parametrize("sample", [1, 3, 16, 1000])
def test_sample_rates_agree(sample):
    t = b"mississippi river missed the mississippi"
    fm = FMIndex(t, sa_sample=sample)
    assert fm.locate(b"issi") == naive_locate(t, b"issi")
    assert fm.locate(b"i") == naive_locate(t, b"i")


def test_absent_and_overlapping():
    fm = FMIndex(b"aaaaa", sa_sample=2)
    assert fm.locate(b"aa") == [0, 1, 2, 3]
    assert fm.count(b"b") == 0 and fm.locate(b"aaaaaa") == []
    assert fm.count(b"\x00") == 0


def test_extract():
    t = rand_text(8, 250, b"xyz")
    for sample in (1, 7):
        fm = FMIndex(t, sa_sample=sample)
        for s, L in [(0, 10), (100, 37), (240, 10), (0, 250), (5, 0)]:
            assert fm.extract(s, L) == t[s:s + L]
    with pytest.raises(IndexError):
        FMIndex(b"abc").extract(2, 5)


def test_sample_reduces_space():
    t = rand_text(9, 500)
    assert FMIndex(t, 16).size_report()["sa_samples"] < FMIndex(t, 1).size_report()["sa_samples"]
