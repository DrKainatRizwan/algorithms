import math
import pytest

from compression_suite import arithmetic, huffman, lz77, pack, unpack
from compression_suite.bitio import BitReader, BitWriter
from compression_suite.entropy import (entropy, conditional_entropy, gen_random,
                                       gen_repetitive, gen_skewed, gen_text, kraft_sum)

SAMPLES = [b"", b"a", b"aaaaaaaaaa", b"abracadabra" * 5, bytes(range(256)), gen_text(3000, 1),
           gen_random(500, 2), gen_repetitive(3000, 3)]


def test_bitio_roundtrip():
    w = BitWriter()
    w.write_bits(0b10110, 5)
    w.write_bits(12345, 20)
    w.write_bit(1)
    assert w.bit_length == 26
    r = BitReader(w.getvalue())
    assert r.read_bits(5) == 0b10110 and r.read_bits(20) == 12345 and r.read_bit() == 1
    with pytest.raises(ValueError):
        w.write_bits(8, 3)


def test_bitreader_eof():
    with pytest.raises(EOFError):
        BitReader(b"\x00").read_bits(9)


def test_entropy_values():
    assert entropy(b"ab" * 50) == pytest.approx(1.0)
    assert entropy(b"") == 0.0
    assert conditional_entropy(b"ab" * 50) == pytest.approx(0.0, abs=0.02)


def test_huffman_optimal_known():
    lengths = huffman.code_lengths({0: 45, 1: 13, 2: 12, 3: 16, 4: 9, 5: 5})
    assert sorted(lengths.values()) == [1, 3, 3, 3, 4, 4]  # CLRS example
    assert kraft_sum(lengths) == pytest.approx(1.0)


def test_huffman_prefix_free_and_within_one_bit_of_entropy():
    data = gen_text(5000, 4)
    from collections import Counter
    f = dict(Counter(data))
    L = huffman.code_lengths(f)
    avg = huffman.expected_length(f, L)
    H = entropy(data)
    assert H <= avg < H + 1
    codes = huffman.canonical_codes(L)
    strs = [format(c, f"0{l}b") for c, l in codes.values()]
    for a in strs:
        assert sum(s.startswith(a) for s in strs) == 1


def test_limit_lengths_kraft():
    f = {i: 2 ** i for i in range(20)}  # Fibonacci-like skew -> deep tree
    L = huffman.code_lengths(f)
    assert max(L.values()) == 19
    L2 = huffman.limit_lengths(L, 8)
    assert max(L2.values()) <= 8 and kraft_sum(L2) <= 1.0


@pytest.mark.parametrize("data", SAMPLES)
def test_huffman_roundtrip(data):
    assert huffman.decompress(huffman.compress(data)) == data


@pytest.mark.parametrize("data", SAMPLES)
def test_lz77_roundtrip(data):
    assert lz77.decompress(lz77.compress(data)) == data


@pytest.mark.parametrize("data", SAMPLES)
@pytest.mark.parametrize("order", [0, 1])
def test_arithmetic_roundtrip(data, order):
    assert arithmetic.decompress(arithmetic.compress(data, order)) == data


def test_lz77_overlapping_match_and_tokens():
    toks = lz77.find_tokens(b"a" * 100)
    assert any(not isinstance(t, int) for t in toks) and len(toks) < 5
    assert lz77.expand(toks) == b"a" * 100
    assert lz77.expand([97, (1, 5)]) == b"aaaaaa"
    with pytest.raises(ValueError):
        lz77.expand([(3, 3)])


def test_lz77_window_respected():
    data = gen_repetitive(4000, 5)
    for t in lz77.find_tokens(data, window=64):
        if not isinstance(t, int):
            assert t[0] <= 64
    assert lz77.expand(lz77.find_tokens(data, window=64)) == data


def test_lz77_lazy_not_worse_much_and_valid():
    data = gen_text(4000, 6)
    a = len(lz77.compress(data, lazy=True))
    b = len(lz77.compress(data, lazy=False))
    assert a <= b * 1.05
    assert lz77.token_stats(lz77.find_tokens(data))["matches"] > 0


def test_lz77_beats_huffman_on_repetitive():
    data = gen_repetitive(8000, 7)
    assert len(lz77.compress(data)) < len(huffman.compress(data)) / 2


def test_arithmetic_near_entropy_on_skewed():
    data = gen_skewed(20000, 8, 0.9)
    bpc = 8 * len(arithmetic.compress(data)) / len(data)
    assert bpc < entropy(data) + 0.1
    # Huffman cannot go below 1 bit/symbol; arithmetic does
    assert bpc < 8 * len(huffman.compress(data)) / len(data)


def test_arithmetic_order1_helps_on_markov_text():
    data = gen_text(8000, 9)
    assert len(arithmetic.compress(data, 1)) < len(arithmetic.compress(data, 0)) * 1.1
    data2 = b"abcabcabd" * 400
    assert len(arithmetic.compress(data2, 1)) < len(arithmetic.compress(data2, 0))


def test_fenwick_model_consistency():
    m = arithmetic.FenwickModel(20)
    for s in [3, 3, 7, 19, 0, 3]:
        m.update(s)
    for s in range(20):
        assert m.cum(s) == sum(m.freq[:s])
        lo = m.cum(s)
        assert m.find(lo) == s and m.find(lo + m.freq[s] - 1) == s


def test_fenwick_rescale_bounds_total():
    m = arithmetic.FenwickModel(10)
    for _ in range(5000):
        m.update(2)
    assert m.total < arithmetic.MAX_TOTAL and all(f >= 1 for f in m.freq)


def test_container_auto_and_stored():
    for data in SAMPLES:
        assert unpack(pack(data)) == data
    r = gen_random(300, 10)
    assert pack(r)[0] == 0 and len(pack(r)) == 301
    rep = gen_repetitive(4000, 11)
    assert len(pack(rep)) < len(rep) / 3
    for m in ("huffman", "lz77", "arith0", "arith1"):
        assert unpack(pack(rep, m)) == rep


def test_random_data_incompressible():
    d = gen_random(4000, 12)
    assert len(arithmetic.compress(d)) > 0.97 * len(d)
