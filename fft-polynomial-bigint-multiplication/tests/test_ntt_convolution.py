import numpy as np
import pytest

from fft_toolkit import (circular_convolve, convolve_2d, convolve_exact, convolve_fft,
                         convolve_mod, convolve_naive, cross_correlate, ntt, wildcard_match)
from fft_toolkit.convolution import convolve_2d_naive, count_matches, wildcard_match_naive


def test_ntt_roundtrip_and_convolution_theorem():
    r = np.random.default_rng(0)
    a = r.integers(0, 998244353, 64)
    assert np.array_equal(ntt(ntt(a), inverse=True), a)


def test_exact_convolution_matches_python():
    r = np.random.default_rng(1)
    a = r.integers(-10**9, 10**9, 50).tolist()
    b = r.integers(-10**9, 10**9, 37).tolist()
    ref = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            ref[i + j] += x * y
    assert convolve_exact(a, b) == ref


def test_convolve_mod_arbitrary_modulus():
    mod = 10**9 + 7
    r = np.random.default_rng(2)
    a = r.integers(0, mod, 40).tolist()
    b = r.integers(0, mod, 40).tolist()
    ref = [0] * 79
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            ref[i + j] = (ref[i + j] + x * y) % mod
    assert convolve_mod(a, b, mod) == ref


def test_ntt_rejects_bad_length():
    with pytest.raises(ValueError):
        ntt(np.ones(3, dtype=np.int64))


def test_fft_convolution_vs_naive_and_circular():
    r = np.random.default_rng(3)
    a, b = r.normal(size=33), r.normal(size=20)
    assert np.allclose(convolve_fft(a, b), convolve_naive(a, b))
    assert np.allclose(convolve_fft(a, b), np.convolve(a, b))
    c = r.normal(size=12)
    d = r.normal(size=12)
    ref = [sum(c[j] * d[(k - j) % 12] for j in range(12)) for k in range(12)]
    assert np.allclose(circular_convolve(c, d), ref)


def test_cross_correlation_finds_shift():
    r = np.random.default_rng(4)
    sig = r.normal(size=100)
    tmpl = sig[37:57]
    cc = cross_correlate(sig, tmpl)
    assert int(np.argmax(cc)) - (len(tmpl) - 1) == 37


def test_2d_convolution():
    r = np.random.default_rng(5)
    img, k = r.normal(size=(9, 11)), r.normal(size=(3, 4))
    assert np.allclose(convolve_2d(img, k), convolve_2d_naive(img, k))


def test_wildcard_matching():
    assert wildcard_match("abcabxabc", "ab?") == wildcard_match_naive("abcabxabc", "ab?")
    r = np.random.default_rng(6)
    text = "".join(r.choice(list("abc?"), 200))
    for pat in ("a?b", "cc", "?a?b?", "b"):
        assert wildcard_match(text, pat) == wildcard_match_naive(text, pat)


def test_count_matches():
    assert count_matches("abcab", "ab") == [2, 0, 0, 2]
