import numpy as np
import pytest

from fft_toolkit import bluestein, dft_naive, fft, fft_iterative, fft_recursive, ifft
from fft_toolkit.fft import bit_reversal_permutation, irfft, rfft, two_real_ffts, fft2


def rand(n, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=n) + 1j * r.normal(size=n)


@pytest.mark.parametrize("n", [1, 2, 8, 64, 256])
def test_radix2_variants_match_naive(n):
    x = rand(n, n)
    ref = dft_naive(x)
    assert np.allclose(fft_recursive(x), ref)
    assert np.allclose(fft_iterative(x), ref)


@pytest.mark.parametrize("n", [3, 5, 7, 12, 100, 127])
def test_bluestein_arbitrary_length(n):
    x = rand(n, n)
    assert np.allclose(bluestein(x), dft_naive(x))
    assert np.allclose(fft(x), np.fft.fft(x))


def test_inverse_roundtrip_and_parseval():
    x = rand(96, 3)
    X = fft(x)
    assert np.allclose(ifft(X), x)
    assert np.isclose(np.sum(np.abs(x) ** 2), np.sum(np.abs(X) ** 2) / 96)


def test_bit_reversal_and_errors():
    assert bit_reversal_permutation(8).tolist() == [0, 4, 2, 6, 1, 5, 3, 7]
    with pytest.raises(ValueError):
        fft_iterative(np.ones(6))


def test_batched_and_2d():
    x = np.random.default_rng(1).normal(size=(5, 16))
    assert np.allclose(fft_iterative(x), np.fft.fft(x, axis=-1))
    m = np.random.default_rng(2).normal(size=(6, 10))
    assert np.allclose(fft2(m), np.fft.fft2(m))
    assert np.allclose(fft2(fft2(m), inverse=True), m)


def test_real_helpers():
    r = np.random.default_rng(4)
    a, b = r.normal(size=20), r.normal(size=20)
    A, B = two_real_ffts(a, b)
    assert np.allclose(A, np.fft.fft(a)) and np.allclose(B, np.fft.fft(b))
    for n in (16, 15):
        x = r.normal(size=n)
        assert np.allclose(rfft(x), np.fft.rfft(x))
        assert np.allclose(irfft(rfft(x), n), x)
