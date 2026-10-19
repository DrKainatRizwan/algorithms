import numpy as np

from fft_toolkit import Polynomial, inverse_series, multiply_coeffs


def test_integer_product_exact_large():
    r = np.random.default_rng(0)
    a = r.integers(-1000, 1000, 80).tolist()
    b = r.integers(-1000, 1000, 90).tolist()
    ref = np.convolve(np.array(a, dtype=object), np.array(b, dtype=object)).tolist()
    assert multiply_coeffs(a, b) == ref


def test_float_product_and_short_path():
    r = np.random.default_rng(1)
    a, b = r.normal(size=60).tolist(), r.normal(size=70).tolist()
    assert np.allclose(multiply_coeffs(a, b), np.convolve(a, b))
    assert multiply_coeffs([1, 2], [3, 4]) == [3, 10, 8]


def test_from_roots_and_evaluation():
    roots = list(range(1, 41))
    p = Polynomial.from_roots(roots)
    assert p.degree == 40
    assert all(p(r) == 0 for r in roots)
    from math import factorial
    assert p(0) == factorial(40)  # (-1)^40 * 40!
    assert p.coeffs[-1] == 1


def test_power_matches_binomial():
    p = Polynomial([1, 1]) ** 30
    from math import comb
    assert p.coeffs == [comb(30, k) for k in range(31)]


def test_inverse_series():
    r = np.random.default_rng(2)
    f = r.normal(size=20)
    f[0] = 2.0
    g = inverse_series(f, 32)
    prod = np.convolve(f, g)[:32]
    assert np.allclose(prod, np.eye(1, 32, 0)[0], atol=1e-8)


def test_division_fft_vs_naive_and_reconstruction():
    r = np.random.default_rng(3)
    a = Polynomial(r.normal(size=60).tolist())
    b = Polynomial(r.normal(size=17).tolist())
    q, rem = a.divmod(b)
    q2, r2 = a.divmod_naive(b)
    assert np.allclose(q.coeffs, q2.coeffs, atol=1e-6)
    assert rem.degree < b.degree or rem.coeffs == [0]
    assert np.allclose((q * b + rem).coeffs, a.coeffs, atol=1e-6)


def test_add_sub_derivative():
    p = Polynomial([1, 2, 3])
    assert (p + p).coeffs == [2, 4, 6]
    assert (p - p).coeffs == [0]
    assert p.derivative().coeffs == [2, 6]


def test_huge_coefficients_fall_back_to_exact_python_ints():
    p = Polynomial.from_roots(list(range(1, 121)))
    from math import factorial
    assert p(0) == factorial(120)
    assert all(p(k) == 0 for k in (1, 60, 120))
