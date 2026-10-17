"""Prime-field arithmetic, polynomial helpers and primality utilities."""
from __future__ import annotations

import random
from typing import Sequence

# 2**127 - 1 and 2**61 - 1 are Mersenne primes.
PRIME_127 = (1 << 127) - 1
PRIME_61 = (1 << 61) - 1

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_probable_prime(n: int) -> bool:
    """Miller-Rabin test; deterministic for n < 3.3e24, strong otherwise."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    """Smallest prime strictly greater than ``n``."""
    c = n + 1
    if c <= 2:
        return 2
    if c % 2 == 0:
        c += 1
    while not is_probable_prime(c):
        c += 2
    return c


def inv_mod(a: int, p: int) -> int:
    """Modular inverse via the extended Euclidean algorithm."""
    a %= p
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in a prime field")
    r0, r1, s0, s1 = p, a, 0, 1
    while r1:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
    if r0 != 1:
        raise ValueError("modulus is not coprime with argument")
    return s0 % p


def poly_eval(coeffs: Sequence[int], x: int, p: int) -> int:
    """Evaluate a polynomial (coeffs[0] is the constant term) with Horner's rule."""
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % p
    return acc


def poly_mul(a: Sequence[int], b: Sequence[int], p: int) -> list[int]:
    """Product of two polynomials over GF(p)."""
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = (out[i + j] + ai * bj) % p
    return out


def poly_divmod(num: Sequence[int], den: Sequence[int], p: int) -> tuple[list[int], list[int]]:
    """Polynomial long division over GF(p). Returns (quotient, remainder)."""
    num = list(num)
    den = list(den)
    while den and den[-1] % p == 0:
        den.pop()
    if not den:
        raise ZeroDivisionError("division by zero polynomial")
    inv_lead = inv_mod(den[-1], p)
    if len(num) < len(den):
        return [0], [c % p for c in num]
    quot = [0] * (len(num) - len(den) + 1)
    for i in range(len(quot) - 1, -1, -1):
        coef = num[i + len(den) - 1] * inv_lead % p
        quot[i] = coef
        if coef:
            for j, d in enumerate(den):
                num[i + j] = (num[i + j] - coef * d) % p
    rem = [c % p for c in num[: len(den) - 1]] or [0]
    return quot, rem


def lagrange_coefficients(xs: Sequence[int], p: int, at: int = 0) -> list[int]:
    """Weights w_i such that f(at) = sum w_i f(x_i) for deg f < len(xs)."""
    if len(set(x % p for x in xs)) != len(xs):
        raise ValueError("evaluation points must be distinct")
    weights = []
    for i, xi in enumerate(xs):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if i != j:
                num = num * (at - xj) % p
                den = den * (xi - xj) % p
        weights.append(num * inv_mod(den, p) % p)
    return weights


def interpolate_at(xs: Sequence[int], ys: Sequence[int], p: int, at: int = 0) -> int:
    """Lagrange interpolation, evaluated at ``at`` (default: the secret at 0)."""
    w = lagrange_coefficients(xs, p, at)
    return sum(wi * yi for wi, yi in zip(w, ys)) % p


def interpolate_poly(xs: Sequence[int], ys: Sequence[int], p: int) -> list[int]:
    """Full coefficient vector of the unique polynomial of degree < len(xs)."""
    result = [0] * len(xs)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        basis = [1]
        den = 1
        for j, xj in enumerate(xs):
            if i != j:
                basis = poly_mul(basis, [-xj % p, 1], p)
                den = den * (xi - xj) % p
        scale = yi * inv_mod(den, p) % p
        for k, c in enumerate(basis):
            result[k] = (result[k] + scale * c) % p
    return result


def random_poly(secret: int, degree: int, p: int, rng: random.Random) -> list[int]:
    """Random polynomial of the given degree with constant term ``secret``."""
    return [secret % p] + [rng.randrange(p) for _ in range(degree)]


def find_safe_prime(bits: int, rng: random.Random) -> tuple[int, int]:
    """Return (p, q) with p = 2q + 1 both prime and p of about ``bits`` bits."""
    while True:
        q = rng.getrandbits(bits - 1) | (1 << (bits - 2)) | 1
        if is_probable_prime(q) and is_probable_prime(2 * q + 1):
            return 2 * q + 1, q
