"""Regev's original LWE public-key encryption (bitwise)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sampling import center, discrete_gaussian, uniform_mod


@dataclass(frozen=True)
class LWEParams:
    """n: secret dimension, q: modulus, m: number of samples, sigma: error std."""
    n: int = 32
    q: int = 3329
    m: int = 96
    sigma: float = 1.0


@dataclass
class LWEKeyPair:
    params: LWEParams
    A: np.ndarray   # (m, n)
    b: np.ndarray   # (m,)  b = A s + e mod q
    s: np.ndarray   # secret (n,)
    e: np.ndarray   # error  (m,)

    @property
    def public(self) -> tuple[np.ndarray, np.ndarray]:
        return self.A, self.b


def lwe_keygen(params: LWEParams, rng: np.random.Generator) -> LWEKeyPair:
    """Generate (A, b = A s + e) with uniform A and Gaussian s, e."""
    A = uniform_mod(rng, params.q, (params.m, params.n))
    s = uniform_mod(rng, params.q, params.n)
    e = discrete_gaussian(rng, params.sigma, params.m)
    b = (A @ s + e) % params.q
    return LWEKeyPair(params, A, b, s, e)


def lwe_encrypt_bit(pub: tuple[np.ndarray, np.ndarray], params: LWEParams, bit: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Encrypt one bit: sum a random subset of samples and add bit*floor(q/2)."""
    A, b = pub
    r = rng.integers(0, 2, size=params.m, dtype=np.int64)
    c1 = (r @ A) % params.q
    c2 = (int(r @ b) + bit * (params.q // 2)) % params.q
    return c1, c2


def lwe_decrypt_bit(s: np.ndarray, params: LWEParams, ct: tuple[np.ndarray, int]) -> int:
    """Decrypt: the phase c2 - <c1, s> is near 0 (bit 0) or near q/2 (bit 1)."""
    c1, c2 = ct
    phase = (int(c2) - int(c1 @ s)) % params.q
    return int(abs(int(center(np.array([phase]), params.q)[0])) > params.q // 4)


def lwe_encrypt_bits(pub, params: LWEParams, bits, rng: np.random.Generator):
    return [lwe_encrypt_bit(pub, params, int(b), rng) for b in bits]


def lwe_decrypt_bits(s, params: LWEParams, cts) -> list[int]:
    return [lwe_decrypt_bit(s, params, ct) for ct in cts]


def decryption_failure_probability(params: LWEParams) -> float:
    """Gaussian approximation: noise = <r, e> has variance ~ (m/2) sigma^2.

    A bit fails when |noise| > q/4, so P = erfc((q/4) / (sqrt(2) * std)).
    """
    std = math.sqrt(params.m / 2.0) * params.sigma
    if std == 0:
        return 0.0
    return math.erfc((params.q / 4.0) / (math.sqrt(2.0) * std))
