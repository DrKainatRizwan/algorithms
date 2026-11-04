"""Ring-LWE public-key encryption in the style of Lyubashevsky-Peikert-Regev."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ring import NegacyclicRing
from .sampling import centered_binomial, center, uniform_mod


@dataclass(frozen=True)
class RLWEParams:
    n: int = 256
    q: int = 7681
    eta: int = 2

    def ring(self) -> NegacyclicRing:
        return NegacyclicRing(self.n, self.q)


def rlwe_keygen(params: RLWEParams, rng: np.random.Generator):
    """Return ((a, b), s) with b = a*s + e in R_q and small s, e."""
    ring = params.ring()
    a = uniform_mod(rng, params.q, params.n)
    s = centered_binomial(rng, params.eta, params.n)
    e = centered_binomial(rng, params.eta, params.n)
    b = ring.add(ring.mul(a, s), e)
    return (a, b), s % params.q


def rlwe_encrypt(pub, params: RLWEParams, bits, rng: np.random.Generator):
    """Encrypt an n-bit message: u = a r + e1, v = b r + e2 + floor(q/2) m."""
    ring = params.ring()
    a, b = pub
    m = np.zeros(params.n, dtype=np.int64)
    bits = np.asarray(bits, dtype=np.int64)
    m[:len(bits)] = bits
    r = centered_binomial(rng, params.eta, params.n)
    e1 = centered_binomial(rng, params.eta, params.n)
    e2 = centered_binomial(rng, params.eta, params.n)
    u = ring.add(ring.mul(a, r), e1)
    v = ring.add(ring.add(ring.mul(b, r), e2), m * (params.q // 2))
    return u, v


def rlwe_decrypt(s, params: RLWEParams, ct, length: int | None = None) -> np.ndarray:
    ring = params.ring()
    u, v = ct
    phase = center(ring.sub(v, ring.mul(u, s)), params.q)
    out = (np.abs(phase) > params.q // 4).astype(np.int64)
    return out if length is None else out[:length]


def noise_of(s, params: RLWEParams, pub_ct, bits) -> int:
    """Max |noise| left after removing the message (for diagnostics/tests)."""
    ring = params.ring()
    u, v = pub_ct
    m = np.zeros(params.n, dtype=np.int64)
    m[:len(bits)] = bits
    resid = ring.sub(ring.sub(v, ring.mul(u, s)), m * (params.q // 2))
    return int(np.abs(center(resid, params.q)).max())
