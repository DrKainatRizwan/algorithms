"""End-to-end BB84 run: preparation, measurement, sifting, QBER estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .channel import Channel
from .photons import encode, measure
from .privacy import privacy_amplify, secure_key_length
from .reconciliation import cascade


@dataclass
class BB84Config:
    """Protocol parameters."""

    n_photons: int = 4000
    sample_fraction: float = 0.2
    abort_qber: float = 0.11
    seed: int = 0
    reconcile: bool = True
    security_eps: float = 1e-6


@dataclass
class BB84Result:
    """Outcome of a protocol run."""

    n_sent: int
    n_detected: int
    n_sifted: int
    qber_estimate: float
    true_qber: float
    aborted: bool
    errors_corrected: int = 0
    leaked_bits: int = 0
    final_key_alice: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    final_key_bob: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    eve_known_fraction: float = 0.0

    @property
    def keys_match(self) -> bool:
        a, b = self.final_key_alice, self.final_key_bob
        return a.shape == b.shape and bool(np.all(a == b))


def sift(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bits: np.ndarray,
    bob_bases: np.ndarray,
    detected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep positions where Bob detected a photon and bases agree.

    Returns ``(alice_key, bob_key, indices)``.
    """
    keep = detected & (alice_bases == bob_bases)
    idx = np.flatnonzero(keep)
    return alice_bits[idx], bob_bits[idx], idx


def estimate_qber(
    a: np.ndarray, b: np.ndarray, fraction: float, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    """Publicly compare a random subset; return (QBER, mask of revealed bits)."""
    n = a.shape[0]
    k = max(1, int(round(fraction * n))) if n else 0
    mask = np.zeros(n, dtype=bool)
    if k:
        mask[rng.choice(n, size=min(k, n), replace=False)] = True
        qber = float(np.mean(a[mask] != b[mask]))
    else:
        qber = 1.0
    return qber, mask


def run_bb84(config: BB84Config, channel: Channel) -> BB84Result:
    """Simulate one complete BB84 session over ``channel``."""
    rng = np.random.default_rng(config.seed)
    n = config.n_photons
    a_bits = rng.integers(0, 2, n)
    a_bases = rng.integers(0, 2, n)
    b_bases = rng.integers(0, 2, n)

    angles = encode(a_bits, a_bases)
    received, detected, eve_mask = channel.transmit(angles, rng)
    b_bits, _ = measure(received, b_bases, rng)

    ka, kb, idx = sift(a_bits, a_bases, b_bits, b_bases, detected)
    true_qber = float(np.mean(ka != kb)) if ka.size else 1.0
    qber, revealed = estimate_qber(ka, kb, config.sample_fraction, rng)

    eve_known = 0.0
    if channel.eve is not None and idx.size:
        eve_bases = channel.eve.last_bases[idx]
        # Eve knows a sifted bit for sure when she guessed the basis right.
        eve_known = float(np.mean(eve_mask[idx] & (eve_bases == a_bases[idx])))

    res = BB84Result(
        n_sent=n, n_detected=int(detected.sum()), n_sifted=int(ka.size),
        qber_estimate=qber, true_qber=true_qber, aborted=False,
        eve_known_fraction=eve_known,
    )
    if ka.size == 0 or qber > config.abort_qber:
        res.aborted = True
        return res

    ra, rb = ka[~revealed], kb[~revealed]
    leaked = 0
    if config.reconcile:
        rb, fixed, leaked = cascade(ra, rb, max(qber, 0.005), rng)
        res.errors_corrected = fixed
    res.leaked_bits = leaked
    m = secure_key_length(ra.size, qber, leaked, config.security_eps)
    if m <= 0:
        res.aborted = True
        return res
    seed = int(rng.integers(0, 2**31 - 1))
    res.final_key_alice = privacy_amplify(ra, m, seed)
    res.final_key_bob = privacy_amplify(rb, m, seed)
    return res
