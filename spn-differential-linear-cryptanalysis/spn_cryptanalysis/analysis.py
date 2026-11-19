"""Experiments: S-box quality reports, success-rate curves and avalanche tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import differential, linear
from .sbox import (differential_uniformity, is_bijective, linearity,
                   nonlinearity)
from .spn import SPN


@dataclass(frozen=True)
class SboxReport:
    """Summary of the cryptographic quality of a 4-bit S-box."""

    name: str
    bijective: bool
    diff_uniformity: int
    linearity: int
    nonlinearity: int
    best_diff_char: float
    best_linear_bias: float


def sbox_report(name: str, sbox: tuple[int, ...], rounds: int = 3) -> SboxReport:
    """Compute headline S-box metrics and best ``rounds``-round distinguishers."""
    ch = differential.best_characteristic(sbox, rounds)
    tr = linear.best_trail(sbox, rounds)
    return SboxReport(name, is_bijective(sbox), differential_uniformity(sbox),
                      linearity(sbox), nonlinearity(sbox),
                      ch.probability if ch else 0.0, tr.bias if tr else 0.0)


def differential_curve(sbox: tuple[int, ...], rounds: int,
                       ch: differential.Characteristic, pair_counts: list[int],
                       trials: int, seed: int = 0) -> list[tuple[int, float]]:
    """Success rate of the differential attack versus number of chosen pairs."""
    return [(n, differential.success_rate(sbox, rounds, ch, n, trials, seed))
            for n in pair_counts]


def linear_curve(sbox: tuple[int, ...], rounds: int, trail: linear.LinearTrail,
                 sample_counts: list[int], trials: int,
                 seed: int = 0) -> list[tuple[int, float]]:
    """Success rate of the linear attack versus number of known plaintexts."""
    return [(n, linear.success_rate(sbox, rounds, trail, n, trials, seed))
            for n in sample_counts]


def avalanche(cipher: SPN, trials: int, rng: np.random.Generator) -> float:
    """Mean fraction of ciphertext bits flipped by flipping one plaintext bit."""
    p = rng.integers(0, 1 << 16, size=trials).astype(np.uint16)
    total = 0.0
    for bit in range(16):
        d = cipher.encrypt(p) ^ cipher.encrypt(p ^ np.uint16(1 << bit))
        flips = np.zeros(trials)
        for b in range(16):
            flips += (d >> b) & 1
        total += flips.mean() / 16
    return total / 16


def rounds_to_resist(sbox: tuple[int, ...], target_log2: float = -16.0,
                     max_rounds: int = 6) -> int | None:
    """Smallest number of rounds whose best characteristic has ``p < 2^target``.

    A crude rule of thumb for the minimum round count that defeats basic
    differential attacks on a 16-bit block (pairs needed ``~ 1/p``).
    """
    for r in range(1, max_rounds + 1):
        ch = differential.best_characteristic(sbox, r)
        if ch is not None and np.log2(ch.probability) < target_log2:
            return r
    return None
