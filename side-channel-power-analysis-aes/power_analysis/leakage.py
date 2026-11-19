"""Simulated power traces under the Hamming-weight leakage model.

Each trace contains, for every key byte i, two samples of the first round:
``HW(p_i ^ k_i)`` (AddRoundKey) and ``HW(S[p_i ^ k_i])`` (SubBytes), plus
Gaussian noise.  A masked variant leaks the masked S-box output and the mask
in separate samples, as in a first-order Boolean-masked implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aes import HW, SBOX_NP


@dataclass
class TraceSet:
    """Plaintexts and corresponding simulated traces."""

    plaintexts: np.ndarray  # (N, 16) uint8
    traces: np.ndarray      # (N, S) float64
    key: bytes
    masked: bool = False

    def subset(self, n: int) -> "TraceSet":
        return TraceSet(self.plaintexts[:n], self.traces[:n], self.key, self.masked)


def sample_index(byte: int, kind: str, masked: bool = False) -> int:
    """Sample index of the leakage of a given byte.

    kind: 'addkey' / 'sbox' (unmasked), 'mask' / 'masked_sbox' (masked).
    """
    if masked:
        return 2 * byte + {"mask": 0, "masked_sbox": 1}[kind]
    return 2 * byte + {"addkey": 0, "sbox": 1}[kind]


def simulate_traces(key: bytes, n: int, noise_sigma: float = 1.0,
                    seed: int = 0) -> TraceSet:
    """Unprotected implementation: Hamming-weight leakage + Gaussian noise."""
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    k = np.frombuffer(key, dtype=np.uint8)
    x = pts ^ k
    traces = np.empty((n, 32))
    traces[:, 0::2] = HW[x]
    traces[:, 1::2] = HW[SBOX_NP[x]]
    traces += rng.normal(0.0, noise_sigma, size=traces.shape)
    return TraceSet(pts, traces, key, False)


def simulate_masked_traces(key: bytes, n: int, noise_sigma: float = 1.0,
                           seed: int = 0) -> TraceSet:
    """First-order Boolean masking: S-box output is XORed with a fresh mask."""
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    masks = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    k = np.frombuffer(key, dtype=np.uint8)
    s = SBOX_NP[pts ^ k]
    traces = np.empty((n, 32))
    traces[:, 0::2] = HW[masks]
    traces[:, 1::2] = HW[s ^ masks]
    traces += rng.normal(0.0, noise_sigma, size=traces.shape)
    return TraceSet(pts, traces, key, True)
