"""Differential, correlation and second-order power analysis attacks."""
from __future__ import annotations

import numpy as np

from .aes import HW, SBOX_NP
from .leakage import TraceSet


def hypotheses(plaintext_col: np.ndarray, target: str = "sbox") -> np.ndarray:
    """Intermediate values for all 256 key guesses, shape (N, 256)."""
    guesses = np.arange(256, dtype=np.uint8)
    x = plaintext_col[:, None] ^ guesses[None, :]
    return SBOX_NP[x] if target == "sbox" else x


def pearson_columns(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pearson correlation between every column of h (N,G) and t (N,S) -> (G,S)."""
    hc = h - h.mean(axis=0)
    tc = t - t.mean(axis=0)
    num = hc.T @ tc
    den = np.sqrt((hc ** 2).sum(axis=0))[:, None] * np.sqrt((tc ** 2).sum(axis=0))[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return r


def cpa_scores(pt_col: np.ndarray, traces: np.ndarray, target: str = "sbox") -> np.ndarray:
    """Max absolute correlation over samples per key guess (256,)."""
    h = HW[hypotheses(pt_col, target)]
    return np.abs(pearson_columns(h, traces)).max(axis=1)


def cpa_attack(ts: TraceSet, byte: int, samples: slice | None = None) -> np.ndarray:
    """Correlation power analysis on one byte. Returns 256 scores."""
    tr = ts.traces if samples is None else ts.traces[:, samples]
    return cpa_scores(ts.plaintexts[:, byte], tr)


def dpa_attack(ts: TraceSet, byte: int, bit: int = 0,
               samples: slice | None = None) -> np.ndarray:
    """Classic difference-of-means DPA on one S-box output bit. 256 scores."""
    tr = ts.traces if samples is None else ts.traces[:, samples]
    sel = ((hypotheses(ts.plaintexts[:, byte]) >> bit) & 1).astype(np.float64)  # (N,256)
    n1 = sel.sum(axis=0)
    n0 = len(sel) - n1
    valid = (n1 > 0) & (n0 > 0)
    sum1 = sel.T @ tr
    sum0 = tr.sum(axis=0)[None, :] - sum1
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = sum1 / n1[:, None] - sum0 / n0[:, None]
    scores = np.abs(np.nan_to_num(diff)).max(axis=1)
    scores[~valid] = 0.0
    return scores


def dpa_attack_multibit(ts: TraceSet, byte: int) -> np.ndarray:
    """Sum DPA scores over all eight S-box output bits."""
    return sum(dpa_attack(ts, byte, b) for b in range(8))


def centered_product(ts: TraceSet, byte: int) -> np.ndarray:
    """Combine mask and masked-S-box samples into one second-order sample."""
    a = ts.traces[:, 2 * byte] - ts.traces[:, 2 * byte].mean()
    b = ts.traces[:, 2 * byte + 1] - ts.traces[:, 2 * byte + 1].mean()
    return (a * b)[:, None]


def second_order_cpa(ts: TraceSet, byte: int) -> np.ndarray:
    """Second-order CPA via centred product of the two leaking samples."""
    return cpa_scores(ts.plaintexts[:, byte], centered_product(ts, byte))


def recover_key(ts: TraceSet, attack=cpa_attack) -> bytes:
    """Run `attack(ts, byte)` for all 16 bytes and return the best guess."""
    return bytes(int(np.argmax(attack(ts, b))) for b in range(16))


def recover_master_key(ts: TraceSet) -> bytes:
    """CPA against the first-round S-box output yields the master key directly."""
    return recover_key(ts, cpa_attack)
