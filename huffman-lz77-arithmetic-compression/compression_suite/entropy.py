"""Information-theoretic helpers and synthetic data generators."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def entropy(data: bytes) -> float:
    """Zeroth-order empirical entropy in bits per byte."""
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


def conditional_entropy(data: bytes) -> float:
    """First-order conditional entropy H(X_i | X_{i-1}) in bits per byte."""
    if len(data) < 2:
        return 0.0
    pairs = Counter(zip(data[:-1], data[1:]))
    ctx = Counter(data[:-1])
    n = len(data) - 1
    return -sum(c / n * math.log2(c / ctx[a]) for (a, _), c in pairs.items())


def kraft_sum(lengths: dict[int, int]) -> float:
    """Kraft-McMillan sum of a set of code lengths (<= 1 for prefix codes)."""
    return sum(2.0 ** -l for l in lengths.values() if l > 0)


def gen_text(n: int, seed: int = 0) -> bytes:
    """Pseudo-English text from a seeded word-level Markov-ish sampler."""
    rng = np.random.default_rng(seed)
    words = ("the of and to in is that it was for on are as with his they at be "
             "this from have or by one had not but what all were when we there "
             "can an your which their said if do will each about how up out them "
             "then she many some so these would other into has more her two like").split()
    p = 1.0 / np.arange(1, len(words) + 1)
    p /= p.sum()
    out = bytearray()
    while len(out) < n:
        w = words[int(rng.choice(len(words), p=p))]
        out += w.encode() + (b". " if rng.random() < 0.08 else b" ")
    return bytes(out[:n])


def gen_repetitive(n: int, seed: int = 0) -> bytes:
    """Data made of a few long phrases repeated with sparse mutations."""
    rng = np.random.default_rng(seed)
    phrases = [rng.integers(0, 256, size=int(rng.integers(40, 120)), dtype=np.uint8).tobytes()
               for _ in range(6)]
    out = bytearray()
    while len(out) < n:
        ph = bytearray(phrases[int(rng.integers(len(phrases)))])
        if rng.random() < 0.3:
            ph[int(rng.integers(len(ph)))] = int(rng.integers(256))
        out += ph
    return bytes(out[:n])


def gen_random(n: int, seed: int = 0) -> bytes:
    """Incompressible uniform bytes."""
    return np.random.default_rng(seed).integers(0, 256, size=n, dtype=np.uint8).tobytes()


def gen_skewed(n: int, seed: int = 0, p: float = 0.9) -> bytes:
    """i.i.d. bytes where symbol 0 has probability ``p`` and the rest share 1-p over 15 symbols."""
    rng = np.random.default_rng(seed)
    probs = np.full(16, (1 - p) / 15)
    probs[0] = p
    return rng.choice(16, size=n, p=probs).astype(np.uint8).tobytes()
