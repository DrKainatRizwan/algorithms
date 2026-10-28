"""Synthetic rule-set and traffic generation (seeded, ClassBench-flavoured)."""
from __future__ import annotations

import numpy as np

from .rules import ANY_PROTO, MAX_IP, Packet, Rule, prefix_mask

_PREFIX_LENS = np.array([0, 8, 16, 24, 28, 32])
_PREFIX_P = np.array([0.05, 0.1, 0.25, 0.3, 0.1, 0.2])
_WELL_KNOWN = [22, 25, 53, 80, 123, 143, 443, 3306, 5432, 8080]


def _prefix(rng: np.random.Generator, pool: np.ndarray) -> tuple[int, int]:
    ln = int(rng.choice(_PREFIX_LENS, p=_PREFIX_P))
    base = int(rng.choice(pool)) if rng.random() < 0.7 else int(rng.integers(0, MAX_IP + 1))
    return base & prefix_mask(ln), ln


def _ports(rng: np.random.Generator) -> tuple[int, int]:
    u = rng.random()
    if u < 0.4:
        return 0, 65535
    if u < 0.75:
        p = int(rng.choice(_WELL_KNOWN))
        return p, p
    if u < 0.9:
        return 1024, 65535
    lo = int(rng.integers(0, 60000))
    return lo, min(65535, lo + int(rng.integers(1, 5000)))


def generate_rules(n: int, seed: int = 0, default_rule: bool = True) -> list[Rule]:
    """Random rule set whose addresses cluster around a few subnets.

    The last rule is a catch-all deny when ``default_rule`` is set.
    """
    rng = np.random.default_rng(seed)
    pool = rng.integers(0, MAX_IP + 1, size=max(4, n // 20), dtype=np.int64)
    rules = []
    count = n - 1 if default_rule else n
    for i in range(count):
        proto = int(rng.choice([6, 17, 1, ANY_PROTO], p=[0.5, 0.25, 0.05, 0.2]))
        rules.append(Rule(_prefix(rng, pool), _prefix(rng, pool), _ports(rng),
                          _ports(rng), proto,
                          "allow" if rng.random() < 0.5 else "deny", i))
    if default_rule:
        rules.append(Rule((0, 0), (0, 0), (0, 65535), (0, 65535), ANY_PROTO,
                          "deny", count))
    return rules


def generate_packets(rules: list[Rule], n: int, seed: int = 1,
                     hit_fraction: float = 0.8) -> list[Packet]:
    """Packets, most of them sampled from inside random rules' hyper-rectangles."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if rules and rng.random() < hit_fraction:
            rg = rules[int(rng.integers(len(rules)))].ranges()
            vals = [int(rng.integers(lo, hi + 1)) for lo, hi in rg]
        else:
            vals = [int(rng.integers(0, MAX_IP + 1)), int(rng.integers(0, MAX_IP + 1)),
                    int(rng.integers(0, 65536)), int(rng.integers(0, 65536)),
                    int(rng.choice([1, 6, 17]))]
        out.append(Packet(*vals))
    return out
