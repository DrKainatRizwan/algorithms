"""Synthetic network-flow generator with labelled attack families."""

from __future__ import annotations

import numpy as np

TRAFFIC_FEATURES = ["duration", "src_bytes", "dst_bytes", "packets", "dst_ports", "syn_rate", "failed_logins"]
ATTACKS = ("portscan", "dos", "exfiltration", "bruteforce")


def _normal(n: int, rng: np.random.Generator) -> np.ndarray:
    duration = rng.lognormal(0.5, 0.8, n)
    src = rng.lognormal(6.5, 1.0, n)
    dst = src * rng.lognormal(1.0, 0.5, n)
    packets = np.maximum(1, (src + dst) / 900 + rng.poisson(3, n))
    ports = 1 + rng.poisson(0.5, n)
    syn = rng.beta(1.5, 20, n)
    failed = rng.poisson(0.05, n)
    return np.column_stack([duration, src, dst, packets, ports, syn, failed])


def _attack(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    base = _normal(n, rng)
    if kind == "portscan":
        base[:, 4] = rng.integers(40, 400, n)
        base[:, 1] = rng.lognormal(4.0, 0.3, n)
        base[:, 2] = rng.lognormal(3.5, 0.3, n)
        base[:, 5] = rng.beta(8, 3, n)
    elif kind == "dos":
        base[:, 3] = rng.lognormal(9.0, 0.4, n)
        base[:, 5] = rng.beta(12, 2, n)
        base[:, 2] = rng.lognormal(3.0, 0.5, n)
    elif kind == "exfiltration":
        base[:, 0] = rng.lognormal(3.5, 0.4, n)
        base[:, 1] = rng.lognormal(13.0, 0.5, n)
        base[:, 2] = rng.lognormal(5.0, 0.5, n)
    elif kind == "bruteforce":
        base[:, 6] = rng.integers(8, 40, n)
        base[:, 3] = base[:, 3] + rng.poisson(40, n)
    else:
        raise ValueError(f"unknown attack {kind!r}")
    return base


def make_traffic(n_normal: int = 2000, n_attack: int = 100, seed: int = 0,
                 attacks: tuple[str, ...] = ATTACKS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, kind)``; ``y`` is 1 for attacks, ``kind`` an int family id (-1 = normal).

    Rows are shuffled deterministically.
    """
    rng = np.random.default_rng(seed)
    parts = [_normal(n_normal, rng)]
    y = [np.zeros(n_normal, dtype=int)]
    kind = [np.full(n_normal, -1)]
    split = np.array_split(np.arange(n_attack), len(attacks)) if attacks else []
    for k, (name, part) in enumerate(zip(attacks, split)):
        if part.size:
            parts.append(_attack(name, part.size, rng))
            y.append(np.ones(part.size, dtype=int))
            kind.append(np.full(part.size, k))
    X, y, kind = np.vstack(parts), np.concatenate(y), np.concatenate(kind)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], kind[perm]
