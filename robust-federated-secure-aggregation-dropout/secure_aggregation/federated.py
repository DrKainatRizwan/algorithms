"""Federated logistic regression driven by (secure) aggregation.

Three server-side aggregators are provided:

``plain``     mean of clipped client updates (no privacy, reference).
``secure``    mean obtained from the secure-aggregation protocol; the server
              only learns the sum of the surviving clients' updates.
``bucketed``  clients are shuffled into buckets, each bucket is securely
              summed, and the server takes the coordinate-wise median of the
              bucket means.  This keeps individual updates hidden while
              limiting the influence of a minority of Byzantine clients.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .protocol import AbortError, STAGES, run_protocol
from .quantize import clip_l2, decode, encode, max_safe_clients


@dataclass
class Dataset:
    """Client shards plus a held-out test set."""

    shards: list[tuple[np.ndarray, np.ndarray]]
    test: tuple[np.ndarray, np.ndarray]
    w_true: np.ndarray


@dataclass
class FLConfig:
    """Hyper-parameters of a federated training run."""

    rounds: int = 25
    local_steps: int = 5
    lr: float = 0.5
    server_lr: float = 1.0
    clip: float = 1.0
    scale: float = 2.0**16
    aggregator: str = "secure"        # plain | secure | bucketed
    n_buckets: int = 5
    threshold_frac: float = 0.5
    dropout_rate: float = 0.0
    byzantine_frac: float = 0.0
    attack_gain: float = 5.0
    seed: int = 0


def make_dataset(n_clients: int = 30, n_per_client: int = 60, dim: int = 10,
                 shift: float = 0.5, seed: int = 0) -> Dataset:
    """Synthetic logistic data with per-client covariate shift (non-IID)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=dim)
    w /= np.linalg.norm(w) / 3.0

    def draw(n: int, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, dim)) + mu
        p = 1.0 / (1.0 + np.exp(-x @ w))
        return x, (rng.random(n) < p).astype(np.float64)

    shards = [draw(n_per_client, shift * rng.normal(size=dim)) for _ in range(n_clients)]
    return Dataset(shards, draw(2000, np.zeros(dim)), w)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def local_update(w: np.ndarray, x: np.ndarray, y: np.ndarray, steps: int,
                 lr: float) -> np.ndarray:
    """Full-batch gradient descent from ``w``; returns ``w_local - w``."""
    cur = w.copy()
    for _ in range(steps):
        cur -= lr * x.T @ (sigmoid(x @ cur) - y) / len(y)
    return cur - w


def accuracy(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Classification accuracy of the linear model ``w``."""
    return float(np.mean((x @ w > 0) == (y > 0.5)))


def log_loss(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy of the linear model ``w``."""
    z = x @ w
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _secure_mean(updates: dict[int, np.ndarray], cfg: FLConfig, rng: random.Random,
                 round_seed: int) -> tuple[np.ndarray | None, int]:
    """Secure mean of ``updates``; clients may drop at a random stage."""
    ids = sorted(updates)
    if len(ids) < 2:
        return None, 0
    if len(ids) > max_safe_clients(cfg.clip, cfg.scale):
        raise OverflowError("too many clients for the ring size")
    drops: dict[str, set[int]] = {s: set() for s in STAGES}
    for u in ids:
        if rng.random() < cfg.dropout_rate:
            drops[rng.choice(STAGES)].add(u)
    t = max(2, int(np.ceil(cfg.threshold_frac * len(ids))))
    enc = {u: encode(updates[u], cfg.scale) for u in ids}
    try:
        res = run_protocol(enc, t, drops, seed=round_seed)
    except AbortError:
        return None, 0
    return decode(res.total, cfg.scale) / len(res.survivors), len(res.survivors)


def aggregate(updates: dict[int, np.ndarray], cfg: FLConfig, rng: random.Random,
              round_seed: int) -> np.ndarray | None:
    """Combine client updates according to ``cfg.aggregator``."""
    if cfg.aggregator == "plain":
        kept = [u for u in sorted(updates) if rng.random() >= cfg.dropout_rate]
        return np.mean([updates[u] for u in kept], axis=0) if kept else None
    if cfg.aggregator == "secure":
        return _secure_mean(updates, cfg, rng, round_seed)[0]
    if cfg.aggregator == "bucketed":
        ids = sorted(updates)
        rng.shuffle(ids)
        buckets = [ids[i::cfg.n_buckets] for i in range(cfg.n_buckets)]
        means = []
        for k, b in enumerate(buckets):
            m, _ = _secure_mean({u: updates[u] for u in b}, cfg, rng, round_seed * 97 + k)
            if m is not None:
                means.append(m)
        return np.median(means, axis=0) if means else None
    raise ValueError(f"unknown aggregator {cfg.aggregator!r}")


def train(data: Dataset, cfg: FLConfig) -> dict[str, object]:
    """Run federated training and return the final model and metrics."""
    n = len(data.shards)
    dim = data.shards[0][0].shape[1]
    rng = random.Random(cfg.seed)
    n_bad = int(cfg.byzantine_frac * n)
    byz = set(range(n_bad))
    w = np.zeros(dim)
    history = []
    skipped = 0
    for r in range(cfg.rounds):
        updates: dict[int, np.ndarray] = {}
        for i, (x, y) in enumerate(data.shards):
            delta = local_update(w, x, y, cfg.local_steps, cfg.lr)
            if i in byz:
                delta = -cfg.attack_gain * delta
            updates[i + 1] = clip_l2(delta, cfg.clip)
        agg = aggregate(updates, cfg, rng, cfg.seed * 1000 + r)
        if agg is None:
            skipped += 1
        else:
            w = w + cfg.server_lr * agg
        history.append(accuracy(w, *data.test))
    return {"w": w, "accuracy": history[-1], "loss": log_loss(w, *data.test),
            "history": history, "skipped_rounds": skipped}
