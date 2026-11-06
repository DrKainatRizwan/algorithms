"""Gradient-level defences applied by the client before sharing an update."""
from __future__ import annotations

import numpy as np

Grads = dict[str, np.ndarray]


def _norm(g: Grads) -> float:
    return float(np.sqrt(sum((v ** 2).sum() for v in g.values())))


def clip_and_noise(g: Grads, clip: float, sigma: float, rng: np.random.Generator) -> Grads:
    """Clip global L2 norm to ``clip`` then add ``N(0, (sigma*clip)^2)`` noise (DP-SGD style)."""
    scale = min(1.0, clip / (_norm(g) + 1e-12))
    return {k: v * scale + sigma * clip * rng.standard_normal(v.shape) for k, v in g.items()}


def top_k_sparsify(g: Grads, keep_fraction: float) -> Grads:
    """Keep only the globally largest-magnitude ``keep_fraction`` of coordinates."""
    flat = np.concatenate([v.ravel() for v in g.values()])
    k = max(1, int(round(keep_fraction * flat.size)))
    thr = np.partition(np.abs(flat), flat.size - k)[flat.size - k]
    return {key: np.where(np.abs(v) >= thr, v, 0.0) for key, v in g.items()}


def quantize(g: Grads, levels: int) -> Grads:
    """Uniform symmetric quantisation to ``levels`` values per tensor."""
    out = {}
    for k, v in g.items():
        m = np.abs(v).max()
        if m == 0:
            out[k] = v.copy(); continue
        step = 2 * m / (levels - 1)
        out[k] = np.round(v / step) * step
    return out


def apply_defense(g: Grads, name: str, strength: float, rng: np.random.Generator) -> Grads:
    """Dispatch by name: ``none``, ``noise`` (sigma), ``sparsify`` (keep fraction), ``quantize`` (levels)."""
    if name == "none":
        return {k: v.copy() for k, v in g.items()}
    if name == "noise":
        return clip_and_noise(g, clip=max(_norm(g), 1e-12), sigma=strength, rng=rng)
    if name == "sparsify":
        return top_k_sparsify(g, strength)
    if name == "quantize":
        return quantize(g, int(strength))
    raise ValueError(f"unknown defence {name!r}")
