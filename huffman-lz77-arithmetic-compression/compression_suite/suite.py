"""Unified container format and benchmarking across all codecs."""
from __future__ import annotations

import time
from typing import Callable

from . import arithmetic, huffman, lz77
from .entropy import entropy

CODECS: dict[str, tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]]] = {
    "huffman": (huffman.compress, huffman.decompress),
    "lz77": (lz77.compress, lz77.decompress),
    "arith0": (lambda d: arithmetic.compress(d, 0), arithmetic.decompress),
    "arith1": (lambda d: arithmetic.compress(d, 1), arithmetic.decompress),
}
_IDS = {name: i + 1 for i, name in enumerate(CODECS)}
_NAMES = {v: k for k, v in _IDS.items()}
STORED = 0


def pack(data: bytes, method: str = "auto") -> bytes:
    """Compress with one codec, or pick the smallest (falling back to stored)."""
    names = list(CODECS) if method == "auto" else [method]
    best = bytes([STORED]) + data
    for name in names:
        cand = bytes([_IDS[name]]) + CODECS[name][0](data)
        if len(cand) < len(best):
            best = cand
    return best


def unpack(blob: bytes) -> bytes:
    """Invert :func:`pack`."""
    if not blob:
        raise ValueError("empty container")
    tag = blob[0]
    if tag == STORED:
        return blob[1:]
    return CODECS[_NAMES[tag]][1](blob[1:])


def benchmark(datasets: dict[str, bytes]) -> list[dict]:
    """Time every codec on every dataset and verify round trips."""
    rows = []
    for dname, data in datasets.items():
        for cname, (comp, decomp) in CODECS.items():
            t0 = time.perf_counter()
            blob = comp(data)
            t1 = time.perf_counter()
            ok = decomp(blob) == data
            t2 = time.perf_counter()
            rows.append({
                "dataset": dname, "codec": cname, "n": len(data), "size": len(blob),
                "ratio": len(blob) / max(1, len(data)), "bpc": 8 * len(blob) / max(1, len(data)),
                "entropy": entropy(data), "enc_s": t1 - t0, "dec_s": t2 - t1, "ok": ok,
            })
    return rows
