"""Cascade information reconciliation (Brassard & Salvail, 1993)."""

from __future__ import annotations

import numpy as np


def parity(bits: np.ndarray, lo: int, hi: int) -> int:
    """Parity of ``bits[lo:hi]``."""
    return int(bits[lo:hi].sum() & 1)


def _binary_search(
    alice: np.ndarray, bob: np.ndarray, idx: np.ndarray, counter: list[int]
) -> int:
    """Locate one error inside a block with a known parity mismatch.

    ``idx`` lists positions of the block; every parity query leaks one bit.
    Returns the global position of the flipped bit.
    """
    lo, hi = 0, idx.size
    while hi - lo > 1:
        mid = (lo + hi) // 2
        left = idx[lo:mid]
        counter[0] += 1
        if (alice[left].sum() - bob[left].sum()) & 1:
            hi = mid
        else:
            lo = mid
    return int(idx[lo])


def cascade(
    alice: np.ndarray,
    bob: np.ndarray,
    qber: float,
    rng: np.random.Generator,
    passes: int = 4,
) -> tuple[np.ndarray, int, int]:
    """Correct Bob's key towards Alice's.

    Returns ``(corrected_bob, n_errors_fixed, parities_disclosed)``.
    Pass 1 uses blocks of ``~0.73/qber`` bits; each later pass doubles the
    block size after a random permutation, and corrections trigger the
    classic back-tracking through earlier passes.
    """
    alice = np.asarray(alice, dtype=np.int64)
    bob = np.array(bob, dtype=np.int64)
    n = alice.size
    if n == 0:
        return bob, 0, 0
    k1 = int(max(4, min(n, np.ceil(0.73 / max(qber, 1e-3)))))
    counter = [0]
    fixed = 0
    blocks_by_pass: list[list[np.ndarray]] = []

    def blocks_of(perm: np.ndarray, k: int) -> list[np.ndarray]:
        return [perm[i:i + k] for i in range(0, n, k)]

    def flip(pos: int, upto: int) -> None:
        """Flip ``pos`` and re-check blocks of earlier passes containing it."""
        nonlocal fixed
        bob[pos] ^= 1
        fixed += 1
        for p in range(upto):
            for blk in blocks_by_pass[p]:
                if pos in blk:
                    counter[0] += 1
                    if (alice[blk].sum() - bob[blk].sum()) & 1:
                        flip(_binary_search(alice, bob, blk, counter), p)
                    break

    k = k1
    for p in range(passes):
        perm = np.arange(n) if p == 0 else rng.permutation(n)
        blocks = blocks_of(perm, k)
        blocks_by_pass.append(blocks)
        for blk in blocks:
            counter[0] += 1
            if (alice[blk].sum() - bob[blk].sum()) & 1:
                flip(_binary_search(alice, bob, blk, counter), p)
        k = min(n, k * 2)
    return bob, fixed, counter[0]
