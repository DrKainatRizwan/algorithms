"""Rainbow tables: time-memory trade-off for inverting a hash over a small
password space (Oechslin, 2003)."""
from __future__ import annotations

import random
from typing import Callable, Optional

__all__ = ["RainbowTable"]


class RainbowTable:
    """Rainbow table over fixed-length strings drawn from ``alphabet``.

    Parameters
    ----------
    hash_fn:
        Maps password bytes to a digest.
    alphabet, length:
        Define the password space ``alphabet**length``.
    chains, chain_len:
        Table geometry.  Storage is ``chains`` (start, end) pairs.
    seed:
        Seeds the choice of chain start points.
    """

    def __init__(self, hash_fn: Callable[[bytes], bytes], alphabet: str, length: int,
                 chains: int, chain_len: int, seed: int = 0) -> None:
        self.hash_fn = hash_fn
        self.alphabet = alphabet
        self.length = length
        self.chain_len = chain_len
        self.space = len(alphabet) ** length
        self.table: dict[int, int] = {}
        rng = random.Random(seed)
        starts = rng.sample(range(self.space), min(chains, self.space))
        for s in starts:
            self.table.setdefault(self._walk(s, 0, chain_len - 1), s)

    # -- helpers --------------------------------------------------------------
    def index_to_pw(self, idx: int) -> bytes:
        """Decode an integer in ``[0, space)`` into a password."""
        base = len(self.alphabet)
        chars = []
        for _ in range(self.length):
            idx, r = divmod(idx, base)
            chars.append(self.alphabet[r])
        return "".join(reversed(chars)).encode()

    def reduce(self, digest: bytes, column: int) -> int:
        """Column-dependent reduction from digest to password index."""
        return (int.from_bytes(digest, "big") + column) % self.space

    def _walk(self, idx: int, first: int, last: int) -> int:
        """Apply columns ``first..last-1`` of hash/reduce starting at ``idx``."""
        for col in range(first, last):
            idx = self.reduce(self.hash_fn(self.index_to_pw(idx)), col)
        return idx

    # -- queries --------------------------------------------------------------
    def lookup(self, digest: bytes) -> Optional[bytes]:
        """Return a password hashing to ``digest`` or ``None``."""
        for start_col in range(self.chain_len - 2, -1, -1):
            idx = self.reduce(digest, start_col)
            end = self._walk(idx, start_col + 1, self.chain_len - 1)
            start = self.table.get(end)
            if start is None:
                continue
            cand = self._walk(start, 0, start_col)
            pw = self.index_to_pw(cand)
            if self.hash_fn(pw) == digest:
                return pw
        return None

    def __len__(self) -> int:
        return len(self.table)
