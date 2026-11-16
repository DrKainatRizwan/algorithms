"""Fiat-Shamir transcript: domain-separated, length-prefixed hashing."""
from __future__ import annotations

import hashlib
from typing import Iterable


class Transcript:
    """Running hash of labelled messages from which challenges are derived."""

    def __init__(self, domain: str) -> None:
        self._h = hashlib.sha256()
        self._absorb_bytes(b"domain", domain.encode())

    def _absorb_bytes(self, label: bytes, data: bytes) -> None:
        self._h.update(len(label).to_bytes(4, "big") + label)
        self._h.update(len(data).to_bytes(4, "big") + data)

    def absorb_int(self, label: str, x: int) -> None:
        n = max(1, (x.bit_length() + 7) // 8)
        self._absorb_bytes(label.encode(), x.to_bytes(n, "big"))

    def absorb_ints(self, label: str, xs: Iterable[int]) -> None:
        xs = list(xs)
        self.absorb_int(label + ".len", len(xs))
        for i, x in enumerate(xs):
            self.absorb_int(f"{label}[{i}]", x)

    def challenge(self, q: int, label: str = "challenge") -> int:
        """Squeeze a challenge in Z_q (with 128 bits of slack to kill bias)."""
        h = self._h.copy()
        h.update(label.encode())
        need = (q.bit_length() + 128 + 7) // 8
        out = b""
        ctr = 0
        while len(out) < need:
            hh = h.copy()
            hh.update(ctr.to_bytes(4, "big"))
            out += hh.digest()
            ctr += 1
        return int.from_bytes(out[:need], "big") % q

    def fork(self) -> "Transcript":
        t = Transcript.__new__(Transcript)
        t._h = self._h.copy()
        return t
