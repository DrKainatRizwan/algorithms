"""FM-index: backward search over a wavelet-tree-encoded BWT with sampled suffix array."""
from __future__ import annotations
from .bwt import suffix_array, bwt_from_sa, _terminated
from .wavelet import WaveletTree


class FMIndex:
    """Self-index supporting count, locate and extract.

    Parameters
    ----------
    text: bytes without 0x00.
    sa_sample: keep every ``sa_sample``-th text position's suffix-array entry; locate
        walks LF until it lands on a sampled row (space/time trade-off).
    """

    def __init__(self, text: bytes, sa_sample: int = 8) -> None:
        if sa_sample < 1:
            raise ValueError("sa_sample must be >= 1")
        t = _terminated(text)
        self.n = len(t)
        self.sa_sample = sa_sample
        sa = suffix_array(t)
        self.last = bwt_from_sa(t, sa)
        self.wt = WaveletTree(self.last)
        counts = [0] * 257
        for c in self.last:
            counts[c + 1] += 1
        self.C = [0] * 257  # C[c] = number of symbols < c
        for c in range(256):
            self.C[c + 1] = self.C[c] + counts[c + 1]
        # sampled SA: row -> text position, for positions divisible by sa_sample
        self.samples = {row: p for row, p in enumerate(sa) if p % sa_sample == 0}
        # sampled inverse SA for extract
        self.isa_samples = {p: row for row, p in enumerate(sa) if p % sa_sample == 0}

    def _lf(self, row: int) -> int:
        c = self.last[row]
        return self.C[c] + self.wt.rank(c, row)

    def search_range(self, pattern: bytes) -> tuple[int, int]:
        """Half-open SA interval ``[lo, hi)`` of suffixes prefixed by ``pattern``."""
        lo, hi = 0, self.n
        for c in reversed(pattern):
            if c == 0:
                return (0, 0)
            base = self.C[c]
            lo = base + self.wt.rank(c, lo)
            hi = base + self.wt.rank(c, hi)
            if lo >= hi:
                return (0, 0)
        return lo, hi

    def count(self, pattern: bytes) -> int:
        """Number of occurrences of ``pattern`` in the text."""
        lo, hi = self.search_range(pattern)
        if not pattern:
            return self.n  # includes the empty suffix at each position + end
        return hi - lo

    def _resolve(self, row: int) -> int:
        steps = 0
        while row not in self.samples:
            row = self._lf(row)
            steps += 1
        return self.samples[row] + steps

    def locate(self, pattern: bytes) -> list[int]:
        """Sorted text offsets of all occurrences of ``pattern``."""
        if not pattern:
            return list(range(self.n))
        lo, hi = self.search_range(pattern)
        return sorted(self._resolve(r) for r in range(lo, hi))

    def extract(self, start: int, length: int) -> bytes:
        """Recover ``text[start:start+length]`` from the index alone."""
        if start < 0 or length < 0 or start + length > self.n - 1:
            raise IndexError("extract range outside text")
        if length == 0:
            return b""
        end = start + length  # exclusive; row of suffix at `end`
        k = self.sa_sample
        p = min(((end + k - 1) // k) * k, self.n - 1)  # nearest sampled position >= end
        row = 0 if p == self.n - 1 else self.isa_samples[p]  # row 0 is the sentinel suffix
        out = bytearray()
        for _ in range(p - start):
            out.append(self.last[row])  # char preceding the suffix at position p
            row = self._lf(row)
        out.reverse()
        return bytes(out[:length])

    def size_report(self) -> dict[str, int]:
        """Rough size accounting in bytes of the main components."""
        wt_bits = self.n * max(1, (len(self.wt.alphabet) - 1).bit_length())
        return {"text": self.n, "wavelet_bits": wt_bits // 8,
                "sa_samples": len(self.samples) * 8, "full_sa": self.n * 8}
