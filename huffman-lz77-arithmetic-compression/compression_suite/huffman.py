"""Canonical Huffman coding for byte streams."""
from __future__ import annotations

import heapq
from collections import Counter

from .bitio import BitReader, BitWriter


def code_lengths(freqs: dict[int, int]) -> dict[int, int]:
    """Optimal prefix-code lengths via the two-queue/heap Huffman construction.

    A single-symbol alphabet receives length 1 so that it still costs one bit.
    """
    if not freqs:
        return {}
    if len(freqs) == 1:
        return {next(iter(freqs)): 1}
    # heap items: (weight, tiebreak, [symbols])
    heap = [(w, s, [s]) for s, w in sorted(freqs.items())]
    heapq.heapify(heap)
    lengths = {s: 0 for s in freqs}
    tie = max(freqs) + 1
    while len(heap) > 1:
        w1, _, s1 = heapq.heappop(heap)
        w2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] += 1
        heapq.heappush(heap, (w1 + w2, tie, s1 + s2))
        tie += 1
    return lengths


def limit_lengths(lengths: dict[int, int], max_len: int) -> dict[int, int]:
    """Clamp lengths to ``max_len`` and repair the Kraft inequality.

    Over-long codes are shortened, then the least-frequent-looking (longest)
    codes are lengthened greedily until the Kraft sum is at most one.
    """
    out = {s: min(l, max_len) for s, l in lengths.items()}
    unit = 1 << max_len
    kraft = sum(unit >> l for l in out.values())
    while kraft > unit:
        # lengthen the longest code that is still below max_len
        cand = [s for s, l in out.items() if l < max_len]
        s = max(cand, key=lambda x: (out[x], x))
        kraft -= unit >> out[s]
        out[s] += 1
        kraft += unit >> out[s]
    return out


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Assign canonical codes: ``symbol -> (code, length)``."""
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    prev = 0
    for sym, l in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= l - prev
        codes[sym] = (code, l)
        code += 1
        prev = l
    return codes


class HuffmanDecoder:
    """Canonical decoder walking one bit at a time using per-length tables."""

    def __init__(self, lengths: dict[int, int]) -> None:
        self.max_len = max(lengths.values())
        self.first_code = [0] * (self.max_len + 2)
        self.first_index = [0] * (self.max_len + 2)
        self.count = [0] * (self.max_len + 2)
        self.symbols = [s for s, _ in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))]
        for l in lengths.values():
            self.count[l] += 1
        code = idx = 0
        for l in range(1, self.max_len + 1):
            self.first_code[l] = code
            self.first_index[l] = idx
            code = (code + self.count[l]) << 1
            idx += self.count[l]

    def decode(self, reader: BitReader) -> int:
        code = 0
        for l in range(1, self.max_len + 1):
            code = (code << 1) | reader.read_bit()
            off = code - self.first_code[l]
            if 0 <= off < self.count[l]:
                return self.symbols[self.first_index[l] + off]
        raise ValueError("invalid Huffman code")


def write_lengths(w: BitWriter, lengths: dict[int, int], alphabet: int) -> None:
    """Serialise lengths for ``alphabet`` symbols, 5 bits each (0 = unused)."""
    for s in range(alphabet):
        w.write_bits(lengths.get(s, 0), 5)


def read_lengths(r: BitReader, alphabet: int) -> dict[int, int]:
    out = {}
    for s in range(alphabet):
        l = r.read_bits(5)
        if l:
            out[s] = l
    return out


def compress(data: bytes) -> bytes:
    """Huffman-compress ``data``. Header: 32-bit length + 256 five-bit lengths."""
    w = BitWriter()
    w.write_bits(len(data), 32)
    if not data:
        return w.getvalue()
    lengths = limit_lengths(code_lengths(dict(Counter(data))), 31)
    write_lengths(w, lengths, 256)
    codes = canonical_codes(lengths)
    for b in data:
        c, l = codes[b]
        w.write_bits(c, l)
    return w.getvalue()


def decompress(blob: bytes) -> bytes:
    r = BitReader(blob)
    n = r.read_bits(32)
    if n == 0:
        return b""
    dec = HuffmanDecoder(read_lengths(r, 256))
    return bytes(dec.decode(r) for _ in range(n))


def expected_length(freqs: dict[int, int], lengths: dict[int, int]) -> float:
    """Average code length in bits per symbol under the empirical distribution."""
    n = sum(freqs.values())
    return sum(freqs[s] * lengths[s] for s in freqs) / n
